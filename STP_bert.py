import gc
import torch
import torch.nn as nn
from tqdm.auto import tqdm

def token_gradients(model, init, device):
    embed_weights = model.get_input_embeddings().weight ##vicuna
    one_hot = torch.zeros(
        init.shape[0],# len_of_token
        embed_weights.shape[0],# size_of_vocab
        device = model.device,
        dtype = embed_weights.dtype
    )
    #print(embed_weights.dtype)
    one_hot.scatter_(
        1, 
        init.unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()#finish create one hot
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)#batch,len,weight
    logits = model(inputs_embeds=input_embeds).logits
    shift_logits = logits[..., -1, :].contiguous()
    loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), torch.tensor([2]).to(device))  ### token as added one, end token.
    loss.backward()
    return one_hot.grad.clone()#computate grad only

def sample_control(tokenizer, model_bert, tokenizer_bert, device, init, grad, batch_size, topk=5, topk_semanteme=10):

    top_indices=torch.zeros(grad.shape[0], topk, dtype=init.dtype).to(device)
    for i in range(1,init.shape[0]):
        arr1 = init[0:i]
        arr2 = init[i+1:]
        original_token = tokenizer.decode([init[i]])
        arr1_string = tokenizer.decode(arr1)
        arr2_string = tokenizer.decode(arr2)
        bert_string = arr1_string + "[MASK]" + arr2_string

        inputs = tokenizer_bert(bert_string, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model_bert(**inputs).logits
        # retrieve index of [MASK]
        mask_token_index = (inputs.input_ids == tokenizer_bert.mask_token_id)[0].nonzero(as_tuple=True)[0]
        vals, indices = logits[0, mask_token_index].topk(k=topk_semanteme, dim=1, largest=True, sorted=True)
        indices = indices.squeeze(0)

        temp=[]
        similar = []
        for j in range(topk_semanteme):
            topk_element = tokenizer_bert.decode([indices[j]]).replace("##", "")
            # print(topk_element)
            # print("---------------")
            if topk_element == original_token:
                # print(original_token)
                continue
            # just use the first token
            topk_element_id = tokenizer.encode(topk_element)[1]
            temp.append(((grad)[i][topk_element_id], topk_element_id))
        temp.sort()
        
        for j in range(topk): # in topksemanteme choose topk to use
            similar.append(temp[j][1])
        top_indices[i]=torch.tensor(similar)
    # print(top_indices)

    """
    [[1,5,9,...],
    [1,5,9,...],
    [1,5,9,...],
    ...]
    """
    #thinking: if make adv directly,[0,0,0,1,0,0,...] will - grda*learning_rate,trending to another one hot vector.

    #[!,!,!,...]
    original_control_toks = init.repeat(batch_size, 1)
    """
    [[!,!,!,...],
    [!,!,!,...],
    [!,!,!,...],
    ...]batch_size
    ! is index of vocab
    """
    new_token_pos = torch.arange(      ##don't touch <s>,gpt2 has no <s> so from 0
        1, 
        len(init), 
        (len(init)-1) / batch_size,      ## this place has to change as well
        device=grad.device
    ).type(torch.int64)
    """
    [...,...,...]
    """

    new_token_val = torch.gather(
        top_indices[new_token_pos], 1, 
        torch.randint(0, topk, (batch_size, 1),device=grad.device)
    )
    """
    [[top_indices],
    [124],
    [124],
    [124],
    ...]
    """
    new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)
    """
    self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
    self[i][index[i][j]] = src[i][j]  # if dim == 1
    self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2

    every control[!,!,!,...] be changed one position.
    """
    return new_control_toks

def step_bert(model, tokenizer, model_bert, tokenizer_bert, device, init, batch_size=1024, topk=5, topk_semanteme=10):

    main_device = model.device
    # Aggregate gradients.universal needs to add all gradient.
    grad = token_gradients(model, init, device)

    with torch.no_grad():
        control_cand = sample_control(tokenizer, model_bert, tokenizer_bert, device, init, grad, batch_size, topk, topk_semanteme)
    del grad ; gc.collect()

    # Search
    loss = torch.zeros( batch_size).to(main_device)
    prob = torch.zeros( batch_size).to(main_device)
    with torch.no_grad():
        for j, cand in enumerate(control_cand):
            full_input=cand.unsqueeze(0)
            logits = model(full_input).logits
            shift_logits = logits[..., -1, :].contiguous()
            loss[j] = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), torch.tensor([2]).to(device))  ###token as added one,end token
            prob[j] = torch.nn.functional.softmax(logits[0,-1,:],dim=0)[2] #end token
        min_idx = loss.argmin()
        next_control, cand_loss, cand_prob= control_cand[min_idx], loss[min_idx], prob[min_idx]

    del  loss,prob ; gc.collect()
    return next_control,cand_loss,cand_prob