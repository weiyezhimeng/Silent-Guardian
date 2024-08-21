import gc
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from sentence_transformers.util import (semantic_search,
                                        dot_score,
                                        normalize_embeddings)

def token_gradients_agg(model_1, model_2, init):
    embed_weights_1 = model_1.get_input_embeddings().weight
    embed_weights_2 = model_2.get_input_embeddings().weight ##vicuna
    one_hot = torch.zeros(
        init.shape[0],# len_of_token
        embed_weights_1.shape[0],# size_of_vocab
        device=model_1.device,
        dtype=embed_weights_1.dtype
    )
    #print(embed_weights.dtype)
    one_hot.scatter_(
        1, 
        init.unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model_1.device, dtype=embed_weights_1.dtype)
    )
    one_hot.requires_grad_()#finish create one hot

    input_embeds_1 = (one_hot @ embed_weights_1).unsqueeze(0)#batch,len,weight
    logits_1 = model_1(inputs_embeds = input_embeds_1).logits
    shift_logits_1 = logits_1[..., -1, :].contiguous()
    loss_1 = nn.CrossEntropyLoss()(shift_logits_1.view(-1, shift_logits_1.size(-1)), torch.tensor([2]).cuda())  ### token as added one, end token.
    
    input_embeds_2 = (one_hot @ embed_weights_2).unsqueeze(0)#batch,len,weight
    logits_2 = model_2(inputs_embeds = input_embeds_2).logits
    shift_logits_2 = logits_2[..., -1, :].contiguous()
    loss_2 = nn.CrossEntropyLoss()(shift_logits_2.view(-1, shift_logits_2.size(-1)), torch.tensor([2]).cuda())  ### token as added one, end token.

    loss = (loss_1 + loss_2)/2
    loss.backward()
    return one_hot.grad.clone()#computate grad only

def sample_control_agg(model_1, init, grad, batch_size,topk=5, topk_semanteme=10):
    curr_embeds_1 = model_1.get_input_embeddings()(init.unsqueeze(0))
    curr_embeds_1 = torch.nn.functional.normalize(curr_embeds_1,dim=2)               # queries

    embedding_matrix_1 = model_1.get_input_embeddings().weight
    embedding_matrix_1 = normalize_embeddings(embedding_matrix_1)      # corpus

    top_indices = torch.zeros(grad.shape[0], topk, dtype=init.dtype).to("cuda")
    for i in range(init.shape[0]):
        similar=[]    
        temp=[]                 
        query = curr_embeds_1[0][i]                 #query
        corpus = embedding_matrix_1                #corpus
        hits = semantic_search(query, 
                               corpus, 
                               top_k=topk_semanteme+1,
                               score_function=dot_score)
        for hit in hits:
            for dic in hit[1:]:## don't choose same token as before
                temp.append(((grad)[i][dic["corpus_id"]],dic["corpus_id"]))
        temp.sort()
        for j in range(topk): # in topksemanteme choose topk to use
            similar.append(temp[j][1])
        top_indices[i]=torch.tensor(similar)

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
    new_token_pos = torch.arange(      ##don't touch <s>
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
    [1],
    [1],
    [1],
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

def step_agg(model_1, model_2, init, batch_size=1024, topk=5 ,topk_semanteme=10):
    main_device = model_1.device
    # Aggregate gradients.universal needs to add all gradient.
    grad = token_gradients_agg(model_1, model_2, init)

    with torch.no_grad():
        control_cand = sample_control_agg(model_1, init, grad, batch_size, topk, topk_semanteme)
    del grad ; gc.collect()

    # Search
    loss = torch.zeros(batch_size).to(main_device)
    loss_1 = torch.zeros(batch_size).to(main_device)
    loss_2 = torch.zeros(batch_size).to(main_device)
    prob_1 = torch.zeros(batch_size).to(main_device)
    prob_2 = torch.zeros(batch_size).to(main_device)
    with torch.no_grad():
        for j, cand in enumerate(control_cand):
            full_input = cand.unsqueeze(0)
            logits_1 = model_1(full_input).logits
            shift_logits_1 = logits_1[..., -1, :].contiguous()
            logits_2 = model_2(full_input).logits
            shift_logits_2 = logits_2[..., -1, :].contiguous()
            loss_1[j] = nn.CrossEntropyLoss()(shift_logits_1.view(-1, shift_logits_1.size(-1)), torch.tensor([2]).cuda())  ###token as added one,end token
            loss_2[j] = nn.CrossEntropyLoss()(shift_logits_2.view(-1, shift_logits_2.size(-1)), torch.tensor([2]).cuda())  ###token as added one,end token
            loss[j] = (loss_1[j] + loss_2[j])/2
            prob_1[j] = torch.nn.functional.softmax(logits_1[0,-1,:],dim=0)[2] #end token
            prob_2[j] = torch.nn.functional.softmax(logits_2[0,-1,:],dim=0)[2] #end token
        min_idx = loss.argmin()
        next_control, cand_loss, cand_loss_1, cand_loss_2, cand_prob_1, cand_prob_2 = control_cand[min_idx], loss[min_idx], loss_1[min_idx], loss_2[min_idx], prob_1[min_idx], prob_2[min_idx]

    del loss, loss_1, loss_2, prob_1, prob_2 ; gc.collect()
    return next_control, cand_loss, cand_loss_1, cand_loss_2, cand_prob_1, cand_prob_2