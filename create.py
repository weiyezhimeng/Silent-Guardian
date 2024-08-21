import torch
import os
import json
import datetime
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BertForMaskedLM
from STP import step
from STP_bert import step_bert
from STP_loss_agg import step_agg
from STP_inistructions import step_instructions
import argparse

class defend_manger:
    def __init__(self, STP, path, bert_path, agg_path, target_file, instructions_file, epoch, batch_size, topk, topk_semanteme):
        self.STP = STP
        self.path = path
        self.bert_path = bert_path
        self.agg_path = agg_path
        self.target_file = target_file
        self.instructions_file = instructions_file
        self.epoch = epoch
        self.batch_size = batch_size
        self.topk = topk
        self.topk_semanteme = topk_semanteme
        self.result_name = self.final_result_file(self.path, self.target_file)
    
    def final_result_file(self, path, target_file):
        current_time = datetime.datetime.now()
        time_str = current_time.strftime("%Y%m%d_%H%M%S")
        filename1, _ = os.path.splitext(os.path.basename(path))
        filename2, _ = os.path.splitext(os.path.basename(target_file))
        final_result_file_name = "{}_{}_{}.json".format(filename1, filename2, time_str)
        return final_result_file_name
    
    def main(self):
        device = "cuda:0"
        tokenizer_path = model_path = self.path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)

        targets = json.load(open(self.target_file, 'r'))
        predictions = []
        for i, target in tqdm(list(enumerate(targets))):
            init_token = torch.tensor(tokenizer.encode(target)).to(device)
            for i in range(self.epoch):
                temp={}
                init_token, loss, prob = step(model, init_token, batch_size=self.batch_size, topk=self.topk, topk_semanteme=self.topk_semanteme)
                temp["origin"] = target
                temp["adv"] = tokenizer.decode(init_token)
                temp["loss"] = float(loss)
                temp["prob"] = float(prob)
                print(temp["adv"], flush=True)
                print(temp["prob"], flush=True)
                print("==============")
                predictions.append(temp)
        if not os.path.exists('commom_method'):
            os.makedirs('commom_method')
        
        json_path = os.path.join('commom_method', self.result_name)
        json_file = open(json_path, mode='w', encoding='utf-8')
        list_json = json.dumps(predictions, indent=4, ensure_ascii=False)
        json_file.write(list_json)
        json_file.close()

    def main_withbert(self):
        device = "cuda:0"
        tokenizer_path = model_path = self.path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device) 

        tokenizer_path_bert = model_path_bert = self.bert_path
        tokenizer_bert = AutoTokenizer.from_pretrained(tokenizer_path_bert)
        model_bert = BertForMaskedLM.from_pretrained(model_path_bert,torch_dtype=torch.float16).to(device) 

        targets = json.load(open(self.target_file, 'r'))
        predictions = []
        for i, target in tqdm(list(enumerate(targets))):
            init_token = torch.tensor(tokenizer.encode(target)).to(device)
            for i in range(self.epoch):
                temp={}
                init_token, loss, prob = step_bert(model, tokenizer, model_bert, tokenizer_bert, device, init_token, batch_size=self.batch_size, topk=self.topk,topk_semanteme=self.topk_semanteme)
                temp["origin"] = target
                temp["adv"] = tokenizer.decode(init_token)
                temp["loss"] = float(loss)
                temp["prob"] = float(prob)
                print(temp["adv"], flush=True)
                print(temp["prob"], flush=True)
                print("==============")
                predictions.append(temp)
        if not os.path.exists('with_bert'):
            os.makedirs('with_bert')
        json_path = os.path.join('with_bert', self.result_name)
        json_file = open(json_path, mode='w', encoding='utf-8')
        list_json = json.dumps(predictions, indent=4, ensure_ascii=False)
        json_file.write(list_json)
        json_file.close()

    def main_loss_agg(self):
        device = "cuda:0"
        tokenizer_path_1 = model_path_1 = self.path
        tokenizer_1 = AutoTokenizer.from_pretrained(tokenizer_path_1)
        model_1 = AutoModelForCausalLM.from_pretrained(model_path_1, torch_dtype=torch.float16).to(device) 

        tokenizer_path_2 = model_path_2 = self.agg_path
        # tokenizer_2 = AutoTokenizer.from_pretrained(tokenizer_path_2)
        model_2 = AutoModelForCausalLM.from_pretrained(model_path_2, torch_dtype=torch.float16).to(device) 

        targets = json.load(open(self.target_file, 'r'))
        predictions = []
        for i, target in tqdm(list(enumerate(targets))):
            init_token = torch.tensor(tokenizer_1.encode(target)).to(device)
            for i in range(self.epoch):
                temp={}
                init_token, loss, loss_1, loss_2, prob_1, prob_2 = step_agg(model_1, model_2, init_token, batch_size=self.batch_size, topk=self.topk,topk_semanteme=self.topk_semanteme)
                temp["origin"] = target
                temp["adv"] = tokenizer_1.decode(init_token)
                temp["loss"] = float(loss)
                temp["loss1"] = float(loss_1)
                temp["loss2"] = float(loss_2)
                temp["prob_1"] = float(prob_1)
                temp["prob_2"] = float(prob_2)
                print(temp["adv"], flush=True)
                print(temp["prob_1"], flush=True)
                print(temp["prob_2"], flush=True)
                print("==============")
                predictions.append(temp)
        if not os.path.exists('aggation'):
            os.makedirs('aggation')
        json_path = os.path.join('aggation', self.result_name)
        json_file = open(json_path, mode='w', encoding='utf-8')
        list_json = json.dumps(predictions, indent=4, ensure_ascii=False)
        json_file.write(list_json)
        json_file.close()

    def main_common_with_instruction(self):
        device = "cuda:0"
        tokenizer_path = model_path = self.path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)  

        targets = json.load(open(self.target_file, 'r'))
        instructions = json.load(open(self.instructions_file, 'r'))
        predictions = []
        for i, target in tqdm(list(enumerate(targets))):
            init_token = torch.tensor(tokenizer.encode(target)).to(device)
            for i in range(self.epoch):
                temp={}
                init_token, loss, prob = step_instructions(model, tokenizer, init_token, instructions, batch_size=self.batch_size, topk=self.topk, topk_semanteme=self.topk_semanteme)
                temp["origin"] = target
                temp["adv"] = tokenizer.decode(init_token)
                temp["loss"] = float(loss)
                temp["prob"] = prob
                print(temp["adv"], flush=True)
                print(temp["prob"], flush=True)
                print("==============")
                predictions.append(temp)
        if not os.path.exists('commom_method_instructions'):
            os.makedirs('commom_method_instructions')
        json_path = os.path.join('commom_method_instructions', self.result_name)
        json_file = open(json_path, mode='w', encoding='utf-8')
        list_json = json.dumps(predictions, indent=4, ensure_ascii=False)
        json_file.write(list_json)
        json_file.close()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--STP", type=str, default="STP")
    parser.add_argument("--path", type=str, default="vicuna")
    parser.add_argument("--bert_path", type=str, default="bert")
    parser.add_argument("--agg_path", type=str, default="llama")
    parser.add_argument("--target_file", type=str, default="target.json")
    parser.add_argument("--instructions_file", type=str, default="instructions.json")
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--topk_semanteme", type=int, default=10)

    return parser.parse_args()

if __name__=='__main__':

    args = get_args()
    defend_manger_main = defend_manger(args.STP, args.path, args.bert_path, args.agg_path, args.target_file, args.instructions_file, args.epoch, args.batch_size, args.topk, args.topk_semanteme)
    if args.STP == "STP":
        defend_manger_main.main()

    if args.STP == "STP_bert":
        defend_manger_main.main_withbert()

    if args.STP == "STP_agg":
        defend_manger_main.main_loss_agg()

    if args.STP == "STP_instructions":
        defend_manger_main.main_common_with_instruction()