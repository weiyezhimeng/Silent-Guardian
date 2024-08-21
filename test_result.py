import json
import torch
import math
import os
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import Levenshtein
import tqdm
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_path", type=str, default="universal_encoder")
    parser.add_argument("--target_path", type=str, default="vicuna_novel_.json")

    return parser.parse_args()

def final_result_file(target_file):
    filename, _ = os.path.splitext(os.path.basename(target_file))
    final_result_file_name = "{}.json".format(filename)
    return final_result_file_name

def test_result(encoder_path, target_path):
    embed = hub.load(encoder_path)
    path = target_path
    targets = json.load(open(path, 'r'))
    result = []

    with tf.device('/CPU:0'):
        for i, target in tqdm(list(enumerate(targets))):
            cat={
                "loss":0,
                "prob":0,
                "similarity":0,
                "edit":0
                }
            
            loss = target["loss"]
            prob = target["prob"]
            origin = [target["origin"]]
            adv = [target["adv"][3:]]
            cos_similarity = torch.cosine_similarity(torch.tensor(np.array(embed(origin))), torch.tensor(np.array(embed(adv))), dim=1)
            final_similarity = 1-torch.acos(cos_similarity)/math.pi
            edit = Levenshtein.distance(target["origin"], target["adv"][3:])/len(target["origin"])
            similarity = float(final_similarity)

            cat["loss"] = loss
            cat["prob"] = prob
            cat["similarity"] = similarity
            cat["edit"] = edit
            print(cat,flush=True)
            print("=============")
            result.append(cat)
    result_file_name = final_result_file(target_path)
    json_path = os.path.join('test_result', result_file_name)
    json_file = open(json_path, mode='w',encoding='utf-8')
    list_json=json.dumps(result, indent=4,ensure_ascii=False)
    json_file.write(list_json)
    json_file.close()

if __name__=='__main__':
    args = get_args()
    test_result(args.encoder_path, args.target_path)