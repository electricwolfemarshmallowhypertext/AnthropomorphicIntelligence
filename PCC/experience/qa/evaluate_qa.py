## Copyright (c) Microsoft Corporation.
## Licensed under the MIT license.

import argparse
import json
import os

import torch
from datasets import load_dataset, load_from_disk
from model.model import PCC
from torch.cuda.amp import autocast
from tqdm import tqdm
from transformers import AutoTokenizer

from ..dataclass import Config
from .utils import exact_match_score, qa_f1_score


def cal_avg_token(example: dict,
                  lm_tokenizer: AutoTokenizer,
                  dataset: str):
    context = None

    if dataset == "nq":
        context = [text['text'] for text in example['positive_passages']]
        context = "\n\n".join(context)
    elif dataset in ["hotpotqa", "squad", "adqa"]:
        context = example['context']
    else:
        raise NotImplementedError(f"dataset {dataset} not supported!")
    return {"sum_token": len(lm_tokenizer(context)['input_ids'])} 

def run(config: Config):
    dataset = config.dataset
    decoder_model = config.decoder_model
    lm_tokenizer = AutoTokenizer.from_pretrained(decoder_model)
    filter_token = 0
    results = []
    
    if dataset == "nq":
        dataset = load_dataset("Tevatron/wikipedia-nq")['dev']
        filter_token = 512
    elif dataset == "hotpotqa":
        dataset = load_dataset("BroAlanTaps/Stage2-PCC-SFT-HotpotQA")['test']
        filter_token = 256
    elif dataset == "squad":
        dataset = load_dataset("BroAlanTaps/Stage2-PCC-Lite-SFT-Squad")['test']
        filter_token = 256
    elif dataset == "adqa":
        dataset = load_dataset("UCLNLP/adversarial_qa","adversarialQA")['validation']
    else:
        raise NotImplementedError(f"dataset {dataset} not supported!")

    dataset = dataset.map(cal_avg_token, num_proc=64, fn_kwargs={"lm_tokenizer": lm_tokenizer, "dataset": config.dataset})
    dataset = dataset.filter(lambda x: x['sum_token'] > filter_token)
    
    model = PCC(config).to(config.device).eval()
    tokenizer = model.compressor.tokenizer
    for idx,data in tqdm(enumerate(dataset), total=len(dataset)):
        if config.dataset == "nq" and data['sum_token'] > 8000:
            continue
        
        context = None
        
        if config.dataset == "nq":
            context = [text['text'] for text in data['positive_passages']]
            context = "\n\n".join(context)
        else:
            context = data['context']
            
        compress_ids = tokenizer(context,return_tensors="pt",truncation=False)['input_ids'].to(config.device)
        
        question = data["query"] if config.dataset == "nq" else data["question"]
        prompt = f"Question: {question}\n\nAnswer: "
        
        with torch.no_grad():
            with autocast(dtype=torch.bfloat16):
                embedding = model(compress_ids=compress_ids,llm_ids=None,get_embedding=True).to(config.device)
                output = model.decoder.generate(input_embedding=embedding,prompt_text=prompt,max_new_token=30)
        output = output.strip()

        if idx%100==0:
            print(f"{idx}/{len(dataset)}")
            
        results.append({
            "question": question,
            "generate": output,
            'label':data['answers']
        })

    if not os.path.exists("./result/"):
        os.makedirs("./result/")
    
    model_type = None
    if config.use_lora == None or config.use_lora == False:
        model_type = "PCC-Lite"
    else:
        model_type = "PCC-Large"
    output_file = f"./result/{config.dataset}-{model_type}-{256//config.embed_len}x.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    avg_f1_score = []
    avg_em_score = []
    for item in results:
        f1_score, em_score = 0, 0
        predict, labels = item['generate'], item['label']
        if config.dataset == "nq":
            f1_score = max([qa_f1_score(predict, label) for label in labels])
            em_score = max([exact_match_score(predict,label) for label in labels])
        else:
            f1_score = qa_f1_score(predict,labels) 
            em_score = exact_match_score(predict,labels)
        avg_f1_score.append(f1_score)
        avg_em_score.append(em_score)
        
    print('-'*50 + "result" + '-'*50)
    print(f"avg_f1_score:{sum(avg_f1_score)/len(avg_f1_score)}")
    print(f"avg_em_score:{sum(avg_em_score)/len(avg_em_score)}")    
    print('-'*100)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="QA Task")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--use_lora', type=bool, required=False)
    parser.add_argument('--adapter_model', type=str, required=False)
    parser.add_argument('--compress_model_path', type=str,required=True)
    parser.add_argument('--converter_model_path', type=str,required=True)
    parser.add_argument('--decoder_model', type=str,required=True)
    parser.add_argument('--compress_ratio',type=int,required=True)
    parser.add_argument('--write',type=bool,default=True)
    parser.add_argument('--segment_length',type=int,default=256)
    parser.add_argument('--compressor_gradient_checkpoint', type=bool, default=False)
    parser.add_argument('--decoder_gradient_checkpoint', type=bool, default=False)
    
    args = parser.parse_args()
    config = Config(
            device="cuda:0",
            dataset=args.dataset,
            compress_model=args.compress_model_path,
            adapter_model=args.adapter_model,
            converter_model=args.converter_model_path,
            decoder_model=args.decoder_model,
            embed_len=(256 // args.compress_ratio),
            write=args.write,
            segment_length=args.segment_length,
            use_lora=args.use_lora,
            compressor_gradient_checkpoint=args.compressor_gradient_checkpoint,
            decoder_gradient_checkpoint=args.decoder_gradient_checkpoint
    )
    print(config)

    run(config=config)