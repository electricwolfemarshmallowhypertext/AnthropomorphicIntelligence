## Copyright (c) Microsoft Corporation.
## Licensed under the MIT license.

import argparse
import json
import os

import nltk
import torch
from datasets import load_dataset, load_from_disk
from model.model import PCC
from torch.cuda.amp import autocast
from tqdm import tqdm

from ..dataclass import Config
from .utils import metrics

nltk.download('punkt_tab')

def _load_data(data_path: str):
    if os.path.exists(data_path):
        try:
            pattern = os.path.join(data_path, "test-*.parquet")
            print(f"Loading parquet files from: {pattern}")
            dataset = load_dataset("parquet", data_files=pattern)['train']
        except Exception as e:
            raise RuntimeError(
                f"Error: {e} Failed to load dataset from {data_path}."
                f"Make sure the directory contains 'test-*.parquet' files."
            ) from e
    else:
        dataset = load_dataset(data_path)['test']
    return dataset

def run(config: Config):
    model = PCC(config).to(config.device).eval()
    
    # load_data
    dataset = _load_data(config.dataset)
    
    bleu_list,bleu1_list,bleu2_list,bleu3_list,bleu4_list,rougeL_list = [],[],[],[],[],[]
    ori_text_list = []
    cons_text_list = []
    data_list = []
    file_name = f"{256 // config.embed_len}x_large" + f"generated_text.json"
    with open(file_name, "w") as file:
        with tqdm(range(len(dataset))) as pbar:
            for i in pbar:
                ori_text = dataset[i]['text']
                if config.use_lora:
                    compress_ids = dataset[i]['input_ids']
                else:
                    compress_ids = dataset[i]['compress_ids']
                ori_text_list.append(ori_text)
                
                with torch.no_grad():
                    with autocast(dtype=torch.bfloat16):
                        cons_text = model.generate(compress_ids, "<ae>", max_new_token=300)
                cons_text_list.append(cons_text)
                
                bleu, bleu1, bleu2, bleu3, bleu4, rougeL = metrics.cal_bleu(ori_text, cons_text)
            
                bleu_list.append(bleu)
                bleu1_list.append(bleu1)
                bleu2_list.append(bleu2)
                bleu3_list.append(bleu3)
                bleu4_list.append(bleu4)
                rougeL_list.append(rougeL)
                
                avg_bleu = sum(bleu_list) / len(bleu_list)
                avg_bleu1 = sum(bleu1_list) / len(bleu1_list)
                avg_bleu2 = sum(bleu2_list) / len(bleu2_list)
                avg_bleu3 = sum(bleu3_list) / len(bleu3_list)
                avg_bleu4 = sum(bleu4_list) / len(bleu4_list)
                avg_rougeL = sum(rougeL_list) / len(rougeL_list)
                print(f"""
                    Avg BLEU": {avg_bleu:.6f}\n
                    Avg BLEU-1": f"{avg_bleu1:.6f}\n
                    Avg BLEU-2": f"{avg_bleu2:.6f}\n
                    Avg BLEU-3": f"{avg_bleu3:.6f}\n
                    Avg BLEU-4": f"{avg_bleu4:.6f}\n
                    Avg ROUGE-L": f"{avg_rougeL:.6f}
                    """)
                print("-"*25)
            
                pbar.set_postfix({
                    "Avg BLEU": f"{avg_bleu:.2f}",
                    "Avg BLEU-4": f"{avg_bleu4:.2f}",
                    "Avg ROUGE-L": f"{avg_rougeL:.2f}"
                })
                data_list.append({
                    "ori_text": ori_text,
                    "cons_text": cons_text,
                    "bleu": bleu,
                    "bleu-1": bleu1,
                    "bleu-2": bleu2,
                    "bleu-3": bleu3,
                    "bleu-4": bleu4,
                    "rougeL": rougeL
                })
                
        json.dump(data_list, file, ensure_ascii=False, indent=2)
        print("-"*25+"Result"+"-"*25)
        print(f"""
              Avg BLEU": {avg_bleu:.6f}\n
              Avg BLEU-1": f"{avg_bleu1:.6f}\n
              Avg BLEU-2": f"{avg_bleu2:.6f}\n
              Avg BLEU-3": f"{avg_bleu3:.6f}\n
              Avg BLEU-4": f"{avg_bleu4:.6f}\n
              Avg ROUGE-L": f"{avg_rougeL:.6f}
              """)
        print("-"*25)
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
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
        dataset=args.data_path,
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
    