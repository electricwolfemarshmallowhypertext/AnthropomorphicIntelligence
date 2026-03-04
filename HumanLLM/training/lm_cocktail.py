import os
import shutil
import argparse

import torch
import random
import numpy as np
from typing import List, Dict, Any

from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, models


def parse_args():
    parser = argparse.ArgumentParser(description="data process")
    parser.add_argument(
        "--model_names_or_paths", type=str, help="split by ','", required=True
    )
    parser.add_argument(
        "--model_type", type=str, help="", required=True, choices=["decoder", "encoder", "reranker"]
    )
    parser.add_argument(
        "--weights", type=str, help="split by ','", required=True
    )
    parser.add_argument(
        "--output_path", type=str, help=""
    )
    parser.add_argument(
        "--sentence_pooling_method", type=str, default='cls', help="the pooling method, should be cls, mean or last", choices=['cls', 'mean', 'last']
    )
    parser.add_argument(
        "--normlized", action='store_true', help=""
    )
    args = parser.parse_args()
    return args

def load_llm(model_name:str, trust_remote_code:bool):
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=trust_remote_code, device_map = {"": "cpu"}, torch_dtype=torch.bfloat16)
    return model

def load_embedder(model_name:str, trust_remote_code:bool):
    model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code, device_map = {"": "cpu"})
    return model

def load_reranker(model_name:str, trust_remote_code:bool):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=trust_remote_code, device_map = {"": "cpu"})
    return model

def load_seq2seq_model(model_name:str, trust_remote_code:bool):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    return model

def load_model(model_name:str, model_type:str, trust_remote_code:bool=True):
    if model_type == 'decoder':
        model = load_llm(model_name, trust_remote_code=trust_remote_code)
    elif model_type == 'encoder':
        model = load_embedder(model_name, trust_remote_code=trust_remote_code)
    elif model_type == 'reranker':
        model = load_reranker(model_name, trust_remote_code=trust_remote_code)
    elif model_type == 'encoder-decoder':      
        model = load_seq2seq_model(model_name, trust_remote_code=trust_remote_code)
    else:
        raise NotImplementedError(f"not support this model_type: {model_type}")
    return model

def get_model_param_list(model_names: List[str], model_type:str):
    model_param_list = []
    for name in model_names:
        print(f"loading {name} -----------------")
        model = load_model(name, model_type=model_type)
        model_param_list.append(model.state_dict())
    return model_param_list

def merge_param(model_param_list: List[Dict], weights: List[float]):
    new_param = {}
    for k in model_param_list[0].keys():
        for w, param in zip(weights, model_param_list):
            if param[k].dtype == torch.int64 or param[k].dtype == torch.int32:
                new_param[k] = param[k]
            elif k not in new_param:
                new_param[k] = w * param[k]
            else:
                new_param[k] += w * param[k]
    return new_param

def save_ckpt_for_sentence_transformers(ckpt_dir, pooling_mode: str = 'cls', normalized: bool = True):
    word_embedding_model = models.Transformer(ckpt_dir)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode=pooling_mode)
    if normalized:
        normalized_layer = models.Normalize()
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model, normalized_layer], device='cpu')
    else:
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device='cpu')
    model.save(ckpt_dir)


def mix_models(model_names_or_paths: List[str], 
               model_type: str, 
               weights: List[float], 
               output_path: str=None,
               sentence_pooling_method: str='cls',
               normlized: bool=True):
    """_summary_
    mix models based on given weights
    Args:
        model_names_or_paths (List[str]): a list of names or paths to models
        model_type (str): type of model to mix, should be in ["decoder", "encoder", "reranker"]
        weights (List[float]): a list of mixing weights. The sum of weights should be equal to 1.
        output_path (str, optional): path to save the mixed model. Defaults to None.

    Returns:
        new model
    """
    
    assert len(model_names_or_paths) == len(weights)
    assert model_type in ['decoder', 'encoder', 'reranker']
    assert sum(weights) - 1 <= 1e-3
    
    param_list = get_model_param_list(model_names_or_paths, model_type=model_type)
    new_param = merge_param(param_list, weights=weights)
    
    print("***weight for each model***: ")
    for w, n in zip(weights, model_names_or_paths):
        print(n, w)
    
    model = load_model(model_names_or_paths[0], model_type=model_type)
    model.load_state_dict(new_param)
    
    if output_path is not None:
        print(f"Saving the new model to {output_path}")
        model.save_pretrained(output_path)
        tokenizer = AutoTokenizer.from_pretrained(model_names_or_paths[0], trust_remote_code=True)
        tokenizer.save_pretrained(output_path)
        
        if model_type == "encoder":
            print(f"Transform the model to the format of 'sentence_transformers' (pooling_method='{sentence_pooling_method}', normalized={normlized})")
            save_ckpt_for_sentence_transformers(ckpt_dir=output_path, pooling_mode=sentence_pooling_method, normalized=normlized)
    return model


if __name__ == "__main__":
    args = parse_args()
    model_names_or_paths = args.model_names_or_paths.split(',')
    weights = [float(w) for w in args.weights.split(',')]
    mix_models(model_names_or_paths, args.model_type, weights, args.output_path, args.sentence_pooling_method, args.normlized)