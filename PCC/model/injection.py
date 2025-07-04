## Copyright (c) Microsoft Corporation.
## Licensed under the MIT license.

import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

def run(args):
    base_model_path = args.base_model
    save_path = args.save_path

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16)
    tokenizer.eos_token_id = 128001
    tokenizer.pad_token_id = 128002

    print(f"eos_token: {tokenizer.eos_token}, id: {tokenizer.eos_token_id}") 
    print(f"pad_token: {tokenizer.pad_token}, id: {tokenizer.pad_token_id}")

    patch = torch.load(args.patch_path)
    new_tokens = patch["tokens"]
    embed_weight = patch["embedding"].to(model.dtype)
    lm_head_weight = patch["lm_head"].to(model.dtype)

    special_tokens_dict = {"additional_special_tokens": new_tokens}
    num_added = tokenizer.add_special_tokens(special_tokens_dict)
    assert num_added == len(new_tokens)
    model.resize_token_embeddings(len(tokenizer))

    new_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in new_tokens]
    for i, token_id in enumerate(new_token_ids):
        model.get_input_embeddings().weight.data[token_id] = embed_weight[i]
        model.lm_head.weight.data[token_id] = lm_head_weight[i]

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print("✅ Patched model with special token embedding + lm_head.")


if __name__ == "__main__":
    # python injection.py --base_model meta-llama/Meta-Llama-3-8B-Instruct --patch_path patch/llama3_8b_special_token_patch.pt --save_path ./decoder
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--patch_path", type=str, default="patch/llama3_8b_special_token_patch.pt")
    parser.add_argument("--save_path", type=str, default="./decoder")
    args = parser.parse_args()

    run(args)
