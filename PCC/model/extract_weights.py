## Copyright (c) Microsoft Corporation.
## Licensed under the MIT license.

import os
import torch
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM

def run(args):
    os.environ["TRANSFORMERS_NO_DEEPSPEED"] = "1"
    modified_model_path = args.modified_model

    mod_tokenizer = AutoTokenizer.from_pretrained(modified_model_path)
    mod_model = AutoModelForCausalLM.from_pretrained(modified_model_path, torch_dtype=torch.bfloat16)

    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    new_tokens = list(set(mod_tokenizer.get_vocab()) - set(base_tokenizer.get_vocab()))
    new_token_ids = [mod_tokenizer.convert_tokens_to_ids(t) for t in new_tokens]


    embed_weight = mod_model.get_input_embeddings().weight.data[new_token_ids, :]          # shape: [N, hidden_dim]
    lm_head_weight = mod_model.lm_head.weight.data[new_token_ids, :]                       # shape: [N, hidden_dim]

    torch.save({
        "tokens": new_tokens,
        "embedding": embed_weight,
        "lm_head": lm_head_weight
    }, args.save_path)

    print("✅ Saved special token patch (embedding + lm_head).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--modified_model", type=str, default="BroAlanTaps/PCC-Decoder-Llama3-8B-Instruct")
    parser.add_argument("--save_path", type=str, default="llama3_special_token_patch.pt")
    args = parser.parse_args()

    run(args)