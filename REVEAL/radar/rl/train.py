import os
import json
import re
import torch
from datasets import Dataset
from trl import (
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from deepspeed.runtime.zero.config import ZeroStageEnum
from deepspeed.runtime.fp16.loss_scaler import LossScaler 
from deepspeed.utils.tensor_fragment import fragment_address
from radar.utils.prompts import build_2class_prompt, build_3class_prompt, build_4class_prompt
from radar.rl.reward import format_reward, answer_reward, consistency_reward

if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([ZeroStageEnum])
    torch.serialization.add_safe_globals([LossScaler])
    torch.serialization.add_safe_globals([fragment_address])

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    JSONL_FILE_PATHS = [path.strip() for path in script_args.dataset_name.split(",")]
    INSTRUCTION = build_4class_prompt()

    def load_jsonl(paths):
        data = []
        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    data.append(json.loads(line))
        return data

    def process_data(data):
        processed_records = []
        for item in data:
            text = item["text"]
            solution = item["label"]
            full_prompt = f"{INSTRUCTION}\n\nText:\n{text}"
            
            processed_records.append({
                "prompt": [{"role": "user", "content": full_prompt}],
                "solution": solution,
                "text": text
            })
        return processed_records

    print("Loading and processing dataset...")
    raw_data = load_jsonl(JSONL_FILE_PATHS)
    dataset_list = process_data(raw_data)
    dataset = Dataset.from_list(dataset_list)

    if training_args.eval_strategy != "no":
        dataset = dataset.train_test_split(
            test_size=0.05,
            seed=42,
            shuffle=True
        )
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
    else:
        train_dataset = dataset
        eval_dataset = None
    
    print(f"Dataset loaded. Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")
    
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        reward_funcs=[format_reward, answer_reward, consistency_reward], 
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)