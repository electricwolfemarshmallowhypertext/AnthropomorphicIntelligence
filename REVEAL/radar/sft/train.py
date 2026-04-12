import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import load_dataset
from deepspeed.runtime.zero.config import ZeroStageEnum
from deepspeed.runtime.fp16.loss_scaler import LossScaler 
from deepspeed.utils.tensor_fragment import fragment_address
from torch.nn import CrossEntropyLoss
from radar.utils.prompts import build_2class_prompt, build_3class_prompt, build_4class_prompt
import wandb
import argparse

if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([ZeroStageEnum])
    torch.serialization.add_safe_globals([LossScaler])
    torch.serialization.add_safe_globals([fragment_address])

ANSWER_START_TAG = "<answer>"
ANSWER_END_TAG = "</answer>"

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss_weights = inputs.pop("loss_weights", None)
        
        outputs = model(**inputs)
        
        if "loss" in outputs and loss_weights is None:
            return (outputs["loss"], outputs) if return_outputs else outputs["loss"]
            
        logits = outputs.get("logits")
        labels = inputs.get("labels")
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        if loss_weights is not None:
            shift_weights = loss_weights[..., 1:].contiguous()
        else:
            shift_weights = torch.ones_like(shift_labels, dtype=torch.float32)

        loss_fct = CrossEntropyLoss(reduction='none')
        shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_weights = shift_weights.view(-1)
        
        raw_loss = loss_fct(shift_logits, shift_labels)

        valid_mask = (shift_labels != -100).float()
        weighted_loss = raw_loss * shift_weights * valid_mask
        
        sum_weights = (shift_weights * valid_mask).sum()
        loss = weighted_loss.sum() / (sum_weights + 1e-8)

        return (loss, outputs) if return_outputs else loss

def main(args):   
    if args.num_classes == 2:
        instruction = build_2class_prompt()
    elif args.num_classes == 3:
        instruction = build_3class_prompt()
    elif args.num_classes == 4:
        instruction = build_4class_prompt()
    else:
        raise ValueError(
            f"Unsupported num_classes={args.num_classes}. "
            "Only support {2, 3, 4}."
        )

    print("Loading datasets...")
    raw_dataset = load_dataset("json", data_files=args.train_data_path, split="train")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def create_sft_prompt(sample):
        text = sample["text"]
        prompt = f"{instruction}\n\nText:\n{text}"
        answer = sample["response"]
        
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer}
        ]

        tokenized_ids = tokenizer.apply_chat_template(
            messages,
            max_length=args.max_seq_length,
            truncation=True,
            add_special_tokens=False,
            add_generation_prompt=False,
        )

        prompt_messages = messages[:-1]
        prompt_ids = tokenizer.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True
        )
        prompt_length = len(prompt_ids)

        labels = list(tokenized_ids)
        effective_prompt_length = min(prompt_length, len(labels))
        labels[:effective_prompt_length] = [-100] * effective_prompt_length
        
        weights = [1.0] * len(labels)
        
        start_tag_idx = answer.find(ANSWER_START_TAG)
        end_tag_idx = answer.find(ANSWER_END_TAG)
        if start_tag_idx != -1 and end_tag_idx != -1:
            ans_start_ids = tokenizer.encode(ANSWER_START_TAG, add_special_tokens=False)
            ans_end_ids = tokenizer.encode(ANSWER_END_TAG, add_special_tokens=False)
            
            def find_subsequence(seq, subseq):
                n = len(seq)
                m = len(subseq)
                for i in range(n - m + 1):
                    if seq[i : i + m] == subseq:
                        return i
                return -1

            search_area = tokenized_ids[effective_prompt_length:]
            idx_start = find_subsequence(search_area, ans_start_ids)
            idx_end = find_subsequence(search_area, ans_end_ids)
            
            if idx_start != -1 and idx_end != -1:
                abs_start = effective_prompt_length + idx_start
                abs_end = effective_prompt_length + idx_end + len(ans_end_ids)
                for i in range(abs_start, abs_end):
                    weights[i] = args.weight

        attention_mask = [1] * len(tokenized_ids)

        return {
            "input_ids": tokenized_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "loss_weights": weights
        }

    train_dataset = raw_dataset.map(create_sft_prompt, remove_columns=list(raw_dataset.features))

    sample = train_dataset[0]
    ids = sample["input_ids"]
    labels = sample["labels"]
    weights = sample["loss_weights"]
    print(f"Total Sequence Length: {len(ids)}")
    
    print("-" * 20 + " Full Decoded Text " + "-" * 20)
    print(tokenizer.decode(ids, skip_special_tokens=False))
    print("\n" + "-" * 20 + " Token Detail View " + "-" * 20)
    print(f"{'IDX':<5} | {'TokenID':<8} | {'Label':<6} | {'Weight':<6} | {'Token String'}")
    print("-" * 60)
    
    for i, (tid, lbl, w) in enumerate(zip(ids, labels, weights)):
        token_str = tokenizer.decode([tid]).replace('\n', '\\n')
        marker = "  <--- WEIGHT BOOST" if w > 1.0 else ""
        print(f"{i:<5} | {tid:<8} | {lbl:<6} | {w:<6.1f} | {token_str}{marker}")
    print("="*80 + "\n")

    def weighted_data_collator(features):
        weights = [f.pop("loss_weights") for f in features]
        batch = data_collator(features)
        max_len = batch["input_ids"].shape[1]
        padded_weights = []
        for w in weights:
            pad_len = max_len - len(w)
            padded_weights.append(w + [0.0] * pad_len)
            
        batch["loss_weights"] = torch.tensor(padded_weights, dtype=torch.float32)
        return batch

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.float32,
        trust_remote_code=True,
        use_cache=False
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, 
        deepspeed=args.deepspeed,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=weighted_data_collator
    )
    
    print("Start Training with Weighted Loss...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT Training Script")
    parser.add_argument("--model_name", type=str, required=True, help="Base model name from Hugging Face.")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the training data JSON file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints and final model.")
    
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=10)
    parser.add_argument("--logging_steps", type=int, default=10)
    
    parser.add_argument("--bf16", action='store_true', help="Use bf16")
    parser.add_argument("--fp16", action='store_true', help="Use fp16")

    parser.add_argument("--deepspeed", type=str, required=True)

    parser.add_argument("--weight", type=int, default=20)
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes")
    
    args = parser.parse_args()
    main(args)