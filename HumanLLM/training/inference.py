import argparse
import json
import os
from tqdm import tqdm
from typing import List, Optional
from dataclasses import dataclass
from trl import TrlParser
from utils import load_inference_datasets, get_model

@dataclass
class ScriptArguments:
    dataset_file: str = None #Optional[List[str]] = None
    domains: Optional[List[str]] = None
    max_samples_per_task: int = 2
    model_name_or_path: str = None
    temperature: float = 0.7
    top_p: float = 1.0
    output_file: str = None
    max_seq_length: int = 8192
    cal_acc: str = "none"  # "none", "item selection", "multi-choice"

def main(script_args):
    model, tokenizer = get_model(script_args)
    dataset = load_inference_datasets(script_args)

    outputs = []
    for example in tqdm(dataset["test"]):
        input_prompt = example["messages"][:-1]
        try:
            response = model.chat.completions.create(
                model=script_args.model_name_or_path,  
                messages=input_prompt,
                max_tokens=script_args.max_seq_length-len(tokenizer.apply_chat_template(input_prompt, tokenize=True, add_generation_prompt=True, enable_thinking=False)),
                temperature=script_args.temperature,
                top_p=script_args.top_p,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False}
                },
            )
            response_text = response.choices[0].message.content
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
            response_text = ""
            break
        except Exception as e:
            print(f'[EXCEPTION] error: {e}')
            response_text = ""
            
        outputs.append({"uid": example["uid"], "source": example["source"], "generated_answer": response_text, "messages": example["messages"]})
    
    with open(script_args.output_file, "w") as f:
        for output in outputs:
            f.write(json.dumps(output) + "\n")

    # calculate acc
    if script_args.cal_acc=="item selection":
        r_c=0
        for output in outputs:
            if output["messages"][-1]["content"].strip().lower() in output["generated_answer"].strip().lower():
                r_c+=1
        print(f"{r_c}/{len(outputs)}={r_c/len(outputs)}")
        with open(script_args.output_file.replace(".json", "_acc.txt"), "w") as f:
            f.write(f"{r_c}/{len(outputs)}={r_c/len(outputs)}\n")

def make_parser():
    dataclass_types = (ScriptArguments)
    parser = TrlParser(dataclass_types)
    return parser

if __name__ == "__main__":
    parser = make_parser()
    script_args = parser.parse_args_and_config()[0]
    os.makedirs(os.path.dirname(script_args.output_file), exist_ok=True)
    main(script_args)