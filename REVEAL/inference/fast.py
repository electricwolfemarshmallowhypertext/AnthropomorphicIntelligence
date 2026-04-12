import os
import math
import re
import argparse

parser = argparse.ArgumentParser(description="AI Detector FAST Inference Script")
parser.add_argument("--model_path", "-m", type=str, required=True, help="Path to the model directory (must contain 2class, 3class, or 4class in name)")
parser.add_argument("--text", "-t", type=str, required=True, help="Input text to detect.")
parser.add_argument("--device", "-d", type=str, default="0", help="CUDA_VISIBLE_DEVICES setting")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from radar.utils.prompts import build_2class_prompt, build_3class_prompt, build_4class_prompt

model_name_lower = args.model_path.lower()
if "2class" in model_name_lower:
    class_type = 2
    target_classes = ["Human", "AI"]
    system_prompt = build_2class_prompt()
elif "3class" in model_name_lower:
    class_type = 3
    target_classes = ["Human", "AI", "Polish"]
    system_prompt = build_3class_prompt()
elif "4class" in model_name_lower:
    class_type = 4
    target_classes = ["Human", "AI", "Polish", "Meaningless"] 
    system_prompt = build_4class_prompt()
else:
    raise ValueError("The model_path must contain '2class', '3class', or '4class' to auto-detect task type.")

print(f"[*] Detected Task Type: {class_type}-Class (FAST Mode)")
print(f"[*] Target Categories: {target_classes}")

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
class_token_ids = {}
for cls_name in target_classes:
    t_id = tokenizer.encode(cls_name, add_special_tokens=False)[0]
    class_token_ids[cls_name] = t_id
print(f"[*] Target Token IDs: {class_token_ids}")

llm = LLM(
    model=args.model_path,
    gpu_memory_utilization=0.80,
    trust_remote_code=True
)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=128,  
    logprobs=20      
)

def build_chat_prompt(user_text):
    messages = [
        {"role": "user", "content": f"{system_prompt}\n\nText:\n{user_text}"}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return prompt

def extract_answer_and_scores(request_output):
    full_text = request_output.outputs[0].text
    token_ids = request_output.outputs[0].token_ids
    logprobs_list = request_output.outputs[0].logprobs

    m = re.search(r"<answer>\s*(.*?)\s*</answer>", full_text, re.DOTALL)
    text_answer = m.group(1).strip() if m else "Unknown"
    
    scores = {cls: 0.0 for cls in target_classes}

    target_search_id = class_token_ids.get(text_answer)
    
    if target_search_id is not None:
        found_idx = -1
        for i in range(len(token_ids) - 1, -1, -1):
            if token_ids[i] == target_search_id:
                found_idx = i
                break
        
        if found_idx != -1:
            step_logprobs = logprobs_list[found_idx]
            raw_probs = {}
            
            for cls_name, t_id in class_token_ids.items():
                item = step_logprobs.get(t_id)
                val = item.logprob if item is not None else -999.0
                raw_probs[cls_name] = math.exp(val)
            
            total = sum(raw_probs.values())
            if total > 0:
                for cls_name in target_classes:
                    scores[cls_name] = raw_probs[cls_name] / total

    return text_answer, scores, full_text

def print_results(answer, scores, raw_text):
    print("\n========================")
    print(f"Model Raw Output: {raw_text.strip()}")
    print("------------------------")
    if answer in scores:
        print(f"Decision: **{answer}**")
        print("Confidence Scores (Normalized):")
        
        best_cls = max(scores, key=scores.get)
        for cls_name, score in scores.items():
            trophy = '🏆' if cls_name == best_cls else ''
            print(f"  {cls_name:<11}: {score:.4f}  {trophy}")
    else:
        print(f"Decision: {answer} (Could not extract valid predefined token)")
    print("========================")

print(f"[*] Processing provided text...")
prompt = build_chat_prompt(args.text)
output = llm.generate([prompt], sampling_params=sampling_params, use_tqdm=False)[0]
answer, scores, raw_text = extract_answer_and_scores(output)
print_results(answer, scores, raw_text)