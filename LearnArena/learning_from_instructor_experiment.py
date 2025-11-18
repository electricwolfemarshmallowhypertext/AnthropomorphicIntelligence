import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

from tqdm import tqdm
from openai import OpenAI

from utils.utils import (
    chat_completion,
    read_jsonl,
    start_vllm_server,
    stop_vllm_server,
    write_jsonl,
)

SYSTEM_PROMPT = """You are a mathematician. Solve the following math problem with accurate, complete, and clear explanations."""

CHECK_SYSTEM_MESSAGE = """You are a helpful AI assistant. You will use your coding and language skills to verify the answer.\nYou are given:\n    1. A problem.\n    2. A reply with the answer to the problem.\n    3. A ground truth answer.\nPlease do the following:\n1. Extract the answer in the reply: "The answer is <answer extracted>".\n2. Check whether the answer in the reply matches the ground truth answer. When comparison is not obvious (for example, 3*\\sqrt(6) and 7.348), you may write code to check the answer and wait for the user to execute the code.\n3. After everything is done, please choose a reply from the following options:\n    - "The answer is correct."\n    - "The answer is approximated but should be correct. Correct Answer: <ground truth answer> | Answer extracted: <answer extracted>."\n    - "The answer is incorrect. Correct Answer: <ground truth answer> | Answer extracted: <answer extracted>."\n    - "The reply doesn't contain an answer." """



def parse_dataset_list(datasets: str) -> List[str]:
    """Split and validate the comma-separated dataset list."""
    names = [item.strip() for item in datasets.split(',') if item.strip()]
    if not names:
        raise ValueError("No dataset names provided. Use --datasets 'file1.jsonl,file2.jsonl'.")
    return names


def scorer(response: str) -> bool:
    response = response.lower()
    return "the answer is correct" in response or "the answer is approximated but should be correct" in response


def process_generation_item(data_item: Dict, api_base: str, model_name: str, max_tokens: int, temperature: float, api_key: str = "xxx") -> Dict:
    question = data_item.get("question")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    response = chat_completion(
        api_base=api_base,
        model_name=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        api_key=api_key,
    )
    data_item["llm_answer"] = response
    return data_item


def process_evaluation_item(data_item: Dict, api_base: str, model_name: str, max_tokens: int, temperature: float, api_key: str = "xxx") -> Dict:
    reference_answer = data_item.get("answer")
    llm_answer = data_item.get("llm_answer")
    question = data_item.get("question")

    user_prompt = f"Problem: {question}\n\nReply: {llm_answer}\n\nGround truth answer: {reference_answer}"
    messages = [
        {"role": "system", "content": CHECK_SYSTEM_MESSAGE},
        {"role": "user", "content": user_prompt},
    ]
    response = chat_completion(
        api_base=api_base,
        model_name=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        api_key=api_key,
    )
    is_correct = scorer(response)
    return {
        "question": question,
        "llm_answer": llm_answer,
        "reference_answer": reference_answer,
        "eval_feedback": response,
        "eval_result": is_correct,
    }


def ensure_parent_dir(file_path: str) -> None:
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def run_generation_for_file(
    input_file: str,
    output_file: str,
    api_base: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    threads: int,
    api_key: str = "xxx",
) -> None:
    data = read_jsonl(input_file)
    ensure_parent_dir(output_file)
    results = []
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [
            executor.submit(process_generation_item, item, api_base, model_name, max_tokens, temperature, api_key)
            for item in data
        ]
        for future in tqdm(futures, desc=f"Generating ({os.path.basename(input_file)})", total=len(futures)):
            results.append(future.result())
    write_jsonl(output_file, results)
    print(f"[INFO] Generated answers saved to {output_file}")


def run_evaluation_for_file(
    input_file: str,
    output_file: str,
    api_base: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    threads: int,
    api_key: str = "xxx",
) -> Tuple[int, int]:
    data = read_jsonl(input_file)
    ensure_parent_dir(output_file)
    wins = 0
    total = 0
    results = []
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [
            executor.submit(process_evaluation_item, item, api_base, model_name, max_tokens, temperature, api_key)
            for item in data
        ]
        for future in tqdm(futures, desc=f"Evaluating ({os.path.basename(input_file)})", total=len(futures)):
            record = future.result()
            wins += int(record.get("eval_result"))
            total += 1
            results.append(record)
    write_jsonl(output_file, results)
    accuracy = (wins / total * 100) if total else 0.0
    print(f"[INFO] Evaluation results saved to {output_file}")
    print(f"[INFO] Accuracy: {accuracy:.2f}% ({wins}/{total})")
    return wins, total


class VLLMServerContext:
    """Context manager for starting/stopping a vLLM server."""

    def __init__(self, model_path: str, model_name: str, port: int, gpu: int):
        self.model_path = model_path
        self.model_name = model_name
        self.port = port
        self.gpu = gpu
        self.process_id = None

    def __enter__(self):
        if not self.model_path:
            return None
        self.process_id = start_vllm_server(self.model_path, self.model_name, self.port, self.gpu)
        if self.process_id is None:
            raise RuntimeError(f"Failed to start vLLM server for {self.model_name} on port {self.port}")
        print(f"[INFO] Started vLLM server for {self.model_name} (PID: {self.process_id})")
        return self.process_id

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.process_id is not None:
            stop_vllm_server(self.process_id)
            print(f"[INFO] Stopped vLLM server for {self.model_name}")


def build_io_lists(dataset_names: List[str], input_dir: str, output_dir: str) -> List[Tuple[str, str]]:
    pairs = []
    for name in dataset_names:
        input_path = os.path.join(input_dir, name)
        output_path = os.path.join(output_dir, name)
        pairs.append((input_path, output_path))
    return pairs


def validate_files_exist(file_paths: List[str]) -> None:
    missing = [path for path in file_paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")


def run_generation_stage(args, dataset_names: List[str]) -> List[str]:
    data_pairs = build_io_lists(dataset_names, args.data_dir, args.gen_output_dir)
    input_paths = [pair[0] for pair in data_pairs]
    validate_files_exist(input_paths)

    generated_paths = []
    for input_file, output_file in data_pairs:
        if os.path.exists(output_file) and not args.overwrite:
            print(f"[INFO] Skipping generation for {output_file} - already exists")
            generated_paths.append(output_file)
            continue
        run_generation_for_file(
            input_file=input_file,
            output_file=output_file,
            api_base=args.gen_api_base,
            model_name=args.gen_model_name,
            max_tokens=args.gen_max_tokens,
            temperature=args.gen_temperature,
            threads=args.gen_threads,
            api_key=args.gen_api_key,
        )
        generated_paths.append(output_file)
    return generated_paths


def run_evaluation_stage(args, dataset_names: List[str], generated_paths: List[str]) -> Dict[str, Dict[str, float]]:
    eval_pairs = build_io_lists(dataset_names, args.gen_output_dir, args.eval_output_dir)
    accuracy_summary: Dict[str, Dict[str, float]] = {}

    gen_lookup = {os.path.basename(path): path for path in generated_paths}

    for dataset_name, (_, output_file) in zip(dataset_names, eval_pairs):
        gen_file = gen_lookup.get(dataset_name)
        if not gen_file or not os.path.exists(gen_file):
            print(f"[WARNING] Generated file missing for {dataset_name}, skipping evaluation")
            continue

        eval_output = output_file
        if os.path.exists(eval_output) and not args.overwrite:
            print(f"[INFO] Skipping evaluation for {eval_output} - already exists")
            continue

        wins, total = run_evaluation_for_file(
            input_file=gen_file,
            output_file=eval_output,
            api_base=args.eval_api_base,
            model_name=args.eval_model_name,
            max_tokens=args.eval_max_tokens,
            temperature=args.eval_temperature,
            threads=args.eval_threads,
            api_key=args.eval_api_key,
        )
        accuracy_summary[dataset_name] = {
            "wins": wins,
            "total": total,
            "accuracy": (wins / total * 100) if total else 0.0,
        }
    return accuracy_summary


def print_summary(accuracy_summary: Dict[str, Dict[str, float]]) -> None:
    if not accuracy_summary:
        print("[WARNING] No evaluation results to summarize.")
        return
    print("\n========================================")
    print("Learning from Instructor - Math Evaluation Summary")
    print("========================================")
    aggregate_wins = 0
    aggregate_total = 0
    for dataset, stats in accuracy_summary.items():
        print(
            f"{dataset}: {stats['accuracy']:.2f}% accuracy "
            f"({stats['wins']}/{stats['total']})"
        )
        aggregate_wins += stats['wins']
        aggregate_total += stats['total']
    if aggregate_total:
        overall = aggregate_wins / aggregate_total * 100
        print(f"Overall: {overall:.2f}% ({aggregate_wins}/{aggregate_total})")


def main():
    parser = argparse.ArgumentParser(
        description="Learning from Instructor: Instruction-tuning evaluation on mathematical reasoning tasks"
    )
    parser.add_argument("--datasets", type=str, required=True, help="Comma-separated list of dataset file names located in --data_dir")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing raw dataset JSONL files")
    parser.add_argument("--gen_output_dir", type=str, default="./gen_output", help="Directory to store generated answers")
    parser.add_argument("--eval_output_dir", type=str, default="./eval_output", help="Directory to store evaluation results")
    parser.add_argument("--stage", type=str, choices=["generate", "evaluate", "pipeline"], default="pipeline", help="Select which stage to run")
    parser.add_argument("--overwrite", action="store_true", help="Re-run stages even if outputs already exist")
    
    # Mode selection
    parser.add_argument("--mode", type=str, default="vllm", choices=["vllm", "api"],
                       help="Mode: 'vllm' for local vLLM servers, 'api' for external API endpoints")

    # Generation parameters
    parser.add_argument("--gen_model_name", type=str, help="Generation model name")
    parser.add_argument("--gen_api_base", type=str, default="http://localhost:8000", help="Generation API base URL")
    parser.add_argument("--gen_api_key", type=str, help="API key for generation model (API mode, or set API_KEY_1 env var)")
    parser.add_argument("--gen_max_tokens", type=int, default=512)
    parser.add_argument("--gen_temperature", type=float, default=0.7)
    parser.add_argument("--gen_threads", type=int, default=32)
    parser.add_argument("--gen_model_path", type=str, help="Path to local generation model (vLLM mode)")
    parser.add_argument("--gen_port", type=int, default=8000, help="Port for generation model server")
    parser.add_argument("--gen_gpu", type=int, default=2, help="Number of GPUs for generation model")

    # Evaluation parameters
    parser.add_argument("--eval_model_name", type=str, help="Evaluation model name")
    parser.add_argument("--eval_api_base", type=str, default="http://localhost:8001", help="Evaluation API base URL")
    parser.add_argument("--eval_api_key", type=str, help="API key for evaluation model (API mode, or set API_KEY_0 env var)")
    parser.add_argument("--eval_max_tokens", type=int, default=256)
    parser.add_argument("--eval_temperature", type=float, default=0.7)
    parser.add_argument("--eval_threads", type=int, default=16)
    parser.add_argument("--eval_model_path", type=str, help="Path to local evaluation model (vLLM mode)")
    parser.add_argument("--eval_port", type=int, default=8001, help="Port for evaluation model server")
    parser.add_argument("--eval_gpu", type=int, default=2, help="Number of GPUs for evaluation model")

    args = parser.parse_args()

    # Validate mode-specific requirements
    if args.mode == "vllm":
        if args.stage in ("generate", "pipeline") and not args.gen_model_path:
            parser.error("--gen_model_path is required for vLLM mode generation")
        if args.stage in ("evaluate", "pipeline") and not args.eval_model_path:
            parser.error("--eval_model_path is required for vLLM mode evaluation")
    elif args.mode == "api":
        if args.stage in ("generate", "pipeline") and not args.gen_api_base:
            parser.error("--gen_api_base is required for API mode generation")
        if args.stage in ("evaluate", "pipeline") and not args.eval_api_base:
            parser.error("--eval_api_base is required for API mode evaluation")
    
    # Get API keys from environment if not provided
    if args.mode == "api":
        if not args.gen_api_key:
            args.gen_api_key = os.getenv("API_KEY_1", "your_api_key_here")
        if not args.eval_api_key:
            args.eval_api_key = os.getenv("API_KEY_0", "your_api_key_here")
    else:
        # vLLM mode uses dummy keys
        args.gen_api_key = "xxx"
        args.eval_api_key = "xxx"

    dataset_names = parse_dataset_list(args.datasets)

    if args.stage in ("generate", "pipeline") and not args.gen_model_name:
        parser.error("--gen_model_name is required when stage includes generation")
    if args.stage in ("evaluate", "pipeline") and not args.eval_model_name:
        parser.error("--eval_model_name is required when stage includes evaluation")

    generated_paths: List[str] = []

    if args.stage in ("generate", "pipeline"):
        if args.mode == "vllm":
            with VLLMServerContext(args.gen_model_path, args.gen_model_name, args.gen_port, args.gen_gpu):
                generated_paths = run_generation_stage(args, dataset_names)
        else:
            # API mode - no server context needed
            generated_paths = run_generation_stage(args, dataset_names)

    if args.stage in ("evaluate", "pipeline"):
        # If evaluation only, assume generation outputs already exist
        if args.stage == "evaluate":
            generated_paths = [
                os.path.join(args.gen_output_dir, name) for name in dataset_names
            ]
        validate_files_exist(generated_paths)
        if args.mode == "vllm":
            with VLLMServerContext(args.eval_model_path, args.eval_model_name, args.eval_port, args.eval_gpu):
                accuracy_summary = run_evaluation_stage(args, dataset_names, generated_paths)
        else:
            # API mode - no server context needed
            accuracy_summary = run_evaluation_stage(args, dataset_names, generated_paths)
        print_summary(accuracy_summary)


if __name__ == "__main__":
    main()
