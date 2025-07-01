import requests
import json
import argparse
from tqdm import tqdm
import transformers
import torch
import os
from vllm import LLM, SamplingParams
import numpy as np
import time


SERVER_URL = "http://localhost:{}/v1/chat/completions"  # Dynamic port in the URL


def get_parser():
    parser = argparse.ArgumentParser(description="Evaluation of MotiveBench")
    parser.add_argument('--dataset', required=True, choices=['Persona', 'Amazon', 'Blog'])
    parser.add_argument('--llm', required=True)
    parser.add_argument('--cot', action='store_true')
    parser.add_argument('--port', type=int, default=4000, help="Port to run the server on")
    parser.add_argument('--parse_mode', action='store_true', help="Enable parse mode (default: False)")
    return parser

def load_questions(input_file, order):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        questions = []
        answers = []

        for item in data:
            if args.cot:
                question_1 = "The following is a Motivational Reasoning Question. Based on the content of the given question, please infer the most likely answer and output the answer index.\n\n" + item["Motivation Reasoning Question"] + "\nOptions: " + str(rearrange_options(item["Options 1"], order)) + "\n\nPlease first think step by step, conduct analysis on the answers to the questions, output the reasoning process, and finally output the most likely option letter. **The last line of your reply should only contain one character of your final choice.**"
                question_2 = "The following is a Behavioral Reasoning Question. Based on the content of the given question, please infer the most likely answer and output the answer index.\n\n" + item["Behavior Reasoning Question"] + "\nOptions: " + str(rearrange_options(item["Options 2"], order)) + "\n\nPlease first think step by step, conduct analysis on the answers to the questions, output the reasoning process, and finally output the most likely option letter. **The last line of your reply should only contain one character of your final choice.**"
                question_3 = "The following is a Motivational and Behavioral Reasoning Question. Based on the content of the given question, please infer the most likely answer and output the answer index.\n\n" + item["Motivation and Behavior Reasoning Question"] + "\nOptions: " + str(rearrange_options(item["Options 3"], order)) + "\n\nPlease first think step by step, conduct analysis on the answers to the questions, output the reasoning process, and finally output the most likely option letter. **The last line of your reply should only contain one character of your final choice.**"
            else:
                question_1 = "The following is a Motivational Reasoning Question. Based on the content of the given question, please infer the most likely answer and output the answer index.\n\n" + item["Motivation Reasoning Question"] + "\nOptions: " + str(rearrange_options(item["Options 1"], order)) + "\n\nPlease answer this multiple-choice question. The result can only return **one character without any other explanation**."
                question_2 = "The following is a Behavioral Reasoning Question. Based on the content of the given question, please infer the most likely answer and output the answer index.\n\n" + item["Behavior Reasoning Question"] + "\nOptions: " + str(rearrange_options(item["Options 2"], order)) + "\n\nPlease answer this multiple-choice question. The result can only return **one character without any other explanation**."
                question_3 = "The following is a Motivational and Behavioral Reasoning Question. Based on the content of the given question, please infer the most likely answer and output the answer index.\n\n" + item["Motivation and Behavior Reasoning Question"] + "\nOptions: " + str(rearrange_options(item["Options 3"], order)) + "\n\nPlease answer this multiple-choice question. The result can only return **one character without any other explanation**."
            
            correct_answer_1 = get_new_correct_answer(item["Correct Answer 1"], order)
            correct_answer_2 = get_new_correct_answer(item["Correct Answer 2"], order)
            correct_answer_3 = get_new_correct_answer(item["Correct Answer 3"], order)
            questions.append([question_1, question_2, question_3])
            answers.append([correct_answer_1, correct_answer_2, correct_answer_3])

        return questions, answers

def rearrange_options(options, order):
    # List of alphabet letters to update the labels
    labels = ['A', 'B', 'C', 'D', 'E', 'F']
    # Rearranging the options based on the given order
    rearranged_options = [options[i-1] for i in order]
    # Updating the labels
    updated_options = [f"{labels[i]}. {option.split('. ', 1)[1]}" for i, option in enumerate(rearranged_options)]
    return updated_options

def get_new_correct_answer(original_correct, order):
    # Get the original index of the correct answer (A=0, B=1, etc.)
    original_index = ord(original_correct) - ord('A')
    # Find the new index of the correct answer based on the rearranged order
    new_index = order.index(original_index + 1)
    # Return the new correct letter
    return chr(new_index + ord('A'))

def save_answers(answers, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(answers, f, ensure_ascii=False, indent=4)

def create_chat_request(messages, max_tokens):
    return {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "repetition_penalty": 1.05,
    }

def get_model_response(question):
    url = SERVER_URL.format(args.port)  # Use the port from the command-line argument
    headers = {
        "Content-Type": "application/json"
    }

    messages = [
        {"role": "system", "content": "You ara a helpful AI assistant."},
        {"role": "user", "content": question}
    ]
    data = create_chat_request(messages, max_tokens=4096)
    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

    
def get_accuracy(answers, model_answers):
    correct_1 = 0
    correct_2 = 0
    correct_3 = 0
    all_correct = 0
    for i in range(len(answers)):
        if answers[i][0] == model_answers[i][0]:
            correct_1 += 1
        if answers[i][1] == model_answers[i][1]:
            correct_2 += 1
        if answers[i][2] == model_answers[i][2]:
            correct_3 += 1
        if answers[i][0] == model_answers[i][0] and answers[i][1] == model_answers[i][1] and answers[i][2] == model_answers[i][2]:
            all_correct += 1
    return correct_1 / len(answers), correct_2 / len(answers), correct_3 / len(answers), all_correct / len(answers)

def main():
    order_list = [[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1], [3, 1, 6, 5, 4, 2], [2, 3, 5, 6, 1, 4], [5, 4, 1, 2, 6, 3], [4, 6, 2, 1, 3, 5]]
    order_list.extend([[1, 4, 3, 2, 5, 6], [2, 5, 4, 3, 6, 1], [3, 1, 2, 5, 4, 6], [2, 3, 5, 4, 1, 6], [3, 4, 1, 2, 6, 5], [5, 6, 2, 1, 3, 4]]) 
    order_list.extend([[1, 3, 2, 4, 5, 6], [6, 5, 4, 2, 3, 1], [3, 1, 5, 6, 4, 2], [2, 3, 6, 5, 1, 4], [5, 4, 1, 6, 2, 3], [4, 2, 6, 1, 3, 5]])


    if args.cot:
        eval_type = 'cot'
    else:
        eval_type = 'base'
    current_directory = os.getcwd()  # Get current working directory
    input_file_path = os.path.join(current_directory, args.dataset, f"{args.dataset}_questions.json")
    output_file_path = os.path.join(current_directory, "results", f"{args.dataset}_{args.llm.split('/')[-1]}_{eval_type}.log")
    ourdir = os.path.dirname(output_file_path)
    os.makedirs(ourdir, exist_ok=True)
    
    flag = False
    while not flag:
        try:
            test_answer = get_model_response("What is your name?")
            print("@@ service is ready")
            print(test_answer)
            flag = True
        except:
            print("@@ service is not ready, retrying after 10 seconds")
            time.sleep(10)
        
    print(test_answer)

    if args.parse_mode:
        model_id = "Qwen/Qwen2.5-7B-Instruct"
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        parse_template = "Extract the letters of the option from the given text and return only ONE letter as the answer. The answer can only return **one character without any other explanation**."


    f = open(output_file_path, 'w', encoding='utf-8')
    acc_1 = 0
    acc_2 = 0
    acc_3 = 0
    acc_all = 0
    acc_1_list = []
    acc_2_list = []
    acc_3_list = []
    acc_all_list = []

    for idx, order in enumerate(order_list):
        questions, answers = load_questions(input_file_path, order)
        model_answers = []

        if args.parse_mode:
            for question in tqdm(questions):
                question_1, question_2, question_3 = question
                answer_1 = get_model_response(question_1).strip('assistant\n\n').strip()
                messages = [
                    {"role": "system", "content": parse_template},
                    {"role": "user", "content": answer_1},
                ]

                answer_1 = pipeline(messages, max_new_tokens=256, pad_token_id = pipeline.tokenizer.eos_token_id)[0]["generated_text"][-1]['content'].strip()[0].upper()
      
                answer_2 = get_model_response(question_2).strip('assistant\n\n').strip()
                messages = [
                    {"role": "system", "content": parse_template},
                    {"role": "user", "content": answer_2},
                ]
                answer_2 = pipeline(messages, max_new_tokens=256, pad_token_id = pipeline.tokenizer.eos_token_id)[0]["generated_text"][-1]['content'].strip()[0].upper()

                answer_3 = get_model_response(question_3).strip('assistant\n\n').strip()
                messages = [
                    {"role": "system", "content": parse_template},
                    {"role": "user", "content": answer_3},
                ]
                answer_3 = pipeline(messages, max_new_tokens=256, pad_token_id = pipeline.tokenizer.eos_token_id)[0]["generated_text"][-1]['content'].strip()[0].upper()

                model_answers.append([answer_1, answer_2, answer_3])
                print(f"Question 1: {question_1}\nAnswer 1: {answer_1}\n\nQuestion 2: {question_2}\nAnswer 2: {answer_2}\n\nQuestion 3: {question_3}\nAnswer 3: {answer_3}\n\n")
        else:
            if args.cot:
                for question in tqdm(questions, desc=f"Run {idx}"):
                    question_1, question_2, question_3 = question
                    answer_1 = get_model_response(question_1).strip('assistant\n\n').strip()
                    answer_1 = answer_1.strip('.').strip('\'').strip('\"').strip('*').strip('*')[-1].upper()
                    answer_2 = get_model_response(question_2).strip('assistant\n\n').strip()
                    answer_2 = answer_2.strip('.').strip('\'').strip('\"').strip('*').strip('*')[-1].upper()
                    answer_3 = get_model_response(question_3).strip('assistant\n\n').strip()
                    answer_3 = answer_3.strip('.').strip('\'').strip('\"').strip('*').strip('*')[-1].upper()
                    model_answers.append([answer_1, answer_2, answer_3])
                    print(f"Question 1: {question_1}\nAnswer 1: {answer_1}\n\nQuestion 2: {question_2}\nAnswer 2: {answer_2}\n\nQuestion 3: {question_3}\nAnswer 3: {answer_3}\n\n")
            else:
                for question in tqdm(questions, desc=f"Run {idx}"):
                    question_1, question_2, question_3 = question
                    # print(get_model_response(question_1))
                    answer_1 = get_model_response(question_1).strip('assistant\n\n').strip()
                    answer_1 = answer_1.strip('.').strip('\'').strip('\"').strip('*').strip('*')[0].upper()
                    answer_2 = get_model_response(question_2).strip('assistant\n\n').strip()
                    answer_2 = answer_2.strip('.').strip('\'').strip('\"').strip('*').strip('*')[0].upper()
                    answer_3 = get_model_response(question_3).strip('assistant\n\n').strip()
                    answer_3 = answer_3.strip('.').strip('\'').strip('\"').strip('*').strip('*')[0].upper()
                    model_answers.append([answer_1, answer_2, answer_3])
                    print(f"Question 1: {question_1}\nAnswer 1: {answer_1}\n\nQuestion 2: {question_2}\nAnswer 2: {answer_2}\n\nQuestion 3: {question_3}\nAnswer 3: {answer_3}\n\n")


        accuracy_1, accuracy_2, accuracy_3, accuracy_all = get_accuracy(answers, model_answers)
        acc_1 += accuracy_1
        acc_2 += accuracy_2
        acc_3 += accuracy_3
        acc_all += accuracy_all
        
        acc_1_list.append(accuracy_1)
        acc_2_list.append(accuracy_2)
        acc_3_list.append(accuracy_3)
        acc_all_list.append(accuracy_all)
        

        f.write("------------------------------------------------------------\n")
        f.write(f"Order: {order}\n")
        f.write(f"Model answers: {model_answers}\n")
        f.write(f"Accuracy of question 1: {accuracy_1}\n")
        f.write(f"Accuracy of question 2: {accuracy_2}\n")
        f.write(f"Accuracy of question 3: {accuracy_3}\n")
        f.write(f"Accuracy of all questions: {accuracy_all}\n")
        f.write("------------------------------------------------------------\n")

    # average variance 
    f.write("------------------------------------------------------------\n")
    f.write(f"Variance of question 1: {np.var(acc_1_list)}\n")
    f.write(f"Variance of question 2: {np.var(acc_2_list)}\n")
    f.write(f"Variance of question 3: {np.var(acc_3_list)}\n")
    f.write(f"Variance of all questions: {np.var(acc_all_list)}\n")
    f.write("------------------------------------------------------------\n")
     

    # average accuracy
    f.write("------------------------------------------------------------\n")
    f.write(f"Average accuracy of question 1: {acc_1 / len(order_list)}\n")
    f.write(f"Average accuracy of question 2: {acc_2 / len(order_list)}\n")
    f.write(f"Average accuracy of question 3: {acc_3 / len(order_list)}\n")
    f.write(f"Average accuracy of all questions: {acc_all / len(order_list)}\n")
    f.write("------------------------------------------------------------\n")

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main()