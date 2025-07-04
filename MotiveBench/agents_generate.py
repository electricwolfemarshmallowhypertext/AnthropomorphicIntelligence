## Copyright (c) Microsoft Corporation.
## Licensed under the MIT license.

from typing import Literal
import requests
import json
import re
from tqdm import tqdm
import random
import string
from pydantic import BaseModel
from openai import OpenAI

from agents_prompt import questioner_prompt, critic_prompt_1, critic_prompt_2, critic_prompt_3, modifier_prompt, adjuster, adjuster_1, adjuster_2, adjuster_3

 
SERVER_URL = "http://localhost:5000/chat/"
 
def create_chat_request(messages, max_tokens):
    return {
        "messages": messages,
        "max_tokens": max_tokens
    }

def chat_with_server(messages, max_tokens=4096):
    url = SERVER_URL
    headers = {
        "Content-Type": "application/json"
    }
    data = create_chat_request(messages, max_tokens)
    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

def get_model_response(system_prompt, user_prompt):
    messages = [
        {"role": "system", "content": "You ara a helpful AI assistant."},
        {"role": "user", "content": system_prompt + "\n\n" + user_prompt}
    ]
    response = chat_with_server(messages)
    return response["response_message"].strip("```").strip().strip("json").strip()


def get_parsed_response(system_prompt, user_prompt):
    while True:
        try:
            answer = get_model_response(system_prompt, user_prompt)
            answer = json.loads(answer)
            break  
        except Exception as e:
            print(f"Error: {e}")
    return answer


def main():
    # with open("/home/v-xixianyong/HumanSimulator/motivation/yelp/testset_v2/yelp_scenarios_1000.json", "r") as f:
    #     scenario_list = json.load(f)

    with open("~/HumanSimulator/motivation/amazon/amazon_scenarios_1000.json", "r") as f:
        scenario_list = json.load(f)
 
    # scenario_list = scenario_list[::50]

    random.shuffle(scenario_list)

    testset = []
    testset_init = []

    for scenario in tqdm(scenario_list[0:100]):
        scenario = scenario["story"]
        print(scenario)

        # substitute random letter as name
        random_letter = random.choice(string.ascii_lowercase).upper()
        questioner_prompt_completed = questioner_prompt.replace("{RANDOM_LETTER}", random_letter)
        # generate initial question
        question = get_parsed_response(questioner_prompt_completed, scenario)

        testset_init.append(question)

        with open("~/Data/benchmark/amazon/amazon_testset_init_241007.json", "w") as f:
            json.dump(testset_init, f, ensure_ascii=False, indent=4)

        iter_num = 0
        while iter_num < 2:
            suggestion_1 = get_parsed_response(critic_prompt_1, str(question))["suggestion"]
            suggestion_2 = get_parsed_response(critic_prompt_2, str(question))["suggestion"]
            suggestion_3 = get_parsed_response(critic_prompt_3, str(question))["suggestion"]

            if suggestion_1 == "No issues." and suggestion_2 == "No issues." and suggestion_3 == "No issues.":
                break

            suggestions = f"Suggestions:\n1. {suggestion_1}\n2. {suggestion_2}\n3. {suggestion_3}"
            question = str(question)
            modifier_answer_dict = get_parsed_response(modifier_prompt, f"Question:\n{question}\n\n{suggestions}")
            question = modifier_answer_dict
            print(suggestions)
            print(modifier_answer_dict)
            iter_num += 1
            print(f"iter_num: {iter_num}")


        question_1 = "Motivation Inference Question:\n" + question["Motivation Inference Question"] + "\n\nOptions:\n" + str(question["Options 1"]) + "\n\nCorrect Answer:\n" + question["Correct Answer 1"]
        question_2 = "Behavior Inference Question:\n" + question["Behavior Inference Question"] + "\n\nOptions:\n" + str(question["Options 2"]) + "\n\nCorrect Answer:\n" + question["Correct Answer 2"]
        question_3 = "Motivation and Behavior Inference Question:\n" + question["Motivation and Behavior Inference Question"] + "\n\nOptions:\n" + str(question["Options 3"]) + "\n\nCorrect Answer:\n" + question["Correct Answer 3"]
        final_queation_1 = get_parsed_response(adjuster_1, question_1)
        final_queation_2 = get_parsed_response(adjuster_2, question_2)
        final_queation_3 = get_parsed_response(adjuster_3, question_3)

        final_queation = {
            "scenario": scenario,
            "Motivation Inference Question": final_queation_1["Motivation Inference Question"],
            "Options 1": final_queation_1["Options"],
            "Correct Answer 1": final_queation_1["Correct Answer"],
            "Question Analysis 1": final_queation_1["Question Analysis"],
            "Behavior Inference Question": final_queation_2["Behavior Inference Question"],
            "Options 2": final_queation_2["Options"],
            "Correct Answer 2": final_queation_2["Correct Answer"],
            "Question Analysis 2": final_queation_2["Question Analysis"],
            "Motivation and Behavior Inference Question": final_queation_3["Motivation and Behavior Inference Question"],
            "Options 3": final_queation_3["Options"],
            "Correct Answer 3": final_queation_3["Correct Answer"],
            "Question Analysis 3": final_queation_3["Question Analysis"]
        }
        testset.append(final_queation)
        with open("~/Data/benchmark/amazon/amazon_testset_241007.json", "w") as f:
            json.dump(testset, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()