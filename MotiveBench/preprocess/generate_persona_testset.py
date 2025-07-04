## Copyright (c) Microsoft Corporation.
## Licensed under the MIT license.

import random
import re
import string
import requests
import json

from tqdm import tqdm
import jsonlines

url = "http://0.0.0.0:4000/v1/chat/completions"
headers = {
    "Content-Type": "application/json"
}

def get_model_response(question):
    data = {
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": question}
        ],
        "max_length": 8192,
        "top_p": 0.85,
        "temperature": 0.8,
        "repetition_penalty": 1.1
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        return f"Failed to get response: {response.status_code}"
    
def get_parsed_response(prompt):
    while True:
        try:
            answer = get_model_response(prompt).strip("```").strip("json").strip()
            answer = json.loads(answer)
            break
        except Exception as e:
            print(f"Error: {e}")
    return answer

questioner_prompt = """
Consider the four elements: scenario, character identity, motivation, and behavior. In a given scenario, a character with a specific identity performs a behavior based on a particular motivation.

**You are a professional psychologist and sociologist, specializing in creating challenging reasoning questions to test participants' motivation and behavior inference abilities.**

Please create three questions based on the given scenario:
1. Motivation Inference Question: Based on a complex scenario, a specific character identity, and a given behavior, infer the most likely motivation behind the character's behavior. The question must not include any descriptions related to the predicted motivation.
2. Behavior Inference Question: Based on a complex scenario, a specific character identity, and a given motivation, infer the most likely behavior that the character would perform based on that motivation. The question must not include any descriptions related to the predicted behavior.
3. Motivation and Behavior Inference Question: This is a more difficult question. Only a complex scenario and a character identity are provided. Infer the most likely behavior the character would perform next and its corresponding motivation.
In summary, all three questions are based on the same story scenario and character identity. In motivation inference questions, an additional behavior is provided in the question, and the task is to infer the motivation behind that behavior. In behavior inference questions, an additional motivation is provided in the question, and the task is to infer the behavior that could result from that motivation. For combined motivation and behavior inference questions, no additional information is provided, and the task is to infer what behavior the character is most likely to exhibit next based on a specific motivation. However, it's essential to ensure that the story scenario and character identity in the question provide enough information to support inferring the correct answer.

Notes:
1. You will be given a simple persona description. Please generate a scenario, motivation and behavior based on the person. Correct any logical inconsistencies and add richer details through appropriate associations, making the scenario, character identity, motivation, and behavior more vivid and complex.
2. Select the most appropriate motivation and behavior to create the question. However, note that the motivation and behavior must only be related to real-life needs and not linked to any product or POI mentioned in the given scenario.
3. The three questions are independent of each other and should be answered separately, meaning that the reasoning should be based only on the prompt of each question, without access to any other information. Therefore, ensure that each of the three questions contains rich and complex scenario and character identity information.
4. Each question must have only one correct answer, along with five plausible but incorrect options. The incorrect options should be related to some part of the information provided in the prompt. Analyze the reasons why each option is correct or incorrect.
5. Distracting and redundant information related to the incorrect options must be included in the question stem, making each question very challenging. The correct answer should require reasoning based on the question’s information, social psychology knowledge, and common sense.

Please only return the answer in JSON format without any further explanations:
{
    "Motivation Inference Question": "A detailed description of a scenario and character identity. Considering a specific behavior of this character, what is the most likely motivation?",
    "Options 1": ["A. motivation_1", "B. motivation_2", "C. motivation_3", "D. motivation_4", "E. motivation_5", "F. motivation_6"],
    "Correct Answer 1": "{ONE LETTER}",
    "Question Analysis 1": ["A. analysis_1", "B. analysis_2", "C. analysis_3", "D. analysis_4", "E. analysis_5", "F. analysis_6"],
    "Behavior Inference Question": "A detailed description of a scenario and character identity. Considering a particular motivation of the character, what kind of behavior is he most likely to exhibit?",
    "Options 2": ["A. behavior_1", "B. behavior_2", "C. behavior_3", "D. behavior_4", "E. behavior_5", "F. behavior_6"],
    "Correct Answer 2": "{ONE LETTER}",
    "Question Analysis 2": ["A. analysis_1", "B. analysis_2", "C. analysis_3", "D. analysis_4", "E. analysis_5", "F. analysis_6"],
    "Motivation and Behavior Inference Question": "A detailed description of a scenario and character identity. What kind of behavior is this character most likely to exhibit next, and what is the motivation behind it?",
    "Options 3": ["A. motivation_1, behavior_1", "B. motivation_2, behavior_2", "C. motivation_3, behavior_3", "D. motivation_4, behavior_4", "E. motivation_5, behavior_5", "F. motivation_6, behavior_6"]
    "Correct Answer 3": "{ONE LETTER}",
    "Question Analysis 3": ["A. analysis_1", "B. analysis_2", "C. analysis_3", "D. analysis_4", "E. analysis_5", "F. analysis_6"]
}

Persona:
"""

critic_prompt_1 = """
You are a strict and discerning psychologist and sociologist, capable of accurately identifying issues and providing improvement suggestions for given behavior and motivation inference questions.

I will provide you with three behavior and motivation inference questions, and I would like you to evaluate them from the following aspect:
**Reasonableness of Information**:
Whether the information and type of question are correct. Specifically, all three questions include concrete scenarios and character identities. The motivation inference questions additionally include information about character behavior. The behavior inference questions also provide information about the character's motivation. However, in the motivation and behavior inference questions, no direct information about the motivation or behavior is given.

Please provide specific suggestions for revisions regarding the questions, addressing the question creator in a reasonable tone. Write the final revision suggestions in one paragraph. If there are no revision suggestions, return "No issues." Only return the response in JSON format:
{
  "suggestion": "..."
}

Question:
"""

critic_prompt_2 = """
You are a strict and discerning psychologist and sociologist, capable of accurately identifying issues and providing improvement suggestions for given behavior and motivation inference questions.

I will provide you with three behavior and motivation inference questions, and I would like you to evaluate them from the following aspect:
**Logical Consistency and Reasonableness**:
1. Evaluate whether the quadruple {scenario, character identity, motivation, behavior} in the questions is logically consistent and reasonable. Specifically, assess whether a character with a specific identity would exhibit this behavior based on the given motivation in the provided scenario.
2. Carefully assess whether each of the three questions contains sufficient information to infer the best answer. If not, revise the question information or options. Specifically, you can enrich or supplement the scenario or character identity as needed.

Please provide specific suggestions for revisions regarding the questions, addressing the question creator in a reasonable tone. Write the final revision suggestions in one paragraph. If there are no revision suggestions, return "No issues." Only return the response in JSON format:
{
  "suggestion": "..."
}

Question:
"""

critic_prompt_3 = """
You are a strict and discerning psychologist and sociologist, capable of accurately identifying issues and providing improvement suggestions for given behavior and motivation inference questions.

I will provide you with three behavior and motivation inference questions, and I would like you to evaluate them from the following aspect:
**Difficulty and Challenge**:
1. Assess whether the question is difficult and challenging. The correct answer should not be explicitly stated in the question information and should only be derived through a lengthy reasoning process. The incorrect options should be misleading and correspond to the distracting information in the question.
2. All three questions should contain enough distracting and redundant information to make them difficult, while ensuring that this information does not interfere with the reasoning process required to find the correct answer.
3. Assess whether the questions are presented neutrally and objectively, without any hints towards the correct answer.

Please provide specific suggestions for revisions regarding the questions, addressing the question creator in a reasonable tone. Write the final revision suggestions in one paragraph. If there are no revision suggestions, return "No issues." Only return the response in JSON format:
{
  "suggestion": "..."
}

Question:
"""

modifier_prompt = """
Consider the four elements: scenario, character identity, motivation, and behavior. In a given scenario, a character with a specific identity performs a behavior based on a particular motivation. You are a professional psychologist and sociologist, specializing in creating and refining motivation and behavior inference test questions, along with providing relevant feedback to improve the test.

The specific types of the three questions are:
1. Motivation Inference Question: Based on a complex scenario, a specific character identity, and a given behavior, infer the most likely motivation behind the character's behavior. The question must not include any descriptions related to the predicted motivation.
2. Behavior Inference Question: Based on a complex scenario, a specific character identity, and a given motivation, infer the most likely behavior that the character would perform based on that motivation. The question must not include any descriptions related to the predicted behavior.
3. Motivation and Behavior Inference Question: This is a more difficult question. Only a complex scenario and a character identity are provided. Infer the most likely behavior the character would perform next and its corresponding motivation.
In summary, all three questions are based on the same story scenario and character identity. In motivation inference questions, an additional behavior is provided in the question, and the task is to infer the motivation behind that behavior. In behavior inference questions, an additional motivation is provided in the question, and the task is to infer the behavior that could result from that motivation. For combined motivation and behavior inference questions, no additional information is provided, and the task is to infer what behavior the character is most likely to exhibit next based on a specific motivation. However, it's essential to ensure that the story scenario and character identity in the question provide enough information to support inferring the correct answer.

Specific requirements:
1. Based on the given questions, carefully consider each suggestion and selectively choose reasonable ones to modify the questions.
2. Do not remove the distracting scenario and character identity information related to the incorrect answers; ensure that the questions remain highly challenging.
3. The three questions are independent of each other and are answered separately, meaning the reasoning should be based solely on the individual question without access to any other information. Therefore, make sure each question contains rich and complex scenario and character identity information.
4. After making the revisions, reanalyze each option to determine why it is correct or incorrect. If there are issues, modify the questions again to ensure that the correct answer is uniquely identifiable.
5. Only return the modified answers in JSON format:
{
    "Motivation Inference Question": "A detailed description of a scenario and character identity. Considering a specific behavior of this character, what is the most likely motivation?",
    "Options 1": ["A. motivation_1", "B. motivation_2", "C. motivation_3", "D. motivation_4", "E. motivation_5", "F. motivation_6"],
    "Correct Answer 1": "{ONE LETTER}",
    "Question Analysis 1": ["A. analysis_1", "B. analysis_2", "C. analysis_3", "D. analysis_4", "E. analysis_5", "F. analysis_6"],
    "Behavior Inference Question": "A detailed description of a scenario and character identity. Considering a particular motivation of the character, what kind of behavior is he most likely to exhibit?",
    "Options 2": ["A. behavior_1", "B. behavior_2", "C. behavior_3", "D. behavior_4", "E. behavior_5", "F. behavior_6"],
    "Correct Answer 2": "{ONE LETTER}",
    "Question Analysis 2": ["A. analysis_1", "B. analysis_2", "C. analysis_3", "D. analysis_4", "E. analysis_5", "F. analysis_6"],
    "Motivation and Behavior Inference Question": "A detailed description of a scenario and character identity. What kind of behavior is this character most likely to exhibit next, and what is the motivation behind it?",
    "Options 3": ["A. motivation_1, behavior_1", "B. motivation_2, behavior_2", "C. motivation_3, behavior_3", "D. motivation_4, behavior_4", "E. motivation_5, behavior_5", "F. motivation_6, behavior_6"]
    "Correct Answer 3": "{ONE LETTER}",
    "Question Analysis 3": ["A. analysis_1", "B. analysis_2", "C. analysis_3", "D. analysis_4", "E. analysis_5", "F. analysis_6"]
}

"""

adjuster = """
You are a professional social and psychological researcher, and I will provide you with three motivation and behavior reasoning questions. Please make appropriate adjustments according to the following steps:

1. In motivation inference questions, no content should describe the motivation. In behavior inference questions, no content should describe the behavior. For questions involving both motivation and behavior inference, there should be no descriptions of either motivation or behavior. Only the scenario and character identity traits should be described. If any such descriptions exist, remove them.
2. Next, check whether the correct answer is explicitly indicated in the question stem. If it is, remove the corresponding content.
3. Based on the correct answer, enrich the descriptions of the scenario and character identity traits in all three questions to ensure that the correct answer remains unique.
4. Be sure to further add a significant amount of distracting and redundant information in all three questions to increase their level of difficulty.
5. Modify any options that are too easily identified as incorrect to make them more easily confused with the correct answer. All incorrect options must correspond to the distracting information in the question stem. If no such information exists, add it.
6. Finally, ensure that the correct answer is the best possible answer. If it isn’t, revise the question stem or the options.
7. Analyze the options for each question step by step. Each analysis must be as detailed as possible, and the correct answer must be justified by referring to the information provided in the question.

Only return the modified answers in JSON format:
{
    "Motivation Inference Question": "A detailed description of a scenario and character identity. Considering a specific behavior of this character, what is the most likely motivation?",
    "Options 1": ["A. motivation_1", "B. motivation_2", "C. motivation_3", "D. motivation_4", "E. motivation_5", "F. motivation_6"],
    "Correct Answer 1": "{ONE LETTER}",
    "Question Analysis 1": ["A. analysis_1", "B. analysis_2", "C. analysis_3", "D. analysis_4", "E. analysis_5", "F. analysis_6"],
    "Behavior Inference Question": "A detailed description of a scenario and character identity. Considering a particular motivation of the character, what kind of behavior is he most likely to exhibit?",
    "Options 2": ["A. behavior_1", "B. behavior_2", "C. behavior_3", "D. behavior_4", "E. behavior_5", "F. behavior_6"],
    "Correct Answer 2": "{ONE LETTER}",
    "Question Analysis 2": ["A. analysis_1", "B. analysis_2", "C. analysis_3", "D. analysis_4", "E. analysis_5", "F. analysis_6"],
    "Motivation and Behavior Inference Question": "A detailed description of a scenario and character identity. What kind of behavior is this character most likely to exhibit next, and what is the motivation behind it?",
    "Options 3": ["A. motivation_1, behavior_1", "B. motivation_2, behavior_2", "C. motivation_3, behavior_3", "D. motivation_4, behavior_4", "E. motivation_5, behavior_5", "F. motivation_6, behavior_6"]
    "Correct Answer 3": "{ONE LETTER}",
    "Question Analysis 3": ["A. analysis_1", "B. analysis_2", "C. analysis_3", "D. analysis_4", "E. analysis_5", "F. analysis_6"]
}
"""

adjuster_1 = """
You are a professional social and psychological researcher, and I will provide you with a motivation reasoning question.

Motivation Inference Question: Based on a complex scenario, a specific character identity, and a given behavior, infer the most likely motivation behind the character's behavior. The question must not include any descriptions related to the predicted motivation.

Please make appropriate adjustments step by step:
1. In motivation inference questions, no content should describe the motivation. In behavior inference questions, no content should describe the behavior. For questions involving both motivation and behavior inference, there should be no descriptions of either motivation or behavior. Only the scenario and character identity traits should be described. If any such descriptions exist, remove them.
2. Next, check whether the correct answer is explicitly indicated in the question stem. If it is, remove the corresponding content.
3. Based on the correct answer, enrich the descriptions of the scenario and character identity traits in all three questions to ensure that the correct answer remains unique.
4. Be sure to further add a significant amount of distracting and redundant information in all three questions to increase their level of difficulty.
5. **Modify any options that are too easily identified as incorrect to make them more easily confused with the correct answer. All incorrect options must correspond to the distracting information in the question stem. If no such information exists, add it.**
6. Finally, ensure that the correct answer is the best possible answer. If it isn’t, revise the question stem or the options.
7. Analyze the options for each question step by step. Each analysis must be as detailed as possible, and the correct answer must be justified by referring to the information provided in the question.

Only return the modified answers in JSON format:
{
    "Motivation Inference Question": "A detailed description of a scenario and character identity. Considering a specific behavior of this character, what is the most likely motivation?",
    "Options": ["A. motivation_1", "B. motivation_2", "C. motivation_3", "D. motivation_4", "E. motivation_5", "F. motivation_6"],
    "Correct Answer": "{ONE LETTER}",
    "Question Analysis": ["A. analysis_1", "B. analysis_2", "C. analysis_3", "D. analysis_4", "E. analysis_5", "F. analysis_6"]
}

"""

adjuster_2 = """
You are a professional social and psychological researcher, and I will provide you with a behavior reasoning question.

Behavior Inference Question: Based on a complex scenario, a specific character identity, and a given motivation, infer the most likely behavior that the character would perform based on that motivation. The question must not include any descriptions related to the predicted behavior.

Please make appropriate adjustments step by step:
1. In motivation inference questions, no content should describe the motivation. In behavior inference questions, no content should describe the behavior. For questions involving both motivation and behavior inference, there should be no descriptions of either motivation or behavior. Only the scenario and character identity traits should be described. If any such descriptions exist, remove them.
2. Next, check whether the correct answer is explicitly indicated in the question stem. If it is, remove the corresponding content.
3. Based on the correct answer, enrich the descriptions of the scenario and character identity traits in all three questions to ensure that the correct answer remains unique.
4. Be sure to further add a significant amount of distracting and redundant information in all three questions to increase their level of difficulty.
5. **Modify any options that are too easily identified as incorrect to make them more easily confused with the correct answer. All incorrect options must correspond to the distracting information in the question stem. If no such information exists, add it.**
6. Finally, ensure that the correct answer is the best possible answer. If it isn’t, revise the question stem or the options.
7. Analyze the options for each question step by step. Each analysis must be as detailed as possible, and the correct answer must be justified by referring to the information provided in the question.

Only return the modified answers in JSON format:
{
    "Behavior Inference Question": "A detailed description of a scenario and character identity. Considering a particular motivation of the character, what kind of behavior is he most likely to exhibit?",
    "Options": ["A. behavior_1", "B. behavior_2", "C. behavior_3", "D. behavior_4", "E. behavior_5", "F. behavior_6"],
    "Correct Answer": "{ONE LETTER}",
    "Question Analysis": ["A. analysis_1", "B. analysis_2", "C. analysis_3", "D. analysis_4", "E. analysis_5", "F. analysis_6"]
}

"""

adjuster_3 = """
You are a professional social and psychological researcher, and I will provide you with a motivation and behavior reasoning question.

Motivation and Behavior Inference Question: This is a more difficult question. Only a complex scenario and a character identity are provided. Infer the most likely behavior the character would perform next and its corresponding motivation.

Please make appropriate adjustments step by step:
1. In motivation inference questions, no content should describe the motivation. In behavior inference questions, no content should describe the behavior. For questions involving both motivation and behavior inference, there should be no descriptions of either motivation or behavior. Only the scenario and character identity traits should be described. If any such descriptions exist, remove them.
2. Next, check whether the correct answer is explicitly indicated in the question stem. If it is, remove the corresponding content.
3. Based on the correct answer, enrich the descriptions of the scenario and character identity traits in all three questions to ensure that the correct answer remains unique.
4. Be sure to further add a significant amount of distracting and redundant information in all three questions to increase their level of difficulty.
5. **Modify any options that are too easily identified as incorrect to make them more easily confused with the correct answer. All incorrect options must correspond to the distracting information in the question stem. If no such information exists, add it.**
6. Finally, ensure that the correct answer is the best possible answer. If it isn’t, revise the question stem or the options.
7. Analyze the options for each question step by step. Each analysis must be as detailed as possible, and the correct answer must be justified by referring to the information provided in the question.

Only return the modified answers in JSON format:
{
    "Motivation and Behavior Inference Question": "A detailed description of a scenario and character identity. What kind of behavior is this character most likely to exhibit next, and what is the motivation behind it?",
    "Options": ["A. motivation_1, behavior_1", "B. motivation_2, behavior_2", "C. motivation_3, behavior_3", "D. motivation_4, behavior_4", "E. motivation_5, behavior_5", "F. motivation_6, behavior_6"]
    "Correct Answer": "{ONE LETTER}",
    "Question Analysis": ["A. analysis_1", "B. analysis_2", "C. analysis_3", "D. analysis_4", "E. analysis_5", "F. analysis_6"]
}

"""

def main():
    with open("HumanSimulator/benchmark/persona/persona.jsonl", "r") as f:
        persona_list = f.readlines()

    random.shuffle(persona_list)

    testset = []

    for persona in tqdm(persona_list[0:200]):
        persona = json.loads(persona)["persona"]
        # generate question
        questioner_prompt_completed = questioner_prompt + persona
        question = get_parsed_response(questioner_prompt_completed)

        iter_num = 0
        while iter_num < 2:
            suggestion_1 = get_parsed_response(critic_prompt_1 + str(question))["suggestion"]
            suggestion_2 = get_parsed_response(critic_prompt_2 + str(question))["suggestion"]
            suggestion_3 = get_parsed_response(critic_prompt_3 + str(question))["suggestion"]

            if suggestion_1 == "No issues." and suggestion_2 == "No issues." and suggestion_3 == "No issues.":
                break

            suggestions = f"Suggestions:\n1. {suggestion_1}\n2. {suggestion_2}\n3. {suggestion_3}"
            question = str(question)
            modifier_answer_dict = get_parsed_response(modifier_prompt + f"Question:\n{question}\n\n{suggestions}")
            question = modifier_answer_dict

            iter_num += 1

        question_1 = "Motivation Inference Question:\n" + question["Motivation Inference Question"] + "\n\nOptions:\n" + str(question["Options 1"]) + "\n\nCorrect Answer:\n" + question["Correct Answer 1"]
        question_2 = "Behavior Inference Question:\n" + question["Behavior Inference Question"] + "\n\nOptions:\n" + str(question["Options 2"]) + "\n\nCorrect Answer:\n" + question["Correct Answer 2"]
        question_3 = "Motivation and Behavior Inference Question:\n" + question["Motivation and Behavior Inference Question"] + "\n\nOptions:\n" + str(question["Options 3"]) + "\n\nCorrect Answer:\n" + question["Correct Answer 3"]
        final_queation_1 = get_parsed_response(adjuster_1 + question_1)
        final_queation_2 = get_parsed_response(adjuster_2 + question_2)
        final_queation_3 = get_parsed_response(adjuster_3 + question_3)

        final_queation = {
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
        print(testset[-1])
        with open("HumanSimulator/benchmark/preprocess/Persona_testset.json", "w") as f:
            json.dump(testset, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()