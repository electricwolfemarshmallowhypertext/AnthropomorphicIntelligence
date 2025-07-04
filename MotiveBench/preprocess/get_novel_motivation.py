## Copyright (c) Microsoft Corporation.
## Licensed under the MIT license.

import re
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


f=open('/home/aiscuser/ds/HumanSimulator/benchmark/novel/To Kill a Mockingbird.txt','rb')
lines=f.readlines()

novel_context = ""
num = 0
total_num = 0
BREAK_NUM = 100

prompt = """
In specific scenarios, characters with specific identities develop different motivations, which in turn lead to different behaviors. In novel texts, there are abundant scenarios, character identities, and behavioral information, and extracting the motivations behind these behaviors is of great significance. Suppose you are a professional psychologist and sociologist capable of completing character analysis based on given requirements.

I will provide you with a passage from a novel, assuming you have never read this novel before. Please extract high-quality quadruples {scenario, character identity, motivation, behavior} from the text. The specific requirements are as follows:
1. The text you will receive may be lengthy. Please conduct a holistic understanding and extraction. Focus on identifying the character's identity, motivation, and behavior on a macro level.
2. The quadruples must be logically coherent and reasonable. The scenario and character identity should give rise to a reasonable motivation, and the behavior should reasonably follow from the motivation.
3. The scenario and character identity should be as vivid, specific, and detailed as possible. The motivation and behavior should be as clear and explicit as possible.
4. High-quality quadruples mean that the extracted motivation should uncover deeper needs or intentions behind the character, with the motivation being grounded in real life and having valuable and meaningful insights.
5. If a high-quality quadruple can be extracted, return only the following fixed JSON format:
{
  "scenario": {DETAILED_SCENARIO},
  "profile": {DETAILED_PROFILE},
  "motivation": {MOTIVATION}, 
  "behavior": {BEHAVIOR}
}
If no high-quality quadruple can be extracted, return only:
{
  "answer": "Unable to extract."
}

Passage Content:
"""

motivations = []

for line in tqdm(lines):
    if num == BREAK_NUM:
        novel_context = ""
        num = 0
    else:
        num += 1
    total_num += 1
    if total_num == 12500:
        break
    line = line.lstrip(b"\xa1").rstrip(b"\n")
    novel_context = novel_context + ' ' + line.decode("utf-8")

    if num == BREAK_NUM:
        # print(novel_context)
        prompt_completed = prompt + novel_context
        answer = get_model_response(prompt_completed).strip("```").strip("json").strip()
        try:
            answer = json.loads(answer)
            if "scenario" in answer.keys():
                motivation = {
                    "scenario": answer["scenario"],
                    "profile": answer["profile"],
                    "motivation": answer["motivation"],
                    "behavior": answer["behavior"]
                }
                motivations.append(motivation)

                with open("HumanSimulator/benchmark/preprocess/to_kill_a_mockingbird_motivations.json", "w") as f:
                    json.dump(motivations, f, ensure_ascii=False, indent=4)
        except:
            continue
