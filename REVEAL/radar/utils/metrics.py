import re

def extract_answer(text):
    if not text:
        return None
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip().lower() if match else None