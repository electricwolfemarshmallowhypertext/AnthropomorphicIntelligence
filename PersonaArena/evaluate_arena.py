import json
import os
import numpy as np
from tqdm import tqdm
from openai import OpenAI  
import time
import re
import csv
import argparse
from utils import utils
from yacs.config import CfgNode
from langchain.schema import HumanMessage
from filelock import FileLock
import threading
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Optional, NamedTuple

os.environ["OPENAI_API_VERSION"] = "2024-12-01-preview"

lock = threading.Lock()

METRICS = [
    "Knowledge Accuracy",
    "Emotional Expression",
    "Personality Traits",
    "Behavioral Accuracy",
    "Immersion",
    "Adaptability",
    "Behavioral Coherence",
    "Interaction Richness",
]

def extract_scores(response):
    regex = (
        r"Knowledge Accuracy:\s*\[?\s*(\d+)\s*\]?.*?"
        r"Emotional Expression:\s*\[?\s*(\d+)\s*\]?.*?"
        r"Personality Traits:\s*\[?\s*(\d+)\s*\]?.*?"
        r"Behavioral Accuracy:\s*\[?\s*(\d+)\s*\]?.*?"
        r"Immersion:\s*\[?\s*(\d+)\s*\]?.*?"
        r"Adaptability:\s*\[?\s*(\d+)\s*\]?.*?"
        r"Behavioral Coherence:\s*\[?\s*(\d+)\s*\]?.*?"
        r"Interaction Richness:\s*\[?\s*(\d+)\s*\]?.*?"
    )
    match = re.search(regex, response or "", re.DOTALL)
    if not match:
        return tuple([-1] * len(METRICS))
    return tuple(int(x) for x in match.groups())

def extract_scores_json(response: str):
    try:
        obj = json.loads((response or "").strip())
        key_map = {
            "knowledge accuracy": "Knowledge Accuracy",
            "emotional expression": "Emotional Expression",
            "personality traits": "Personality Traits",
            "behavioral accuracy": "Behavioral Accuracy",
            "immersion": "Immersion",
            "adaptability": "Adaptability",
            "behavioral coherence": "Behavioral Coherence",
            "interaction richness": "Interaction Richness",
        }
        out = []
        for k in METRICS:
            v = None
            if k in obj:
                v = obj[k]
            else:
                for kk in obj.keys():
                    if key_map.get(kk.lower(), kk) == k:
                        v = obj[kk]
                        break
            if not isinstance(v, int):
                return tuple([-1]*len(METRICS))
            out.append(int(v))
        return tuple(out)
    except Exception:
        return tuple([-1]*len(METRICS))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config_file", type=str, default="config/evaluate.yaml", help="Path to config file"
    )
    parser.add_argument(
        "-l", "--log_file", type=str, default="", help="Path to log file"
    )
    parser.add_argument(
        "-n", "--log_name", type=str, default=str(os.getpid()), help="Name of logger"
    )
    args = parser.parse_args()
    return args

def strip_reasoning_blocks(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()

def _call_llm(LLM, prompt: str) -> str:
    if 'Chat' in type(LLM).__name__:
        raw = LLM([HumanMessage(content=prompt)]).content
    else:
        raw = LLM(prompt)
    return strip_reasoning_blocks(raw)

def _pick_judge_creds(j: dict, config: CfgNode):
    
    mdl = j.get("model") or j.get("llm") or j.get("name")
    ak = j.get("api_key", config.get("api_key", "empty"))
    ab = j.get("api_base", config.get("api_base", ""))
    if ak in (None, ""):
        ak = "empty"
    return mdl, ak, ab
# =====================================================================

def critic(LLM,scene,character,actions):
    prompt = f"""
Please execute the following role-play and identify any issues based on these strict evaluation criteria:

- Scene Description: 
{scene}

- Character Description:
{character}

- Character Actions: 
{actions}

Strict Evaluation Criteria:
1. Factual Accuracy: Flag elements that conflict with the historical or factual backdrop.
2. Character Consistency: Identify mismatches between actions/dialogue and defined traits or goals.
3. Logical Coherence: Point out fallacies or contradictions with established context or character logic.
4. Content Redundancy: Note repetitive dialogue or actions that reduce engagement or realism.
5. Emotional Expression: Judge whether emotions are appropriate and convincingly conveyed; mark discrepancies.
6. Interaction Adaptability: Critique unnatural or contextually inappropriate responses to others.
7. Creativity and Originality: Call out generic or unoriginal actions or replies.
8. Detail Handling: Scrutinize scene and behavior details for depth and accuracy gaps.
9. Style Consistency: Ensure narrative and language style remain consistent; note deviations.
10. Fluency and Quality: Critically assess the smoothness and quality of the text, highlighting any grammatical errors or awkward phrasings.
Condense the issues into one paragraph.
    """
    return _call_llm(LLM, prompt)

def get_num_tokens(text: str) -> int:
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    return len(tokenizer.encode(text))

def _normalize_judges(config: CfgNode):

    out = []
    global_key = config.get('api_key')
    global_base = config.get('api_base')
    global_provider = config.get('provider', None)
    global_api_version = config.get('api_version', None)

    def _push(model, name=None, api_key=None, api_base=None, provider=None, api_version=None):
        if not model:
            return
        out.append({
            "model": model,
            "name": (name or model),
            "api_key": (api_key if api_key not in (None, "") else global_key),
            "api_base": (api_base if api_base not in (None, "") else global_base),
            "provider": (provider if provider is not None else global_provider),
            "api_version": (api_version if api_version is not None else global_api_version),
        })

    judges_cfg = config.get('judges', None)
    if judges_cfg:
        if isinstance(judges_cfg, str):
            _push(judges_cfg)
        elif isinstance(judges_cfg, dict):
            model = judges_cfg.get("model") or judges_cfg.get("llm") or judges_cfg.get("name")
            _push(model,
                  name=judges_cfg.get("name"),
                  api_key=judges_cfg.get("api_key"),
                  api_base=judges_cfg.get("api_base"),
                  provider=judges_cfg.get("provider"),
                  api_version=judges_cfg.get("api_version"))
        elif isinstance(judges_cfg, (list, tuple)):
            for item in judges_cfg:
                if isinstance(item, str):
                    _push(item)
                elif isinstance(item, dict):
                    model = item.get("model") or item.get("llm") or item.get("name")
                    _push(model,
                          name=item.get("name"),
                          api_key=item.get("api_key"),
                          api_base=item.get("api_base"),
                          provider=item.get("provider"),
                          api_version=item.get("api_version"))
    else:
        jd = config.get('judger_llm', "")
        if jd:
            _push(jd)

    return out

def _strip_code_fences(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"^\s*(?:json)?\s*", "", s.strip(), flags=re.IGNORECASE)
    s = re.sub(r"\s*\s*$", "", s)
    return s

def _first_json_block(s: str):
    if not s:
        return None
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    return m.group(0) if m else None

def _coerce_1to5(v) -> int:
    try:
        if isinstance(v, bool):
            v = int(v)
        elif isinstance(v, (int, float, str)):
            v = int(float(v))
        else:
            return 3
    except Exception:
        return 3
    return max(1, min(5, v))

def normalize_scores_dict(obj: dict) -> tuple[int, ...]:
    key_map = {
        "knowledge accuracy": "Knowledge Accuracy",
        "emotional expression": "Emotional Expression",
        "personality traits": "Personality Traits",
        "behavioral accuracy": "Behavioral Accuracy",
        "immersion": "Immersion",
        "adaptability": "Adaptability",
        "behavioral coherence": "Behavioral Coherence",
        "interaction richness": "Interaction Richness",
    }
    out = []
    for k in METRICS:
        v = None
        if k in obj:
            v = obj[k]
        else:
            for kk in obj.keys():
                if key_map.get(str(kk).lower(), kk) == k:
                    v = obj[kk]
                    break
        if v is None:
            v = 3  
        out.append(_coerce_1to5(v))
    return tuple(out)

def robust_extract_scores(response: str) -> tuple[int, ...]:
    try:
        txt = _strip_code_fences(strip_reasoning_blocks(response or ""))
        block = _first_json_block(txt) or txt
        obj = json.loads(block)
        if not isinstance(obj, dict):
            return tuple([-1]*len(METRICS))
        return normalize_scores_dict(obj)
    except Exception:
        scores = extract_scores_json(response)
        if scores[0] != -1:
            return tuple(_coerce_1to5(x) for x in scores)
        scores = extract_scores(response)
        if scores[0] != -1:
            return tuple(_coerce_1to5(x) for x in scores)
        return tuple([-1]*len(METRICS))

def build_scoring_criteria_block() -> str:
    return """
[Scoring Criteria]:
1. Knowledge Accuracy:
   - 1: Often incorrect/irrelevant; conflicts with background.
   - 3: Generally accurate; occasional errors or weak relevance.
   - 5: Always accurate and highly relevant; shows deep knowledge.
2. Emotional Expression:
   - 1: Monotonous or inappropriate to content/context.
   - 3: Moderately varied but lacks depth/subtlety.
   - 5: Rich, nuanced, highly consistent with context.
3. Personality Traits:
   - 1: Conflicts with or lacks consistency to setup.
   - 3: Generally matches; occasional inconsistencies.
   - 5: Consistently matches core traits; shows uniqueness.
4. Behavioral Accuracy:
   - 1: Fails to capture behaviors/linguistic habits.
   - 3: Reflects partially; not precise/complete.
   - 5: Accurately mimics specific behaviors and phrases.
5. Immersion:
   - 1: Inconsistent portrayal; hard to immerse.
   - 3: Mostly consistent; some contradictions.
   - 5: Always consistent; enhances immersion/self-awareness.
6. Adaptability:
   - 1: Lacks adaptability to new situations.
   - 3: Adapts in most cases; sometimes inflexible.
   - 5: Always adapts while maintaining consistency.
7. Behavioral Coherence:
   - 1: Responses often illogical to plot/dialogue.
   - 3: Generally coherent; some unreasonable parts.
   - 5: Always logical; adjusts with plot progression.
8. Interaction Richness:
   - 1: Repeats nearly identical statements; little progress.
   - 3: Occasionally varies with some new info.
   - 5: Consistently fresh, varied, advances conversation.

You MUST output ONLY a JSON object with EXACTLY these 9 keys and integer values 1-5:
{
  "Knowledge Accuracy": <int>,
  "Emotional Expression": <int>,
  "Personality Traits": <int>,
  "Behavioral Accuracy": <int>,
  "Immersion": <int>,
  "Adaptability": <int>,
  "Behavioral Coherence": <int>,
  "Interaction Richness": <int>,
}
No other text, no commentary, no code fences.
"""

def repair_to_json(JLLM, raw_text: str) -> str:
    prompt = (
        "Convert the following content into EXACTLY the required JSON with the 9 keys and integer values 1-5. "
        "Return ONLY the JSON object, with no extra text or code fences.\n\n"
        "CONTENT START\n"
        f"{raw_text}\n"
        "CONTENT END\n"
        + build_scoring_criteria_block()
        + "\n[Response]:\n"
    )
    return _call_llm(JLLM, prompt)

def ask_fresh_json(JLLM, base_prompt: str, critique: str) -> str:
    prompt = base_prompt + f"[Critique]:\n{critique}\n" + build_scoring_criteria_block() + "\n[Response]:\n"
    return _call_llm(JLLM, prompt)

class Dispute(NamedTuple):
    metric_idx: int
    metric_name: str
    variance: float
    judge_scores: List[Tuple[int, int]]

def _find_disputes(
    all_scores: List[Tuple[int, ...]],
    metric_names: List[str],
    var_threshold: float,
    topk: int
) -> List[Dispute]:
    if not all_scores:
        return []
    disputes: List[Dispute] = []
    num_metrics = len(metric_names)
    for m in range(num_metrics):
        col = [(j, all_scores[j][m]) for j in range(len(all_scores)) if all_scores[j][m] != -1]
        if len(col) < 2:
            continue
        scores = [s for _, s in col]
        var = float(np.var(scores))
        if var > var_threshold:
            disputes.append(Dispute(
                metric_idx=m,
                metric_name=metric_names[m],
                variance=var,
                judge_scores=col
            ))
    disputes.sort(key=lambda d: d.variance, reverse=True)
    return disputes[:topk]

JUDGE_STATEMENT_PROMPT = """
You are Judge {judge_name}. Provide a critic statement for the disputed metric.

[Metric]: {metric_name}
[Scale]: 1 (poor) to 5 (excellent), integers only.
[Assigned Score]: {score}

[Scene]
{scene_text}

[Character]
{character_info}

[Actions]
{actions}

Task:
Return ONLY a JSON object with exactly these keys:
{{
  "score": <int>,
  "justification": "<short text>",
  "evidence": ["<excerpt 1>", "<excerpt 2>"]
}}
Rules:
- Keep the score unchanged.
- Evidence excerpts should be short quotes copied from [Actions] that support the score.
- If no evidence is available, use an empty list.
"""

JUDGE_STATEMENT_REPAIR_PROMPT = """
Convert the following content into EXACTLY the required JSON with the keys:
{{
  "score": <int>,
  "justification": "<short text>",
  "evidence": ["<excerpt 1>", "<excerpt 2>"]
}}
Return ONLY the JSON object, with no extra text or code fences.

CONTENT START
{raw}
CONTENT END
"""

DEBATE_ARBITER_PROMPT = """
You are an impartial referee. Given the scene, character, actions, a disputed metric, and the judges' critic statements:

[Metric]: {metric_name}
[Scale]: 1 (poor) to 5 (excellent), integers only.
[Variance]: {variance}

[Scene]
{scene_text}

[Character]
{character_info}

[Actions]
{actions}

[Judge Statements]
{judge_statements}

Task:
1) Synthesize the statements into a unified rationale for the metric.
2) Output a SINGLE final integer in 1..5 as the reconciled score for this metric.
3) Strictly follow the output format:

Final Score: [X]
Unified Rationale: (1-3 sentences)
"""

_DEBATE_SCORE_RX = re.compile(r"Final\s*Score\s*:\s*\[?\s*([1-5])\s*\]?", re.IGNORECASE)
_DEBATE_RATIONALE_RX = re.compile(r"Unified\s*Rationale\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)

def _parse_judge_statement(resp: str, fallback_score: int) -> Tuple[Dict[str, Any], bool]:
    raw = _strip_code_fences(resp or "")
    block = _first_json_block(raw) or raw
    try:
        obj = json.loads(block)
    except Exception:
        return {"score": fallback_score, "justification": raw.strip(), "evidence": []}, False
    score = obj.get("score", fallback_score)
    try:
        score = int(score)
    except Exception:
        score = fallback_score
    evidence = obj.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = []
    evidence = [str(x) for x in evidence][:3]
    justification = str(obj.get("justification", "")).strip()
    return {"score": score, "justification": justification, "evidence": evidence}, True

def _repair_judge_statement(judge_llm, raw_resp: str) -> str:
    prompt = JUDGE_STATEMENT_REPAIR_PROMPT.format(raw=raw_resp or "")
    return _call_llm(judge_llm, prompt)

def _build_judge_statement(
    judge_llm,
    judge_name: str,
    metric_name: str,
    score: int,
    scene_text: str,
    character_info: str,
    actions: str,
    logger=None,
) -> Dict[str, Any]:
    prompt = JUDGE_STATEMENT_PROMPT.format(
        judge_name=judge_name,
        metric_name=metric_name,
        score=score,
        scene_text=scene_text.strip(),
        character_info=character_info.strip(),
        actions=actions.strip(),
    )
    try:
        resp = _call_llm(judge_llm, prompt) or ""
    except Exception:
        resp = ""
    stmt, parsed_ok = _parse_judge_statement(resp, score)
    if not parsed_ok and resp:
        try:
            repaired = _repair_judge_statement(judge_llm, resp)
        except Exception:
            repaired = ""
        stmt2, parsed_ok2 = _parse_judge_statement(repaired, score)
        if parsed_ok2:
            stmt = stmt2
        if logger:
            try:
                logger.warning(
                    f"[judge_statement] parse_failed name={judge_name} metric={metric_name} "
                    f"repaired={parsed_ok2}"
                )
            except Exception:
                pass
    stmt["judge_name"] = judge_name
    stmt["score"] = score
    return stmt

def _run_debate_once(
    referee_llm,
    metric_name: str,
    scene_text: str,
    character_info: str,
    actions: str,
    judge_statements: List[Dict[str, Any]],
    variance: float,
) -> Optional[Tuple[int, str]]:
    prompt = DEBATE_ARBITER_PROMPT.format(
        metric_name=metric_name,
        scene_text=scene_text.strip(),
        character_info=character_info.strip(),
        actions=actions.strip(),
        judge_statements=json.dumps(judge_statements, ensure_ascii=True, indent=2),
        variance=f"{variance:.4f}",
    )
    try:
        resp = _call_llm(referee_llm, prompt) or ""
    except Exception:
        return None
    m = _DEBATE_SCORE_RX.search(resp)
    if not m:
        return None

    score = int(m.group(1))
    reason_match = _DEBATE_RATIONALE_RX.search(resp)
    reason = reason_match.group(1).strip() if reason_match else ""
    return score, reason

def _is_target_character(char_key, actions, target_cid_str: str) -> bool:
    try:
        key_norm = str(int(char_key))
    except Exception:
        key_norm = str(char_key).strip()
    try:
        item_norm = str(int(actions[0].get("character_id", char_key)))
    except Exception:
        v = actions[0].get("character_id", char_key)
        item_norm = str(v).strip()
    return (key_norm == target_cid_str) or (item_norm == target_cid_str)

def run_eval_detail(config, logger, character_record, SAVE_PATH, done):
    narrator = config['narrator_llm']
    character_model = config['character_llm'] 
    scene_id = config['scene_id']
    print("DEBUG: scene_id from config =", scene_id)
    max_rounds = config['max_rounds']
    title = config['title']

    target_cid_cfg = config.get("target_character_id", 0)
    try:
        TARGET_CID = str(int(target_cid_cfg))
    except Exception:
        TARGET_CID = str(target_cid_cfg).strip()

    debate_enabled = bool(config.get("debate_enabled", True))
    debate_var_threshold = float(config.get("debate_var_threshold", config.get("debate_gap", 2)))
    debate_topk = int(config.get("debate_topk", 8))
    debate_referee = config.get("debate_referee", None)  

    judges = _normalize_judges(config)
    if not judges:
        raise ValueError("No judge models provided. Set 'judges' or 'judger_llm' in config.")

    judge_llms, judge_names = [], []
    for j in judges:
        mdl, ak, ab = _pick_judge_creds(j, config)
        cfg_j = deepcopy(config)
        if j.get("provider") is not None:
            cfg_j["provider"] = j["provider"]
        if j.get("api_version") is not None:
            cfg_j["api_version"] = j["api_version"]
        judge_llms.append(utils.get_llm(mdl, cfg_j, logger, ak, ab))
        judge_names.append(j.get("name", mdl))
    print("Judges:", judge_names)

    ids = list(character_record.keys())
    print(f"Scene ID: {scene_id}\n Characters:{character_record.keys()}")

    lock_path = SAVE_PATH + '.lock'
    file_lock = FileLock(lock_path)

    with open(SAVE_PATH, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Title','Judger','Narrator','Model','SceneID', 'Round',
                      'SceneInfo','CharacterInfo','Critic','JudgeScores'] + METRICS
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore') 

        csvfile.seek(0, 2)
        if csvfile.tell() == 0:
            writer.writeheader()
        
        found_target = False
        for id in tqdm(ids, desc=f"Scene {scene_id} Characters"):
            actions = character_record[id]
            if not actions:
                continue
            if not _is_target_character(id, actions, TARGET_CID):
                continue
            found_target = True

            name = actions[0]['character_name']
            print("Processing (protagonist): ", actions[0]['character_name'])

            event = actions[0]["detail"]['event']
            scene_time = actions[0]["detail"]['time']
            location = actions[0]["detail"]['location']
            description = actions[0]["detail"]['description']

            scene = (
                f"Scenario Information:\n"
                f"Event: {event}\n"
                f"Time: {scene_time}\n"
                f"Location: {location}\n"
                f"Description: {description}\n"
            )

            character_static_info = (
                f"Name: {name}\n"
                f"Description: {actions[0]['detail']['character_description']}\n")

            if character_static_info in done:
                print(f"{character_model}: Skipping character {name} as it is already evaluated.")
                continue

            prompt_head = (
                "Please evaluate the role-playing ability of the character based on actions across multiple turns "
                "based on scene, character information, critique and evaluation criteria.\n"
                f"[Scene]:\n{scene}\n[Character]:\n{character_static_info}\n[Multi-turn Actions]:\n"
            )
            last_round = 0
            prompt_actions = ""
            behaviors = ""
            for action in actions:
                round_id = action.get('round', 0)
                if round_id > max_rounds:
                    break
                if round_id != last_round:
                    last_round = round_id
                    prompt_actions += "Round: " + str(round_id) + "\n"
                obs = (action.get('detail', {}).get('observation') or "")
                txt = (action.get('detail', {}).get('text') or "")
                prompt_actions += f"Observation: {obs}\n"
                behavior = "Action:\n"
                if action.get('type') == 'dialogue' and txt:
                    behavior += f"{name}: {txt}\n"
                else:
                    behavior += f"{txt}\n"
                prompt_actions += f"{behavior}\n"
                behaviors += f"Observation: {obs}\n{behavior}\n"

            panel_sum = np.zeros(len(METRICS), dtype=float)
            panel_cnt = 0
            critic_list: List[str] = []

            per_judge_scores: List[Tuple[int, ...]] = []
            per_judge_critics: List[str] = []

            criteria = build_scoring_criteria_block()

            for jmeta, JLLM in zip(judges, judge_llms):
                jname = jmeta.get('name', jmeta.get('model'))
                try:
                    problem = critic(JLLM, scene, character_static_info, behaviors)
                except Exception as e:
                    problem = "(no critique due to error)"

                full_prompt = prompt_head + prompt_actions + f"[Critique]:\n{problem}\n" + criteria + "\n[Response]:\n"

                scores = tuple([-1]*len(METRICS))
                parse_ok = False
                last_response = ""

                for _try in range(3):
                    try:
                        last_response = _call_llm(JLLM, full_prompt)
                    except Exception:
                        last_response = ""
                    scores = robust_extract_scores(last_response)
                    if scores[0] != -1:
                        parse_ok = True
                        break
                    time.sleep(0.2 * (_try + 1))

                if not parse_ok:
                    try:
                        fixed = repair_to_json(JLLM, last_response or "")
                        scores = robust_extract_scores(fixed)
                        if scores[0] != -1:
                            parse_ok = True
                    except Exception:
                        pass

                if not parse_ok:
                    try:
                        fresh = ask_fresh_json(JLLM, prompt_head + prompt_actions, problem)
                        scores = robust_extract_scores(fresh)
                        if scores[0] != -1:
                            parse_ok = True
                    except Exception:
                        pass

                if not parse_ok:
                    scores = tuple([3]*len(METRICS))

                per_judge_scores.append(scores)
                per_judge_critics.append(problem)
                panel_sum += np.array(scores, dtype=float)
                panel_cnt += 1
                critic_list.append(f"[{jname}] {problem}")

            if panel_cnt == 0:
                continue

            final_metric_scores: Dict[int, int] = {}

            if debate_enabled:

                # === DEBUG START ===
                try:
                    print("Judges:", judge_names)
                    print("debate_enabled=", debate_enabled, "debate_var_threshold=", debate_var_threshold, "debate_topk=", debate_topk, flush=True)
                    print("per_judge_scores:", per_judge_scores, flush=True)
                except Exception:
                    pass
                # === DEBUG END ===

                disputes = _find_disputes(per_judge_scores, METRICS, var_threshold=debate_var_threshold, topk=debate_topk)

                try:
                    print("disputes_found:", [(d.metric_name, round(d.variance, 4)) for d in disputes], flush=True)
                except Exception:
                    pass

                if disputes:

                    if debate_referee:
                        referee_llm = utils.get_llm(
                            debate_referee, config, logger,
                            config.get("debate_referee_api_key", config.get("api_key", "empty")),
                            config.get("debate_referee_api_base", config.get("api_base", "")),
                            "debate_referee"
                        )
                        referee_name = f"referee:{debate_referee}"
                    else:
                        referee_llm = judge_llms[0]
                        referee_name = f"referee:{judge_names[0]}"

                    debate_notes = []
                    for d in disputes:
                        m_idx = d.metric_idx
                        judge_statements = []
                        for j_idx, score in d.judge_scores:
                            stmt = _build_judge_statement(
                                judge_llm=judge_llms[j_idx],
                                judge_name=judge_names[j_idx],
                                metric_name=METRICS[m_idx],
                                score=score,
                                scene_text=scene,
                                character_info=character_static_info,
                                actions=behaviors,
                                logger=logger,
                            )
                            judge_statements.append(stmt)

                        final_m, final_reason = _run_debate_once(
                            referee_llm=referee_llm,
                            metric_name=METRICS[m_idx],
                            scene_text=scene,
                            character_info=character_static_info,
                            actions=behaviors,
                            judge_statements=judge_statements,
                            variance=d.variance,
                        )
                        if final_m is None:
                            continue
                        final_metric_scores[m_idx] = final_m

                        note = (
                            f"[Debate/{referee_name}] Metric={METRICS[m_idx]} variance={d.variance:.4f} "
                            f"Final:{final_m} Reason:{final_reason}"
                        )
                        debate_notes.append(note)

                    if debate_notes:
                        for n in debate_notes:
                            try:
                                logger.info(n)
                            except Exception:
                                pass
                        critic_list.append("\n".join(debate_notes))
            # ===== End Debate =====

            panel_avg = (panel_sum / max(panel_cnt, 1)).tolist()
            for m_idx, final_score in final_metric_scores.items():
                panel_avg[m_idx] = float(final_score)

            judge_scores_map = {}
            for jname, scores in zip(judge_names, per_judge_scores):
                judge_scores_map[jname] = {m: int(scores[i]) for i, m in enumerate(METRICS)}

            data_row = {
                'Title': title,
                'Judger': ", ".join(judge_names), 
                'Narrator': narrator,
                'Model': character_model,
                'SceneID': scene_id,
                'Round': last_round,
                'SceneInfo': scene,
                'CharacterInfo': character_static_info,
                'Critic': "\n\n".join(critic_list),
                'JudgeScores': json.dumps(judge_scores_map, ensure_ascii=True),
            }
            for i, m in enumerate(METRICS):
                data_row[m] = round(panel_avg[i], 3) 

            with file_lock:
                writer.writerow(data_row)

        if not found_target:
            try:
                logger.warning(f"[eval] target character not found (target_character_id={TARGET_CID}).")
            except Exception:
                pass

    return scene_id

if __name__ == '__main__':

    args = parse_args() 
    
    config = CfgNode(new_allowed=True)
    config = utils.add_variable_to_config(config, "log_file", args.log_file)
    config = utils.add_variable_to_config(config, "log_name", args.log_name)
    config.merge_from_file(args.config_file)

    judges = _normalize_judges(config)
    narrator = config['narrator_llm']
    character = config['character_llm']
    title = config['title']
    scene_id = config['scene_id']

    target_cid_cfg = config.get("target_character_id", 0)
    try:
        TARGET_CID = str(int(target_cid_cfg))
    except Exception:
        TARGET_CID = str(target_cid_cfg).strip()

    debate_enabled = bool(config.get("debate_enabled", True))
    debate_var_threshold = float(config.get("debate_var_threshold", config.get("debate_gap", 0)))
    debate_topk = int(config.get("debate_topk", 8))
    debate_referee = config.get("debate_referee", None) 

    utils.ensure_dir("output/evaluation/multi/"+title)

    RECORD_PATH = f"output/record/{title}/{narrator}_{character}_{scene_id}_character.json"

    SAVE_PATH = f"output/evaluation/multi/{title}/{narrator}_{scene_id}_character_evaluation_avg.csv"
    max_rounds = config['max_rounds'] 
    character_record = []

    if args.log_file == "":
        utils.ensure_dir("output/log/evaluation/multi/"+title)
        args.log_file = f"evaluation/multi/{title}/{narrator}_{character}_{scene_id}_character_evaluation_avg.log"
    logger = utils.set_logger(args.log_file, args.log_name)
    logger.info(f"os.getpid()={os.getpid()}")
    logger.info(f"\n{config}")

    with open(RECORD_PATH, 'r', encoding='utf-8') as f:
        character_record = json.load(f)
    
    if not judges:
        judges = [{"name": config['judger_llm'], "model": config['judger_llm']}]

    judge_llms, judge_names = [], []
    for j in judges:
        mdl, ak, ab = _pick_judge_creds(j, config)
        cfg_j = deepcopy(config)
        if j.get("provider") is not None:
            cfg_j["provider"] = j["provider"]
        if j.get("api_version") is not None:
            cfg_j["api_version"] = j["api_version"]
        judge_llms.append(utils.get_llm(mdl, cfg_j, logger, ak, ab))
        judge_names.append(j.get("name", mdl))
    print("Judges(main):", judge_names)

    ids = list(character_record.keys())

    with open(SAVE_PATH, 'a', newline='', encoding='utf-8') as csvfile:

        fieldnames = ['Title','Judger','Narrator','Model','SceneID', 'Round'] + METRICS + ['Average', 'DebateCount', 'DebatedMetrics']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')  
        csvfile.seek(0, 2)
        if csvfile.tell() == 0:
            writer.writeheader()

        avg_scores = {k: 0.0 for k in METRICS}
        valid_count = 0
        last_round_global = 0

        criteria = build_scoring_criteria_block()

        debate_count = 0
        debate_metrics_set = set()

        def _is_target_character_summary(char_key, actions, target_cid_str: str) -> bool:
            try:
                key_norm = str(int(char_key))
            except Exception:
                key_norm = str(char_key).strip()
            try:
                item_norm = str(int(actions[0].get("character_id", char_key)))
            except Exception:
                v = actions[0].get("character_id", char_key)
                item_norm = str(v).strip()
            return (key_norm == target_cid_str) or (item_norm == target_cid_str)

        for id in tqdm(ids):
            actions = character_record[id]
            if not actions:
                continue
            if not _is_target_character_summary(id, actions, TARGET_CID):
                continue

            name = actions[0]['character_name']
            print("Processing (protagonist): ", actions[0]['character_name'])

            event = actions[0]["detail"]['event']
            scene_time = actions[0]["detail"]['time']
            location = actions[0]["detail"]['location']
            description = actions[0]["detail"]['description']

            scene = (
                f"Scenario Information:\n"
                f"Event: {event}\n"
                f"Time: {scene_time}\n"
                f"Location: {location}\n"
                f"Description: {description}\n"
            )

            character_static_info = (
                f"Name: {name}\n"
                f"Description: {actions[0]['detail']['character_description']}\n")
            
            prompt_head = (
                "please evaluate the role-playing ability of the character based on actions across multiple turns "
                "based on scene, character information, critique and evaluation criteria.\n"
                f"{scene}\n{character_static_info}\nMulti-turn Actions as follows:\n"
            )
            last_round = 0
            prompt_actions = ""
            behaviors = ""

            for action in actions:
                round_id = action.get('round', 0)
                if round_id > max_rounds:
                    break
                if round_id != last_round:
                    last_round = round_id
                    prompt_actions += "Round: " + str(round_id) + "\n"

                obs = (action.get('detail', {}).get('observation') or "")
                txt = (action.get('detail', {}).get('text') or "")
                prompt_actions += f"Observation: {obs}\n"

                behavior = "Action:\n"
                if action.get('type') == 'dialogue' and txt:
                    behavior += f"{name}: {txt}\n"
                else:
                    behavior += f"{txt}\n"
                
                prompt_actions += f"{behavior}\n"
                behaviors += f"Observation: {obs}\n{behavior}\n"

            last_round_global = max(last_round_global, last_round)

            panel_sum = np.zeros(len(METRICS), dtype=float)
            panel_cnt = 0
            per_judge_scores: List[Tuple[int, ...]] = []
            per_judge_critics: List[str] = []
            final_metric_scores: Dict[int, int] = {}

            for jmeta, JLLM in zip(judges, judge_llms):

                try:
                    problem = critic(JLLM, scene, character_static_info, behaviors)
                except Exception:
                    problem = "(no critique due to error)"

                full_prompt = f"{prompt_head}{prompt_actions}[Critique]:\n{problem}\n{criteria}\n[Response]:\n"

                scores = tuple([-1]*len(METRICS))
                parse_ok = False
                last_response = ""

                for _try in range(3):
                    try:
                        last_response = _call_llm(JLLM, full_prompt)
                    except Exception:
                        last_response = ""
                    scores = robust_extract_scores(last_response)
                    if scores[0] != -1:
                        parse_ok = True
                        break
                    time.sleep(0.2 * (_try + 1))

                if not parse_ok:
                    try:
                        fixed = repair_to_json(JLLM, last_response or "")
                        scores = robust_extract_scores(fixed)
                        if scores[0] != -1:
                            parse_ok = True
                    except Exception:
                        pass

                if not parse_ok:
                    try:
                        fresh = ask_fresh_json(JLLM, prompt_head + prompt_actions, problem)
                        scores = robust_extract_scores(fresh)
                        if scores[0] != -1:
                            parse_ok = True
                    except Exception:
                        pass

                if not parse_ok:
                    scores = tuple([3]*len(METRICS))

                per_judge_scores.append(scores)
                per_judge_critics.append(problem)
                panel_sum += np.array(scores, dtype=float)
                panel_cnt += 1

            if panel_cnt == 0:
                continue

            if debate_enabled:
                disputes = _find_disputes(per_judge_scores, METRICS, var_threshold=debate_var_threshold, topk=debate_topk)
                if disputes:
                    if debate_referee:
                        referee_llm = utils.get_llm(
                            debate_referee, config, logger,
                            config.get("api_key", "empty"), config.get("api_base", "")
                        )
                    else:
                        referee_llm = judge_llms[0]

                    for d in disputes:
                        m_idx = d.metric_idx
                        judge_statements = []
                        for j_idx, score in d.judge_scores:
                            stmt = _build_judge_statement(
                                judge_llm=judge_llms[j_idx],
                                judge_name=judge_names[j_idx],
                                metric_name=METRICS[m_idx],
                                score=score,
                                scene_text=scene,
                                character_info=character_static_info,
                                actions=behaviors,
                                logger=logger,
                            )
                            judge_statements.append(stmt)

                        final_m, final_reason = _run_debate_once(
                            referee_llm=referee_llm,
                            metric_name=METRICS[m_idx],
                            scene_text=scene,
                            character_info=character_static_info,
                            actions=behaviors,
                            judge_statements=judge_statements,
                            variance=d.variance,
                        )
                        if final_m is None:
                            continue

                        final_metric_scores[m_idx] = final_m

                      
                        debate_count += 1
                        debate_metrics_set.add(METRICS[m_idx])

                        
                        try:
                            logger.info(
                                f"[Debate/Summary] Metric={METRICS[m_idx]} variance={d.variance:.4f} "
                                f"Final:{final_m} Reason:{final_reason}"
                            )
                        except Exception:
                            pass
            # ===== End Debate =====

            panel_avg = (panel_sum / panel_cnt).tolist()
            for m_idx, final_score in final_metric_scores.items():
                panel_avg[m_idx] = float(final_score)

          
            for i, m in enumerate(METRICS):
                avg_scores[m] += panel_avg[i]
            valid_count += 1
        
       
        if valid_count > 0:
            avg_per_metric = {m: (avg_scores[m]/valid_count) for m in METRICS}
            avg_all = sum(avg_per_metric.values())/len(METRICS)

            row = {
                'Title': title,
                'Judger': ", ".join(judge_names),
                'Narrator': narrator,
                'Model': character,
                'SceneID': scene_id,
                'Round': last_round_global,
            }
            row.update(avg_per_metric)
            row['Average'] = avg_all

     
            row['DebateCount'] = debate_count
            row['DebatedMetrics'] = ";".join(sorted(debate_metrics_set)) if debate_metrics_set else ""

            writer.writerow(row)
        else:
            print("No valid protagonist scored in summary stage; nothing written.")
