#!/usr/bin/env python3
import argparse
import asyncio
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from openai import AsyncOpenAI

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

CRITIC_PROMPT_TEMPLATE = """
Please execute the following role-play and identify any issues based on these strict evaluation criteria:

- Target Character Name:
{target_character_name}

- Scene Description: 
{scene}

- Character Description:
{character}

- Character Actions: 
{actions}

Strict Evaluation Criteria:
1. Factual Accuracy: Identify and point out any elements that do not accurately match the historical or factual backdrop.
2. Character Consistency: Explicitly highlight inconsistencies between the character's actions, dialogues, and their predefined traits and goals.
3. Logical Coherence: Point out any logical fallacies or actions that contradict the established context or character logic.
4. Content Redundancy: Identify repetitions in dialogue or action that could detract from engagement and realism.
5. Emotional Expression: Assess whether emotional responses and expressions are appropriate and convincingly portrayed, highlighting any discrepancies.
6. Interaction Adaptability: Critique the character's interactions with others, noting any unnatural or contextually inappropriate responses.
7. Creativity and Originality: Evaluate the creativity of responses and actions, pointing out generic or unoriginal content.
8. Detail Handling: Scrutinize the level of detail in scene setting and character enactment, marking areas lacking depth or accuracy.
9. Style Consistency: Ensure that the narrative and linguistic style remains consistent, identifying any deviations.
10. Fluency and Quality: Critically assess the smoothness and quality of the text, highlighting any grammatical errors or awkward phrasings.
Important Scope Constraint:
- Evaluate ONLY the target character: {target_character_name}.
- Mentions of other characters are context only and MUST NOT be scored as their own performance.
- If other characters are discussed, tie the analysis back to how {target_character_name} responds/adapts.
Condense the issues into one paragraph.
"""

DEBATE_ARBITER_PROMPT = """
You are an impartial referee. Given the scene, character, actions, a disputed metric, and two judges' scores with their critiques:

[Target Character Name]: {target_character_name}
[Metric]: {metric_name}
[Scale]: 1 (poor) to 5 (excellent), integers only.

[Scene]
{scene_text}

[Character]
{character_info}

[Actions]
{actions}

[Judge High] name={high_name}, score={high_score}
Critique:
{high_critic}

[Judge Low] name={low_name}, score={low_score}
Critique:
{low_critic}

Task:
1) Briefly weigh whose reasoning better fits the metric definition.
2) Output a SINGLE final integer in 1..5 as the reconciled score for this metric.
3) Scoring scope is ONLY the target character ({target_character_name}); other roles are context.
4) Strictly follow the output format:

Final Score: [X]
Reason: (one short sentence)
"""

_DEBATE_SCORE_RX = re.compile(r"Final\s*Score\s*:\s*\[?\s*([1-5])\s*\]?", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Async evaluator for output/record/**/*_character.jsonl or *_character.jsnol trajectories."
    )
    p.add_argument("--config", default="config/evaluate.yaml", help="Path to evaluate.yaml")
    p.add_argument(
        "--input_dir",
        default="output/record",
        help="Root directory containing scene subfolders",
    )
    p.add_argument(
        "--output_dir",
        default="output_all/evaluation/by_model",
        help="Output base dir. If ends with 'by_model', results go to by_model/<character_llm>",
    )
    p.add_argument(
        "--glob",
        default="**/persona_detail/*_character.*",
        help="Recursive glob under input_dir",
    )
    p.add_argument("--max_files", type=int, default=0, help="0 means all files")
    p.add_argument("--index_range", default="", help="1-based inclusive file index range, e.g. 100-150")
    p.add_argument("--concurrency", type=int, default=4, help="File-level concurrency")
    p.add_argument("--judge_timeout", type=float, default=120.0, help="Per-judge timeout seconds")
    p.add_argument("--retry", type=int, default=2, help="Retries per judge request")
    p.add_argument("--target_character_id", type=int, default=None, help="Override config target_character_id")
    p.add_argument("--resume", action="store_true", help="Resume by skipping files already in output jsonl")
    p.add_argument("--verbose", action="store_true", help="Enable verbose output")
    return p.parse_args()


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_index_range(spec: str, total: int) -> Tuple[int, int]:
    m = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", spec or "")
    if not m:
        raise ValueError(f"Invalid --index_range format: {spec!r}. Expected like 100-150")
    start = int(m.group(1))
    end = int(m.group(2))
    if start < 1 or end < 1 or start > end:
        raise ValueError("--index_range must be 1-based and start<=end")
    return max(0, start - 1), min(total, end)


def normalize_judges(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    gk = cfg.get("api_key")
    gb = cfg.get("api_base")

    def push(model: Optional[str], name: Optional[str] = None, api_key: Optional[str] = None, api_base: Optional[str] = None, extra_body: Optional[dict] = None):
        if not model:
            return
        out.append(
            {
                "model": model,
                "name": name or model,
                "api_key": api_key if api_key not in (None, "") else (gk or "empty"),
                "api_base": api_base if api_base not in (None, "") else (gb or ""),
                "extra_body": extra_body or None,
            }
        )

    judges = cfg.get("judges")
    if isinstance(judges, list):
        for j in judges:
            if isinstance(j, str):
                push(j)
            elif isinstance(j, dict):
                push(
                    j.get("model") or j.get("llm") or j.get("name"),
                    j.get("name"),
                    j.get("api_key"),
                    j.get("api_base"),
                    j.get("extra_body")
                )
    elif isinstance(judges, dict):
        push(judges.get("model") or judges.get("llm") or judges.get("name"), judges.get("name"), judges.get("api_key"), judges.get("api_base"), judges.get("extra_body"))
    elif isinstance(judges, str):
        push(judges)
    else:
        push(cfg.get("judger_llm"), cfg.get("judger_llm"), cfg.get("api_key"), cfg.get("api_base"))
    return out


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

You MUST output ONLY a JSON object with EXACTLY these 8 keys and integer values 1-5:
{
  "Knowledge Accuracy": <int>,
  "Emotional Expression": <int>,
  "Personality Traits": <int>,
  "Behavioral Accuracy": <int>,
  "Immersion": <int>,
  "Adaptability": <int>,
  "Behavioral Coherence": <int>,
  "Interaction Richness": <int>
}
No other text, no commentary, no code fences.
"""


def strip_reasoning_blocks(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"【思考】.*?【/思考】", "", text, flags=re.DOTALL)
    return text.strip()


def extract_post_think_text(text: Any) -> str:
    s = "" if text is None else str(text)
    # If model output contains a visible </think> boundary, only keep final answer part.
    parts = re.split(r"</think>", s, flags=re.IGNORECASE)
    if len(parts) > 1:
        tail = parts[-1].strip()
        return tail if tail else s.strip()
    return s.strip()


def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"```$", "", t).strip()
    return t


def _first_json_block(text: str) -> Optional[str]:
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return m.group(0) if m else None


def _coerce_1to5(v: Any) -> int:
    try:
        x = int(float(v))
    except Exception:
        return 3
    return max(1, min(5, x))


def extract_scores_regex(raw: str) -> Tuple[int, ...]:
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
    m = re.search(regex, raw or "", re.DOTALL)
    if not m:
        return tuple([-1] * len(METRICS))
    return tuple(_coerce_1to5(x) for x in m.groups())


def robust_extract_scores(raw: str) -> Tuple[int, ...]:
    txt = _strip_code_fences(strip_reasoning_blocks(raw or ""))
    block = _first_json_block(txt) or txt
    try:
        obj = json.loads(block)
        if not isinstance(obj, dict):
            return tuple([-1] * len(METRICS))
        out = []
        for k in METRICS:
            if k not in obj:
                return tuple([-1] * len(METRICS))
            out.append(_coerce_1to5(obj[k]))
        return tuple(out)
    except Exception:
        return extract_scores_regex(raw)


def make_prompt_parts(scene: str, character: str, actions: str, target_character_name: str) -> Tuple[str, str]:
    prompt_head = (
        "Please evaluate the role-playing ability of the character based on actions across multiple turns "
        "based on scene, character information, critique and evaluation criteria.\n"
        f"[Target Character Name]:\n{target_character_name}\n"
        "Important: Score ONLY this target character. Other characters are context only.\n"
        f"[Scene]:\n{scene}\n"
        f"[Character]:\n{character}\n"
        "[Multi-turn Actions]:\n"
    )
    return prompt_head, actions + "\n"


def render_scene_character_actions(item: Dict[str, Any], target_cid: str, max_rounds: int) -> Optional[Dict[str, Any]]:
    record = item.get("record")
    if not isinstance(record, dict):
        return None

    all_actions: List[Dict[str, Any]] = []
    for rk in sorted(record.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
        v = record.get(rk)
        if isinstance(v, list):
            all_actions.extend(v)
    if not all_actions:
        return None

    selected = []
    for a in all_actions:
        cid = a.get("character_id")
        try:
            cid_norm = str(int(cid))
        except Exception:
            cid_norm = str(cid).strip()
        if cid_norm == target_cid:
            selected.append(a)
    if not selected:
        return None

    first = selected[0]
    d0 = first.get("detail", {}) if isinstance(first.get("detail"), dict) else {}
    name = str(first.get("character_name", d0.get("name", "Unknown")))
    scene = (
        "Scenario Information:\n"
        f"Event: {d0.get('event', '')}\n"
        f"Time: {d0.get('time', '')}\n"
        f"Location: {d0.get('location', '')}\n"
        f"Description: {d0.get('description', item.get('scene', ''))}\n"
    )
    character = f"Name: {name}\nDescription: {d0.get('character_description', '')}\n"

    lines: List[str] = []
    last_round = 0
    for a in selected:
        rd = int(a.get("round", 0) or 0)
        if rd > max_rounds:
            continue
        if rd != last_round:
            lines.append(f"Round: {rd}")
            last_round = rd
        det = a.get("detail", {}) if isinstance(a.get("detail"), dict) else {}
        obs = det.get("observation", "")
        txt = extract_post_think_text(det.get("text", ""))
        lines.append(f"Observation: {obs}")
        if a.get("type") == "dialogue" and txt:
            lines.append(f"Action:\n{name}: {txt}\n")
        else:
            lines.append(f"Action:\n{txt}\n")

    if not lines:
        return None

    return {
        "scene": scene,
        "character": character,
        "actions": "\n".join(lines),
        "character_name": name,
        "round": last_round,
    }


async def chat_once(client: AsyncOpenAI, model: str, prompt: str, timeout_s: float, temperature: float, max_tokens: int, extra_body: Optional[dict] = None) -> str:
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if extra_body:
        kwargs["extra_body"] = extra_body

    resp = await asyncio.wait_for(
        client.chat.completions.create(**kwargs),
        timeout=timeout_s,
    )
    return strip_reasoning_blocks((resp.choices[0].message.content or "").strip())


async def eval_with_judge(
    judge: Dict[str, Any],
    client: AsyncOpenAI,
    scene: str,
    character: str,
    target_character_name: str,
    actions: str,
    prompt_head: str,
    prompt_actions: str,
    timeout_s: float,
    temperature: float,
    max_tokens: int,
    retry: int,
) -> Dict[str, Any]:
    critique = "(no critique)"
    raw = ""
    scores = tuple([-1] * len(METRICS))
    err = ""
    critique_err = ""

    cp = CRITIC_PROMPT_TEMPLATE.format(
        target_character_name=target_character_name,
        scene=scene,
        character=character,
        actions=actions,
    )
    critic_attempts = max(1, int(retry) + 1)
    for i in range(critic_attempts):
        try:
            candidate = await chat_once(client, judge["model"], cp, timeout_s, temperature, max_tokens, judge.get("extra_body"))
            candidate = (candidate or "").strip()
            if candidate and candidate.lower() not in {"(no critique)", "(no critique due to error)"}:
                critique = candidate
                critique_err = ""
                break
            critique_err = "empty_critique_response"
        except Exception as e:
            critique_err = str(e)
        if i < critic_attempts - 1:
            await asyncio.sleep(min(1.0, 0.2 * (2**i)))

    if critique.startswith("(no critique") and critique_err:
        err = f"critique_error: {critique_err}"

    criteria = build_scoring_criteria_block()
    score_consistency_guard = (
        "[Scoring Consistency Rules]:\n"
        "1) Please score with reference to YOUR OWN critique above (same judge, same sample).\n"
        "2) Consider the issues you mentioned when assigning the 8 metric scores.\n"
        "3) Return ONLY the required JSON object.\n"
    )
    full_prompt = (
        prompt_head
        + prompt_actions
        + f"[Critique by You]:\n{critique}\n"
        + score_consistency_guard
        + criteria
        + "\n[Response]:\n"
    )

    parse_ok = False
    for _ in range(3):
        try:
            raw = await chat_once(client, judge["model"], full_prompt, timeout_s, temperature, max_tokens, judge.get("extra_body"))
            scores = robust_extract_scores(raw)
            if scores[0] != -1:
                parse_ok = True
                break
        except Exception as e:
            err = str(e)
        await asyncio.sleep(0.2)

    if not parse_ok:
        repair_prompt = (
            "Convert the following content into EXACTLY the required JSON with the 8 keys and integer values 1-5. "
            "Return ONLY the JSON object, with no extra text or code fences.\n\n"
            "CONTENT START\n"
            f"{raw}\n"
            "CONTENT END\n"
            + criteria
            + "\n[Response]:\n"
        )
        try:
            repaired = await chat_once(client, judge["model"], repair_prompt, timeout_s, temperature, max_tokens, judge.get("extra_body"))
            scores = robust_extract_scores(repaired)
            if scores[0] != -1:
                parse_ok = True
        except Exception:
            pass

    if not parse_ok:
        fresh_prompt = (
            prompt_head
            + prompt_actions
            + f"[Critique by You]:\n{critique}\n"
            + score_consistency_guard
            + criteria
            + "\n[Response]:\n"
        )
        try:
            fresh = await chat_once(client, judge["model"], fresh_prompt, timeout_s, temperature, max_tokens, judge.get("extra_body"))
            scores = robust_extract_scores(fresh)
            if scores[0] != -1:
                parse_ok = True
        except Exception:
            pass

    if not parse_ok:
        scores = tuple([3] * len(METRICS))

    return {
        "judge_name": judge["name"],
        "judge_model": judge["model"],
        "scores": list(scores),
        "critic": critique,
        "raw_response": raw,
        "error": err,
    }


def find_disputes(per_scores: List[List[int]], gap: int, topk: int) -> List[Tuple[int, int, int, int, int]]:
    disputes = []
    if len(per_scores) < 2:
        return disputes
    for m in range(len(METRICS)):
        col = [(j, int(per_scores[j][m])) for j in range(len(per_scores))]
        low_j, low_s = min(col, key=lambda x: x[1])
        high_j, high_s = max(col, key=lambda x: x[1])
        if (high_s - low_s) >= gap:
            disputes.append((m, low_j, high_j, low_s, high_s))
    disputes.sort(key=lambda x: (x[4] - x[3]), reverse=True)
    return disputes[:topk]


async def run_debate_once(
    referee_client: AsyncOpenAI,
    referee_model: str,
    timeout_s: float,
    temperature: float,
    max_tokens: int,
    target_character_name: str,
    metric_name: str,
    scene_text: str,
    character_info: str,
    actions: str,
    high_name: str,
    high_score: int,
    high_critic: str,
    low_name: str,
    low_score: int,
    low_critic: str,
) -> Optional[Tuple[int, str]]:
    prompt = DEBATE_ARBITER_PROMPT.format(
        target_character_name=target_character_name,
        metric_name=metric_name,
        scene_text=scene_text,
        character_info=character_info,
        actions=actions,
        high_name=high_name,
        high_score=high_score,
        high_critic=high_critic,
        low_name=low_name,
        low_score=low_score,
        low_critic=low_critic,
    )
    try:
        resp = await chat_once(referee_client, referee_model, prompt, timeout_s, temperature, max_tokens)
    except Exception:
        return None
    m = _DEBATE_SCORE_RX.search(resp or "")
    if not m:
        return None
    score = int(m.group(1))
    rm = re.search(r"Reason\s*:\s*(.*)", resp or "", re.IGNORECASE | re.DOTALL)
    return score, (rm.group(1).strip() if rm else "")


async def evaluate_file(
    path: Path,
    judges: List[Dict[str, Any]],
    clients: Dict[str, AsyncOpenAI],
    target_cid: str,
    max_rounds: int,
    timeout_s: float,
    temperature: float,
    max_tokens: int,
    debate_enabled: bool,
    debate_gap: int,
    debate_topk: int,
    referee_client: Optional[AsyncOpenAI],
    referee_model: Optional[str],
    retry: int,
    narrator_llm_cfg: str,
    character_llm_cfg: str,
) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        item = json.loads(first_line)
    except Exception as e:
        return {"file": str(path), "title": "", "character_name": "", "error": f"parse_error: {e}"}

    title = str(item.get("title", ""))
    scene_id = int(item.get("scene_id", 0) or 0)
    narrator_llm_infer, character_llm_infer = infer_models_from_path(path)
    narrator_llm = narrator_llm_cfg or narrator_llm_infer
    character_llm = character_llm_cfg or character_llm_infer
    payload = render_scene_character_actions(item, target_cid, max_rounds)
    if payload is None:
        return {
            "file": str(path),
            "title": title,
            "character_name": "",
            "error": f"target_character_id={target_cid} not found or no actions",
        }

    prompt_head, prompt_actions = make_prompt_parts(
        payload["scene"], payload["character"], payload["actions"], payload["character_name"]
    )

    judge_tasks = []
    for j in judges:
        judge_tasks.append(
            eval_with_judge(
                judge=j,
                client=clients[j["name"]],
                scene=payload["scene"],
                character=payload["character"],
                target_character_name=payload["character_name"],
                actions=payload["actions"],
                prompt_head=prompt_head,
                prompt_actions=prompt_actions,
                timeout_s=timeout_s,
                temperature=temperature,
                max_tokens=max_tokens,
                retry=retry,
            )
        )
    judge_results = await asyncio.gather(*judge_tasks)

    if debate_enabled and len(judge_results) >= 2 and referee_client is not None and referee_model:
        per_scores = [list(r.get("scores", [3] * len(METRICS))) for r in judge_results]
        disputes = find_disputes(per_scores, gap=debate_gap, topk=debate_topk)
        notes: List[str] = []
        for m_idx, low_j, high_j, low_s, high_s in disputes:
            out = await run_debate_once(
                referee_client=referee_client,
                referee_model=referee_model,
                timeout_s=timeout_s,
                temperature=temperature,
                max_tokens=max_tokens,
                target_character_name=payload["character_name"],
                metric_name=METRICS[m_idx],
                scene_text=payload["scene"],
                character_info=payload["character"],
                actions=payload["actions"],
                high_name=judge_results[high_j]["judge_name"],
                high_score=high_s,
                high_critic=judge_results[high_j].get("critic", ""),
                low_name=judge_results[low_j]["judge_name"],
                low_score=low_s,
                low_critic=judge_results[low_j].get("critic", ""),
            )
            if out is None:
                continue
            final_score, reason = out
            # Apply debate reconciliation to all judges for this metric,
            # not only the high/low pair that triggered arbitration.
            for jr in judge_results:
                if isinstance(jr.get("scores"), list) and len(jr["scores"]) == len(METRICS):
                    jr["scores"][m_idx] = final_score
            notes.append(
                f"[Debate] Metric={METRICS[m_idx]} {judge_results[low_j]['judge_name']}:{low_s} vs "
                f"{judge_results[high_j]['judge_name']}:{high_s} -> Final:{final_score} Reason:{reason}"
            )
        if notes:
            note_txt = "\n".join(notes)
            for jr in judge_results:
                jr["critic"] = (jr.get("critic", "") + "\n\n" + note_txt).strip()

    matrix = [r["scores"] for r in judge_results if r.get("scores")]
    avg = [round(sum(col) / len(col), 3) for col in zip(*matrix)] if matrix else [3.0] * len(METRICS)
    avg_map = {m: avg[i] for i, m in enumerate(METRICS)}

    return {
        "file": str(path),
        "title": title,
        "scene_id": scene_id,
        "narrator_llm": narrator_llm,
        "character_llm": character_llm,
        "character_name": payload["character_name"],
        "target_character_id": target_cid,
        "round": payload["round"],
        "scene_info": payload["scene"],
        "character_info": payload["character"],
        "critic": "\n\n".join([f"[{x['judge_name']}] {x.get('critic','')}" for x in judge_results]),
        "judges": judge_results,
        "average_scores": avg_map,
        "error": "",
    }


def safe_title_dir(title: str) -> str:
    t = (title or "unknown_title").strip().replace("/", "_")
    return t or "unknown_title"


def safe_model_dir(name: str) -> str:
    t = (name or "unknown_model").strip()
    t = t.replace("/", "_").replace("\\", "_").replace(":", "_")
    t = re.sub(r"\s+", "_", t)
    t = re.sub(r"[^A-Za-z0-9._-]", "_", t)
    return t or "unknown_model"


# Global cache for persona names
_PERSONA_NAMES_CACHE: Optional[List[str]] = None


def load_persona_names(persona_file: str = "persona_data/1000_persona.en.jsonl") -> List[str]:
    """Load all user_name from persona file and cache the result."""
    global _PERSONA_NAMES_CACHE
    if _PERSONA_NAMES_CACHE is not None:
        return _PERSONA_NAMES_CACHE

    persona_path = Path(persona_file)
    if not persona_path.exists():
        return []

    names = []
    try:
        with persona_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    name = data.get("user_name", "")
                    if name:
                        names.append(name)
                except json.JSONDecodeError:
                    continue
    except Exception:
        return []

    _PERSONA_NAMES_CACHE = names
    return names


def infer_models_from_path(path: Path, persona_file: str = "persona_data/1000_persona.en.jsonl") -> Tuple[str, str]:
    narrator = ""
    character = ""

    # Typical record layout:
    # output/record/<title>/persona_detail/<narrator>_<character>_character(.jsonl)
    # Example: autogen_scene_<persona_name>_<character_model>_<index>/persona_detail/<narrator>_<character>_character.jsonl
    try:
        title_dir = path.parents[1].name
    except Exception:
        title_dir = ""

    # Extract character_llm from directory name by matching persona names
    if title_dir.startswith("autogen_scene_"):
        # Remove the "autogen_scene_" prefix
        dir_suffix = title_dir[len("autogen_scene_"):]

        # Load all persona names
        persona_names = load_persona_names(persona_file)

        # Try to find matching persona name in the directory name
        matched_persona = None
        for pname in persona_names:
            # Convert persona name to underscore format (e.g., "Anna Castillo" -> "Anna_Castillo")
            underscore_name = pname.replace(" ", "_")
            if dir_suffix.startswith(underscore_name + "_"):
                matched_persona = underscore_name
                break

        if matched_persona:
            # Extract the part after the persona name
            # Format: <persona_name>_<character_model> or <persona_name>_<character_model>_<index>
            rest = dir_suffix[len(matched_persona) + 1:]  # +1 for the underscore
            parts = rest.split("_")

            if parts:
                # The first part after persona name should be the character model
                character = parts[0]

                # If there's a second part and it's a number, it's an index
                # If there are more parts, they might be part of the model name
                if len(parts) > 1:
                    # Check if the last part is a numeric index
                    if parts[-1].isdigit():
                        # Everything between persona and the index is the model name
                        character = "_".join(parts[:-1])
                    else:
                        # No index, everything after persona is the model name
                        character = "_".join(parts)

    # Extract narrator from filename using the character model we already found
    stem = path.stem
    m = re.fullmatch(r"(.+)_character(?:_\d+)?", stem)
    if m:
        prefix = m.group(1)
        # Expected format: <narrator_llm>_<character_llm>
        # Since we already know the character_llm, we can strip it from the end
        if character:
            # Remove the character_llm suffix (with underscore) to get narrator
            if prefix.endswith("_" + character):
                narrator = prefix[:-(len(character) + 1)]
            else:
                # Fallback: try rsplit if exact match fails
                parts = prefix.rsplit("_", 1)
                if len(parts) == 2:
                    narrator = parts[0]
        else:
            # No character model found, use simple rsplit
            parts = prefix.rsplit("_", 1)
            if len(parts) == 2:
                narrator = parts[0]

    return narrator, character


def resolve_output_dir(out_base: Path, character_llm: str) -> Path:
    # by_model/<character_llm>/... when model is known; otherwise write under by_model directly.
    if out_base.name == "by_model" and character_llm:
        return out_base / safe_model_dir(character_llm)
    return out_base


def write_detail_csv(result: Dict[str, Any], out_dir: Path) -> None:
    if result.get("error"):
        return
    title = safe_title_dir(result.get("title", ""))
    detail_dir = out_dir / "detail" / title
    detail_dir.mkdir(parents=True, exist_ok=True)

    src_file = Path(result.get("file", ""))
    stem = src_file.stem
    cname = str(result.get("character_name", "")).replace(" ", "_")
    prefix = cname + "_"
    core = stem[len(prefix) :] if cname and stem.startswith(prefix) else stem
    csv_path = detail_dir / f"{core}_evaluation_detail.csv"

    fieldnames = [
        "Title",
        "Judger",
        "Narrator",
        "Model",
        "SceneID",
        "Round",
        "SceneInfo",
        "CharacterInfo",
        "Critic",
        "JudgeScores",
    ] + METRICS

    judge_scores_obj: Dict[str, Dict[str, int]] = {}
    for j in result.get("judges", []):
        jn = j.get("judge_name", "")
        sc = j.get("scores", [])
        if not jn or not isinstance(sc, list) or len(sc) != len(METRICS):
            continue
        judge_scores_obj[jn] = {METRICS[i]: int(sc[i]) for i in range(len(METRICS))}

    row = {
        "Title": result.get("title", ""),
        "Judger": ", ".join([j.get("judge_name", "") for j in result.get("judges", [])]),
        "Narrator": result.get("narrator_llm", ""),
        "Model": result.get("character_llm", ""),
        "SceneID": result.get("scene_id", 0),
        "Round": result.get("round", 0),
        "SceneInfo": result.get("scene_info", ""),
        "CharacterInfo": result.get("character_info", ""),
        "Critic": result.get("critic", ""),
        "JudgeScores": json.dumps(judge_scores_obj, ensure_ascii=False),
    }
    for m in METRICS:
        row[m] = result.get("average_scores", {}).get(m, "")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerow(row)


async def run_all(args: argparse.Namespace) -> None:
    cfg = load_yaml(args.config)
    judges = normalize_judges(cfg)
    if not judges:
        raise ValueError("No judges found in config.")

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob(args.glob))
    # Accept *_character.jsonl, *_character_1.jsonl and typo extension *.jsnol.
    valid_suffixes = {".jsonl", ".jsnol"}
    files = [p for p in files if p.is_file() and p.suffix.lower() in valid_suffixes and "_character" in p.stem]
    total_all = len(files)
    if not files:
        raise ValueError(f"No files found in {input_dir} with glob={args.glob}")

    if args.index_range:
        lo, hi = parse_index_range(args.index_range, total_all)
        files = files[lo:hi]
    if args.max_files and args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise ValueError("No files selected after applying --index_range/--max_files filters.")

    target_cid = str(args.target_character_id if args.target_character_id is not None else cfg.get("target_character_id", 0))
    max_rounds = int(cfg.get("max_rounds", 20))
    temperature = float(cfg.get("temperature", 0))
    max_tokens = int(cfg.get("max_token", 1500))
    narrator_llm_cfg = str(cfg.get("narrator_llm", "")).strip()
    character_llm_cfg = str(cfg.get("character_llm", "")).strip()

    debate_enabled = bool(cfg.get("debate_enabled", True))
    debate_gap = int(cfg.get("debate_gap", 2))
    debate_topk = int(cfg.get("debate_topk", 8))
    debate_referee = cfg.get("debate_referee", None)
    debate_referee_api_base = cfg.get("debate_referee_api_base", cfg.get("api_base", ""))
    debate_referee_api_key = cfg.get("debate_referee_api_key", cfg.get("api_key", "empty"))

    # Verbose mode: command line flag takes precedence over config file
    verbose = args.verbose or bool(cfg.get("verbose", False))

    clients: Dict[str, AsyncOpenAI] = {}
    for j in judges:
        clients[j["name"]] = AsyncOpenAI(api_key=j.get("api_key", "empty"), base_url=j.get("api_base", "") or None)

    referee_client: Optional[AsyncOpenAI] = None
    referee_model: Optional[str] = None
    if debate_enabled:
        if debate_referee:
            referee_client = AsyncOpenAI(api_key=debate_referee_api_key, base_url=debate_referee_api_base or None)
            referee_model = str(debate_referee)
        else:
            referee_client = clients[judges[0]["name"]]
            referee_model = str(judges[0]["model"])

    out_base = Path(args.output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    existing_results: List[Dict[str, Any]] = []
    done_files = set()
    resume_jsonls: List[Path] = []
    if out_base.name == "by_model":
        resume_jsonls.extend(sorted(out_base.glob("*/evaluation_results.jsonl")))
    else:
        resume_jsonls.append(out_base / "evaluation_results.jsonl")

    if args.resume:
        for out_jsonl in resume_jsonls:
            if not out_jsonl.exists():
                continue
            with out_jsonl.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    existing_results.append(obj)
                    fp = obj.get("file")
                    if isinstance(fp, str) and fp:
                        done_files.add(fp)
        if done_files:
            files = [p for p in files if str(p) not in done_files]

    print(f"[select] total_candidates={total_all}")
    if args.index_range:
        print(f"[select] index_range={args.index_range}")
    if args.max_files:
        print(f"[select] max_files={args.max_files}")
    if args.resume:
        print(f"[select] resume_skip={len(done_files)}")
    print(f"[select] pending_files={len(files)}")

    if not files:
        print("[done] nothing to evaluate after resume/filter.")
        return

    sem = asyncio.Semaphore(max(1, args.concurrency))

    async def worker(p: Path) -> Dict[str, Any]:
        async with sem:
            if verbose:
                print(f"[verbose] Evaluating: {p}")
            result = await evaluate_file(
                path=p,
                judges=judges,
                clients=clients,
                target_cid=target_cid,
                max_rounds=max_rounds,
                timeout_s=args.judge_timeout,
                temperature=temperature,
                max_tokens=max_tokens,
                debate_enabled=debate_enabled,
                debate_gap=debate_gap,
                debate_topk=debate_topk,
                referee_client=referee_client,
                referee_model=referee_model,
                retry=args.retry,
                narrator_llm_cfg=narrator_llm_cfg,
                character_llm_cfg=character_llm_cfg,
            )
            if verbose:
                if result.get("error"):
                    print(f"[verbose] ❌ Error: {p} - {result.get('error', '')}")
                else:
                    print(f"[verbose] ✅ Done: {p}")
            return result

    tasks = [asyncio.create_task(worker(p)) for p in files]
    results: List[Dict[str, Any]] = []

    done = 0
    for t in asyncio.as_completed(tasks):
        r = await t
        results.append(r)
        per_out_dir = resolve_output_dir(out_base, str(r.get("character_llm", "")))
        per_out_dir.mkdir(parents=True, exist_ok=True)
        write_detail_csv(r, per_out_dir)
        done += 1
        if done % 20 == 0 or done == len(files):
            print(f"[progress] {done}/{len(files)}")

    all_results = existing_results + results

    grouped_results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    if out_base.name == "by_model":
        for r in all_results:
            grouped_results[str(r.get("character_llm", "") or "")].append(r)
    else:
        grouped_results[""] = all_results

    written_dirs: List[Path] = []
    for model_name, group in grouped_results.items():
        dst_dir = resolve_output_dir(out_base, model_name)
        dst_dir.mkdir(parents=True, exist_ok=True)
        written_dirs.append(dst_dir)

        out_jsonl = dst_dir / "evaluation_results.jsonl"
        out_csv = dst_dir / "evaluation_results_avg.csv"

        with out_jsonl.open("w", encoding="utf-8") as f:
            for r in group:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        with out_csv.open("w", newline="", encoding="utf-8") as f:
            fieldnames = ["file", "title", "character_name", "error"] + METRICS
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in group:
                row = {
                    "file": r.get("file", ""),
                    "title": r.get("title", ""),
                    "character_name": r.get("character_name", ""),
                    "error": r.get("error", ""),
                }
                avg = r.get("average_scores", {})
                for m in METRICS:
                    row[m] = avg.get(m, "")
                w.writerow(row)

    ok = sum(1 for r in all_results if not r.get("error"))
    fail = len(all_results) - ok
    print(f"[done] files_total={len(all_results)} ok={ok} fail={fail}")
    print(f"[done] files_newly_evaluated={len(results)}")
    for d in sorted(set(written_dirs)):
        print(f"[done] jsonl={d / 'evaluation_results.jsonl'}")
        print(f"[done] csv={d / 'evaluation_results_avg.csv'}")
        print(f"[done] detail_dir={d / 'detail'}")


def main() -> None:
    args = parse_args()
    asyncio.run(run_all(args))


if __name__ == "__main__":
    main()
