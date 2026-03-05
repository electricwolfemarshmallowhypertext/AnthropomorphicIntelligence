# -*- coding: utf-8 -*-
import os
import json
import argparse
import threading
import concurrent.futures
from tqdm import tqdm
from yacs.config import CfgNode
import pandas as pd
import re

from utils import utils
from evaluate_arena import run_eval_detail

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

_DEBATE_TAG = re.compile(r"\[Debate/([^\]]+)\]", re.IGNORECASE)
_DEBATE_METRIC = re.compile(r"Metric\s*=\s*([A-Za-z ]+)", re.IGNORECASE)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--character_llm", type=str, default=None, help="override character llm")
    parser.add_argument("--narrator_llm", type=str, default=None, help="override narrator llm")
    parser.add_argument("--title", type=str, default=None, help="evaluate a single title")
    parser.add_argument("--titles", type=str, default=None, help="comma-separated list of titles")
    parser.add_argument("--scene_path", type=str, default=None, help="unused (reserved)")
    parser.add_argument("-c", "--config_file", type=str, default="config/evaluate.yaml", help="Path to config file")
    parser.add_argument("-o", "--output_file", type=str, default="message.json", help="Path to output file")
    parser.add_argument("-l", "--log_file", type=str, default="", help="Path to log file")
    parser.add_argument("-n", "--log_name", type=str, default=str(os.getpid()), help="Name of logger")
    parser.add_argument("-p", "--play_role", type=int, default=-1, help="Add a user controllable role")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="Modify config options from command line")
    return parser.parse_args()

def run_eval_with_config(config, logger, scene, SAVE_PATH, done):
    return run_eval_detail(config, logger, scene, SAVE_PATH, done)

def _build_detail_path(title: str, character: str, narrator: str) -> str:
    utils.ensure_dir(f"output/evaluation/detail/{title}")
    return f"output/evaluation/detail/{title}/{character}_{narrator}_character_evaluation_detail.csv"
    

def _build_multi_path(title: str, narrator: str, scene_id: int) -> str:
    utils.ensure_dir(f"output/evaluation/multi/{title}")
    return f"output/evaluation/multi/{title}/{narrator}_{scene_id}_character_evaluation_avg.csv"
    
def _find_record_path(title: str, narrator: str, character: str) -> str:
    candidates = [
        f"output/record/{title}/persona_detail/{narrator}_{character}_character.jsonl",
        f"output/record/{title}/character/{narrator}_{character}_character.jsonl",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return ""


def _append_scene_summary_from_detail(detail_csv: str, multi_csv: str, narrator: str, character: str, title: str, scene_id: int):

    if not os.path.exists(detail_csv):
        print(f"[summary] detail file not found: {detail_csv}")
        return

    try:
        df = pd.read_csv(detail_csv)
    except Exception as e:
        print(f"[summary] failed reading detail: {detail_csv}, err={e}")
        return

    if df.empty:
        print(f"[summary] detail is empty: {detail_csv}")
        return

    def _norm(s):
        return s.astype(str).fillna("").str.strip()

    if "Title" not in df.columns:
        df["Title"] = ""

    df["_SceneID_s"] = _norm(df.get("SceneID"))
    df["_Model_s"]   = _norm(df.get("Model"))
    df["_Narr_s"]    = _norm(df.get("Narrator"))
    df["_Title_s"]   = _norm(df.get("Title"))

    scene_id_s = str(scene_id).strip()
    character_s = str(character).strip()
    narrator_s  = str(narrator).strip()
    title_s     = str(title).strip()

    mask_scene   = (df["_SceneID_s"] == scene_id_s)
    mask_model   = (df["_Model_s"]   == character_s)
    mask_narr    = (df["_Narr_s"]    == narrator_s)
    mask_title   = (df["_Title_s"]   == title_s)

    mask = mask_scene & mask_model & mask_narr & mask_title
    sdf = df[mask].copy()

    if sdf.empty:
        print(f"[summary] no rows for scene={scene_id} in detail; skip multi")

        try:
            seen_sids = sorted(df["_SceneID_s"].unique().tolist())
            print(f"          seen SceneIDs in detail: {seen_sids[:30]}{' ...' if len(seen_sids)>30 else ''}")
        except Exception:
            pass

        try:
            print(f"          unique Model: {sorted(df['_Model_s'].unique().tolist())[:10]}")
            print(f"          unique Narrator: {sorted(df['_Narr_s'].unique().tolist())[:10]}")
            print(f"          unique Title: {sorted(df['_Title_s'].unique().tolist())[:5]}")
        except Exception:
            pass

        tmp = df[mask_scene]
        print(f"          rows with same SceneID only: {len(tmp)}")
        return

    avg_per_metric = {}
    for m in METRICS:
        try:
            avg_per_metric[m] = pd.to_numeric(sdf[m], errors="coerce").mean()
        except Exception:
            avg_per_metric[m] = float("nan")

    avg_all = sum([v for v in avg_per_metric.values() if pd.notna(v)]) / len(METRICS)

    all_critics = " \n ".join(sdf.get("Critic", "").astype(str).tolist())
    debate_tags = _DEBATE_TAG.findall(all_critics)
    debate_count = len(debate_tags)

    debated_metrics = set()
    for m in _DEBATE_METRIC.findall(all_critics):
        debated_metrics.add(m.strip())
    debated_metrics_str = "; ".join(sorted(debated_metrics)) if debated_metrics else ""

    cols = ['Title','Judger','Narrator','Model','SceneID','Round'] + METRICS + ['Average','DebateCount','DebatedMetrics']
    try:
        last_round = int(pd.to_numeric(sdf.get("Round"), errors="coerce").max())
    except Exception:
        try:
            last_round = int(pd.to_numeric(sdf.get("Round").iloc[0], errors="coerce"))
        except Exception:
            last_round = 0

    out_row = {
        'Title': title_s,
        'Judger': str(sdf.get("Judger").iloc[0]) if "Judger" in sdf.columns and len(sdf) else "",
        'Narrator': narrator_s,
        'Model': character_s,
        'SceneID': int(scene_id) if str(scene_id).isdigit() else scene_id_s,
        'Round': last_round,
        'Average': avg_all,
        'DebateCount': debate_count,
        'DebatedMetrics': debated_metrics_str,
    }
    for m in METRICS:
        out_row[m] = avg_per_metric[m]

    header_needed = not os.path.exists(multi_csv) or os.path.getsize(multi_csv) == 0
    import csv
    with open(multi_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        if header_needed:
            writer.writeheader()
        writer.writerow(out_row)

    print(f"[summary] scene {scene_id} -> {multi_csv}")

def main():
    args = parse_args()

    config = CfgNode(new_allowed=True)
    output_file = os.path.join("output/message", args.output_file)
    config = utils.add_variable_to_config(config, "output_file", output_file)
    config = utils.add_variable_to_config(config, "log_file", args.log_file)
    config = utils.add_variable_to_config(config, "log_name", args.log_name)
    config = utils.add_variable_to_config(config, "play_role", args.play_role)
    config.merge_from_file(args.config_file)

    global_logger = utils.set_logger("quick_start.log", "quick_start")

    titles = ['autogen_scene_xx_deepseek-v2','autogen_scene_xx_gpt-4_1','autogen_scene_xx_gpt-4o','autogen_scene_xx_gpt-4o-mini','autogen_scene_xx_gpt-35-turbo','autogen_scene_xx_gpt-oss','autogen_scene_xx_llama3_1_8b','autogen_scene_xx_llama3_2_3b','autogen_scene_xx_mistral-small3_2','autogen_scene_xx_phi4','autogen_scene_xx_qwen3_1_7b','autogen_scene_xx_qwen3_4b','autogen_scene_xx_qwen3_8b','autogen_scene_xx_qwen3_14b','autogen_scene_xx_qwen3-32b']
    character_llms = ['deepseek-v2','gpt-4.1','gpt-4o','gpt-4o-mini','gpt-35-turbo','gpt-oss','llama3.1:8b','llama3.2:3b','mistral-small3.2','phi4','qwen3:1.7b','qwen3:4b','qwen3:8b','qwen3:14b','qwen3-32b']

    if args.character_llm:
        character_llms = [args.character_llm]
    if args.narrator_llm:
        config['narrator_llm'] = args.narrator_llm
    if args.title:
        titles = [args.title]
    elif args.titles:
        titles = [t.strip() for t in args.titles.split(",") if t.strip()]

    max_scenes = config.get('max_scenes', 5)
    scene_ids = [i for i in range(0, 9)]

    for model_path in tqdm(character_llms, desc="Models"):
        print("Current Model:", model_path)
        for title in tqdm(titles, desc="Titles"):
            print("Current Title:", title)

            character = model_path.split("/")[-1]
            config['title'] = title
            config['character_llm'] = character

            narrator = config.get('narrator_llm', '')
            detail_csv = _build_detail_path(title, character, narrator)

            record_path = _find_record_path(title, narrator, character)
            if not record_path:
                print(f"[driver] record not found for title={title}, narrator={narrator}, character={character}")
                continue

            eval_done = {character: {sid: [] for sid in scene_ids}}
            if os.path.exists(detail_csv):
                try:
                    df_exist = pd.read_csv(detail_csv)
                    for _, row in df_exist.iterrows():
                        model_key = row.get('Model', character) or character
                        sid = row.get('SceneID', None)
                        cinfo = row.get('CharacterInfo', "")
                        if sid is None:
                            continue
                        if model_key not in eval_done:
                            eval_done[model_key] = {}
                        if sid not in eval_done[model_key]:
                            eval_done[model_key][sid] = []
                        eval_done[model_key][sid].append(cinfo)
                except Exception as e:
                    print("Warning: failed to read existing detail file, start fresh. Err:", e)

            scenes = {}
            with open(record_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for idx in range(len(lines) - 1, -1, -1):
                    record = json.loads(lines[idx])
                    if record['scene_id'] not in scenes:
                        scenes[record['scene_id']] = record

            available_ids = sorted(list(scenes.keys()))
            if max_scenes and len(available_ids) > max_scenes:
                available_ids = available_ids[-max_scenes:]

            utils.ensure_dir(f"output/log/evaluation/detail/{title}")
            futures = []
            with concurrent.futures.ThreadPoolExecutor() as executor_eval:
                for sid in available_ids:
                    scene = scenes[sid]
                    config['scene_id'] = scene['scene_id']
                    log_file = f"evaluation/detail/{title}/{config.get('narrator_llm','')}_{config.get('character_llm','')}_{config.get('scene_id')}_evaluation.log"
                    logger = utils.set_logger(log_file, args.log_name)
                    logger.info(f"os.getpid()={os.getpid()}")
                    logger.info(f"\n{config}")

                    done_list = eval_done.get(config['character_llm'], {}).get(config['scene_id'], [])
                    futures.append(
                        executor_eval.submit(run_eval_with_config, config.copy(), logger, scene['record'], detail_csv, done_list)
                    )

                for future in concurrent.futures.as_completed(futures):
                    try:
                        result_sid = future.result()
                        print(f"SCENE DONE: {result_sid}")
                    except Exception as exc:
                        print(f'Generated an exception: {exc}')

            print(f"[{title}] detail saved -> {detail_csv}")

            for sid in available_ids:
                multi_csv = _build_multi_path(title, narrator, sid)
                _append_scene_summary_from_detail(
                    detail_csv=detail_csv,
                    multi_csv=multi_csv,
                    narrator=narrator,
                    character=character,
                    title=title,
                    scene_id=sid
                )

if __name__ == "__main__":
    main()

# # ************************** Single-Judge Implementation Script **************************
# # -*- coding: utf-8 -*-
# import os
# import json
# import argparse
# import threading
# import concurrent.futures
# from tqdm import tqdm
# from yacs.config import CfgNode
# import pandas as pd
# import re
# import csv

# from utils import utils
# from evaluate_demo import run_eval_detail

# os.environ["OPENAI_API_VERSION"] = "2024-12-01-preview"

# lock = threading.Lock()

# METRICS = [
#     "Knowledge Accuracy",
#     "Emotional Expression",
#     "Personality Traits",
#     "Behavioral Accuracy",
#     "Immersion",
#     "Adaptability",
#     "Behavioral Coherence",
#     "Interaction Richness",
# ]

# _DEBATE_TAG = re.compile(r"\[Debate/([^\]]+)\]", re.IGNORECASE)
# _DEBATE_METRIC = re.compile(r"Metric\s*=\s*([A-Za-z ]+)", re.IGNORECASE)


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--character_llm", type=str, default=None, help="override character llm")
#     parser.add_argument("--scene_path", type=str, default=None, help="unused (reserved)")
#     parser.add_argument("-c", "--config_file", type=str, default="config/evaluate.yaml", help="Path to config file")
#     parser.add_argument("-o", "--output_file", type=str, default="message.json", help="Path to output file")
#     parser.add_argument("-l", "--log_file", type=str, default="", help="Path to log file")
#     parser.add_argument("-n", "--log_name", type=str, default=str(os.getpid()), help="Name of logger")
#     parser.add_argument("-p", "--play_role", type=int, default=-1, help="Add a user controllable role")
#     parser.add_argument("--run-tag", type=str, default="",
#                         help="Suffix tag for output files, e.g. 'single'. Empty = no suffix")
#     parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="Modify config options from command line")
#     return parser.parse_args()


# def _with_tag(path: str, tag: str) -> str:
#     tag = (tag or "").strip()
#     if not tag:
#         return path
#     base, ext = os.path.splitext(path)
#     return f"{base}_{tag}{ext}" if ext else f"{path}_{tag}"


# def run_eval_with_config(config, logger, scene, SAVE_PATH, done):
#     return run_eval_detail(config, logger, scene, SAVE_PATH, done)


# def _build_detail_path(title: str, character: str, narrator: str) -> str:
#     utils.ensure_dir(f"output/evaluation/detail/{title}")
#     return f"output/evaluation/detail/{title}/{character}_{narrator}_character_evaluation_detail.csv"


# def _build_multi_path(title: str, narrator: str, scene_id: int) -> str:
#     utils.ensure_dir(f"output/evaluation/multi/{title}")
#     return f"output/evaluation/multi/{title}/{narrator}_{scene_id}_character_evaluation_avg.csv"


# def _append_scene_summary_from_detail(
#     detail_csv: str,
#     multi_csv: str,
#     narrator: str,
#     character: str,
#     title: str,
#     scene_id: int,
#     run_tag: str = ""
# ):

#     if not os.path.exists(detail_csv):
#         print(f"[summary] detail file not found: {detail_csv}")
#         return

#     try:
#         df = pd.read_csv(detail_csv)
#     except Exception as e:
#         print(f"[summary] failed reading detail: {detail_csv}, err={e}")
#         return

#     if df.empty:
#         print(f"[summary] detail is empty: {detail_csv}")
#         return

#     def _norm(s):
#         return s.astype(str).fillna("").str.strip()

#     if "Title" not in df.columns:
#         df["Title"] = ""

#     df["_SceneID_s"] = _norm(df.get("SceneID"))
#     df["_Model_s"]   = _norm(df.get("Model"))
#     df["_Narr_s"]    = _norm(df.get("Narrator"))
#     df["_Title_s"]   = _norm(df.get("Title"))

#     scene_id_s = str(scene_id).strip()
#     character_s = str(character).strip()
#     narrator_s  = str(narrator).strip()
#     title_s     = str(title).strip()

#     mask_scene   = (df["_SceneID_s"] == scene_id_s)
#     mask_model   = (df["_Model_s"]   == character_s)
#     mask_narr    = (df["_Narr_s"]    == narrator_s)
#     mask_title   = (df["_Title_s"]   == title_s)

#     mask = mask_scene & mask_model & mask_narr & mask_title
#     sdf = df[mask].copy()

#     if sdf.empty:

#         print(f"[summary] no rows for scene={scene_id} in detail; skip multi")
#         try:
#             seen_sids = sorted(df["_SceneID_s"].unique().tolist())
#             print(f"          seen SceneIDs in detail: {seen_sids[:30]}{' ...' if len(seen_sids)>30 else ''}")
#         except Exception:
#             pass
#         try:
#             print(f"          unique Model: {sorted(df['_Model_s'].unique().tolist())[:10]}")
#             print(f"          unique Narrator: {sorted(df['_Narr_s'].unique().tolist())[:10]}")
#             print(f"          unique Title: {sorted(df['_Title_s'].unique().tolist())[:5]}")
#         except Exception:
#             pass
#         tmp = df[mask_scene]
#         print(f"          rows with same SceneID only: {len(tmp)}")
#         return

#     avg_per_metric = {}
#     for m in METRICS:
#         try:
#             avg_per_metric[m] = pd.to_numeric(sdf[m], errors="coerce").mean()
#         except Exception:
#             avg_per_metric[m] = float("nan")

#     valid_vals = [v for v in avg_per_metric.values() if pd.notna(v)]
#     avg_all = (sum(valid_vals) / len(valid_vals)) if valid_vals else float("nan")

#     if "Critic" in sdf.columns:
#         critics_series = sdf["Critic"].astype(str)
#     else:
#         critics_series = pd.Series([], dtype=str)
#     all_critics = " \n ".join(critics_series.tolist())

#     debate_tags = _DEBATE_TAG.findall(all_critics)
#     debate_count = len(debate_tags)

#     debated_metrics = set()
#     for m in _DEBATE_METRIC.findall(all_critics):
#         debated_metrics.add(m.strip())
#     debated_metrics_str = "; ".join(sorted(debated_metrics)) if debated_metrics else ""

#     cols = ['Title','Judger','Narrator','Model','SceneID','Round'] + METRICS + ['Average','DebateCount','DebatedMetrics','RunTag']
#     try:
#         last_round = int(pd.to_numeric(sdf.get("Round"), errors="coerce").max())
#     except Exception:
#         try:
#             last_round = int(pd.to_numeric(sdf.get("Round").iloc[0], errors="coerce"))
#         except Exception:
#             last_round = 0

#     out_row = {
#         'Title': title_s,
#         'Judger': str(sdf.get("Judger").iloc[0]) if "Judger" in sdf.columns and len(sdf) else "",
#         'Narrator': narrator_s,
#         'Model': character_s,
#         'SceneID': int(scene_id) if str(scene_id).isdigit() else scene_id_s,
#         'Round': last_round,
#         'Average': avg_all,
#         'DebateCount': debate_count,
#         'DebatedMetrics': debated_metrics_str,
#         'RunTag': (run_tag or ""),
#     }
#     for m in METRICS:
#         out_row[m] = avg_per_metric[m]

#     header_needed = not os.path.exists(multi_csv) or os.path.getsize(multi_csv) == 0
#     with open(multi_csv, "a", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=cols)
#         if header_needed:
#             writer.writeheader()
#         writer.writerow(out_row)

#     print(f"[summary] scene {scene_id} -> {multi_csv}")


# def main():
#     args = parse_args()
#     run_tag = (args.run_tag or "").strip()

#     config = CfgNode(new_allowed=True)

#     output_file = _with_tag(os.path.join("output/message", args.output_file), run_tag)
#     config = utils.add_variable_to_config(config, "output_file", output_file)

#     config = utils.add_variable_to_config(config, "log_file", args.log_file)
#     config = utils.add_variable_to_config(config, "log_name", args.log_name)
#     config = utils.add_variable_to_config(config, "play_role", args.play_role)
#     config.merge_from_file(args.config_file)

#     global_logger = utils.set_logger("quick_start.log", "quick_start")

#     # titles = [

#     # ]
#     # titles = [

#     # ]
#     # # character_llms = ['']  
#     # character_llms = [

#     # ]
#     titles = [

#     ]
#     # character_llms = ['']  
#     character_llms = [
#         'deepseek-r1:8b'
#     ]


#     if args.character_llm:
#         character_llms = [args.character_llm]

#     max_scenes = config.get('max_scenes', 5)
#     scene_ids = [i for i in range(0, 9)]

#     for model_path in tqdm(character_llms, desc="Models"):
#         print("Current Model:", model_path)
#         for title in tqdm(titles, desc="Titles"):
#             print("Current Title:", title)

#             character = model_path.split("/")[-1]
#             config['title'] = title
#             config['character_llm'] = character

#             narrator = config.get('narrator_llm', '')

#             detail_csv_raw = _build_detail_path(title, character, narrator)
#             detail_csv = _with_tag(detail_csv_raw, run_tag)

#             record_path = f"output/record/{title}/character/{narrator}_{character}_character.jsonl"
#             if not os.path.exists(record_path):
#                 print(f"[driver] record not found: {record_path}")
#                 continue

#             eval_done = {character: {sid: [] for sid in scene_ids}}
#             if os.path.exists(detail_csv):
#                 try:
#                     df_exist = pd.read_csv(detail_csv)
#                     for _, row in df_exist.iterrows():
#                         model_key = row.get('Model', character) or character
#                         sid = row.get('SceneID', None)
#                         cinfo = row.get('CharacterInfo', "")
#                         if sid is None:
#                             continue
#                         if model_key not in eval_done:
#                             eval_done[model_key] = {}
#                         if sid not in eval_done[model_key]:
#                             eval_done[model_key][sid] = []
#                         eval_done[model_key][sid].append(cinfo)
#                 except Exception as e:
#                     print("Warning: failed to read existing detail file, start fresh. Err:", e)
#             scenes = {}
#             with open(record_path, 'r', encoding='utf-8') as f:
#                 lines = f.readlines()
#                 for idx in range(len(lines) - 1, -1, -1):
#                     record = json.loads(lines[idx])
#                     if record['scene_id'] not in scenes:
#                         scenes[record['scene_id']] = record

#             available_ids = sorted(list(scenes.keys()))
#             if max_scenes and len(available_ids) > max_scenes:
#                 available_ids = available_ids[-max_scenes:]

#             log_dir = f"output/log/evaluation/detail/{title}_{run_tag or 'default'}"
#             utils.ensure_dir(log_dir)
#             futures = []
#             with concurrent.futures.ThreadPoolExecutor() as executor_eval:
#                 for sid in available_ids:
#                     scene = scenes[sid]
#                     config['scene_id'] = scene['scene_id']

#                     log_file = f"evaluation/detail/{title}_{run_tag or 'default'}/{config.get('narrator_llm','')}_{config.get('character_llm','')}_{config.get('scene_id')}_evaluation.log"
#                     logger = utils.set_logger(log_file, args.log_name)
#                     logger.info(f"os.getpid()={os.getpid()}")
#                     logger.info(f"\n{config}")

#                     done_list = eval_done.get(config['character_llm'], {}).get(config['scene_id'], [])
#                     futures.append(
#                         executor_eval.submit(run_eval_with_config, config.copy(), logger, scene['record'], detail_csv, done_list)
#                     )

#                 for future in concurrent.futures.as_completed(futures):
#                     try:
#                         result_sid = future.result()
#                         print(f"SCENE DONE: {result_sid}")
#                     except Exception as exc:
#                         print(f'Generated an exception: {exc}')

#             print(f"[{title}] detail saved -> {detail_csv}")

#             for sid in available_ids:
#                 multi_csv_raw = _build_multi_path(title, narrator, sid)
#                 multi_csv = _with_tag(multi_csv_raw, run_tag)
#                 _append_scene_summary_from_detail(
#                     detail_csv=detail_csv,
#                     multi_csv=multi_csv,
#                     narrator=narrator,
#                     character=character,
#                     title=title,
#                     scene_id=sid,
#                     run_tag=run_tag
#                 )


# if __name__ == "__main__":
#     main()
