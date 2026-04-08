import os, openai, json, functools, argparse
from datasets import load_dataset
from proactvl.utils.multiprocessor import local_mt
import re
from typing import Dict, Union

def extract_scores(text: str) -> Dict[str, Union[int, float]]:
    # Each line should look like: key: value
    pattern = re.compile(r'^\s*([^:\n]+?)\s*:\s*([+-]?\d+(?:\.\d+)?)\s*$', re.M)
    out = {}
    for k, v in pattern.findall(text):
        v_num = float(v)
        out[k.strip()] = int(v_num) if v_num.is_integer() else v_num
    return out

# ========================= Initialize GPT Client =========================
model_id = ''
try:
    from evaluation.gaming.gpt_client_utils import GPTClientFactory
except ImportError:
    from gpt_client_utils import GPTClientFactory


deployment_name = "gpt-5.1_2025-11-13"

try:
    gpt = GPTClientFactory().build()
except Exception as e:
    raise RuntimeError(f"Error: failed to initialize GPT client. {e}") from e
# ======================== End Initialize GPT Client =========================

video_event_id_to_gt_asr, video_start_end_to_video_event_id, video_event_id_to_model_pred = {}, {}, {}
video_event_id_to_dataset_name = {}
video_event_id_to_tag = {}
video_event_id_to_context = {}
video_event_id_to_gt_asr_clip = {}
video_event_id_to_model_pred_clip = {}

def judge_liveu(context, model_response, ground_truth):
    prompt = r"""Task Description:
You are an expert in evaluating the **Delivery Dynamics** of real-time, second-by-second streaming commentary for video clips.
Assess the model based on scenario constraints, timing/rhythm, and real-time listenability.

You will be provided:
1) Context Text (previous commentary or background context; may also contain prior lines from the same commentator)
2) Label Timeline Text (time-aligned reference timeline; includes all on-clip information such as [USER] questions, [SPEAKER*] commentary, and the current commentator’s lines; use mainly as a reference for rhythm and salient windows)
3) Prediction Timeline Text (the current commentator’s output to be evaluated)

Speaker tags (apply to all inputs):
- [ASSISTANT] = the current commentator’s lines (may appear in Context as prior commentary; and appears in Label/Prediction as on-clip current commentator)
- [USER] = user questions (reference/context only; may appear in Label Timeline)
- [SPEAKER*] = other commentators (reference/context only; may appear in Label Timeline)

What to score:
- ONLY score the content in Prediction Timeline Text.
- Any tagged lines in Context or Label Timeline are reference/context signals only and must NOT be scored.

Scenario & speaking constraints (infer from Label Timeline):
- Solo: normal streaming commentary flow.
- Multi-commentator: generally avoid speaking when [SPEAKER*] is speaking (avoid overlap), unless clearly necessary for a key moment. A brief 1-second overlap is acceptable; penalize overlap mainly when it is sustained (e.g., 2+ consecutive seconds), frequent (repeated across the clip), or clearly disruptive.
- Guidance: when [USER] asks a question, provide step-by-step guidance; avoid "one huge dump" of info.

Important notes:
- Prediction may be event-triggered: it may speak only on some seconds; missing seconds are silence. "SILENCE"/empty = silence.
- Label Timeline is a reference only; no verbatim match required. Use it mainly to judge speaking rhythm, salient windows, and whether Prediction contradicts key events.
- Subjective/expressive commentary is acceptable (brief opinions, reactions, humor, hype, atmosphere-building), as long as it does not contradict key events and does not cause disruptive chattering.

Scoring rules:
- Output integer scores in [1..10] (no 0).
- Score bands: 10 excellent; 7–9 good; 4–6 mixed; 2–3 very poor but somewhat usable; 1 unusable.

Dimension 1. Time (Synchronicity: timing + coverage + overlap etiquette):
Scoring criteria:
- 10: Excellent rhythm; enters within salient windows; reacts promptly to key moments and [USER] questions; stays silent when appropriate; strong coverage; overlap is rare and not sustained/disruptive.
- 7-9: Generally good timing; minor drift or small gaps; occasional brief overlap is acceptable; sustained overlap is rare.
- 4-6: Unstable rhythm; noticeable late/early speaking; multiple unnecessary lines or noticeable gaps; partial coverage; overlap becomes noticeable (e.g., repeated or sometimes sustained).
- 2-3: Poor; frequently misses obvious speaking moments or speaks when silence/etiquette requires not speaking; overlap is frequent or often sustained/disruptive.
- 1: Unusable timing; chaotic speaking/silence patterns or near-constant disruptive overlap that breaks the live experience.

Coverage tolerance rule (apply to all bands):
- Window shift is acceptable: if Label indicates a salient speaking window (e.g., 3–7s), speaking in any reasonably overlapping window such as 4–6, 4–7, 4–8, or 2–5 can be considered acceptable (exact boundaries not required).

Dimension 2. Rate (Cadence: density + pacing + anti-dump):
Scoring criteria:
- 10: Perfect real-time pacing; short and listenable per-second/burst outputs; no dense dumps; minimal filler; high signal-to-noise; Guidance answers delivered step-by-step (not a single large dump).
- 7-9: Mostly well paced; occasional slightly long bursts or mild redundancy/info-clumping, but still listenable.
- 4-6: Noticeable pacing issues; multiple long/dense bursts, repetitive filler, or fragmented delivery; Guidance sometimes turns into a dump; signal-to-noise often low.
- 2-3: Very poor pacing; frequent extremely long/dense outputs OR clearly under-speaking such that salient moments are missed.
- 1: Unusable pacing; consistently unlistenable due to extreme length/density or severe fragmentation.

Dimension 3. TextU (Streaming usability: spoken-form clarity):
Scoring criteria:
- 10: Consistently clear and well-formed as live speech; natural spoken wording; minimal grammar issues; easy to follow in real time; Guidance is step-by-step and actionable in delivery.
- 7-9: Mostly clear; minor awkwardness; occasional light fragmentation but overall usable as live speech.
- 4-6: Mixed; frequent awkward/incomplete phrasing or noticeable fragmentation; live usability is noticeably harmed.
- 2-3: Poor; many lines hard to understand, overly fragmented, or off-scenario (e.g., does not engage a [USER] question in Guidance); live usability is low.
- 1: Unusable; mostly incoherent/garbled or consistently impossible to follow as live speech.

STRICT output format (no extra characters):
Time: <1-10>
Rate: <1-10>
TextU: <1-10>

Context: {context}
Label Timeline: {label_timeline}
Prediction Timeline: {prediction_timeline}
""".format(context=context, label_timeline=ground_truth, prediction_timeline=model_response)
    return prompt

def judge_finalq(context, model_response, ground_truth):
    prompt = r"""Task Description:
You are an expert in evaluating the **Narrative Integrity** and overall quality of consolidated video commentary.
Assess the final script’s readability, coherence, and usefulness, while only penalizing direct event contradictions.

You will be provided:
1) Context Text (previous commentary or background context; may also contain prior lines from the same commentator)
2) Label Final Text (raw ASR final text from the original video; reference only)
3) Prediction Text (the final model output to be evaluated)

Speaker definitions (apply to Context, Label Final, and Prediction):
- [ASSISTANT] = current commentator’s lines (may appear in Context as prior commentary)
- [USER] = user questions (reference/context only; may appear in Context and/or Label Final)
- [SPEAKER*] = other commentators (reference/context only; may appear in Context and/or Label Final)

What to score:
- ONLY score the content in Prediction Text.
- Any tagged lines in Context / Label Final are reference/context signals only and must NOT be scored.

Evaluation principles:
1) Scenario alignment: in Multi-commentator mode, check whether [ASSISTANT] complements [SPEAKER*] logically; in Guidance mode, check whether the [USER] query is actually addressed/resolved.
2) Directional compatibility: use Label Final as a reference for major events; do NOT require word-by-word match; penalize **direct contradictions** (major event reversal, incompatible outcomes, or clearly wrong situation claims).
3) Fidelity (lightweight): expressive commentary is allowed, but avoid unjustified concrete specifics (e.g., names, numbers, causes, outcomes) that would mislead; major invented events that conflict with Label should be penalized.

Scoring rules:
- Output integer scores in [1..10] (no 0).
- Score bands: 10 excellent; 7–9 good; 4–6 mixed; 2–3 very poor but somewhat usable; 1 unusable.

Dimension 1. Fidelity (Event compatibility / non-contradiction):
Criteria: compatibility with Label Final and avoidance of misleading concrete claims.
- 10: Fully compatible; captures core events accurately; no major contradictions; any added specifics remain non-misleading and do not conflict with Context/Label.
- 7-9: Generally compatible; core events align; some added detail/emphasis is acceptable and does not contradict the situation.
- 4-6: Mixed; includes notable mismatches or unjustified concrete specifics that could mislead; weakens trust.
- 2-3: Largely incompatible; many statements contradict the reference or introduce incompatible events/claims.
- 1: Unusable; completely unrelated or fundamentally opposite to the reference facts.

Dimension 2. Continuity (Coherence with Context, co-hosts, and user thread):
Criteria: logical flow and referential integrity across entities, stance, and conversation.
- 10: Seamless continuity; consistent topics/entities/stance; complements [SPEAKER*] logically; clear referents (no confusing pronouns); smooth transitions; avoids obvious stitched repetition (e.g., repeating the same conclusion multiple times).
- 7-9: Mostly consistent; follows the narrative direction; minor awkwardness in transitions or small continuity slips; limited repetition.
- 4-6: Noticeable logic drift; repeats points already established; confusing references to entities; clunky transitions; coherence is weakened.
- 2-3: Poor; ignores [USER] questions in Guidance or contradicts important prior context; heavy repetition/looping; hard to follow logically.
- 1: Unusable; fundamental breakdown of context awareness or narrative logic.

Dimension 3. Substance (Text quality + usefulness + redundancy control):
Criteria: informational value, guidance effectiveness, and consolidated-script usability (structure, readability, low redundancy).
- 10: High value; clear structure; strong readability; minimal redundancy; provides helpful insights; directly and effectively addresses [USER] queries when present; adds value beyond generic filler.
- 7-9: Clear and informative; engaging voice; minor redundancy; covers the [USER] request reasonably well; mostly avoids clichés and empty looping.
- 4-6: Mediocre; generic/cliché; partially answers [USER] queries; noticeable repetition or bloated sections; usefulness is limited.
- 2-3: Low value; hollow/robotic; fails to address the [USER] or provides disjoint segments; heavy looping/redundancy; mostly filler.
- 1: Unusable due to severe readability failure or near-total lack of substance.

STRICT output format (no extra characters):
Fidelity: <1-10>
Continuity: <1-10>
Substance: <1-10>

Context: {context}
Label Final: {label_final}
Prediction: {prediction}
""".format(context=context, label_final=ground_truth, prediction=model_response)
    return prompt

def llm_score(video_event_id):
    model_pred_clip = video_event_id_to_model_pred_clip[video_event_id]
    model_pred = video_event_id_to_model_pred[video_event_id]
    gt_asr_clip = video_event_id_to_gt_asr_clip[video_event_id]
    gt_asr = video_event_id_to_gt_asr[video_event_id]
    context = video_event_id_to_context[video_event_id]
    score_map = {}
    prompt_liveu = judge_liveu(context, model_pred_clip, gt_asr_clip)


    prompt_final = judge_finalq(context, model_pred, gt_asr)
    # score_map_finalq = {}
    while True:
        try:
            resp = gpt.chat.completions.create(
                # model='gpt-4o-2024-08-06',
                # model='gpt-4.1',
                model=deployment_name,
                messages=[{"role": "user", "content": [{'type': 'text', 'text': prompt_liveu}]}],
                seed=42,
                temperature=0,
            ).choices[0].message.content
            liveU_score = extract_scores(resp)
            if 'Time' in liveU_score and 'Rate' in liveU_score and 'TextU' in liveU_score:
                break
            else:
                print(f'Incomplete scores received: {liveU_score}, retrying...')
        except Exception as e:
            print('Failed to get response...', e)

    while True:
        try:
            resp_final = gpt.chat.completions.create(
                # model='gpt-4o-2024-08-06',
                # model='gpt-4.1',
                model=deployment_name,
                messages=[{"role": "user", "content": [{'type': 'text', 'text': prompt_final}]}],
                seed=42,
                temperature=0,
            ).choices[0].message.content
            finalQ_score = extract_scores(resp_final)
            if 'Fidelity' in finalQ_score and 'Continuity' in finalQ_score and 'Substance' in finalQ_score:
                break
            else:
                print(f'Incomplete scores received: {finalQ_score}, retrying...')
        except Exception as e:
            print('Failed to get response...', e)
    # Merge the two scoring results
    score_map.update(liveU_score)
    score_map.update(finalQ_score)
    return score_map

def judge(video_event_id, model_id):
    gt_asr = video_event_id_to_gt_asr[video_event_id]
    dataset_name = video_event_id_to_dataset_name[video_event_id]
    tag = video_event_id_to_tag[video_event_id]
    return {
        'video_event_id': video_event_id, 
        'score': llm_score(video_event_id), 
        'dataset_name': dataset_name,
        'tag': tag
    }

def merge_commentary(commentary_dict):
    sorted_seconds = sorted(commentary_dict.keys(), key=int)
    merged_commentary = ' '.join([commentary_dict[sec].replace(' ...', '') for sec in sorted_seconds])
    return merged_commentary.strip()

def pred2standard(pred):
    sorted_seconds = sorted(pred.keys(), key=int)
    asrs = []
    asr_lines = []
    for sec in sorted_seconds:
        line = pred[sec]
        # Remove trailing ' ...'
        if line.endswith(' ...'):
            line = line[:-4]
        if line.strip() == '':
            continue
        else:
            asr_lines.append(f'[t={sec}] {line}')
            asrs.append(line)
    return '\n'.join(asr_lines), ' '.join(asrs)

def gt2standard(annotations):
    active_speaker = annotations['active_speaker']['name']
    asrs = []
    asr_lines = []
    for item in annotations['annotations']:
        # if item['speaker'] != active_speaker:
        #     continue
        sec = item['start']
        line = item['text'] if 'text' in item else item['query']

        if line.strip() == '':
            continue
        else:
            if item['speaker'] == active_speaker:
                
                asr_lines.append(f'[t={sec}][ASSISTANT]:{line}')
            elif item['speaker'] == 'user':
                asr_lines.append(f'[t={sec}][USER]:{line}')
            else:
                asr_lines.append(f'[t={sec}][{item["speaker"]}]:{line}')
            asrs.append(line)
    return '\n'.join(asr_lines), ' '.join(asrs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True, help='Model name to compare against baseline')
    parser.add_argument('--prediction_jsonl', type=str, required=True, help='Path to model predictions in JSONL format')
    parser.add_argument('--output_dir', type=str, default='evaluation/gaming/judges/', help='Directory to save judgment results')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--asr_jsonl', type=str, default=None, help='Path to ASR transcriptions in JSONL format')
    args = parser.parse_args()

    model_id = args.model_id
    prediction_jsonl = args.prediction_jsonl
    output_dir = args.output_dir


    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'llm_score_{model_id}.jsonl')

    # prediction_jsonl = 'evaluation/gaming/gaming/Qwen2.5-Omni-7B.jsonl'
    prediction_jsonl = args.prediction_jsonl



    for line in open(prediction_jsonl):
        datum = json.loads(line)
        # if datum['dataset_name'] not in args.dataset_name_list:
        #     continue
        dataset_name = datum['dataset_name']
        if dataset_name == 'ego4d_goal_step':
            dataset_name = 'ego4d'
        video_event_id = f'{datum["video_id"]}_{datum["begin"]}_{datum["end"]}_{dataset_name}_{datum["idx"]}'
        if video_event_id in video_event_id_to_model_pred:
            print(f'Warning: duplicate video_event_id {video_event_id} in predictions')
        pred_clip, pred = pred2standard(datum['pred'])
        video_event_id_to_model_pred_clip[video_event_id] = pred_clip
        video_event_id_to_model_pred[video_event_id] = pred
        video_event_id_to_dataset_name[video_event_id] = datum['dataset_name']
        video_event_id_to_tag[video_event_id] = datum['tag']

    for line in open(args.asr_jsonl):
        datum = json.loads(line)
        video_event_id = f'{os.path.basename(datum["video_path"])}_{datum["video_begin"]}_{datum["video_end"]}_{datum["dataset_name"]}_{datum["idx"]}'
        pred_clip, pred = gt2standard(datum)
        video_event_id_to_gt_asr_clip[video_event_id] = pred_clip
        video_event_id_to_gt_asr[video_event_id] = pred
        video_event_id_to_context[video_event_id] = datum['history'] if 'history' in datum else ''

    # assert len(video_event_id_to_model_pred) == len(video_event_id_to_gt_asr), "Mismatch between predictions and ASR ground truths"
    if len(video_event_id_to_model_pred) != len(video_event_id_to_context):
        print(f'Length of predictions: {len(video_event_id_to_model_pred)}, Length of contexts: {len(video_event_id_to_context)}')
        # Iterate and print missing video_event_id values
        missing_ids = set(video_event_id_to_context.keys()) - set(video_event_id_to_model_pred.keys())
        print(f'Number of missing contexts: {len(missing_ids)}')
        for missing_id in missing_ids:
            print(f'Missing context for video_event_id: {missing_id}')
        # raise ValueError("Mismatch between predictions and contexts")
    winner_results = local_mt(
        video_event_id_to_model_pred_clip.keys(),
        functools.partial(judge, model_id=model_id), 
        desc=f'LLM Score Call gpt4o for {model_id}', 
        num_workers=args.num_workers
    )
    
    # For liveu, aggregate results by game, by tag, and overall
    with open(os.path.join(output_dir, f'llm_score_liveu_{model_id}.jsonl'), 'w') as f:
        for winner_result in winner_results:
            liveu_score = {k: v for k, v in winner_result['score'].items() if k in ['Time', 'Rate', 'TextU']}
            output_dict = {
                'video_event_id': winner_result['video_event_id'],
                'liveu_score': liveu_score,
                'dataset_name': winner_result['dataset_name'],
                'tag': winner_result['tag']
            }
            f.write(json.dumps(output_dict) + '\n')
    # For finalq, aggregate results by game, by tag, and overall
    with open(os.path.join(output_dir, f'llm_score_finalq_{model_id}.jsonl'), 'w') as f:
        for winner_result in winner_results:
            finalq_score = {k: v for k, v in winner_result['score'].items() if k in ['Fidelity', 'Continuity', 'Substance']}
            output_dict = {
                'video_event_id': winner_result['video_event_id'],
                'finalq_score': finalq_score,
                'dataset_name': winner_result['dataset_name'],
                'tag': winner_result['tag']
            }
            f.write(json.dumps(output_dict) + '\n')
    # Aggregate liveu_score and finalq_score by game, by tag, then overall
    summary_by_game = {}
    for winner_result in winner_results:
        dataset_name = winner_result['dataset_name']
        if dataset_name not in summary_by_game:
            summary_by_game[dataset_name] = {
                'liveu_score': {'Time': 0, 'Rate': 0, 'TextU': 0},
                'finalq_score': {'Fidelity': 0, 'Continuity': 0, 'Substance': 0},
                'count': 0
            }
        liveu_score = {k: v for k, v in winner_result['score'].items() if k in ['Time', 'Rate', 'TextU']}
        finalq_score = {k: v for k, v in winner_result['score'].items() if k in ['Fidelity', 'Continuity', 'Substance']}
        for k in liveu_score:
            summary_by_game[dataset_name]['liveu_score'][k] += liveu_score[k]
        for k in finalq_score:
            summary_by_game[dataset_name]['finalq_score'][k] += finalq_score[k]
        summary_by_game[dataset_name]['count'] += 1
    # summary by game: output mean scores
    for dataset_name, summary in summary_by_game.items():
        count = summary['count']
        liveu_avg = {k: v / count for k, v in summary['liveu_score'].items()}
        finalq_avg = {k: v / count for k, v in summary['finalq_score'].items()}
        output = f'Average LiveU Score for {model_id} on {dataset_name}: ' + ', '.join([f'{k}: {v:.2f}' for k, v in liveu_avg.items()])
        print(output)
        with open(os.path.join(output_dir, 'log.txt'), 'a') as f:
            f.write(output + '\n')
        output = f'Average FinalQ Score for {model_id} on {dataset_name}: ' + ', '.join([f'{k}: {v:.2f}' for k, v in finalq_avg.items()])
        print(output)
        with open(os.path.join(output_dir, 'log.txt'), 'a') as f:
            f.write(output + '\n')
    # summary by tag
    summary_by_tag = {}
    for winner_result in winner_results:
        tag = winner_result['tag']
        if tag not in summary_by_tag:
            summary_by_tag[tag] = {
                'liveu_score': {'Time': 0, 'Rate': 0, 'TextU': 0},
                'finalq_score': {'Fidelity': 0, 'Continuity': 0, 'Substance': 0},
                'count': 0
            }
        liveu_score = {k: v for k, v in winner_result['score'].items() if k in ['Time', 'Rate', 'TextU']}
        finalq_score = {k: v for k, v in winner_result['score'].items() if k in ['Fidelity', 'Continuity', 'Substance']}
        for k in liveu_score:
            summary_by_tag[tag]['liveu_score'][k] += liveu_score[k]
        for k in finalq_score:
            summary_by_tag[tag]['finalq_score'][k] += finalq_score[k]
        summary_by_tag[tag]['count'] += 1
    # summary by tag: output mean scores
    for tag, summary in summary_by_tag.items():
        count = summary['count']
        liveu_avg = {k: v / count for k, v in summary['liveu_score'].items()}
        finalq_avg = {k: v / count for k, v in summary['finalq_score'].items()}
        output = f'Average LiveU Score for {model_id} on tag {tag}: ' + ', '.join([f'{k}: {v:.2f}' for k, v in liveu_avg.items()])
        print(output)
        with open(os.path.join(output_dir, 'log.txt'), 'a') as f:
            f.write(output + '\n')
        output = f'Average FinalQ Score for {model_id} on tag {tag}: ' + ', '.join([f'{k}: {v:.2f}' for k, v in finalq_avg.items()])
        print(output)
        with open(os.path.join(output_dir, 'log.txt'), 'a') as f:
            f.write(output + '\n')
    # overall average
    total_count = len(winner_results)
    total_liveu_score = {'Time': 0, 'Rate': 0, 'TextU': 0}
    total_finalq_score = {'Fidelity': 0, 'Continuity': 0, 'Substance': 0}
    for winner_result in winner_results:
        liveu_score = {k: v for k, v in winner_result['score'].items() if k in ['Time', 'Rate', 'TextU']}
        finalq_score = {k: v for k, v in winner_result['score'].items() if k in ['Fidelity', 'Continuity', 'Substance']}
        for k in liveu_score:
            total_liveu_score[k] += liveu_score[k]
        for k in finalq_score:
            total_finalq_score[k] += finalq_score[k]
    overall_liveu_avg = {k: v / total_count for k, v in total_liveu_score.items()}
    overall_finalq_avg = {k: v / total_count for k, v in total_finalq_score.items()}
    output = f'Overall Average LiveU Score for {model_id}: ' + ', '.join([f'{k}: {v:.2f}' for k, v in overall_liveu_avg.items()])
    print(output)
    with open(os.path.join(output_dir, 'log.txt'), 'a') as f:
        f.write(output + '\n')
    output = f'Overall Average FinalQ Score for {model_id}: ' + ', '.join([f'{k}: {v:.2f}' for k, v in overall_finalq_avg.items()])
    print(output)
    with open(os.path.join(output_dir, 'log.txt'), 'a') as f:
        f.write(output + '\n')  
