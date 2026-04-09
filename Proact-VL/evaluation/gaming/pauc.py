import argparse, json, os
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Set, Tuple, Dict, Any
import numpy as np
try:
    from evaluation.gaming.gpt_client_utils import GPTClientFactory
except ImportError:
    from gpt_client_utils import GPTClientFactory


def _make_question_id(dataset_name: str, video_id: str) -> str:
    dataset_name = (dataset_name or "").strip()
    video_id = (video_id or "").strip()
    if dataset_name:
        # Avoid conflict with '=' separator used in custom_id
        return f"{dataset_name}::{video_id}"
    return video_id

def _make_segment_id(seg: dict) -> str:
    """
    Unique ID for segment-jsonl: prefer idx (if present), otherwise begin-end.
    Note: avoid '=' because custom_id uses '=' as a separator.
    """
    dataset_name = str(seg.get("dataset_name", "") or "").strip()
    video_id = str(seg.get("video_id", "") or "").strip()
    idx = seg.get("idx", None)
    if idx is not None:
        return f"{dataset_name}::{video_id}::idx={idx}".replace("=", "_")
    begin = seg.get("begin", "")
    end = seg.get("end", "")
    return f"{dataset_name}::{video_id}::span={begin}-{end}".replace("=", "_")

def _load_json_or_jsonl(fname: str):
    """
    Read JSON or JSONL:
    - JSON: return json.load result
    - JSONL: return list[dict] (json.loads per line)
    """
    try:
        with open(fname, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        data = []
        with open(fname, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
        return data


def _is_segment_jsonl_item(obj: dict) -> bool:
    # segment format: one item per line {video_id, begin, end, pred:{ts:txt}, label:{ts:txt}, dataset_name?}
    if not isinstance(obj, dict):
        return False
    if "video_id" not in obj or "begin" not in obj or "end" not in obj:
        return False
    if "pred" not in obj or "label" not in obj:
        return False
    return isinstance(obj.get("pred"), dict) and isinstance(obj.get("label"), dict)

def _is_pred_segment_jsonl_item(obj: dict) -> bool:
    """pred-jsonl row: must contain video_id/begin/end/pred(dict); label may be missing or None."""
    if not isinstance(obj, dict):
        return False
    if "video_id" not in obj or "begin" not in obj or "end" not in obj:
        return False
    if "pred" not in obj:
        return False
    return isinstance(obj.get("pred"), dict)


def _load_reference_labels(reference_file: str) -> Dict[Tuple[str, str, Any, Any], dict]:
    """
    Load label data from reference file (indexed by (video_id, dataset_name, begin, end)).
    Reference format: one JSON per line, containing at least video_id/dataset_name/begin/end/label(dict).
    """
    ref = {}
    if not reference_file:
        return ref
    with open(reference_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            vid = item.get("video_id")
            dataset = item.get("dataset_name")
            begin = item.get("begin")
            end = item.get("end")
            label = item.get("label")
            if not vid:
                continue
            if not isinstance(label, dict) or not label:
                continue
            ref[(vid, dataset, begin, end)] = label
    return ref


def _build_label_lookup_from_segments(seg_list: list) -> Dict[Tuple[str, str, Any, Any], dict]:
    """Extract label mapping from segment list: (video_id, dataset_name, begin, end) -> label(dict)."""
    m = {}
    if not isinstance(seg_list, list):
        return m
    for seg in seg_list:
        if not isinstance(seg, dict):
            continue
        vid = seg.get("video_id")
        dataset = seg.get("dataset_name")
        begin = seg.get("begin")
        end = seg.get("end")
        label = seg.get("label")
        if not vid:
            continue
        if isinstance(label, dict) and label:
            m[(vid, dataset, begin, end)] = label
    return m


def _join_ts_text_map(ts2text: dict) -> str:
    """Sort {timestamp(str): text} by time and join into one paragraph."""
    if not ts2text:
        return ""
    items = []
    for k, v in ts2text.items():
        try:
            t = float(k)
        except (TypeError, ValueError):
            continue
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        items.append((t, s))
    items.sort(key=lambda x: x[0])
    return " ".join([s for _, s in items]).strip()


def _merge_adjacent_pred_sents(all_pred_sents, delta_s: float = 1.0):
    """
    Merge predicted sentences whose adjacent time difference is delta_s:
    - Merge content (if previous chunk ends with "..." or "…", strip it before concatenation)
    - Use the start time of the continuous sequence as timestamp

    Input/output format: list[tuple[time(float), content(str)]]
    """
    if not all_pred_sents:
        return []

    def _strip_trailing_ellipsis(s: str) -> str:
        s = (s or "").rstrip()
        if s.endswith("..."):
            return s[:-3].rstrip()
        if s.endswith("…"):
            return s[:-1].rstrip()
        return s

    merged = []
    start_t, prev_t = all_pred_sents[0][0], all_pred_sents[0][0]
    buf = str(all_pred_sents[0][1]).strip()

    for t, sent in all_pred_sents[1:]:
        sent = str(sent).strip()
        # Merge only items whose adjacent time difference is delta_s
        if round(t - prev_t, 6) == round(delta_s, 6):
            buf = _strip_trailing_ellipsis(buf)
            if buf and sent:
                buf = f"{buf} {sent}"
            else:
                buf = buf + sent
            prev_t = t
        else:
            merged.append((start_t, buf))
            start_t, prev_t = t, t
            buf = sent

    merged.append((start_t, buf))
    return merged


def _get_openai_client():
    """Build GPT client with Entra ID or API Key authentication."""
    return GPTClientFactory().build()


_thread_local = threading.local()


def _get_thread_client():
    """Use one client per thread to avoid potential thread-safety issues."""
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = _get_openai_client()
        _thread_local.client = client
    return client


def _judge_one_request(
    req: Dict[str, Any],
    openai_model: Optional[str],
    temperature: float,
    max_tokens: int,
    max_retries: int,
    retry_sleep_s: float,
) -> Tuple[str, Dict[str, Any]]:
    """
    Call `chat.completions` for a single request and return (custom_id, output_line_dict).
    Never raises exceptions (writes the minimum score "1" on failure).
    """
    custom_id = req.get("custom_id", "")
    body = req.get("body", {}) or {}
    model = body.get("model") or openai_model
    messages = body.get("messages")

    content = "1"
    if model and messages:
        last_err = None
        for attempt in range(max_retries + 1):
            try:
                client = _get_thread_client()
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    # max_tokens=max_tokens,
                )
                content = resp.choices[0].message.content or "1"
                break
            except Exception as e:
                last_err = e
                if attempt == max_retries:
                    content = "1"
                    print(f"[WARN] judge failed custom_id={custom_id}: {repr(last_err)}")
                else:
                    time.sleep(retry_sleep_s)

    out_line = {
        "custom_id": custom_id,
        "response": {
            "body": {
                "choices": [
                    {"message": {"content": str(content)}}
                ]
            }
        },
    }
    return custom_id, out_line


def _sync_judge_from_requests(
    requests_jsonl: str,
    responses_jsonl: str,
    openai_model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4,
    max_retries: int = 3,
    retry_sleep_s: float = 2.0,
    resume: bool = True,
    concurrency: int = 8,
    judge_limit: int = -1,
    print_every: int = 50,
):
    """
    Synchronously call `chat.completions` for each request in the JSONL generated by
    `create_openai_batch_input`, and write a response JSONL compatible with OpenAI Batch output
    so that `process_openai_batch_output` can be reused.
    """
    # Resume support: skip already completed custom_id entries
    done_custom_ids: Set[str] = set()
    if resume and os.path.exists(responses_jsonl):
        try:
            for line in open(responses_jsonl, encoding="utf-8"):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    cid = obj.get("custom_id")
                    if cid:
                        done_custom_ids.add(cid)
                except json.JSONDecodeError:
                    continue
        except OSError:
            pass

    out_dir = os.path.dirname(responses_jsonl)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    total = 0
    skipped = 0
    written = 0
    scheduled = 0  # Actual number of newly scheduled judge requests (excluding resume skips)
    if concurrency < 1:
        concurrency = 1

    # Use thread pool for concurrent requests; main thread writes files to avoid write conflicts
    with open(responses_jsonl, "a", encoding="utf-8") as f_out, ThreadPoolExecutor(max_workers=concurrency) as ex:
        in_flight = set()

        def _drain_some(min_done: int = 1):
            nonlocal written
            done_cnt = 0
            for fut in as_completed(in_flight):
                in_flight.remove(fut)
                _, out_line = fut.result()
                f_out.write(json.dumps(out_line, ensure_ascii=False) + "\n")
                done_cnt += 1
                written += 1
                if print_every > 0 and written % print_every == 0:
                    print(f"[judge] written={written}, skipped={skipped}, total_seen={total}, in_flight={len(in_flight)}")
                if done_cnt >= min_done:
                    break
        for line in open(requests_jsonl, encoding="utf-8"):
            if not line.strip():
                continue
            req = json.loads(line)
            custom_id = req.get("custom_id", "")
            total += 1

            if custom_id in done_custom_ids:
                skipped += 1
                continue
            if judge_limit is not None and judge_limit > 0 and scheduled >= judge_limit:
                break
            # print(requests_jsonl)
            fut = ex.submit(
                _judge_one_request,
                req,
                openai_model,
                temperature,
                max_tokens,
                max_retries,
                retry_sleep_s,
            )
            in_flight.add(fut)
            scheduled += 1

            # Control queue length to avoid memory spikes
            if len(in_flight) >= concurrency * 2:
                _drain_some(min_done=concurrency)

        # Finalize: flush all remaining items
        while in_flight:
            _drain_some(min_done=1)

    print(
        f"[judge done] written={written}, skipped={skipped}, total_seen={total}, "
        f"scheduled={scheduled}, output={responses_jsonl}"
    )


def one_step_eval(
    pred_fname: str,
    gold_fname: Optional[str],
    output_fname: str,
    openai_model: str,
    n_examples: int = -1,
    max_scores: int = 3,
    resume: bool = False,
    concurrency: int = 8,
    judge_limit: int = 100,
    reference_fname: Optional[str] = None,
    start_score: float = 0.5,
):
    """
    Complete in one step:
    1) Generate judge requests (JSONL)
    2) Synchronously call OpenRouter `chat.completions` to generate judge responses (JSONL)
    3) Aggregate into structured JSON (compatible with `stat_metric`)
    4) Print final metrics
    """
    out_dir = os.path.dirname(output_fname)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    stem, _ = os.path.splitext(output_fname)
    requests_jsonl = stem + "judge_requests.jsonl"
    responses_jsonl = stem + "judge_responses.jsonl"

    if not gold_fname:
        gold_fname = pred_fname

    create_openai_batch_input(
        pred_fname=pred_fname,
        gold_fname=gold_fname,
        output_fname=requests_jsonl,
        n_examples=n_examples,
        openai_model=openai_model,
        reference_fname=reference_fname,
    )
    _sync_judge_from_requests(
        requests_jsonl=requests_jsonl,
        responses_jsonl=responses_jsonl,
        openai_model=openai_model,
        resume=resume,
        concurrency=concurrency,
        judge_limit=judge_limit,
    )
    process_openai_batch_output(
        pred_fname=pred_fname,
        gold_fname=gold_fname,
        input_fname=responses_jsonl,
        output_fname=output_fname,
        n_examples=n_examples,
        reference_fname=reference_fname,
    )
    def _count_judged_points(scored_json: str) -> Dict[str, int]:
        data = json.load(open(scored_json, encoding="utf-8"))
        num_examples = len(data) if isinstance(data, list) else 0
        num_turns = 0
        num_points = 0
        for ex in data or []:
            for turn in ex.get("answer", []) or []:
                num_turns += 1
                js = turn.get("judge_scores") or {}
                num_points += len(js)
        return {"num_examples": num_examples, "num_turns": num_turns, "num_judged_points": num_points}

    summary = {
        "pred_fname": pred_fname,
        "gold_fname": gold_fname,
        "reference_fname": reference_fname,
        "openai_model": openai_model,
        "judge_limit": judge_limit,
        "concurrency": concurrency,
        "resume": resume,
        "output_fname": output_fname,
        "start_score": start_score,
    }
    summary.update(_count_judged_points(output_fname))

    print(f"NOTE: max_score={max_scores}, so the largest y-axis is {max_scores-1}")
    for omega in [0, 0.5, 1]:
        res = stat_metric(output_fname, omega=omega, max_score=max_scores - 1, start_score=start_score)
        print(round(res["mean_auc"] * 100, 1), end=" & ")
        summary[f"mean_auc_omega_{omega}"] = float(res["mean_auc"])
    print("num_examples:", res["num_videos"])
    print()

    # New: aggregate by tag (following evaluate_end_to_end.py by_tag pattern)
    try:
        res_by_tag = stat_metric_by_tag(
            input_fname=output_fname,
            omegas=(0, 0.5, 1),
            max_score=max_scores - 1,
            start_score=start_score,
            ignore_empty=False,
        )
        summary["pauc_by_tag"] = res_by_tag
        if res_by_tag.get("by_tag"):
            print("By tag (PAUC):")
            for tag, stats in res_by_tag["by_tag"].items():
                print(f"{tag}:")
                for omega in ["0", "0.5", "1"]:
                    v = stats["mean_auc"].get(omega)
                    if v == v:  # not NaN
                        print(f"  omega={omega}: mean_auc={v:.6f} ({stats['num_turns'].get(omega, 0)} turns)")
                    else:
                        print(f"  omega={omega}: mean_auc=nan ({stats['num_turns'].get(omega, 0)} turns)")
            print()
    except Exception as e:
        print(f"[WARN] stat_metric_by_tag failed in one_step_eval: {repr(e)}")

    # Additional metric computed only on judged turns, useful for judge_limit small-sample sanity checks
    print(f"NOTE (judged_only): ignore empty turns (no judge_scores)")
    for omega in [0, 0.5, 1]:
        res_j = stat_metric(output_fname, omega=omega, max_score=max_scores - 1, ignore_empty=True, start_score=start_score)
        # If judge_limit is too small, res_j['num_turns'] may be 0 and np.mean can become nan
        mean_auc = res_j["mean_auc"]
        if mean_auc == mean_auc:  # not NaN
            print(round(mean_auc * 100, 1), end=" & ")
        else:
            print("nan", end=" & ")
        summary[f"mean_auc_judged_only_omega_{omega}"] = float(mean_auc) if mean_auc == mean_auc else None
    print("num_turns_used:", res_j["num_turns"])
    print()

    # by_tag stats for judged_only (for small-sample/resume sanity checks)
    try:
        res_by_tag_j = stat_metric_by_tag(
            input_fname=output_fname,
            omegas=(0, 0.5, 1),
            max_score=max_scores - 1,
            start_score=start_score,
            ignore_empty=True,
        )
        summary["pauc_by_tag_judged_only"] = res_by_tag_j
    except Exception as e:
        print(f"[WARN] stat_metric_by_tag(ignore_empty=True) failed: {repr(e)}")

    summary_fname = os.path.splitext(output_fname)[0] + "summary.json"
    with open(summary_fname, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[saved] {summary_fname}")


def load_model_preds(pred_fname):
    """
    Load model prediction data.
    Supported formats:
    1. Legacy format: one JSON object per line, with `question_id` and `model_response_list`
    2. New format: a JSON array, each element has `video_id` and a `label` list, where each label may include `pred`
    3. segment-jsonl: one segment per line, including `video_id`/`begin`/`end`/`pred(dict)`
    """
    res_dict = dict()
    
    # Try loading as JSON / JSONL (new format or segment-jsonl)
    try:
        data = _load_json_or_jsonl(pred_fname)
        if isinstance(data, list) and data:
            # 1) segment-jsonl: one segment per line
            if _is_segment_jsonl_item(data[0]):
                for seg in data:
                    if not _is_segment_jsonl_item(seg):
                        continue
                    video_id = seg.get("video_id")
                    if not video_id:
                        continue
                    dataset_name = seg.get("dataset_name", "")
                    question_id = _make_question_id(dataset_name, video_id)
                    pred_map = seg.get("pred") or {}
                    for ts, content in pred_map.items():
                        try:
                            t = float(ts)
                        except (ValueError, TypeError):
                            continue
                        if content is None:
                            continue
                        s = str(content).strip()
                        if not s:
                            continue
                        res_dict.setdefault(question_id, []).append((t, s))
                return res_dict

            # 2) Legacy-new format: JSON array (each element is a video; label is a list; label_seg contains pred dict)
            first_item = data[0]
            if isinstance(first_item, dict) and "video_id" in first_item and isinstance(first_item.get("label"), list):
                for video_data in data:
                    video_id = video_data.get('video_id')
                    if video_id is None:
                        continue
                    dataset_name = video_data.get("dataset_name", "")
                    question_id = _make_question_id(dataset_name, video_id)

                    pred_list = []
                    label_list = video_data.get('label', [])
                    for label_seg in label_list:
                        if 'pred' in label_seg and label_seg['pred']:
                            for timestamp_str, content in label_seg['pred'].items():
                                try:
                                    timestamp = float(timestamp_str)
                                    pred_list.append((timestamp, content))
                                except (ValueError, TypeError):
                                    continue

                    if pred_list:
                        res_dict[question_id] = pred_list
                return res_dict
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Old format: one JSON object per line
    for line in open(pred_fname, encoding='utf-8'):
        example = json.loads(line)
        res_dict[example['question_id']] = [(e['time'], e['content']) for e in example['model_response_list'] if e.get('role', 'assistant') == 'assistant']
        # model_response_list may contain user input in some cases
    return res_dict


def create_openai_batch_input(pred_fname, gold_fname, output_fname, n_examples, openai_model, reference_fname: Optional[str] = None):
    game_instruction = (
        "You are an evaluator for a gaming commentary system. Your task is to rate the whether the predicted answer covers the key points of the ground truth answer. Use the following scale to assign a score:\n"
        "- 3: Mostly covered; the predicted answer covers all key information in the ground truth answer, though it may have minor inaccuracies or rephrases.\n"
        "- 2: Partially covered; the predicted answer has some correct information, but also contains significant inaccuracies or missing key points.\n"
        "- 1: Incorrect; the predicted answer may be related to the ground truth answer, but most of the information is missing, or the predicted answer is of very poor quality."

        "The similarity between past and current Predicted Answers must be considered, and penalties will be applied to both cases."
        "Output the score only, do not add more explanations."
    )

    # In segment-jsonl mode, do not pre-aggregate by video, otherwise overlapping windows get mixed.
    # So defer load_model_preds until format detection is complete.
    
    # Load gold (may be pred_fname itself or a separate gold file)
    gold_data_list = _load_json_or_jsonl(gold_fname) if gold_fname else []
    # Optional reference: used to backfill missing labels in pred/gold rows
    reference_labels = _load_reference_labels(reference_fname) if reference_fname else {}
    gold_label_lookup = _build_label_lookup_from_segments(gold_data_list) if isinstance(gold_data_list, list) else {}
    
    # Determine whether format is new or old
    is_new_format = False
    is_segment_format = False
    # Segment detection is more reliable via pred_fname naming (pred-segment-jsonl, label may be missing)
    pred_list_for_detect = _load_json_or_jsonl(pred_fname)
    if isinstance(pred_list_for_detect, list) and pred_list_for_detect:
        if _is_pred_segment_jsonl_item(pred_list_for_detect[0]):
            is_segment_format = True
    if (not is_segment_format) and isinstance(gold_data_list, list) and len(gold_data_list) > 0:
        first_item = gold_data_list[0]
        if _is_segment_jsonl_item(first_item):
            is_segment_format = True
        elif 'video_id' in first_item and 'label' in first_item:
            is_new_format = True
    
    if is_segment_format:
        # segment-jsonl: process line-by-line (iterate pred_fname), evaluate each segment independently
        pred_list = pred_list_for_detect if isinstance(pred_list_for_detect, list) else _load_json_or_jsonl(pred_fname)

        # Create output directory if directory name is not empty
        output_dir = os.path.dirname(output_fname)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        result_list = []
        added_examples = 0
        skipped_no_label = 0
        for seg in (pred_list or []):
            if not _is_pred_segment_jsonl_item(seg):
                continue

            seg_id = _make_segment_id(seg)
            vid = seg.get("video_id")
            dataset = seg.get("dataset_name")
            begin = float(seg.get("begin"))
            end = float(seg.get("end"))
            # Prefer in-segment label; otherwise backfill from gold_fname or reference
            label_dict = None
            if isinstance(seg.get("label"), dict) and seg.get("label"):
                label_dict = seg.get("label")
            else:
                key = (vid, dataset, seg.get("begin"), seg.get("end"))
                label_dict = gold_label_lookup.get(key) or reference_labels.get(key)
            if not isinstance(label_dict, dict) or not label_dict:
                skipped_no_label += 1
                continue
            gold_text = _join_ts_text_map(label_dict)
            if not gold_text:
                skipped_no_label += 1
                continue

            pred_map = seg.get("pred") or {}
            all_pred_sents = []
            for ts, content in (pred_map or {}).items():
                try:
                    t = float(ts)
                except (ValueError, TypeError):
                    continue
                if begin <= t <= end:
                    s = str(content).strip()
                    if s:
                        all_pred_sents.append((t, s))

            all_pred_sents = sorted(all_pred_sents, key=lambda x: x[0])
            all_pred_sents = _merge_adjacent_pred_sents(all_pred_sents, delta_s=1.0)

            # Each segment has only one gold turn, so turn_i is fixed at 0
            turn_i = 0
            added_pred_sents = []
            for time, pred_sent in all_pred_sents:
                if pred_sent in added_pred_sents:
                    continue
                added_pred_sents.append(pred_sent)
                model_input = (
                    f"Ground Truth Answer: {gold_text}\n"
                    f"Predicted Answer: {' '.join(added_pred_sents)}"
                )
                conversation = [
                    {"role": "system", "content": game_instruction},
                    {"role": "user", "content": model_input},
                ]
                # custom_id uses only 3 parts: {segment_id}={turn_i}={time}
                custom_id = f"{seg_id}={turn_i}={time}"
                result_list.append(
                    {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {"model": openai_model, "messages": conversation},
                    }
                )

            added_examples += 1
            if added_examples == n_examples:
                break

        with open(output_fname, "w", encoding="utf-8") as f_out:
            for example in result_list:
                f_out.write(json.dumps(example, ensure_ascii=False) + "\n")
        if skipped_no_label:
            print(f"[segment] skipped segments without label: {skipped_no_label}")
        return

    # Non-segment format: keep original question_id-based aggregation logic
    pred_data_dict = load_model_preds(pred_fname)

    if is_new_format:
        # New format: convert to old-format structure
        gold_data_dict = {}
        for video_data in gold_data_list:
            video_id = video_data.get('video_id')
            if video_id is None:
                continue
            dataset_name = video_data.get('dataset_name', '')
            question_id = _make_question_id(dataset_name, video_id)
            
            # Convert to old format while preserving dataset_name
            gold_data_dict[question_id] = {
                'question_id': question_id,
                'dataset_name': dataset_name,
                'answer': []
            }
            
            label_list = video_data.get('label', [])
            for label_seg in label_list:
                if 'label_sentence' in label_seg:
                    answer_turn = {
                        'reply_timespan': [label_seg['begin'], label_seg['end']],
                        'content': label_seg['label_sentence'],
                        'related_timespan': [label_seg['begin'], label_seg['end']]  # Use same values
                    }
                    gold_data_dict[question_id]['answer'].append(answer_turn)
    else:
        # Old format
        gold_data_dict = {e['question_id']: e for e in gold_data_list}
        # Ensure old format also has dataset_name field (may be empty)
        for question_id in gold_data_dict:
            if 'dataset_name' not in gold_data_dict[question_id]:
                gold_data_dict[question_id]['dataset_name'] = ''
    
    # Create output directory if directory name is not empty
    output_dir = os.path.dirname(output_fname)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    added_examples = 0
    result_list = []
    for question_id, pred_data in pred_data_dict.items():
        if question_id not in gold_data_dict: continue
        gold_data = gold_data_dict[question_id]
        for turn_i, answer_turn in enumerate(gold_data['answer']):
            # find all pred sents in this ground truth answer span
            all_pred_sents = [(time, pred_sent) for time, pred_sent in pred_data \
                              if answer_turn['reply_timespan'][0] <= time <= answer_turn['reply_timespan'][1]]
            # Sort by time
            all_pred_sents = sorted(all_pred_sents, key=lambda x: x[0])
            before_adjust_pred = all_pred_sents
            all_pred_sents = _merge_adjacent_pred_sents(all_pred_sents, delta_s=1.0)

            added_pred_sents = []
            for idx, (time, pred_sent) in enumerate(all_pred_sents):
                if pred_sent in added_pred_sents:
                    continue
                added_pred_sents.append(pred_sent)
                model_input = (
                    # f"Question: {gold_data['conversation'][0]['content']}\n"
                    f"Ground Truth Answer: {answer_turn['content']}\n"
                    f"Predicted Answer: {' '.join(added_pred_sents)}"
                )
                conversation = [
                    {'role': 'system', 'content': game_instruction},
                    {'role': 'user', 'content': model_input},
                ]
                # Include dataset_name in custom_id to ensure uniqueness
                dataset_name = gold_data.get('dataset_name', '')
                if dataset_name:
                    custom_id = f"{dataset_name}={gold_data['question_id']}={turn_i}={time}"
                else:
                    custom_id = f"{gold_data['question_id']}={turn_i}={time}"
                output_example = {
                    "custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions",
                    "body": {"model": openai_model, "messages": conversation}
                }
                result_list.append(output_example)

        added_examples += 1
        if added_examples == n_examples: break

    with open(output_fname, 'w', encoding='utf-8') as f_out:
        for example in result_list:
            f_out.write(json.dumps(example, ensure_ascii=False) + '\n')


def process_openai_batch_output(pred_fname, gold_fname, input_fname, output_fname, n_examples=-1, reference_fname: Optional[str] = None):
    # Load gold (may be pred_fname itself or a separate gold file)
    gold_data_list = _load_json_or_jsonl(gold_fname) if gold_fname else []
    reference_labels = _load_reference_labels(reference_fname) if reference_fname else {}
    gold_label_lookup = _build_label_lookup_from_segments(gold_data_list) if isinstance(gold_data_list, list) else {}
    
    # Determine whether format is new or old
    is_new_format = False
    is_segment_format = False
    pred_list_for_detect = _load_json_or_jsonl(pred_fname)
    if isinstance(pred_list_for_detect, list) and pred_list_for_detect:
        if _is_pred_segment_jsonl_item(pred_list_for_detect[0]):
            is_segment_format = True
    if (not is_segment_format) and isinstance(gold_data_list, list) and len(gold_data_list) > 0:
        first_item = gold_data_list[0]
        if _is_segment_jsonl_item(first_item):
            is_segment_format = True
        elif 'video_id' in first_item and 'label' in first_item:
            is_new_format = True
    
    if is_segment_format:
        # segment-jsonl: process line-by-line; each segment is one example with a single answer turn
        pred_list = pred_list_for_detect if isinstance(pred_list_for_detect, list) else _load_json_or_jsonl(pred_fname)
        gold_data_dict = {}
        for seg in (pred_list or []):
            if not _is_pred_segment_jsonl_item(seg):
                continue
            seg_id = _make_segment_id(seg)
            dataset_name = str(seg.get("dataset_name", "") or "")
            begin = float(seg.get("begin"))
            end = float(seg.get("end"))
            tag = seg.get("tag") or dataset_name
            vid = seg.get("video_id")
            dataset = seg.get("dataset_name")
            label_dict = None
            if isinstance(seg.get("label"), dict) and seg.get("label"):
                label_dict = seg.get("label")
            else:
                key = (vid, dataset, seg.get("begin"), seg.get("end"))
                label_dict = gold_label_lookup.get(key) or reference_labels.get(key)
            gold_text = _join_ts_text_map(label_dict or {})

            pred_map = seg.get("pred") or {}
            preds = []
            for ts, content in (pred_map or {}).items():
                try:
                    t = float(ts)
                except (ValueError, TypeError):
                    continue
                if begin <= t <= end:
                    s = str(content).strip()
                    if s:
                        preds.append((t, s))
            preds = sorted(preds, key=lambda x: x[0])
            preds = _merge_adjacent_pred_sents(preds, delta_s=1.0)

            gold_data_dict[seg_id] = {
                "question_id": seg_id,
                "dataset_name": dataset_name,
                "tag": tag,
                "answer": [
                    {
                        "reply_timespan": [begin, end],
                        "content": gold_text,
                        "related_timespan": [begin, end],
                        "preds": preds,
                        "judge_scores": {},
                    }
                ],
            }

    elif is_new_format:
        # New format: convert to old-format structure
        gold_data_dict = {}
        for video_data in gold_data_list:
            video_id = video_data.get('video_id')
            if video_id is None:
                continue
            dataset_name = video_data.get('dataset_name', '')
            tag = video_data.get("tag") or dataset_name
            question_id = _make_question_id(dataset_name, video_id)
            
            # Convert to old format while preserving dataset_name
            gold_data_dict[question_id] = {
                'question_id': question_id,
                'dataset_name': dataset_name,
                'tag': tag,
                'answer': []
            }
            
            label_list = video_data.get('label', [])
            for label_seg in label_list:
                if 'label_sentence' in label_seg:
                    answer_turn = {
                        'reply_timespan': [label_seg['begin'], label_seg['end']],
                        'content': label_seg['label_sentence'],
                        'related_timespan': [label_seg['begin'], label_seg['end']]  # Use same values
                    }
                    gold_data_dict[question_id]['answer'].append(answer_turn)
    else:
        # Old format
        gold_data_dict = {example['question_id']: example for example in gold_data_list}
        # Ensure old format also has dataset_name field (may be empty)
        for question_id in gold_data_dict:
            if 'dataset_name' not in gold_data_dict[question_id]:
                gold_data_dict[question_id]['dataset_name'] = ''
            if 'tag' not in gold_data_dict[question_id]:
                gold_data_dict[question_id]['tag'] = gold_data_dict[question_id].get('dataset_name', '')
    
    output_dict = dict()

    # Only non-segment format requires question_id-based pred aggregation
    pred_data_dict = None if is_segment_format else load_model_preds(pred_fname)

    for line in open(input_fname):
        example = json.loads(line)
        splits = example['custom_id'].split('=')
        # Parse custom_id: format may be {dataset_name}={question_id}={turn_i}={time} or {question_id}={turn_i}={time}
        if len(splits) == 4:
            # New format: includes dataset_name
            dataset_name, question_id, turn_id, pred_time = splits[0], splits[1], int(splits[2]), float(splits[3])
        elif len(splits) == 3:
            # Old format: no dataset_name
            question_id, turn_id, pred_time = splits[0], int(splits[1]), float(splits[2])
        else:
            # Compatibility handling: use the last three parts
            question_id, turn_id, pred_time = '='.join(splits[:-2]), int(splits[-2]), float(splits[-1])
        
        if question_id not in output_dict:
            gold_data = gold_data_dict[question_id]
            if is_segment_format:
                # Segment mode: gold_data already has prefilled preds/judge_scores
                output_dict[question_id] = gold_data
            else:
                pred_data = pred_data_dict.get(question_id, [])
                for answer_i, answer in enumerate(gold_data['answer']):
                    gold_start_time, gold_end_time = answer['reply_timespan'][0], answer['reply_timespan'][1]
                    answer['preds'] = [(time, sent) for time, sent in pred_data if gold_start_time <= time <= gold_end_time]
                    answer['judge_scores'] = dict()
                output_dict[question_id] = gold_data
        gold_turn = output_dict[question_id]['answer'][turn_id]
        openai_judge_output = example['response']['body']['choices'][0]['message']['content']
        # NOTE: LLM output scores are in {1, 2, 3}, but when calculating PAUC area under curve we should use y \in {0, 1, 2}
        if openai_judge_output[0] in '012345':
            openai_judge_score = int(openai_judge_output[0]) - 1
        else:
            openai_judge_score = None

        if openai_judge_score is not None:
            if gold_turn['reply_timespan'][0] <= pred_time <= gold_turn['reply_timespan'][1]:
                gold_turn['judge_scores'][pred_time] = openai_judge_score

    # Handle missing prediction cases
    for question_id in gold_data_dict.keys():
        if question_id not in output_dict:
            # the model has no predictions in this ground truth reply span.
            gold_example = gold_data_dict[question_id]
            for answer in gold_example['answer']:
                answer['judge_scores'] = dict()
                answer['preds'] = list()
            output_dict[question_id] = gold_example

    # if there are some replies after related_timespan but in reply_timespan, and the score of these replies are lower than the last one in related_timespan,
    # remove the last turn, as this is possiblely from the second turn.
    for question_id, output_example in output_dict.items():
        for answer in output_example['answer']:
            if len(answer['judge_scores']) <= 1: continue
            if 'related_timespan' not in answer:
                continue  # Skip if related_timespan is missing
            judge_score_items = sorted(answer['judge_scores'].items(), key=lambda x: x[0])
            last_score_in_related_timespan = 0
            for time, score in judge_score_items:
                if time <= answer['related_timespan'][1]:
                    last_score_in_related_timespan = score
                elif score < last_score_in_related_timespan:
                    del answer['judge_scores'][time]

    with open(output_fname, 'w', encoding='utf-8') as f_out:
        json.dump(list(output_dict.values()), f_out, ensure_ascii=False, indent=2)


def stat_metric(input_fname, omega=0.5, max_score=2, start_score=0.5, ignore_empty: bool = False):
    pred_data = json.load(open(input_fname))
    auc_list = list()
    for example in pred_data:
        for turn in example['answer']:
            if not len(turn['judge_scores']):
                # if there is no pred reply, use start_score / max_score directly
                if not ignore_empty:
                    auc_list.append(start_score / max_score)
                continue

            points = [[float(key) - turn['reply_timespan'][0], val] for key, val in turn['judge_scores'].items()]
            max_x, max_y = turn['reply_timespan'][1] - turn['reply_timespan'][0], max_score
            auc_list.append(area_under_line_ratio(points, max_x, max_y, omega, start_score))
    return {'mean_auc': np.mean(auc_list), 'num_videos': len(pred_data), 'num_turns': len(auc_list)}

def stat_metric_by_tag(
    input_fname: str,
    omegas=(0, 0.5, 1),
    max_score: int = 2,
    start_score: float = 0.5,
    ignore_empty: bool = False,
):
    """
    Compute PAUC statistics by tag (consistent with the AUC definition in `stat_metric`).
    - overall: mean across all turns (equal weight per turn)
    - by_tag: mean across all turns under the same tag (equal weight per turn)
    """
    pred_data = json.load(open(input_fname, encoding="utf-8"))

    # Aggregation: tag -> omega -> list[auc]
    by_tag = defaultdict(lambda: defaultdict(list))
    overall = defaultdict(list)  # omega -> list[auc]

    for example in pred_data or []:
        tag = example.get("tag") or example.get("dataset_name")
        for turn in example.get("answer", []) or []:
            if not len(turn.get("judge_scores") or {}):
                if ignore_empty:
                    continue
                # No pred reply: use start_score/max_score as this turn's auc
                for omega in omegas:
                    auc = start_score / max_score if max_score else 0
                    overall[omega].append(auc)
                    by_tag[tag][omega].append(auc)
                continue

            points = [[float(key) - turn['reply_timespan'][0], val] for key, val in (turn.get("judge_scores") or {}).items()]
            max_x, max_y = turn['reply_timespan'][1] - turn['reply_timespan'][0], max_score
            for omega in omegas:
                auc = area_under_line_ratio(points, max_x, max_y, omega, start_score)
                overall[omega].append(auc)
                by_tag[tag][omega].append(auc)

    def _mean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    overall_out = {
        "num_examples": len(pred_data) if isinstance(pred_data, list) else 0,
        "num_turns": int(max(len(overall[omegas[0]]), 0)) if omegas else 0,
        "omegas": list(omegas),
        "mean_auc": {str(omega): _mean(overall[omega]) for omega in omegas},
    }

    by_tag_out = {}
    for tag in sorted(by_tag.keys(), key=lambda x: str(x)):
        by_tag_out[str(tag)] = {
            "num_turns": {str(omega): len(by_tag[tag][omega]) for omega in omegas},
            "mean_auc": {str(omega): _mean(by_tag[tag][omega]) for omega in omegas},
        }

    return {"overall": overall_out, "by_tag": by_tag_out}


def area_under_line_ratio(points, max_x, max_y, omega=0.5, start_score=0.5):
    """
    Calculate the area enclosed between the polyline formed by the coordinate points and the x-axis.
    Note that adjacent points are not connected by a diagonal line. Instead, at the x-coordinate of the subsequent point (x_2), the y-value abruptly increases to y_2.

    :param points: A list containing (x, y) tuples, e.g., [(x1, y1), (x2, y2), ...]
    :return: The area (float)
    """
    if not len(points): return 0
    points = sorted(points, key=lambda p: p[0])

    # ----- ADJUSTING THE IMPORTANCE OF TIMELINESS -----
    # Yet, an answer generated near the end of the span may still have its value.  
    # If this hyperparameter is 0, the x-axis remains unmodified—this is a scenario where timeliness is more important than correctness.  
    # If this hyperparameter is 1, all points on the x-axis are shifted to 1—this is a scenario where correctness is more important than timeliness.
    points = [(x * (1 - omega), y) for x, y in points]
    points.append([max_x, points[-1][1]])
    prev_y, prev_x, area = start_score, 0, 0  
    max_area = max_x * max_y
    for i in range(len(points)):
        x1, y1 = points[i]
        area += (x1 - prev_x) * prev_y
        prev_x, prev_y = x1, y1
    return area / max_area


def openai_send_batch(batch_input_fname, description="debug"):
    client = _get_openai_client()
    batch_input_file = client.files.create(file=open(batch_input_fname, "rb"), purpose="batch")
    batch_input_file_id = batch_input_file.id
    batch_metadata = client.batches.create(
        input_file_id=batch_input_file_id, 
        endpoint="/v1/chat/completions", completion_window="24h", 
        metadata={"description": description})
    print(batch_input_fname)
    print(batch_metadata)


def openai_get_batch(output_file_id, output_fname):
    client = _get_openai_client()
    if output_file_id is not None:
        file_response = client.files.content(output_file_id)
        print(f'saving result file {output_file_id} to {output_fname}')
        output_dir = os.path.dirname(output_fname)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_fname, 'w') as f_out:
            f_out.write(file_response.text)
    else:
        print('output_file_id is None, batch not completed')


if __name__ == '__main__':
    # import debugpy
    # debugpy.listen(9501)
    # print("Waiting for debugger to attach...")
    # debugpy.wait_for_client()
    parser = argparse.ArgumentParser()
    parser.add_argument('--func', type=str)
    parser.add_argument('--input_fname', type=str)
    parser.add_argument('--pred_fname', type=str)
    parser.add_argument('--gold_fname', type=str, default=None)
    parser.add_argument('--output_fname', type=str)
    parser.add_argument('--n_examples', type=int, default=-1)
    parser.add_argument('--openai_model', type=str, default='gpt-4.1')
    parser.add_argument('--max_scores', type=int, default=3)
    parser.add_argument('--resume', action='store_true', help='Resume in one_step mode (default: off). When enabled, already generated custom_id entries are skipped')
    parser.add_argument('--concurrency', type=int, default=8, help='Number of concurrent requests (threads) in one_step mode')
    parser.add_argument('--judge_limit', type=int, default=100, help='Maximum number of new judge requests in one_step mode (default: 100; use -1 for all)')
    parser.add_argument('--reference', type=str, default=None, help='Optional reference JSONL used to backfill labels by (video_id, dataset_name, begin, end)')
    parser.add_argument('--start_score', type=float, default=0.5, help='Initial score')
    parser.add_argument('--description', type=str)
    parser.add_argument('--file_id', type=str)
    args = parser.parse_args()
    print(args)

    if args.func == 'batch_input':
        if not args.gold_fname:
            raise ValueError("batch_input 需要 --gold_fname（one_step 模式可省略）")
        create_openai_batch_input(args.pred_fname, args.gold_fname, args.output_fname, args.n_examples, args.openai_model)
    elif args.func == 'batch_output':
        if not args.gold_fname:
            raise ValueError("batch_output 需要 --gold_fname（one_step 模式可省略）")
        process_openai_batch_output(args.pred_fname, args.gold_fname, args.input_fname, args.output_fname, args.n_examples)

    elif args.func == 'stat_metric':
        print(f"NOTE: max_score={args.max_scores}, so the largest y-axis is {args.max_scores-1}")
        for omega in [0, 0.5, 1]:
            res = stat_metric(args.input_fname, omega=omega, max_score=args.max_scores-1, start_score=args.start_score)
            print(round(res['mean_auc'] * 100, 1), end=' & ')
        print('num_examples:', res['num_videos'])
        print()

    elif args.func == 'stat_metric_by_tag':
        max_score = args.max_scores - 1
        res = stat_metric_by_tag(
            input_fname=args.input_fname,
            omegas=(0, 0.5, 1),
            max_score=max_score,
            start_score=args.start_score,
            ignore_empty=False,
        )
        print(f"NOTE: max_score={args.max_scores}, so the largest y-axis is {max_score}")
        # Print order: by_tag first, then overall (same organization as evaluate_end_to_end)
        for tag, stats in res["by_tag"].items():
            print(f"{tag}:")
            for omega in ["0", "0.5", "1"]:
                v = stats["mean_auc"].get(omega)
                if v == v:  # not NaN
                    print(f"  omega={omega}: mean_auc={v:.6f} ({stats['num_turns'].get(omega, 0)} turns)")
                else:
                    print(f"  omega={omega}: mean_auc=nan ({stats['num_turns'].get(omega, 0)} turns)")
        print("Overall:")
        for omega in ["0", "0.5", "1"]:
            v = res["overall"]["mean_auc"].get(omega)
            if v == v:
                print(f"  omega={omega}: mean_auc={v:.6f}")
            else:
                print(f"  omega={omega}: mean_auc=nan")

        # Do not create a separate by_tag_output; write directly to output_fname (in-place if same as input)
        if args.output_fname:
            out_dir = os.path.dirname(args.output_fname)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(args.output_fname, "w", encoding="utf-8") as f_out:
                json.dump(res, f_out, ensure_ascii=False, indent=2)
            print(f"[saved] {args.output_fname}")

    elif args.func == 'one_step':
        one_step_eval(
            pred_fname=args.pred_fname,
            gold_fname=args.gold_fname,
            output_fname=args.output_fname,
            openai_model=args.openai_model,
            n_examples=args.n_examples,
            max_scores=args.max_scores,
            resume=args.resume,
            concurrency=args.concurrency,
            judge_limit=args.judge_limit,
            reference_fname=args.reference,
            start_score=args.start_score,
        )

    elif args.func == 'send_batch':
        openai_send_batch(batch_input_fname=args.input_fname, description=args.description)
    elif args.func == 'get_batch':
        openai_get_batch(output_file_id=args.file_id, output_fname=args.output_fname)

    elif args.func == 'check_batch':
        from datetime import datetime
        client = _get_openai_client()
        for task in client.batches.list(limit=20).data:
            if task.metadata is not None:
                print(task.id, datetime.fromtimestamp(task.created_at), task.metadata.get('description', ''),
                      task.status, f"{task.request_counts.completed} / {task.request_counts.total}",
                      task.output_file_id, end='\n\n')
            else:
                print(task.id, datetime.fromtimestamp(task.created_at), '',
                      task.status, f"{task.request_counts.completed} / {task.request_counts.total}",
                      task.output_file_id, end='\n\n')
    else:
        raise NotImplementedError