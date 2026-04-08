#!/usr/bin/env python3
"""
End-to-end evaluation script: from raw JSONL to Timediff results

Workflow:
1. Read the raw JSONL file
2. Merge data by video_id
3. Compute Timediff metrics
4. Output results (including exp_name)
"""

import json
import argparse
import os
from collections import defaultdict


def load_reference_labels(reference_file):
    """Load label data from a reference file (indexed by (video_id, begin, end, dataset_name, idx))."""
    print(f"  Loading labels from reference file: {reference_file}")
    
    labels_by_key = {}

    with open(reference_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                vid = item.get("video_id")
                dataset = item.get("dataset_name")
                begin = item.get("begin")
                end = item.get("end")
                idx = item.get("idx")
                
                if not vid:
                    continue

                # Use (video_id, begin, end, dataset_name, idx) as the unique key
                key = (vid, begin, end, dataset, idx)
                
                label = item.get("label")
                if label is not None:
                    labels_by_key[key] = {
                        "begin": begin,
                        "end": end,
                        "idx": idx,
                        "label": label,
                        "tag": item.get("tag")
                    }
            except json.JSONDecodeError:
                continue
    
    print(f"  Loaded {len(labels_by_key)} label entries from reference file")
    return labels_by_key


def load_data(input_file, reference_labels=None):
    """
    Load data and match labels by (video_id, begin, end, dataset_name, idx).
    No merging is performed; each record is returned directly.
    """
    print(f"Step 1/3: Reading data: {input_file}")
    
    results = []
    ref_label_used_count = 0

    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                vid = item.get("video_id")
                dataset = item.get("dataset_name")
                if not vid:
                    continue
                
                begin = item.get("begin")
                end = item.get("end")
                idx = item.get("idx")
                tag = item.get("tag")
                pred = item.get("pred")
                label = item.get("label")
                
                # If label is missing, try exact-key lookup from reference
                if label is None and reference_labels:
                    ref_key = (vid, begin, end, dataset, idx)
                    if ref_key in reference_labels:
                        label = reference_labels[ref_key]["label"]
                        ref_label_used_count += 1

                results.append({
                    "video_id": vid,
                    "dataset_name": dataset,
                    "begin": begin,
                    "end": end,
                    "idx": idx,
                    "tag": tag,
                    "pred": pred,
                    "label": label
                })

            except json.JSONDecodeError as e:
                print(f"  Warning: JSON parsing failed at line {line_num}")
                continue

    print(f"  Read {len(results)} records")
    if reference_labels:
        print(f"  Backfilled {ref_label_used_count} labels from reference file")
    
    return results


def extract_timestamps(label_dict):
    """Extract all timestamps from a label dictionary and return (first, last)."""
    if not isinstance(label_dict, dict):
        return None, None
    
    timestamps = []
    for key in label_dict.keys():
        try:
            timestamps.append(int(key))
        except (ValueError, TypeError):
            continue
    
    if not timestamps:
        return None, None
    
    return min(timestamps), max(timestamps)


def calculate_timediff_single(sample, k=2, alpha=0.5, enable_penalty=True):
    """
    Compute timediff for a single sample (with penalty terms).
    
    Args:
        sample: A single record containing video_id, begin, end, idx, tag, pred, label
        k: Tolerance hyperparameter (default: 2)
        alpha: Penalty factor (default: 0.5)
        enable_penalty: Whether to enable the penalty mechanism (default: True)
    
    Returns:
        A timediff result dictionary, or None if it cannot be computed
    """
    video_id = sample.get("video_id")
    tag = sample.get("tag")
    begin = sample.get("begin")
    end = sample.get("end")
    idx = sample.get("idx")
    pred = sample.get("pred")
    label = sample.get("label")
    
    if label is None:
        return None
    
    first_time, last_time = extract_timestamps(label)
    if first_time is None or last_time is None:
        return None
    
    # Compute response time
    response_time = None
    if pred is not None:
        if isinstance(pred, dict):
            timestamps = []
            for key in pred.keys():
                try:
                    t = int(key)
                    if t >= first_time:
                        timestamps.append(t)
                except (ValueError, TypeError):
                    continue
            if timestamps:
                response_time = min(timestamps)
        elif isinstance(pred, str):
            if pred.strip():
                if begin >= first_time:
                    response_time = begin
    
    # Compute raw timediff
    if response_time is None:
        base_timediff = last_time - first_time
    else:
        base_timediff = response_time - first_time
    
    # Count predicted responses outside valid range (only when penalty is enabled)
    out_of_range_count = 0
    penalty = 0
    if enable_penalty and pred is not None:
        valid_range_start = first_time - k
        valid_range_end = last_time + k
        
        if isinstance(pred, dict):
            for key in pred.keys():
                try:
                    t = int(key)
                    if pred[key] and (t < valid_range_start or t > valid_range_end):
                        out_of_range_count += 1
                except (ValueError, TypeError):
                    continue
        elif isinstance(pred, str):
            if pred.strip():
                if begin is not None and (begin < valid_range_start or begin > valid_range_end):
                    out_of_range_count += 1
        
        penalty = alpha * out_of_range_count
    
    # Compute final timediff
    timediff_final = base_timediff + penalty
    
    return {
        "video_id": video_id,
        "tag": tag,
        "begin": begin,
        "end": end,
        "idx": idx,
        "first_time": first_time,
        "last_time": last_time,
        "response_time": response_time,
        "base_timediff": base_timediff,
        "out_of_range_count": out_of_range_count,
        "penalty": penalty,
        "timediff": timediff_final
    }


def calculate_overlap_single(sample):
    """
    Compute Recall, Precision, and F1 for a single sample.
    
    Args:
        sample: A single record containing video_id, begin, end, idx, tag, pred, label
    
    Returns:
        A metric dictionary, or None if it cannot be computed
    """
    video_id = sample.get("video_id")
    tag = sample.get("tag")
    begin = sample.get("begin")
    end = sample.get("end")
    idx = sample.get("idx")
    pred = sample.get("pred")
    label = sample.get("label")
    
    # Get speaking seconds from predict and label
    pred_seconds = get_speaking_seconds_from_content(pred, begin, end)
    label_seconds = get_speaking_seconds_from_content(label, begin, end)
    
    if not label_seconds:
        # No label available, cannot compute
        return None
    
    # Compute intersection (true positives)
    tp = len(pred_seconds & label_seconds)
    
    # Compute Recall and Precision
    recall = tp / len(label_seconds) if label_seconds else 0
    precision = tp / len(pred_seconds) if pred_seconds else 0
    
    # Compute F1
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "video_id": video_id,
        "tag": tag,
        "begin": begin,
        "end": end,
        "idx": idx,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "tp": tp,
        "predict_count": len(pred_seconds),
        "label_count": len(label_seconds)
    }


def get_speaking_seconds_from_content(content, begin, end):
    """
    Extract speaking seconds from pred or label content.
    Returns a set.
    """
    speaking_seconds = set()
    
    if content is None:
        return speaking_seconds
    
    if isinstance(content, dict):
        # Dict format: key is timestamp
        for k in content.keys():
            try:
                t = int(k)
                speaking_seconds.add(t)
            except (ValueError, TypeError):
                continue
    elif isinstance(content, str) and content.strip():
        # String format: assume speaking throughout [begin, end]
        if begin is not None and end is not None:
            for t in range(begin, end + 1):
                speaking_seconds.add(t)
    
    return speaking_seconds     

def safe_div(n, d):
    return n / d if d else 0

def f1_from_pr(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0


def evaluate_metrics(data_list, k=2, alpha=0.5, enable_penalty=True):
    """
    Compute all evaluation metrics (directly iterating through each sample).
    
    Args:
        data_list: List of samples, each containing video_id, begin, end, idx, tag, pred, label
        k: Tolerance hyperparameter (default: 2)
        alpha: Penalty factor (default: 0.5)
        enable_penalty: Whether to enable the penalty mechanism (default: True)
    """
    print(f"\nStep 2/3: Computing evaluation metrics")
    if enable_penalty:
        print(f"  Penalty parameters: k={k}, alpha={alpha} (enabled)")
    else:
        print(f"  Penalty mechanism: disabled")
    
    # Compute Timediff
    print(f"  Computing Timediff...")
    all_timediffs = []
    timediffs_by_tag = defaultdict(list)
    total_samples_by_tag = defaultdict(int)
    valid_timediff_by_tag = defaultdict(int)
    
    for sample in data_list:
        tag = sample.get("tag")
        total_samples_by_tag[tag] += 1
        
        timediff_result = calculate_timediff_single(
            sample, k=k, alpha=alpha, enable_penalty=enable_penalty
        )
        if timediff_result:
            all_timediffs.append(timediff_result)
            timediffs_by_tag[tag].append(timediff_result["timediff"])
            valid_timediff_by_tag[tag] += 1
    
    print(f"    Computed {len(all_timediffs)} timediff values")
    for tag in sorted(total_samples_by_tag.keys()):
        print(f"      {tag}: {valid_timediff_by_tag[tag]}/{total_samples_by_tag[tag]} valid samples")
    
    # Compute Recall, Precision, F1 (sample-level)
    print(f"  Computing Recall, Precision, and F1 (sample-level)...")
    all_overlap_metrics = []
    # Supports both:
    # - macro: average per-sample metrics (equal weight)
    # - micro: globally aggregate TP/Count before P/R/F1 (second-level weighting)
    overlap_by_tag = defaultdict(lambda: {
        "recall_list": [],
        "precision_list": [],
        "f1_list": [],
        "tp_sum": 0,
        "predict_count_sum": 0,
        "label_count_sum": 0,
        "sample_count": 0
    })
    overall_overlap_sums = {"tp_sum": 0, "predict_count_sum": 0, "label_count_sum": 0}
    
    for sample in data_list:
        metrics = calculate_overlap_single(sample)
        if metrics:
            all_overlap_metrics.append(metrics)
            tag = metrics["tag"]
            overlap_by_tag[tag]["recall_list"].append(metrics["recall"])
            overlap_by_tag[tag]["precision_list"].append(metrics["precision"])
            overlap_by_tag[tag]["f1_list"].append(metrics["f1"])
            overlap_by_tag[tag]["tp_sum"] += metrics["tp"]
            overlap_by_tag[tag]["predict_count_sum"] += metrics["predict_count"]
            overlap_by_tag[tag]["label_count_sum"] += metrics["label_count"]
            overlap_by_tag[tag]["sample_count"] += 1

            overall_overlap_sums["tp_sum"] += metrics["tp"]
            overall_overlap_sums["predict_count_sum"] += metrics["predict_count"]
            overall_overlap_sums["label_count_sum"] += metrics["label_count"]
    
    print(f"    Computed metrics for {len(all_overlap_metrics)} samples")
    
    return all_timediffs, timediffs_by_tag, all_overlap_metrics, overlap_by_tag, overall_overlap_sums


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end evaluation: from raw JSONL to Timediff results"
    )
    parser.add_argument("input", help="Input raw JSONL file")
    parser.add_argument("--output", required=True, help="Output result JSON file")
    parser.add_argument("--reference", help="Reference JSONL file (used to fill labels when missing in input)")
    parser.add_argument("--verbose", action="store_true", help="Save detailed timediff data")
    parser.add_argument("--enable-penalty", action="store_true", default=True, help="Enable penalty mechanism (default: enabled)")
    parser.add_argument("--disable-penalty", dest="enable_penalty", action="store_false", help="Disable penalty mechanism")
    parser.add_argument("--k", type=int, default=2, help="Tolerance hyperparameter (default: 2)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Penalty factor (default: 0.5)")
    
    args = parser.parse_args()
    
    # Extract exp_name from file name (remove path and extension)
    exp_name = os.path.splitext(os.path.basename(args.input))[0]
    
    print("=" * 80)
    print(f"Experiment name: {exp_name}")
    print("=" * 80)
    
    # If a reference file is provided, load labels first
    reference_labels = None
    if args.reference:
        reference_labels = load_reference_labels(args.reference)
    
    # Step 1: Load data (no merging, process by sample)
    data_list = load_data(args.input, reference_labels)
    
    # Step 2: Compute metrics
    all_timediffs, timediffs_by_tag, all_overlap_metrics, overlap_by_tag, overall_overlap_sums = evaluate_metrics(
        data_list, k=args.k, alpha=args.alpha, enable_penalty=args.enable_penalty
    )
    
    # Step 3: Generate results
    print(f"\nStep 3/3: Generating results")
    
    # Compute Timediff statistics
    all_timediff_values = [item["timediff"] for item in all_timediffs]
    
    # Compute Overlap statistics
    all_recall = [m["recall"] for m in all_overlap_metrics]
    all_precision = [m["precision"] for m in all_overlap_metrics]
    all_f1 = [m["f1"] for m in all_overlap_metrics]

    # macro: equal-weight average across samples
    macro_recall = sum(all_recall) / len(all_recall) if all_recall else 0
    macro_precision = sum(all_precision) / len(all_precision) if all_precision else 0
    macro_f1 = sum(all_f1) / len(all_f1) if all_f1 else 0

    # micro: compute after aggregating TP/Count (closer to overall second-level performance)
    micro_recall = safe_div(overall_overlap_sums["tp_sum"], overall_overlap_sums["label_count_sum"])
    micro_precision = safe_div(overall_overlap_sums["tp_sum"], overall_overlap_sums["predict_count_sum"])
    micro_f1 = f1_from_pr(micro_precision, micro_recall)
    
    # Compute penalty statistics
    total_penalty = sum(item["penalty"] for item in all_timediffs)
    total_out_of_range = sum(item["out_of_range_count"] for item in all_timediffs)
    
    result = {
        "exp_name": exp_name,
        "penalty_params": {
            "enabled": args.enable_penalty,
            "k": args.k,
            "alpha": args.alpha
        },
        "overall": {
            "timediff": {
                "count": len(all_timediff_values),
                "avg": sum(all_timediff_values) / len(all_timediff_values) if all_timediff_values else 0,
                "min": min(all_timediff_values) if all_timediff_values else 0,
                "max": max(all_timediff_values) if all_timediff_values else 0,
                "total_penalty": total_penalty,
                "total_out_of_range_count": total_out_of_range
            },
            # Backward compatibility: keep original top-level fields as macro values
            "recall": macro_recall,
            "precision": macro_precision,
            "f1": macro_f1,
            # New: output both macro and micro to avoid ambiguity in overall semantics
            "overlap_macro": {
                "sample_count": len(all_overlap_metrics),
                "recall": macro_recall,
                "precision": macro_precision,
                "f1": macro_f1
            },
            "overlap_micro": {
                "sample_count": len(all_overlap_metrics),
                "tp": overall_overlap_sums["tp_sum"],
                "predict_count": overall_overlap_sums["predict_count_sum"],
                "label_count": overall_overlap_sums["label_count_sum"],
                "recall": micro_recall,
                "precision": micro_precision,
                "f1": micro_f1
            }
        },
        "by_tag": {},
        "details": {
            "timediff": all_timediffs if args.verbose else [],
            "overlap": all_overlap_metrics if args.verbose else []
        }
    }
    
    # Aggregate by tag
    all_tags = set(timediffs_by_tag.keys()) | set(overlap_by_tag.keys())
    for tag in sorted(all_tags):
        tag_result = {}
        
        # Timediff statistics
        if tag in timediffs_by_tag:
            values = timediffs_by_tag[tag]
            tag_result["timediff"] = {
                "count": len(values),
                "avg": sum(values) / len(values) if values else 0,
                "min": min(values) if values else 0,
                "max": max(values) if values else 0
            }
        
        # Overlap statistics
        if tag in overlap_by_tag:
            recalls = overlap_by_tag[tag]["recall_list"]
            precisions = overlap_by_tag[tag]["precision_list"]
            f1s = overlap_by_tag[tag]["f1_list"]

            tag_macro_recall = sum(recalls) / len(recalls) if recalls else 0
            tag_macro_precision = sum(precisions) / len(precisions) if precisions else 0
            tag_macro_f1 = sum(f1s) / len(f1s) if f1s else 0

            tag_micro_recall = safe_div(overlap_by_tag[tag]["tp_sum"], overlap_by_tag[tag]["label_count_sum"])
            tag_micro_precision = safe_div(overlap_by_tag[tag]["tp_sum"], overlap_by_tag[tag]["predict_count_sum"])
            tag_micro_f1 = f1_from_pr(tag_micro_precision, tag_micro_recall)

            # Backward compatibility: still expose macro in top-level fields
            tag_result["recall"] = tag_macro_recall
            tag_result["precision"] = tag_macro_precision
            tag_result["f1"] = tag_macro_f1

            tag_result["overlap_macro"] = {
                "sample_count": overlap_by_tag[tag]["sample_count"],
                "recall": tag_macro_recall,
                "precision": tag_macro_precision,
                "f1": tag_macro_f1
            }
            tag_result["overlap_micro"] = {
                "sample_count": overlap_by_tag[tag]["sample_count"],
                "tp": overlap_by_tag[tag]["tp_sum"],
                "predict_count": overlap_by_tag[tag]["predict_count_sum"],
                "label_count": overlap_by_tag[tag]["label_count_sum"],
                "recall": tag_micro_recall,
                "precision": tag_micro_precision,
                "f1": tag_micro_f1
            }
        
        result["by_tag"][tag] = tag_result
    
    # Ensure output directory exists, then save results
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"  Results saved to: {args.output}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"Experiment name: {exp_name}\n")
    
    # Print by tag
    for tag in sorted(all_tags):
        print(f"{tag}:")
        
        if tag in timediffs_by_tag:
            values = timediffs_by_tag[tag]
            avg_timediff = sum(values) / len(values) if values else 0
            print(f"  Timediff: sample_count={len(values)}, avg={avg_timediff:.2f}s, "
                f"min={min(values):.2f}s, max={max(values):.2f}s")
        
        if tag in overlap_by_tag:
            recalls = overlap_by_tag[tag]["recall_list"]
            precisions = overlap_by_tag[tag]["precision_list"]
            f1s = overlap_by_tag[tag]["f1_list"]
            avg_recall = sum(recalls) / len(recalls) if recalls else 0
            avg_precision = sum(precisions) / len(precisions) if precisions else 0
            avg_f1 = sum(f1s) / len(f1s) if f1s else 0

            tag_micro_recall = safe_div(overlap_by_tag[tag]["tp_sum"], overlap_by_tag[tag]["label_count_sum"])
            tag_micro_precision = safe_div(overlap_by_tag[tag]["tp_sum"], overlap_by_tag[tag]["predict_count_sum"])
            tag_micro_f1 = f1_from_pr(tag_micro_precision, tag_micro_recall)

            print(f"  Overlap (macro, equal weight per sample):")
            print(f"    Recall: {avg_recall:.4f}")
            print(f"    Precision: {avg_precision:.4f}")
            print(f"    F1: {avg_f1:.4f}")
            print(f"  Overlap (micro, second-level aggregation):")
            print(f"    Recall: {tag_micro_recall:.4f}")
            print(f"    Precision: {tag_micro_precision:.4f}")
            print(f"    F1: {tag_micro_f1:.4f}")
        
        print()
    
    # Overall statistics
    print("=" * 80)
    print(f"Overall:")
    print(f"  Timediff: sample_count={len(all_timediff_values)}, "
        f"avg={result['overall']['timediff']['avg']:.2f}s")
    print(f"  Overlap (macro, equal weight per sample), sample_count={result['overall']['overlap_macro']['sample_count']}:")
    print(f"    Recall: {result['overall']['overlap_macro']['recall']:.4f}")
    print(f"    Precision: {result['overall']['overlap_macro']['precision']:.4f}")
    print(f"    F1: {result['overall']['overlap_macro']['f1']:.4f}")
    print(f"  Overlap (micro, second-level aggregation):")
    print(f"    Recall: {result['overall']['overlap_micro']['recall']:.4f}")
    print(f"    Precision: {result['overall']['overlap_micro']['precision']:.4f}")
    print(f"    F1: {result['overall']['overlap_micro']['f1']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()

