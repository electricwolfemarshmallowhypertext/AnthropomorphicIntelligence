#!/usr/bin/env python3
"""
Compute overlap metrics only (Recall / Precision / F1) from raw JSONL.

Workflow:
1. Read data (optional label backfill from reference JSONL)
2. Compute overlap metrics and aggregate by tag / overall
"""

import json
import argparse
import os
from collections import defaultdict


def load_reference_labels(reference_file):
    """Load label data from reference file, indexed by (video_id, begin, end, dataset_name, idx)."""
    print(f"  Loading labels from reference file: {reference_file}")

    labels_by_key = {}
    with open(reference_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            vid = item.get("video_id")
            dataset = item.get("dataset_name")
            begin = item.get("begin")
            end = item.get("end")
            idx = item.get("idx")
            if not vid:
                continue

            label = item.get("label")
            if label is not None:
                labels_by_key[(vid, begin, end, dataset, idx)] = label

    print(f"  Loaded {len(labels_by_key)} label entries from reference file")
    return labels_by_key


def load_data(input_file, reference_labels=None):
    """Load data and fill missing labels by exact key when reference is provided."""
    print(f"Step 1/2: Reading data: {input_file}")

    results = []
    ref_label_used_count = 0

    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"  Warning: JSON parsing failed at line {line_num}")
                continue

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

            if label is None and reference_labels:
                ref_key = (vid, begin, end, dataset, idx)
                if ref_key in reference_labels:
                    label = reference_labels[ref_key]
                    ref_label_used_count += 1

            results.append(
                {
                    "video_id": vid,
                    "dataset_name": dataset,
                    "begin": begin,
                    "end": end,
                    "idx": idx,
                    "tag": tag,
                    "pred": pred,
                    "label": label,
                }
            )

    print(f"  Read {len(results)} records")
    if reference_labels:
        print(f"  Backfilled {ref_label_used_count} labels from reference file")

    return results


def get_speaking_seconds_from_content(content, begin, end):
    """Extract speaking seconds from pred or label content as a set[int]."""
    speaking_seconds = set()

    if content is None:
        return speaking_seconds

    if isinstance(content, dict):
        for key in content.keys():
            try:
                speaking_seconds.add(int(key))
            except (TypeError, ValueError):
                continue
    elif isinstance(content, str) and content.strip():
        if begin is not None and end is not None:
            for second in range(begin, end + 1):
                speaking_seconds.add(second)

    return speaking_seconds


def calculate_overlap_single(sample):
    """Compute Recall/Precision/F1 for one sample."""
    pred = sample.get("pred")
    label = sample.get("label")

    pred_seconds = get_speaking_seconds_from_content(pred, sample.get("begin"), sample.get("end"))
    label_seconds = get_speaking_seconds_from_content(label, sample.get("begin"), sample.get("end"))

    if not label_seconds:
        return None

    tp = len(pred_seconds & label_seconds)
    recall = tp / len(label_seconds) if label_seconds else 0
    precision = tp / len(pred_seconds) if pred_seconds else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "video_id": sample.get("video_id"),
        "tag": sample.get("tag"),
        "begin": sample.get("begin"),
        "end": sample.get("end"),
        "idx": sample.get("idx"),
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "tp": tp,
        "predict_count": len(pred_seconds),
        "label_count": len(label_seconds),
    }


def safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0


def f1_from_pr(precision, recall):
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0


def evaluate_f1(data_list):
    """Compute overlap metrics for all samples and aggregate supporting stats."""
    print("\nStep 2/2: Computing overlap metrics (Recall/Precision/F1)")

    all_metrics = []
    by_tag = defaultdict(
        lambda: {
            "recall_list": [],
            "precision_list": [],
            "f1_list": [],
            "tp_sum": 0,
            "predict_count_sum": 0,
            "label_count_sum": 0,
            "sample_count": 0,
        }
    )

    overall = {
        "tp_sum": 0,
        "predict_count_sum": 0,
        "label_count_sum": 0,
    }

    for sample in data_list:
        metric = calculate_overlap_single(sample)
        if not metric:
            continue

        all_metrics.append(metric)
        tag = metric["tag"]

        by_tag[tag]["recall_list"].append(metric["recall"])
        by_tag[tag]["precision_list"].append(metric["precision"])
        by_tag[tag]["f1_list"].append(metric["f1"])
        by_tag[tag]["tp_sum"] += metric["tp"]
        by_tag[tag]["predict_count_sum"] += metric["predict_count"]
        by_tag[tag]["label_count_sum"] += metric["label_count"]
        by_tag[tag]["sample_count"] += 1

        overall["tp_sum"] += metric["tp"]
        overall["predict_count_sum"] += metric["predict_count"]
        overall["label_count_sum"] += metric["label_count"]

    print(f"  Computed metrics for {len(all_metrics)} samples")
    return all_metrics, by_tag, overall


def main():
    parser = argparse.ArgumentParser(description="Compute overlap metrics only (Recall/Precision/F1)")
    parser.add_argument("input", help="Input raw JSONL file")
    parser.add_argument("--output", required=True, help="Output result JSON file")
    parser.add_argument("--reference", help="Reference JSONL file for label backfill")
    parser.add_argument("--verbose", action="store_true", help="Save per-sample overlap details")
    args = parser.parse_args()

    exp_name = os.path.splitext(os.path.basename(args.input))[0]

    print("=" * 80)
    print(f"Experiment name: {exp_name}")
    print("=" * 80)

    reference_labels = load_reference_labels(args.reference) if args.reference else None
    data_list = load_data(args.input, reference_labels)

    all_metrics, by_tag, overall = evaluate_f1(data_list)

    all_recall = [m["recall"] for m in all_metrics]
    all_precision = [m["precision"] for m in all_metrics]
    all_f1 = [m["f1"] for m in all_metrics]

    macro_recall = sum(all_recall) / len(all_recall) if all_recall else 0
    macro_precision = sum(all_precision) / len(all_precision) if all_precision else 0
    macro_f1 = sum(all_f1) / len(all_f1) if all_f1 else 0

    micro_recall = safe_div(overall["tp_sum"], overall["label_count_sum"])
    micro_precision = safe_div(overall["tp_sum"], overall["predict_count_sum"])
    micro_f1 = f1_from_pr(micro_precision, micro_recall)

    result = {
        "exp_name": exp_name,
        "overall": {
            "recall": macro_recall,
            "precision": macro_precision,
            "f1": macro_f1,
            "overlap_macro": {
                "sample_count": len(all_metrics),
                "recall": macro_recall,
                "precision": macro_precision,
                "f1": macro_f1,
            },
            "overlap_micro": {
                "sample_count": len(all_metrics),
                "tp": overall["tp_sum"],
                "predict_count": overall["predict_count_sum"],
                "label_count": overall["label_count_sum"],
                "recall": micro_recall,
                "precision": micro_precision,
                "f1": micro_f1,
            },
        },
        "by_tag": {},
        "details": {"overlap": all_metrics if args.verbose else []},
    }

    for tag in sorted(by_tag.keys()):
        recalls = by_tag[tag]["recall_list"]
        precisions = by_tag[tag]["precision_list"]
        f1s = by_tag[tag]["f1_list"]

        tag_macro_recall = sum(recalls) / len(recalls) if recalls else 0
        tag_macro_precision = sum(precisions) / len(precisions) if precisions else 0
        tag_macro_f1 = sum(f1s) / len(f1s) if f1s else 0

        tag_micro_recall = safe_div(by_tag[tag]["tp_sum"], by_tag[tag]["label_count_sum"])
        tag_micro_precision = safe_div(by_tag[tag]["tp_sum"], by_tag[tag]["predict_count_sum"])
        tag_micro_f1 = f1_from_pr(tag_micro_precision, tag_micro_recall)

        result["by_tag"][tag] = {
            "recall": tag_macro_recall,
            "precision": tag_macro_precision,
            "f1": tag_macro_f1,
            "overlap_macro": {
                "sample_count": by_tag[tag]["sample_count"],
                "recall": tag_macro_recall,
                "precision": tag_macro_precision,
                "f1": tag_macro_f1,
            },
            "overlap_micro": {
                "sample_count": by_tag[tag]["sample_count"],
                "tp": by_tag[tag]["tp_sum"],
                "predict_count": by_tag[tag]["predict_count_sum"],
                "label_count": by_tag[tag]["label_count_sum"],
                "recall": tag_micro_recall,
                "precision": tag_micro_precision,
                "f1": tag_micro_f1,
            },
        }

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
