#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from collections import defaultdict

TARGET_TAGS = {"Solo commentators", "Multiple commentators", "Guidance"}


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] line {ln} parse failed: {e}")
    return rows


def init_stat(fields):
    d = {"cnt": 0}
    for k in fields:
        d[f"sum_{k}"] = 0.0
    return d


def aggregate(rows, score_key: str, fields: list[str]):
    stats_by_tag = defaultdict(lambda: init_stat(fields))
    stats_merged = init_stat(fields)

    for r in rows:
        tag = r.get("tag", "UNKNOWN")
        score = r.get(score_key) or {}

        vals = []
        ok = True
        for k in fields:
            v = safe_float(score.get(k))
            if v is None:
                ok = False
                break
            vals.append(v)

        # Count only rows where all fields exist (avoid skew from dirty data)
        if not ok:
            continue

        # per tag
        s = stats_by_tag[tag]
        s["cnt"] += 1
        for k, v in zip(fields, vals):
            s[f"sum_{k}"] += v

        # merged for target tags
        if tag in TARGET_TAGS:
            stats_merged["cnt"] += 1
            for k, v in zip(fields, vals):
                stats_merged[f"sum_{k}"] += v

    return stats_by_tag, stats_merged


def mean_row(stat, fields):
    cnt = stat["cnt"]
    if cnt <= 0:
        return None

    means = [stat[f"sum_{k}"] / cnt for k in fields]
    overall = sum(means) / len(means)
    return cnt, means, overall


def print_table(stats_by_tag, stats_merged, fields, score_key):
    print(f"\n=== Mean stats for {score_key} by tag ===")
    col_w = 24
    header = f"{'Tag':<{col_w}} {'N':>6} " + " ".join([f"{k:>12}" for k in fields]) + f" {'Overall':>12}"
    print(header)
    print("-" * len(header))

    # sort by count desc
    for tag, stat in sorted(stats_by_tag.items(), key=lambda x: x[1]["cnt"], reverse=True):
        row = mean_row(stat, fields)
        if row is None:
            continue
        cnt, means, overall = row
        line = f"{tag:<{col_w}} {cnt:>6} " + " ".join([f"{m:>12.4f}" for m in means]) + f" {overall:>12.4f}"
        print(line)

    print("\n=== Merged mean for tags: Solo commentators / Multiple commentators / Guidance ===")
    row = mean_row(stats_merged, fields)
    if row is None:
        print("No valid samples found for the three target tags.")
        return
    cnt, means, overall = row
    kv = ", ".join([f"{k}={m:.4f}" for k, m in zip(fields, means)])
    print(f"N={cnt}, {kv}, Overall={overall:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Path to jsonl file (one JSON per line)")
    parser.add_argument("--score_key", required=True, help="Score field name, e.g. liveu_score / finalq_score")
    parser.add_argument("--fields", nargs="+", required=True, help="Fields under score_key, e.g. Time Rate TextU")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    stats_by_tag, stats_merged = aggregate(rows, args.score_key, args.fields)
    print_table(stats_by_tag, stats_merged, args.fields, args.score_key)


if __name__ == "__main__":
    main()

# python summary_llm_score.py  --score_key liveu_score --fields Time Rate TextU -i data.jsonl

