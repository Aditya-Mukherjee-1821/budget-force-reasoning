"""Download, validate, sample GSM8K, and create topic-annotation JSONL.

Run from the project root:
  python data/build_stage_1.py
Requires: datasets (and its normal Hugging Face download dependencies).
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.filters.common import canonical_number, normalize_text, write_json, write_jsonl


def parse_record(row: dict, split: str, index: int) -> tuple[dict | None, str | None]:
    question, answer = str(row.get("question", "")).strip(), str(row.get("answer", ""))
    if not question or "####" not in answer:
        return None, "missing_question_or_final_delimiter"
    reasoning, raw_answer = answer.rsplit("####", 1)
    final_answer = canonical_number(raw_answer)
    if not final_answer:
        return None, "unparseable_final_answer"
    if any(token in question.lower() for token in ("traceback", "api error", "<html")):
        return None, "error_text"
    return {
        "id": f"gsm8k_{split}_{index:05d}", "source": "openai/gsm8k:main", "source_split": split,
        "question": question, "source_reasoning": reasoning.strip(), "source_answer": final_answer,
        "source_answer_raw": answer, "quality_passed": True,
        "length_features": {"question_tokens": len(question.split()), "reasoning_tokens": len(reasoning.split())},
        "base_prediction": None, "base_extracted_answer": None, "base_correct": None,
    }, None


def filter_split(rows, split: str, target: int) -> tuple[list[dict], Counter]:
    rejected: Counter = Counter()
    clean: list[dict] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        record, reason = parse_record(row, split, index)
        if reason:
            rejected[reason] += 1; continue
        key = normalize_text(record["question"])
        if key in seen:
            rejected["duplicate_within_split"] += 1; continue
        seen.add(key); clean.append(record)
    if target > len(clean):
        raise ValueError(f"Requested {target} {split} records but only {len(clean)} passed basic validation.")
    # Deterministic evenly spaced sampling spans the full source order; it does
    # not accidentally take only the first portion when target is large.
    indices = [(index * len(clean)) // target for index in range(target)]
    return [clean[index] for index in indices], rejected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--teacher-model", default=os.environ.get("TEACHER_MODEL", "GPT-5.6 Terra Medium"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    raw_dir = ROOT / "data" / "raw"
    # Prefer the project-local Parquet files. This makes the command work offline
    # after a one-time manual download and leaves those raw files untouched.
    local_files = {
        "train": str(raw_dir / "train-00000-of-00001.parquet"),
        "test": str(raw_dir / "test-00000-of-00001.parquet"),
    }
    if all(Path(path).is_file() for path in local_files.values()):
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise SystemExit("Install pyarrow first: pip install pyarrow") from exc
        dataset = {split: pq.read_table(path).to_pylist() for split, path in local_files.items()}
        print("Using local GSM8K Parquet files from data/raw/.")
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise SystemExit("Install datasets first: pip install datasets") from exc
        dataset = load_dataset("openai/gsm8k", "main", cache_dir=str(raw_dir))
        print("Downloaded/loaded openai/gsm8k:main into data/raw/ cache.")
    train, train_rejected = filter_split(dataset["train"], "train", args.train_size)
    test, test_rejected = filter_split(dataset["test"], "test", args.test_size)
    train_questions = {normalize_text(row["question"]) for row in train}
    test = [row for row in test if normalize_text(row["question"]) not in train_questions]
    if len(test) < args.test_size:
        raise RuntimeError("Cross-split duplicates left too few test candidates; increase --test-size source pool logic.")
    stage = ROOT / "data" / "stage_1_split"
    write_json(stage / "train.json", train, overwrite=args.overwrite)
    write_json(stage / "test.json", test, overwrite=args.overwrite)
    payloads = [{"custom_id": row["id"], "method": "POST", "url": "/v1/responses",
                 "body": {"model": args.teacher_model, "input": [{"role": "system", "content": (ROOT / "data/prompts/stage_1_topics_system.md").read_text(encoding="utf-8")}, {"role": "user", "content": row["question"]}]}}
                for row in train + test]
    write_jsonl(stage / "gpt_topics_input.jsonl", payloads, overwrite=args.overwrite)
    write_json(ROOT / "evaluation/logs/stage_1_audit.json", {"train_selected": len(train), "test_selected": len(test), "train_rejections": train_rejected, "test_rejections": test_rejected}, overwrite=args.overwrite)
    print(f"Wrote {len(train)} train and {len(test)} test candidates plus {len(payloads)} topic payloads.")


if __name__ == "__main__":
    main()
