"""Validate topic labels, balance candidates, and create reasoning JSONL.

Before this command, merge validated teacher topic fields (id, topic,
topic_confidence) into data/stage_1_split/{train,test}.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.filters.balance_and_rank import DROPPED_TOPICS, TOPICS, select_balanced
from data.filters.common import canonical_number, write_json, write_jsonl


def load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_topic_replies(path: Path) -> dict[str, dict]:
    """Read the already-validated, pasted Stage 1 topic replies by ID."""
    replies = load(path)
    by_id = {reply["id"]: reply for reply in replies}
    if len(by_id) != len(replies):
        raise ValueError(f"Duplicate topic-response IDs in {path}")
    return by_id


def attach_topics(records: list[dict], response_path: Path) -> list[dict]:
    replies = load_topic_replies(response_path)
    ids = {record["id"] for record in records}
    if set(replies) != ids:
        raise ValueError(f"Topic response IDs in {response_path} do not exactly match the Stage 1 records.")
    return [{**record, "topic": replies[record["id"]]["topic"], "topic_confidence": replies[record["id"]]["topic_confidence"]} for record in records]


def validate(records: list[dict], split: str) -> None:
    ids = [row.get("id") for row in records]
    allowed_topics = set(TOPICS) | DROPPED_TOPICS
    if len(ids) != len(set(ids)) or any(row.get("topic") not in allowed_topics for row in records):
        raise ValueError(f"{split} needs unique IDs and one valid topic per record.")
    if any(not isinstance(row.get("topic_confidence"), (int, float)) or not 0 <= row["topic_confidence"] <= 1 for row in records):
        raise ValueError(f"{split} has invalid topic_confidence values.")
    for row in records:
        raw_answer = str(row.get("source_answer_raw", ""))
        if "####" not in raw_answer:
            raise ValueError(f"{split} record {row.get('id')} has no raw final-answer delimiter.")
        expected = canonical_number(raw_answer.rsplit("####", 1)[1])
        if expected != row.get("source_answer"):
            raise ValueError(
                f"{split} record {row.get('id')} has corrupted source_answer: "
                f"stored={row.get('source_answer')!r}, raw={expected!r}. Rebuild Stage 1."
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--teacher-model", default=os.environ.get("TEACHER_MODEL", "GPT-5.6 Terra Medium"))
    parser.add_argument("--train-per-topic", type=int, default=50)
    parser.add_argument("--test-per-topic", type=int, default=10)
    args = parser.parse_args()
    stage1 = ROOT / "data/stage_1_split"
    train = attach_topics(load(stage1 / "train.json"), stage1 / "gpt_response_train.json")
    test = attach_topics(load(stage1 / "test.json"), stage1 / "gpt_response_test.json")
    validate(train, "train"); validate(test, "test")
    dropped_train = [row for row in train if row["topic"] in DROPPED_TOPICS]
    dropped_test = [row for row in test if row["topic"] in DROPPED_TOPICS]
    train = [row for row in train if row["topic"] in TOPICS]
    test = [row for row in test if row["topic"] in TOPICS]
    if {row["id"] for row in train} & {row["id"] for row in test}:
        raise ValueError("Train/test ID overlap detected.")
    selected_train = select_balanced(train, per_topic=args.train_per_topic, train=True)
    selected_test = select_balanced(test, per_topic=args.test_per_topic, train=False)
    stage2 = ROOT / "data/stage_2_split"
    write_json(stage2 / "train.json", selected_train, overwrite=args.overwrite)
    write_json(stage2 / "test.json", selected_test, overwrite=args.overwrite)
    prompt = (ROOT / "data/prompts/stage_2_reasoning_system.md").read_text(encoding="utf-8")
    payloads = [{"custom_id": row["id"], "method": "POST", "url": "/v1/responses", "body": {"model": args.teacher_model, "input": [{"role": "system", "content": prompt}, {"role": "user", "content": json.dumps({"id": row["id"], "question": row["question"], "verified_answer": row["source_answer"], "topic": row["topic"]}, ensure_ascii=False)}]}} for row in selected_train + selected_test]
    write_jsonl(stage2 / "gpt_reasoning_input.jsonl", payloads, overwrite=args.overwrite)
    print(f"Dropped excluded-topic records: {len(dropped_train)} train and {len(dropped_test)} test.")
    print(f"Wrote {len(selected_train)} train and {len(selected_test)} test selections plus reasoning payloads.")


if __name__ == "__main__":
    main()
