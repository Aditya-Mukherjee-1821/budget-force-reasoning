"""Topic-balanced, deterministic candidate selection."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

TOPICS = (
    "addition_subtraction_counting", "multiplication_division", "fractions",
    "ratios_and_proportions", "percentages_discount_profit", "rates_time_distance_work",
    "money_and_unit_conversion", "averages_and_mixtures", "geometry_measurement",
)

# Historical Stage 1 responses may contain this label.  It is accepted while
# loading those responses but is deliberately excluded from Stage 2 selection.
DROPPED_TOPICS = {"simple_algebra_and_counting"}


def _score(record: dict[str, Any], train: bool) -> tuple[float, str]:
    features = record.get("length_features", {})
    length = float(features.get("reasoning_tokens", 0))
    moderate_length = max(0.0, 1.0 - abs(length - 100.0) / 100.0)
    confidence = float(record.get("topic_confidence", 0.0))
    correctness_bonus = 1.0 if train and record.get("base_correct") is False else 0.0
    return (2.0 * float(bool(record.get("quality_passed"))) + correctness_bonus + moderate_length + confidence, record["id"])


def select_balanced(records: list[dict[str, Any]], *, per_topic: int, train: bool, allow_short_topics: bool = False) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("topic") in TOPICS:
            grouped[record["topic"]].append(record)
    shortages = {topic: per_topic - len(grouped[topic]) for topic in TOPICS if len(grouped[topic]) < per_topic}
    if shortages and not allow_short_topics:
        raise ValueError(f"Cannot make a balanced split; topic shortages: {shortages}")
    selected: list[dict[str, Any]] = []
    for topic in TOPICS:
        ranked = sorted(grouped[topic], key=lambda row: _score(row, train), reverse=True)
        for score, record in [(_score(row, train), row) for row in ranked[:per_topic]]:
            record = dict(record)
            record["selection_score"] = score[0]
            record["selection_reasons"] = ["quality_passed", "topic_capped", "deterministic_rank"]
            if train and record.get("base_correct") is False:
                record["selection_reasons"].append("base_model_incorrect")
            selected.append(record)
    return selected


def select_top(records: list[dict[str, Any]], *, count: int, train: bool, topic_floor: int = 0) -> list[dict[str, Any]]:
    """Select a fixed total, reserving a small per-topic floor before global ranking."""
    if len(records) < count:
        raise ValueError(f"Need {count} records but only {len(records)} are available.")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["topic"]].append(record)
    ranked_by_topic = {topic: sorted(rows, key=lambda row: _score(row, train), reverse=True) for topic, rows in grouped.items()}
    reserved = [row for topic in TOPICS for row in ranked_by_topic.get(topic, [])[:topic_floor]]
    if len(reserved) > count:
        raise ValueError(f"Topic floor selects {len(reserved)} records, exceeding requested total {count}.")
    reserved_ids = {row["id"] for row in reserved}
    globally_ranked = sorted((row for row in records if row["id"] not in reserved_ids), key=lambda row: _score(row, train), reverse=True)
    chosen = reserved + globally_ranked[:count - len(reserved)]
    selected: list[dict[str, Any]] = []
    for record in chosen:
        record = dict(record)
        record["selection_score"] = _score(record, train)[0]
        record["selection_reasons"] = ["quality_passed", "global_deterministic_rank"]
        if record["id"] in reserved_ids:
            record["selection_reasons"].append("topic_floor")
        if train and record.get("base_correct") is False:
            record["selection_reasons"].append("base_model_incorrect")
        selected.append(record)
    return selected
