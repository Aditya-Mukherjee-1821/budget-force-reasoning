"""Validation for first-pass teacher topic labels."""
from .balance_and_rank import DROPPED_TOPICS, TOPICS


def valid_topic_reply(reply: dict) -> bool:
    confidence = reply.get("topic_confidence")
    return (isinstance(reply.get("id"), str) and reply.get("topic") in set(TOPICS) | DROPPED_TOPICS
            and isinstance(confidence, (int, float)) and 0 <= confidence <= 1)
