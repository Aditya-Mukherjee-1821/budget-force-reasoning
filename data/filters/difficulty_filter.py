"""Base-model difficulty fields are populated in the model-inference phase."""


def is_preferred_training_example(record: dict) -> bool:
    return record.get("base_correct") is False
