"""Simple, reproducible whitespace-token length feature extraction."""


def token_length(text: str) -> int:
    return len(text.split())
