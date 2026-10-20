"""Quality checks are applied by build_stage_1; retained as an extension point."""


def is_nonempty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())
