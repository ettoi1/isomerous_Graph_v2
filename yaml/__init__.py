"""Minimal YAML loader supporting the subset used in configs."""
from __future__ import annotations

from typing import Any, Dict, Tuple


def safe_load(stream: Any) -> Dict[str, Any]:
    """Parse a YAML document into a Python dictionary."""

    if hasattr(stream, "read"):
        text = stream.read()
    else:
        text = str(stream)
    lines = [line.rstrip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    data, _ = _parse_block(lines, 0, 0)
    return data


def _parse_block(lines: list[str], indent: int, index: int) -> Tuple[Dict[str, Any], int]:
    mapping: Dict[str, Any] = {}
    while index < len(lines):
        line = lines[index]
        stripped = line.lstrip()
        current_indent = len(line) - len(stripped)
        if current_indent < indent:
            break
        if ":" not in stripped:
            index += 1
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not value:
            child, index = _parse_block(lines, current_indent + 2, index + 1)
            mapping[key] = child
        else:
            mapping[key] = _convert_value(value)
            index += 1
    return mapping, index


def _convert_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value or "e" in lowered:
            return float(value)
        return int(value)
    except ValueError:
        return value


def safe_dump(data: Dict[str, Any]) -> str:
    raise NotImplementedError("Dumping YAML is not supported in this minimal implementation.")
