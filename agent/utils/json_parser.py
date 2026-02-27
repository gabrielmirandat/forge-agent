"""Utility for extracting JSON tool calls from LLM text responses.

Some models return tool calls as JSON text instead of structured tool_calls.
This module provides a shared utility to parse them.
"""

import json
import re
from typing import Any, Dict, Optional


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON tool call from text, handling markdown code blocks.

    Args:
        text: Text that may contain JSON in code blocks or plain text.

    Returns:
        Parsed JSON dict with "name" and "arguments" keys, or None.
    """
    # Try markdown code blocks first (```json ... ``` or ``` ... ```)
    for match in re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL):
        try:
            parsed = json.loads(match)
            if "name" in parsed and "arguments" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue

    # Fall back to inline JSON object with tool call pattern
    for match in re.findall(
        r'\{[^{}]*"name"[^{}]*"arguments"[^{}]*\{[^{}]*\}[^{}]*\}', text, re.DOTALL
    ):
        try:
            parsed = json.loads(match)
            if "name" in parsed and "arguments" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue

    return None
