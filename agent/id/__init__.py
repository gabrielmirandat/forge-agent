"""Identifier system - OpenCode-compatible ID generation.

Implements the exact same ID system as OpenCode:
- Prefixed IDs (ses_, msg_, per_, que_, usr_, prt_, pty_, tool_)
- Monotonic generation (sortable by creation time)
- Format: {prefix}_{timestamp_hex}{random_base62}
- Algorithm matches OpenCode exactly (BigInt operations)
"""

import secrets
import time
from typing import Literal, Optional

# ID prefixes matching OpenCode exactly
PREFIXES = {
    "session": "ses",
    "message": "msg",
    "permission": "per",
    "question": "que",
    "user": "usr",
    "part": "prt",
    "pty": "pty",
    "tool": "tool",
}

# Length of random part (after timestamp hex)
LENGTH = 26

# State for monotonic ID generation (module-level)
_last_timestamp: int = 0
_counter: int = 0


def _random_base62(length: int) -> str:
    """Generate random base62 string.
    
    Args:
        length: Length of random string
        
    Returns:
        Random base62 string
    """
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    return "".join(secrets.choice(chars) for _ in range(length))


def create(
    prefix: Literal["session", "message", "permission", "question", "user", "part", "pty", "tool"],
    descending: bool = False,
    timestamp: Optional[float] = None,
) -> str:
    """Create a new identifier (matches OpenCode algorithm exactly).
    
    OpenCode algorithm:
    - now = BigInt(currentTimestamp) * BigInt(0x1000) + BigInt(counter)
    - now = descending ? ~now : now
    - Extract 6 bytes (48 bits) as hex: (now >> (40 - 8*i)) & 0xff for i in [0..5]
    - Add random base62 suffix
    
    Args:
        prefix: ID prefix (session, message, etc.)
        descending: If True, newer IDs have lower values (for descending sort)
        timestamp: Optional timestamp in milliseconds (defaults to current time)
        
    Returns:
        Identifier string (e.g., "ses_1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p")
    """
    global _last_timestamp, _counter
    
    prefix_str = PREFIXES[prefix]
    current_timestamp = int(timestamp or (time.time() * 1000))  # Milliseconds
    
    # Monotonic counter (matches OpenCode)
    if current_timestamp != _last_timestamp:
        _last_timestamp = current_timestamp
        _counter = 0
    _counter += 1
    
    # Encode: BigInt(currentTimestamp) * BigInt(0x1000) + BigInt(counter)
    # Python int can handle this (effectively unlimited precision)
    now = (current_timestamp * 0x1000) + _counter
    
    # For descending, invert (matches OpenCode: descending ? ~now : now)
    if descending:
        # Use 64-bit inversion (matches JavaScript ~ operator on BigInt)
        # Limit to 64 bits for proper inversion
        now = (~now) & 0xFFFFFFFFFFFFFFFF
    
    # Convert to 6 bytes (48 bits) as hex (12 hex chars)
    # OpenCode: (now >> BigInt(40 - 8 * i)) & BigInt(0xff)
    # This extracts bits 0-47 (lower 48 bits) from the 64-bit number
    time_bytes = bytearray(6)
    for i in range(6):
        # Extract byte: (now >> (40 - 8*i)) & 0xff
        shift = 40 - (8 * i)
        # Ensure we're working with unsigned 64-bit value
        now_unsigned = now & 0xFFFFFFFFFFFFFFFF
        byte_val = (now_unsigned >> shift) & 0xFF
        time_bytes[i] = byte_val
    
    timestamp_hex = time_bytes.hex()
    
    # Add random base62 suffix (LENGTH - 12 chars)
    random_suffix = _random_base62(LENGTH - 12)
    
    return f"{prefix_str}_{timestamp_hex}{random_suffix}"


def ascending(
    prefix: Literal["session", "message", "permission", "question", "user", "part", "pty", "tool"],
    given: Optional[str] = None,
) -> str:
    """Create an ascending identifier (newer = higher value).
    
    Args:
        prefix: ID prefix
        given: Optional existing ID to validate
        
    Returns:
        Identifier string
    """
    if given:
        prefix_str = PREFIXES[prefix]
        if not given.startswith(prefix_str + "_"):
            raise ValueError(f"ID {given} does not start with {prefix_str}_")
        return given
    return create(prefix, descending=False)


def descending(
    prefix: Literal["session", "message", "permission", "question", "user", "part", "pty", "tool"],
    given: Optional[str] = None,
) -> str:
    """Create a descending identifier (newer = lower value).
    
    Args:
        prefix: ID prefix
        given: Optional existing ID to validate
        
    Returns:
        Identifier string
    """
    if given:
        prefix_str = PREFIXES[prefix]
        if not given.startswith(prefix_str + "_"):
            raise ValueError(f"ID {given} does not start with {prefix_str}_")
        return given
    return create(prefix, descending=True)


def validate(
    prefix: Literal["session", "message", "permission", "question", "user", "part", "pty", "tool"],
    id_value: str,
) -> bool:
    """Validate that an ID has the correct prefix.
    
    Args:
        prefix: Expected prefix
        id_value: ID to validate
        
    Returns:
        True if valid, False otherwise
    """
    prefix_str = PREFIXES[prefix]
    return id_value.startswith(prefix_str + "_")


def timestamp(id_value: str) -> Optional[int]:
    """Extract timestamp from an ascending ID.
    
    Note: Only works with ascending IDs, not descending.
    Matches OpenCode algorithm: encoded / BigInt(0x1000)
    
    Note: Due to 48-bit encoding, the extracted timestamp may be slightly
    different from the original (lower bits preserved, upper bits may be lost).
    This is fine for sorting purposes.
    
    Args:
        id_value: ID string
        
    Returns:
        Timestamp in milliseconds (approximate), or None if invalid
    """
    try:
        # Format: prefix_timestamp_hex_random
        parts = id_value.split("_", 1)
        if len(parts) != 2:
            return None
        
        # Extract hex part (first 12 chars = 6 bytes)
        hex_part = parts[1][:12]
        # Convert hex to int (matches OpenCode: BigInt("0x" + hex))
        encoded = int(hex_part, 16)
        
        # Extract timestamp: encoded / 0x1000 (matches OpenCode)
        # OpenCode: Number(encoded / BigInt(0x1000))
        timestamp_ms = encoded // 0x1000
        return timestamp_ms
    except (ValueError, IndexError):
        return None
