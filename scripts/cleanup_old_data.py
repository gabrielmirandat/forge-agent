"""Script to clean up old data fields from JSON storage files.

Removes deprecated fields:
- plan_result
- execution_result
- pending_approval_steps
- pty_session (if not needed)

This script modifies JSON files in place. Make a backup first!
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict


def clean_message(msg_data: Dict[str, Any]) -> Dict[str, Any]:
    """Clean a single message by removing deprecated fields.
    
    Args:
        msg_data: Message data dictionary
        
    Returns:
        Cleaned message data
    """
    cleaned = {
        "message_id": msg_data["message_id"],
        "session_id": msg_data.get("session_id", ""),
        "role": msg_data["role"],
        "content": msg_data["content"],
        "created_at": msg_data.get("created_at", 0),
    }
    return cleaned


def clean_session_file(file_path: Path) -> bool:
    """Clean a session JSON file.
    
    Args:
        file_path: Path to session JSON file
        
    Returns:
        True if file was modified, False otherwise
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        original_data = json.dumps(data, sort_keys=True)
        
        # Clean messages
        if "messages" in data:
            data["messages"] = [clean_message(msg) for msg in data["messages"]]
        
        # Remove deprecated top-level fields if they exist
        deprecated_fields = ["plan_result", "execution_result", "pending_approval_steps"]
        for field in deprecated_fields:
            data.pop(field, None)
        
        # Keep only essential session fields
        cleaned_data = {
            "session_id": data["session_id"],
            "title": data.get("title", ""),
            "messages": data["messages"],
            "created_at": data.get("created_at", 0),
            "updated_at": data.get("updated_at", 0),
        }
        
        # Keep pty_session if it exists (might still be used)
        if "pty_session" in data:
            cleaned_data["pty_session"] = data["pty_session"]
        
        new_data = json.dumps(cleaned_data, sort_keys=True, indent=2)
        
        if original_data != new_data:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_data)
            return True
        
        return False
    except Exception as e:
        print(f"Error cleaning {file_path}: {e}", file=sys.stderr)
        return False


def main():
    """Main function to clean all session files."""
    # Default storage path
    storage_path = Path.home() / ".forge-agent" / "sessions"
    
    if len(sys.argv) > 1:
        storage_path = Path(sys.argv[1]).expanduser().resolve()
    
    if not storage_path.exists():
        print(f"Storage path does not exist: {storage_path}")
        sys.exit(1)
    
    print(f"Cleaning session files in: {storage_path}")
    
    json_files = list(storage_path.glob("*.json"))
    if not json_files:
        print("No JSON files found.")
        return
    
    modified_count = 0
    for json_file in json_files:
        if clean_session_file(json_file):
            print(f"Cleaned: {json_file.name}")
            modified_count += 1
    
    print(f"\nDone! Modified {modified_count} out of {len(json_files)} files.")


if __name__ == "__main__":
    main()
