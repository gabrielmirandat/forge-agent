#!/usr/bin/env python3
"""
Phase 1 Output Validation Script

Validates JSON outputs and calculates evaluation metrics.

Usage:
    python validate_output.py results/llama3.1-8b-instruct/test1_run1.json
    python validate_output.py results/llama3.1-8b-instruct/ --test 1
    python validate_output.py results/llama3.1-8b-instruct/ --all
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from collections import Counter


# Expected tool names
AVAILABLE_TOOLS = {
    "filesystem": ["read_file", "write_file", "list_directory", "create_file", "delete_file"],
    "git": ["create_branch", "commit", "push", "status", "diff"],
    "github": ["create_pr", "list_prs", "comment_pr"],
    "shell": ["execute_command"],
    "system": ["get_status", "get_info"]
}

ALL_TOOL_NAMES = set()
for tool, operations in AVAILABLE_TOOLS.items():
    ALL_TOOL_NAMES.add(tool)
    for op in operations:
        ALL_TOOL_NAMES.add(f"{tool}.{op}")


def validate_json(content: str) -> tuple[bool, Optional[Dict], Optional[str]]:
    """Validate JSON and return parsed object."""
    try:
        # Try to extract JSON from markdown code blocks
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            content = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            content = content[start:end].strip()
        
        data = json.loads(content)
        return True, data, None
    except json.JSONDecodeError as e:
        return False, None, str(e)
    except Exception as e:
        return False, None, str(e)


def check_required_fields(data: Dict) -> Dict[str, bool]:
    """Check if required fields are present."""
    required = {
        "plan_id": "plan_id" in data,
        "steps": "steps" in data and isinstance(data["steps"], list),
        "estimated_time_seconds": "estimated_time_seconds" in data
    }
    
    if required["steps"]:
        # Check step structure
        step_fields = ["step_id", "tool", "operation", "parameters", "description"]
        for i, step in enumerate(data["steps"]):
            if not isinstance(step, dict):
                required[f"step_{i}_is_dict"] = False
                continue
            for field in step_fields:
                required[f"step_{i}_{field}"] = field in step
    
    return required


def check_tool_accuracy(steps: List[Dict]) -> tuple[int, int, List[str]]:
    """Check for tool hallucinations."""
    total_tool_calls = 0
    hallucinations = 0
    hallucinated_tools = []
    
    for step in steps:
        if not isinstance(step, dict):
            continue
        
        tool = step.get("tool", "")
        operation = step.get("operation", "")
        
        if tool:
            total_tool_calls += 1
            # Check if tool name is valid
            if tool not in AVAILABLE_TOOLS:
                hallucinations += 1
                hallucinated_tools.append(tool)
            # Check if operation is valid for the tool
            elif operation and tool in AVAILABLE_TOOLS:
                if operation not in AVAILABLE_TOOLS[tool]:
                    hallucinations += 1
                    hallucinated_tools.append(f"{tool}.{operation}")
    
    return total_tool_calls, hallucinations, hallucinated_tools


def check_logical_ordering(steps: List[Dict]) -> float:
    """Basic check for logical ordering (simplified)."""
    if len(steps) < 2:
        return 1.0
    
    score = 0.0
    total_checks = 0
    
    # Check for obvious ordering issues
    for i in range(len(steps) - 1):
        current = steps[i]
        next_step = steps[i + 1]
        
        current_tool = current.get("tool", "")
        next_tool = next_step.get("tool", "")
        
        # Git operations should follow: create_branch -> commit -> push
        if current_tool == "git" and next_tool == "git":
            current_op = current.get("operation", "")
            next_op = next_step.get("operation", "")
            
            if current_op == "push" and next_op == "commit":
                score += 0  # Wrong order
            elif current_op == "commit" and next_op == "create_branch":
                score += 0  # Wrong order
            else:
                score += 1  # Seems OK
            total_checks += 1
    
    return score / total_checks if total_checks > 0 else 1.0


def validate_file(filepath: Path) -> Dict[str, Any]:
    """Validate a single result file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    
    result_data = data.get("result", {})
    output = result_data.get("output", "")
    
    # If output is already a dict (parsed JSON), use it directly
    # Otherwise, try to parse it as JSON string
    if isinstance(output, dict):
        parsed_json = output
        is_valid = True
        json_error = None
    elif isinstance(output, str):
        is_valid, parsed_json, json_error = validate_json(output)
    else:
        is_valid = False
        parsed_json = None
        json_error = "Output is not a string or dict"
    
    scores = {
        "json_valid": is_valid,
        "json_error": json_error,
        "structure_compliance": 0.0,
        "tool_accuracy": 0.0,
        "logical_ordering": 0.0,
        "completeness": 0.0,
        "total_tool_calls": 0,
        "hallucinations": 0,
        "hallucinated_tools": []
    }
    
    if not is_valid:
        return scores
    
    # Structure Compliance
    required = check_required_fields(parsed_json)
    present = sum(1 for v in required.values() if v)
    total = len(required)
    scores["structure_compliance"] = present / total if total > 0 else 0.0
    scores["required_fields"] = required
    
    # Tool Accuracy
    steps = parsed_json.get("steps", [])
    total_calls, hallucinations, hallucinated = check_tool_accuracy(steps)
    scores["total_tool_calls"] = total_calls
    scores["hallucinations"] = hallucinations
    scores["hallucinated_tools"] = hallucinated
    scores["tool_accuracy"] = 1.0 - (hallucinations / total_calls) if total_calls > 0 else 1.0
    
    # Logical Ordering
    scores["logical_ordering"] = check_logical_ordering(steps)
    
    # Completeness (simplified - just check if plan has steps)
    scores["completeness"] = 1.0 if len(steps) > 0 else 0.0
    
    return scores


def calculate_consistency(files: List[Path]) -> float:
    """Calculate consistency across multiple runs."""
    if len(files) < 2:
        return 1.0
    
    all_steps = []
    
    for filepath in files:
        with open(filepath, "r") as f:
            data = json.load(f)
        
        result_data = data.get("result", {})
        output = result_data.get("output", "")
        
        is_valid, parsed_json, _ = validate_json(output)
        if not is_valid:
            continue
        
        steps = parsed_json.get("steps", [])
        # Create a signature for the step sequence
        step_sig = tuple(
            (step.get("tool", ""), step.get("operation", ""))
            for step in steps
        )
        all_steps.append(step_sig)
    
    if len(all_steps) < 2:
        return 0.0
    
    # Calculate Jaccard similarity
    similarities = []
    for i in range(len(all_steps)):
        for j in range(i + 1, len(all_steps)):
            set1 = set(all_steps[i])
            set2 = set(all_steps[j])
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            similarity = intersection / union if union > 0 else 0.0
            similarities.append(similarity)
    
    return sum(similarities) / len(similarities) if similarities else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Validate Phase 1 evaluation outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("path", type=Path, help="Result file or directory")
    parser.add_argument("--test", type=int, help="Filter by test number")
    parser.add_argument("--all", action="store_true", help="Process all tests")
    
    args = parser.parse_args()
    
    if not args.path.exists():
        print(f"Error: Path does not exist: {args.path}")
        sys.exit(1)
    
    if args.path.is_file():
        # Validate single file
        print(f"Validating: {args.path}\n")
        scores = validate_file(args.path)
        
        print("Results:")
        print(f"  JSON Valid: {'✓' if scores['json_valid'] else '✗'}")
        if scores['json_error']:
            print(f"  Error: {scores['json_error']}")
        print(f"  Structure Compliance: {scores['structure_compliance']:.2%}")
        print(f"  Tool Accuracy: {scores['tool_accuracy']:.2%}")
        print(f"  Logical Ordering: {scores['logical_ordering']:.2%}")
        print(f"  Completeness: {scores['completeness']:.2%}")
        print(f"  Total Tool Calls: {scores['total_tool_calls']}")
        print(f"  Hallucinations: {scores['hallucinations']}")
        if scores['hallucinated_tools']:
            print(f"  Hallucinated Tools: {', '.join(scores['hallucinated_tools'])}")
    
    elif args.path.is_dir():
        # Validate directory
        test_files = {}
        
        for filepath in sorted(args.path.glob("test*_run*.json")):
            # Extract test number
            parts = filepath.stem.split("_")
            if len(parts) >= 2:
                test_num = int(parts[0].replace("test", ""))
                run_num = int(parts[1].replace("run", ""))
                
                if args.test and test_num != args.test:
                    continue
                
                if test_num not in test_files:
                    test_files[test_num] = []
                test_files[test_num].append((run_num, filepath))
        
        for test_num in sorted(test_files.keys()):
            print(f"\n{'='*60}")
            print(f"Test {test_num}")
            print(f"{'='*60}\n")
            
            files = [f for _, f in sorted(test_files[test_num])]
            
            # Individual scores
            all_scores = []
            for filepath in files:
                scores = validate_file(filepath)
                all_scores.append(scores)
            
            # Aggregate
            json_valid_rate = sum(1 for s in all_scores if s['json_valid']) / len(all_scores)
            avg_structure = sum(s['structure_compliance'] for s in all_scores) / len(all_scores)
            avg_tool_accuracy = sum(s['tool_accuracy'] for s in all_scores) / len(all_scores)
            avg_ordering = sum(s['logical_ordering'] for s in all_scores) / len(all_scores)
            avg_completeness = sum(s['completeness'] for s in all_scores) / len(all_scores)
            total_hallucinations = sum(s['hallucinations'] for s in all_scores)
            total_tool_calls = sum(s['total_tool_calls'] for s in all_scores)
            
            # Consistency
            consistency = calculate_consistency(files)
            
            print("Aggregate Scores:")
            print(f"  JSON Validity: {json_valid_rate:.2%}")
            print(f"  Structure Compliance: {avg_structure:.2%}")
            print(f"  Tool Accuracy: {avg_tool_accuracy:.2%}")
            print(f"  Logical Ordering: {avg_ordering:.2%}")
            print(f"  Completeness: {avg_completeness:.2%}")
            print(f"  Consistency: {consistency:.2%}")
            print(f"  Total Tool Calls: {total_tool_calls}")
            print(f"  Total Hallucinations: {total_hallucinations}")
            if total_tool_calls > 0:
                hallucination_rate = total_hallucinations / total_tool_calls
                print(f"  Hallucination Rate: {hallucination_rate:.2%}")


if __name__ == "__main__":
    main()

