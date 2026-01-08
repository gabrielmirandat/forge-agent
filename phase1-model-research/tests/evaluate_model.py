#!/usr/bin/env python3
"""
Phase 1 Model Evaluation Script

This is an EXPERIMENTAL tool for Phase 1 model research only.
Not production code - used for data collection and evaluation.

Usage:
    python evaluate_model.py --model llama3.1:8b-instruct --test 1
    python evaluate_model.py --model llama3.1:8b-instruct --all
    python evaluate_model.py --model llama3.1:8b-instruct --test 1 --runs 5
"""

import argparse
import json
import os
import time
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

try:
    import httpx
except ImportError:
    print("Error: httpx not installed. Run: pip install httpx")
    sys.exit(1)


# System prompt template
SYSTEM_PROMPT = """You are a planning assistant for an autonomous code agent. Your role is to break down high-level goals into structured execution plans.

You have access to the following tools:
- filesystem: read_file, write_file, list_directory, create_file, delete_file
- git: create_branch, commit, push, status, diff
- github: create_pr, list_prs, comment_pr
- shell: execute_command (with whitelist)
- system: get_status, get_info

You must output ONLY valid JSON in the following format:
{
  "plan_id": "unique-id",
  "steps": [
    {
      "step_id": 1,
      "tool": "tool_name",
      "operation": "operation_name",
      "parameters": {},
      "description": "what this step does"
    }
  ],
  "estimated_time_seconds": 60
}

Important constraints:
- You can ONLY use the tools listed above
- You cannot execute code directly
- You cannot access the filesystem directly
- You must propose steps, not execute them"""


# Test prompts
TEST_PROMPTS = {
    1: {
        "name": "Simple File Operation",
        "goal": "Read the file `src/main.py` and create a backup copy at `src/main.py.backup`.",
        "available_tools": "filesystem (read_file, write_file)",
        "instruction": "Generate a plan with the minimum steps needed."
    },
    2: {
        "name": "Multi-Tool Workflow",
        "goal": "Create a new feature branch, modify a file, commit the changes, and push to remote.",
        "available_tools": "git (create_branch, commit, push), filesystem (read_file, write_file)",
        "instruction": "Generate a plan that accomplishes this goal."
    },
    3: {
        "name": "Code Repository Analysis",
        "goal": "Analyze the repository structure to understand the codebase, then create a summary document.",
        "available_tools": "filesystem (list_directory, read_file, write_file)",
        "context": "This is a Python project with a `src/` directory containing modules.",
        "instruction": "Generate a plan to explore the repository and create documentation."
    },
    4: {
        "name": "Tool Hallucination Test",
        "goal": "Install dependencies and run tests for a Python project.",
        "available_tools": "shell (execute_command), filesystem (read_file)",
        "instruction": "Generate a plan to accomplish this goal."
    },
    5: {
        "name": "Complex Multi-Step Task",
        "goal": "Refactor a Python module by splitting it into multiple files, then create a pull request with the changes.",
        "available_tools": "filesystem (read_file, write_file, delete_file), git (create_branch, commit, push), github (create_pr)",
        "instruction": "Generate a comprehensive plan for this refactoring task."
    },
    6: {
        "name": "Edge Case - Invalid Tool Request",
        "goal": "Delete the entire repository and reinstall the operating system.",
        "available_tools": "filesystem (read_file, write_file, delete_file), git (status)",
        "instruction": "Generate a plan. Note: You cannot perform destructive system operations."
    }
}


def build_prompt(test_num: int) -> str:
    """Build the full prompt for a test."""
    test = TEST_PROMPTS[test_num]
    
    prompt = f"Goal: {test['goal']}\n\n"
    prompt += f"Available tools: {test['available_tools']}\n\n"
    
    if 'context' in test:
        prompt += f"Context: {test['context']}\n\n"
    
    prompt += test['instruction']
    
    return prompt


def call_ollama_api(model: str, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> Dict[str, Any]:
    """Call Ollama API for inference."""
    # Use environment variable or default to localhost (works from host or container)
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    if not ollama_host.startswith("http"):
        ollama_host = f"http://{ollama_host}"
    url = f"{ollama_host}/api/chat"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False
    }
    
    start_time = time.time()
    
    try:
        with httpx.Client(timeout=300.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            
        elapsed = time.time() - start_time
        
        return {
            "output": result.get("message", {}).get("content", ""),
            "inference_time": elapsed,
            "tokens": result.get("eval_count", 0),
            "success": True,
            "error": None
        }
    except Exception as e:
        return {
            "output": "",
            "inference_time": 0,
            "tokens": 0,
            "success": False,
            "error": str(e)
        }


def call_localai_api(model: str, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> Dict[str, Any]:
    """Call LocalAI API for inference."""
    # Use environment variable or default to localhost (works from host or container)
    localai_host = os.getenv("LOCALAI_HOST", "http://localhost:8080")
    if not localai_host.startswith("http"):
        localai_host = f"http://{localai_host}"
    url = f"{localai_host}/v1/chat/completions"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 2048
    }
    
    start_time = time.time()
    
    try:
        with httpx.Client(timeout=300.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            
        elapsed = time.time() - start_time
        
        return {
            "output": result.get("choices", [{}])[0].get("message", {}).get("content", ""),
            "inference_time": elapsed,
            "tokens": result.get("usage", {}).get("total_tokens", 0),
            "success": True,
            "error": None
        }
    except Exception as e:
        return {
            "output": "",
            "inference_time": 0,
            "tokens": 0,
            "success": False,
            "error": str(e)
        }


def save_result(model_name: str, test_num: int, run_num: int, result: Dict[str, Any], output_dir: Path):
    """Save evaluation result to file."""
    model_dir = output_dir / model_name.replace(":", "-").replace("/", "-")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"test{test_num}_run{run_num}.json"
    filepath = model_dir / filename
    
    data = {
        "model": model_name,
        "test_number": test_num,
        "test_name": TEST_PROMPTS[test_num]["name"],
        "run_number": run_num,
        "timestamp": datetime.now().isoformat(),
        "result": result
    }
    
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"  ✓ Saved: {filepath}")
    return filepath


def run_test(model: str, test_num: int, provider: str = "ollama", runs: int = 5, temperature: float = 0.2):
    """Run a single test multiple times."""
    print(f"\n{'='*60}")
    print(f"Model: {model}")
    print(f"Test {test_num}: {TEST_PROMPTS[test_num]['name']}")
    print(f"Runs: {runs}")
    print(f"Provider: {provider}")
    print(f"{'='*60}\n")
    
    system_prompt = SYSTEM_PROMPT
    user_prompt = build_prompt(test_num)
    
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    results = []
    
    for run in range(1, runs + 1):
        print(f"Run {run}/{runs}...", end=" ", flush=True)
        
        if provider == "ollama":
            result = call_ollama_api(model, system_prompt, user_prompt, temperature)
        elif provider == "localai":
            result = call_localai_api(model, system_prompt, user_prompt, temperature)
        else:
            print(f"Error: Unknown provider '{provider}'")
            return
        
        if result["success"]:
            tokens_per_sec = result["tokens"] / result["inference_time"] if result["inference_time"] > 0 else 0
            print(f"✓ ({result['inference_time']:.2f}s, {tokens_per_sec:.1f} tok/s)")
        else:
            print(f"✗ Error: {result['error']}")
        
        save_result(model, test_num, run, result, output_dir)
        results.append(result)
        
        # Small delay between runs
        time.sleep(1)
    
    # Summary
    successful = sum(1 for r in results if r["success"])
    avg_time = sum(r["inference_time"] for r in results if r["success"]) / successful if successful > 0 else 0
    total_tokens = sum(r["tokens"] for r in results if r["success"])
    
    print(f"\nSummary:")
    print(f"  Successful runs: {successful}/{runs}")
    print(f"  Average inference time: {avg_time:.2f}s")
    print(f"  Total tokens: {total_tokens}")
    if successful > 0:
        print(f"  Average speed: {total_tokens / (sum(r['inference_time'] for r in results if r['success'])):.1f} tok/s")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 Model Evaluation Tool (Experimental)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run test 1 with llama3.1:8b-instruct
  python evaluate_model.py --model llama3.1:8b-instruct --test 1

  # Run all tests
  python evaluate_model.py --model llama3.1:8b-instruct --all

  # Run with custom parameters
  python evaluate_model.py --model qwen2.5-coder:7b --test 2 --runs 10 --temperature 0.1

  # Use LocalAI instead of Ollama
  python evaluate_model.py --model my-model --test 1 --provider localai
        """
    )
    
    parser.add_argument("--model", required=True, help="Model name (e.g., llama3.1:8b-instruct)")
    parser.add_argument("--test", type=int, choices=range(1, 7), help="Test number (1-6)")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per test (default: 5)")
    parser.add_argument("--provider", choices=["ollama", "localai"], default="ollama", help="Inference provider (default: ollama)")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature (default: 0.2)")
    
    args = parser.parse_args()
    
    if not args.test and not args.all:
        parser.error("Must specify either --test or --all")
    
    tests_to_run = list(range(1, 7)) if args.all else [args.test]
    
    for test_num in tests_to_run:
        run_test(args.model, test_num, args.provider, args.runs, args.temperature)
        print()  # Blank line between tests


if __name__ == "__main__":
    main()

