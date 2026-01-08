#!/usr/bin/env python3
"""Test script for Phase 2 Planner implementation.

This script tests the Planner component without requiring a running Ollama instance.
It validates schema, imports, and basic functionality.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_schema():
    """Test Planner schema validation."""
    print("=" * 60)
    print("Testing Planner Schema")
    print("=" * 60)
    
    try:
        from agent.runtime.schema import Plan, PlanStep, ToolName, InvalidPlanError
        print("✓ Schema imports OK")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    # Test 1: Valid PlanStep
    try:
        step = PlanStep(
            step_id=1,
            tool=ToolName.FILESYSTEM,
            operation='read_file',
            arguments={'path': 'test.py'},
            rationale='Test step'
        )
        print("✓ PlanStep validation works")
    except Exception as e:
        print(f"✗ PlanStep validation failed: {e}")
        return False
    
    # Test 2: Valid Plan
    try:
        plan = Plan(
            plan_id='test-plan',
            objective='Test objective',
            steps=[step]
        )
        print("✓ Plan validation works")
        print(f"  Plan ID: {plan.plan_id}")
        print(f"  Objective: {plan.objective}")
        print(f"  Steps: {len(plan.steps)}")
    except Exception as e:
        print(f"✗ Plan validation failed: {e}")
        return False
    
    # Test 3: Invalid operation (should fail)
    try:
        invalid_step = PlanStep(
            step_id=1,
            tool=ToolName.FILESYSTEM,
            operation='invalid_operation',
            arguments={},
            rationale='This should fail'
        )
        print("✗ Invalid operation was accepted (should have failed)")
        return False
    except ValueError as e:
        print(f"✓ Invalid operation correctly rejected")
        print(f"  Error: {str(e)[:80]}...")
    
    # Test 4: Non-sequential step IDs (should fail)
    try:
        plan_bad_ids = Plan(
            plan_id='bad-ids',
            objective='Test',
            steps=[
                PlanStep(step_id=1, tool=ToolName.FILESYSTEM, operation='read_file', 
                        arguments={}, rationale='Step 1'),
                PlanStep(step_id=3, tool=ToolName.FILESYSTEM, operation='write_file', 
                        arguments={}, rationale='Step 3'),  # Missing 2
            ]
        )
        print("✗ Non-sequential step IDs were accepted (should have failed)")
        return False
    except ValueError as e:
        print(f"✓ Non-sequential step IDs correctly rejected")
    
    # Test 5: All tools and operations
    print("\n✓ Testing all allowed tools and operations:")
    for tool_name, operations in [
        (ToolName.FILESYSTEM, ['read_file', 'write_file']),
        (ToolName.GIT, ['create_branch', 'commit']),
        (ToolName.GITHUB, ['create_pr']),
        (ToolName.SHELL, ['execute_command']),
        (ToolName.SYSTEM, ['get_status']),
    ]:
        try:
            test_step = PlanStep(
                step_id=1,
                tool=tool_name,
                operation=operations[0],
                arguments={},
                rationale=f'Test {tool_name.value}'
            )
            print(f"  ✓ {tool_name.value}.{operations[0]}")
        except Exception as e:
            print(f"  ✗ {tool_name.value}.{operations[0]}: {e}")
            return False
    
    print("\n✅ All schema tests passed!")
    return True


def test_ollama_provider():
    """Test OllamaProvider initialization."""
    print("\n" + "=" * 60)
    print("Testing OllamaProvider")
    print("=" * 60)
    
    try:
        from agent.llm.ollama import OllamaProvider
        print("✓ OllamaProvider import OK")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    # Test initialization
    try:
        config = {
            'base_url': 'http://localhost:11434',
            'model': 'qwen2.5-coder:7b',
            'temperature': 0.1,
            'timeout': 300
        }
        provider = OllamaProvider(config)
        print(f"✓ OllamaProvider initialized")
        print(f"  Model: {provider.model}")
        print(f"  Temperature: {provider.temperature}")
        print(f"  Base URL: {provider.base_url}")
        print(f"  Timeout: {provider.timeout}")
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False
    
    print("\n✅ OllamaProvider tests passed!")
    print("Note: Full LLM integration test requires running Ollama instance")
    return True


def test_planner():
    """Test Planner class."""
    print("\n" + "=" * 60)
    print("Testing Planner")
    print("=" * 60)
    
    try:
        from agent.runtime.planner import Planner
        from agent.config.loader import AgentConfig
        print("✓ Planner import OK")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    # Test that Planner class can be accessed
    print("✓ Planner class loaded")
    print("✓ All imports successful")
    
    print("\n✅ Planner import tests passed!")
    print("Note: Full integration test requires:")
    print("  1. Running Ollama instance (docker-compose up ollama)")
    print("  2. qwen2.5-coder:7b model downloaded")
    print("  3. Valid agent config file (config/agent.yaml)")
    return True


def main():
    """Run all tests."""
    print("\n🧪 Phase 2 Planner Implementation Tests\n")
    
    results = []
    results.append(("Schema", test_schema()))
    results.append(("OllamaProvider", test_ollama_provider()))
    results.append(("Planner", test_planner()))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:20} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

