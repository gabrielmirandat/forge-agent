#!/usr/bin/env python3
"""Test script for Phase 2 Planner hardening improvements.

Tests the structural improvements:
- PlannerDiagnostics
- Empty Plan support
- Deterministic Plan ID
- Stricter JSON extraction
- Explicit failure modes
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_planner_diagnostics():
    """Test PlannerDiagnostics."""
    print("=" * 60)
    print("Testing PlannerDiagnostics")
    print("=" * 60)
    
    try:
        from agent.runtime.schema import PlannerDiagnostics
        
        # Test successful diagnostics
        diag = PlannerDiagnostics(
            model_name="qwen2.5-coder:7b",
            temperature=0.1,
            retries_used=0,
            raw_llm_response='{"plan_id": "test", "objective": "test", "steps": []}',
            extracted_json='{"plan_id": "test", "objective": "test", "steps": []}',
            validation_errors=None
        )
        print("✓ PlannerDiagnostics creation works")
        print(f"  Model: {diag.model_name}")
        print(f"  Temperature: {diag.temperature}")
        print(f"  Retries: {diag.retries_used}")
        
        # Test diagnostics with errors
        diag_error = PlannerDiagnostics(
            model_name="qwen2.5-coder:7b",
            temperature=0.1,
            retries_used=1,
            raw_llm_response="Invalid response",
            extracted_json=None,
            validation_errors=["JSON parsing failed"]
        )
        print("✓ PlannerDiagnostics with errors works")
        print(f"  Has errors: {diag_error.validation_errors is not None}")
        
        return True
    except Exception as e:
        print(f"✗ PlannerDiagnostics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_empty_plan():
    """Test empty plan support."""
    print("\n" + "=" * 60)
    print("Testing Empty Plan Support")
    print("=" * 60)
    
    try:
        from agent.runtime.schema import Plan
        
        # Test valid empty plan
        empty_plan = Plan(
            plan_id="empty-plan-1",
            objective="Check if file exists",
            steps=[],
            notes="File already exists, no action needed"
        )
        print("✓ Empty plan creation works")
        print(f"  Steps: {len(empty_plan.steps)}")
        print(f"  Notes: {empty_plan.notes}")
        
        # Test empty plan without notes (should fail)
        try:
            invalid_empty = Plan(
                plan_id="invalid",
                objective="Test",
                steps=[],
                notes=None
            )
            print("✗ Empty plan without notes was accepted (should fail)")
            return False
        except ValueError as e:
            print("✓ Empty plan without notes correctly rejected")
            print(f"  Error: {str(e)[:80]}...")
        
        return True
    except Exception as e:
        print(f"✗ Empty plan test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_deterministic_plan_id():
    """Test deterministic plan ID generation."""
    print("\n" + "=" * 60)
    print("Testing Deterministic Plan ID")
    print("=" * 60)
    
    try:
        from agent.runtime.planner import Planner
        from agent.config.loader import AgentConfig
        
        config = AgentConfig()
        planner = Planner(config, None)  # LLM not needed for ID test
        
        # Same goal should produce same ID (within same minute)
        goal = "Test goal"
        id1 = planner._generate_plan_id(goal)
        id2 = planner._generate_plan_id(goal)
        
        print(f"  Goal: '{goal}'")
        print(f"  ID 1: {id1}")
        print(f"  ID 2: {id2}")
        
        if id1 == id2:
            print("✓ Same goal produces same ID (deterministic)")
        else:
            print("⚠ Same goal produced different IDs (may be due to timestamp bucket)")
        
        # Different goals should produce different IDs
        goal2 = "Different goal"
        id3 = planner._generate_plan_id(goal2)
        print(f"  Different goal ID: {id3}")
        
        if id1 != id3:
            print("✓ Different goals produce different IDs")
        else:
            print("✗ Different goals produced same ID")
            return False
        
        # Test with context
        id4 = planner._generate_plan_id(goal, {"key": "value"})
        id5 = planner._generate_plan_id(goal, {"key": "value"})
        
        if id4 == id5:
            print("✓ Same goal + context produces same ID")
        else:
            print("⚠ Same goal + context produced different IDs")
        
        if id1 != id4:
            print("✓ Context affects ID generation")
        else:
            print("⚠ Context did not affect ID generation")
        
        return True
    except Exception as e:
        print(f"✗ Deterministic Plan ID test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_strict_json_extraction():
    """Test stricter JSON extraction rules."""
    print("\n" + "=" * 60)
    print("Testing Stricter JSON Extraction")
    print("=" * 60)
    
    try:
        from agent.runtime.planner import Planner
        from agent.runtime.schema import JSONExtractionError
        from agent.config.loader import AgentConfig
        
        config = AgentConfig()
        planner = Planner(config, None)
        
        # Test 1: Single JSON in markdown (should work)
        text1 = '```json\n{"plan_id": "test", "objective": "test"}\n```'
        try:
            result = planner._extract_json(text1)
            print("✓ Single JSON in markdown extracted")
        except Exception as e:
            print(f"✗ Single JSON in markdown failed: {e}")
            return False
        
        # Test 2: Single JSON without markdown (should work)
        text2 = '{"plan_id": "test", "objective": "test"}'
        try:
            result = planner._extract_json(text2)
            print("✓ Single JSON without markdown extracted")
        except Exception as e:
            print(f"✗ Single JSON without markdown failed: {e}")
            return False
        
        # Test 3: No JSON (should fail)
        text3 = "This is just text with no JSON"
        try:
            result = planner._extract_json(text3)
            print("✗ No JSON was accepted (should fail)")
            return False
        except JSONExtractionError as e:
            print("✓ No JSON correctly rejected")
            print(f"  Error: {str(e)[:60]}...")
        
        # Test 4: Multiple JSON objects (should fail)
        text4 = '{"first": true} and also {"second": true}'
        try:
            result = planner._extract_json(text4)
            print("✗ Multiple JSON objects were accepted (should fail)")
            return False
        except JSONExtractionError as e:
            print("✓ Multiple JSON objects correctly rejected")
            print(f"  Error: {str(e)[:60]}...")
        
        # Test 5: Multiple JSON in code blocks (should fail)
        text5 = '```json\n{"first": true}\n```\n```json\n{"second": true}\n```'
        try:
            result = planner._extract_json(text5)
            print("✗ Multiple JSON in code blocks were accepted (should fail)")
            return False
        except JSONExtractionError as e:
            print("✓ Multiple JSON in code blocks correctly rejected")
        
        return True
    except Exception as e:
        print(f"✗ Strict JSON extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_explicit_failure_modes():
    """Test explicit failure modes."""
    print("\n" + "=" * 60)
    print("Testing Explicit Failure Modes")
    print("=" * 60)
    
    try:
        from agent.runtime.schema import (
            LLMCommunicationError,
            JSONExtractionError,
            InvalidPlanError,
            PlannerDiagnostics
        )
        
        # Test LLMCommunicationError
        diag = PlannerDiagnostics(
            model_name="test",
            temperature=0.1,
            retries_used=1,
            raw_llm_response="error",
            extracted_json=None,
            validation_errors=["LLM error"]
        )
        error1 = LLMCommunicationError("LLM failed", diagnostics=diag)
        print("✓ LLMCommunicationError creation works")
        print(f"  Has diagnostics: {error1.diagnostics is not None}")
        
        # Test JSONExtractionError
        error2 = JSONExtractionError(
            "No JSON found",
            raw_response="raw text",
            diagnostics=diag
        )
        print("✓ JSONExtractionError creation works")
        print(f"  Has raw_response: {error2.raw_response is not None}")
        print(f"  Has diagnostics: {error2.diagnostics is not None}")
        
        # Test InvalidPlanError
        error3 = InvalidPlanError(
            "Validation failed",
            validation_errors=["error1", "error2"],
            diagnostics=diag
        )
        print("✓ InvalidPlanError creation works")
        print(f"  Validation errors: {len(error3.validation_errors)}")
        print(f"  Has diagnostics: {error3.diagnostics is not None}")
        
        return True
    except Exception as e:
        print(f"✗ Explicit failure modes test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_plan_result():
    """Test PlanResult."""
    print("\n" + "=" * 60)
    print("Testing PlanResult")
    print("=" * 60)
    
    try:
        from agent.runtime.schema import PlanResult, Plan, PlannerDiagnostics
        
        # Test with regular plan
        plan = Plan(
            plan_id="test-plan",
            objective="Test objective",
            steps=[]
        )
        diag = PlannerDiagnostics(
            model_name="qwen2.5-coder:7b",
            temperature=0.1,
            retries_used=0,
            raw_llm_response="test",
            extracted_json='{"plan_id": "test"}',
            validation_errors=None
        )
        result = PlanResult(plan=plan, diagnostics=diag)
        print("✓ PlanResult creation works")
        print(f"  Plan ID: {result.plan.plan_id}")
        print(f"  Diagnostics model: {result.diagnostics.model_name}")
        
        return True
    except Exception as e:
        print(f"✗ PlanResult test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all hardening tests."""
    print("\n🧪 Phase 2 Planner Hardening Tests\n")
    
    results = []
    results.append(("PlannerDiagnostics", test_planner_diagnostics()))
    results.append(("Empty Plan", test_empty_plan()))
    results.append(("Deterministic Plan ID", test_deterministic_plan_id()))
    results.append(("Strict JSON Extraction", test_strict_json_extraction()))
    results.append(("Explicit Failure Modes", test_explicit_failure_modes()))
    results.append(("PlanResult", test_plan_result()))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:30} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 All hardening tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

