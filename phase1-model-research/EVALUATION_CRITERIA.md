# Phase 1: Evaluation Criteria

Clear pass/fail criteria for model selection.

## Core Requirements

A model **MUST** pass all of these to be considered:

### 1. JSON Validity (Critical)

**Requirement**: ≥90% of outputs are valid JSON

**Measurement**:
- Parse each output with `json.loads()`
- Count valid vs invalid
- Calculate percentage

**Pass**: ≥90% valid JSON across all test runs
**Fail**: <90% valid JSON

**Rationale**: The Planner must parse LLM output. Invalid JSON breaks the system.

---

### 2. Instruction Following (Critical)

**Requirement**: ≥80% of required fields present and correctly formatted

**Measurement**:
- Check for required fields: `plan_id`, `steps`, `estimated_time_seconds`
- Check step structure: `step_id`, `tool`, `operation`, `parameters`, `description`
- Score: (fields_present / fields_required) × 100

**Pass**: ≥80% field compliance
**Fail**: <80% field compliance

**Rationale**: Missing fields break downstream processing. The Executor needs structured data.

---

### 3. Task Decomposition (Critical)

**Requirement**: ≥70% of steps are logical and executable

**Measurement**:
- Manual review: Are steps reasonable?
- Can each step be executed with available tools?
- Are steps in logical order?
- Score: (executable_steps / total_steps) × 100

**Pass**: ≥70% executable steps
**Fail**: <70% executable steps

**Rationale**: The plan must be actionable. Illogical steps waste resources and fail.

---

### 4. Tool Hallucination (Critical)

**Requirement**: ≤10% of tool calls reference non-existent tools

**Measurement**:
- Compare tool names in plan vs available tools list
- Count hallucinations: tool names not in available list
- Score: (hallucinations / total_tool_calls) × 100

**Pass**: ≤10% hallucination rate
**Fail**: >10% hallucination rate

**Rationale**: Hallucinated tools cause execution failures. The Executor can only use real tools.

---

### 5. Consistency (Important)

**Requirement**: ≥60% similarity across 5 runs of the same prompt

**Measurement**:
- Run same prompt 5 times
- Compare step sequences (Jaccard similarity)
- Average similarity across all 5 runs

**Pass**: ≥60% average similarity
**Fail**: <60% average similarity

**Rationale**: Inconsistent outputs make debugging difficult. We need predictable behavior.

---

### 6. Hardware Fit (Critical)

**Requirement**: Runs comfortably within 12GB VRAM

**Measurement**:
- Load model with quantization
- Monitor VRAM usage during inference
- Test with max context length

**Pass**: Peak VRAM < 11GB (leaving 1GB buffer)
**Fail**: Peak VRAM ≥ 11GB

**Rationale**: Must fit on available hardware. OOM errors break the system.

---

## Secondary Criteria (Nice to Have)

These are evaluated but not blocking:

### 7. Inference Speed

**Target**: ≥10 tokens/second
**Measurement**: Tokens generated / time elapsed
**Note**: Speed is secondary to quality, but very slow models (<5 tokens/s) may be impractical

### 8. Code Understanding

**Target**: Shows understanding of code structure in Test 3
**Measurement**: Manual review of repository analysis plan
**Note**: Important for code-focused tasks, but not critical for basic planning

### 9. Safety Awareness

**Target**: Recognizes dangerous operations in Test 6
**Measurement**: Does not propose destructive operations
**Note**: Good to have, but Executor will enforce safety anyway

---

## Scoring Methodology

### Per-Test Scoring

For each of the 6 test prompts:
1. Run 5 times
2. Score each run on 6 criteria (1-6 above)
3. Average scores across 5 runs
4. Record per-test average

### Overall Model Score

**Formula**:
```
Overall Score = (JSON_Validity × 0.25) +
                (Instruction_Following × 0.25) +
                (Task_Decomposition × 0.20) +
                (Tool_Accuracy × 0.20) +
                (Consistency × 0.10)
```

**Hardware Fit** is binary: Pass/Fail (blocks if Fail)

### Final Decision

A model is **SELECTED** if:
1. ✅ Passes all 6 critical requirements
2. ✅ Has highest overall score among passing models
3. ✅ Meets minimum inference speed (≥5 tokens/s)

If multiple models pass, choose the one with:
- Highest overall score
- Best balance of quality and speed
- Most consistent results

---

## Edge Cases

### What if no model passes all criteria?

**Option 1**: Relax criteria slightly (e.g., 85% instead of 90% JSON validity)
**Option 2**: Use best available model and document limitations
**Option 3**: Consider larger models (if they fit) or wait for better models

### What if multiple models pass equally?

**Tie-breakers** (in order):
1. Consistency (most stable)
2. Inference speed (faster is better)
3. Code understanding (if code-focused tasks are important)
4. Model size (smaller is better for future scaling)

---

## Documentation Requirements

For each model evaluated, document:

1. **Hardware Usage**: Peak VRAM, RAM, inference speed
2. **Per-Test Scores**: All 6 tests, all 5 runs
3. **Failure Modes**: What does this model struggle with?
4. **Optimal Parameters**: Temperature, max_tokens, quantization level
5. **Sample Outputs**: 2-3 example plans (good and bad)

This documentation will inform Phase 2 implementation.

