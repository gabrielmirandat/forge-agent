# Phase 1: Model Evaluation Results

**Model**: [Model Name]  
**Version**: [Version/Quantization]  
**Date**: [YYYY-MM-DD]  
**Evaluator**: [Name]  
**Hardware**: RTX 5070 (12GB VRAM), 32GB RAM, Ubuntu 24.04

---

## Hardware Performance

| Metric | Value | Notes |
|--------|-------|-------|
| Quantization | Q4_K_M / Q5_K_M / Other | |
| Peak VRAM Usage | X GB | |
| Peak RAM Usage | X GB | |
| Average Inference Speed | X tokens/s | |
| Context Length Tested | X tokens | |
| Load Time | X seconds | |

**Hardware Fit**: ✅ PASS / ❌ FAIL (must be <11GB VRAM)

---

## Inference Parameters

```yaml
temperature: 0.2
max_tokens: 2048
top_p: 0.95
repeat_penalty: 1.1
quantization: Q4_K_M
context_length: 4096
```

**Rationale**: [Why these parameters were chosen]

---

## Test Results

### Test 1: Simple File Operation

| Run | JSON Valid | Structure | Tool Accuracy | Logical Order | Completeness | Consistency |
|-----|------------|-----------|---------------|---------------|--------------|-------------|
| 1   | ✅ / ❌    | X/1       | X/1           | X/1           | X/1          | -           |
| 2   | ✅ / ❌    | X/1       | X/1           | X/1           | X/1          | -           |
| 3   | ✅ / ❌    | X/1       | X/1           | X/1           | X/1          | -           |
| 4   | ✅ / ❌    | X/1       | X/1           | X/1           | X/1          | -           |
| 5   | ✅ / ❌    | X/1       | X/1           | X/1           | X/1          | -           |
| **Avg** | **X%** | **X/1** | **X/1** | **X/1** | **X/1** | **X%** |

**Sample Output (Best Run)**:
```json
[Paste best output here]
```

**Sample Output (Worst Run)**:
```json
[Paste worst output here]
```

**Notes**: [Observations, edge cases, patterns]

---

### Test 2: Multi-Tool Workflow

[Same table structure as Test 1]

**Sample Output (Best Run)**:
```json
[Paste best output here]
```

**Notes**: [Observations]

---

### Test 3: Code Repository Analysis

[Same table structure as Test 1]

**Sample Output (Best Run)**:
```json
[Paste best output here]
```

**Notes**: [Observations]

---

### Test 4: Tool Hallucination Test

[Same table structure as Test 1]

**Hallucinations Found**:
- [List any invented tool names]
- [Count: X hallucinations / Y total tool calls = Z%]

**Notes**: [Observations]

---

### Test 5: Complex Multi-Step Task

[Same table structure as Test 1]

**Sample Output (Best Run)**:
```json
[Paste best output here]
```

**Notes**: [Observations]

---

### Test 6: Edge Case - Invalid Tool Request

[Same table structure as Test 1]

**Response Type**: 
- ✅ Refused gracefully
- ✅ Proposed safe alternative
- ❌ Proposed invalid operation
- ❌ Hallucinated tools

**Notes**: [Observations]

---

## Aggregate Scores

| Criterion | Score | Pass/Fail | Notes |
|-----------|-------|-----------|-------|
| JSON Validity | X% | ✅ / ❌ | [≥90% required] |
| Instruction Following | X% | ✅ / ❌ | [≥80% required] |
| Task Decomposition | X% | ✅ / ❌ | [≥70% required] |
| Tool Hallucination | X% | ✅ / ❌ | [≤10% required] |
| Consistency | X% | ✅ / ❌ | [≥60% required] |
| Hardware Fit | ✅ / ❌ | ✅ / ❌ | [<11GB required] |

**Overall Score**: X/36 (weighted average)

---

## Failure Modes

**What this model struggles with**:
1. [Specific issue observed]
2. [Another issue]
3. [Pattern of failures]

**Example failures**:
- [Describe a specific failure case]
- [Another example]

---

## Strengths

**What this model does well**:
1. [Specific strength]
2. [Another strength]
3. [Notable capability]

**Example successes**:
- [Describe a specific success case]
- [Another example]

---

## Comparison Notes

**Compared to other models**:
- [How does this compare to Model X?]
- [Unique characteristics]
- [Trade-offs]

---

## Recommendation

**Status**: ✅ SELECT / ⚠️ CONDITIONAL / ❌ REJECT

**Rationale**: [Why this model should/shouldn't be selected]

**If Selected**:
- Optimal parameters: [list]
- Known limitations: [list]
- Mitigation strategies: [how to work around limitations]

**If Rejected**:
- Primary reason: [why it failed]
- Could be viable if: [what would need to change]

---

## Raw Data

[Optional: Link to raw output files, logs, etc.]

