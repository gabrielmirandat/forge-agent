# Phase 1: Model Research - Evaluation Plan

## Objective

Evaluate local LLMs for suitability as a **reasoning engine** for the Planner component. The selected model will be used exclusively to propose structured execution plans, not to execute actions.

## Hardware Constraints

- **GPU**: NVIDIA RTX 5070 (12GB VRAM)
- **RAM**: 32GB
- **OS**: Ubuntu 24.04
- **Inference**: Local only (no cloud, no paid APIs)

## Models Under Evaluation

### Primary Candidates

1. **llama3.1:8b-instruct** (Meta)
   - Size: ~8B parameters
   - Expected VRAM: ~5-6GB (quantized)
   - Strengths: Strong instruction following, good reasoning
   - Weaknesses: May struggle with code-specific tasks

2. **qwen2.5-coder:7b** (Alibaba)
   - Size: ~7B parameters
   - Expected VRAM: ~4-5GB (quantized)
   - Strengths: Code-focused, good at structured output
   - Weaknesses: Less general reasoning capability

3. **deepseek-coder:6.7b** (DeepSeek)
   - Size: ~6.7B parameters
   - Expected VRAM: ~4-5GB (quantized)
   - Strengths: Excellent code understanding, strong JSON output
   - Weaknesses: May be too specialized

4. **nemotron-mini** (NVIDIA, if available)
   - Size: TBD
   - Expected VRAM: TBD
   - Strengths: Optimized for NVIDIA hardware
   - Weaknesses: Availability and size unknown

### Evaluation Approach

We will test each model with the same set of prompts and measure:
- JSON validity rate
- Instruction following accuracy
- Task decomposition quality
- Tool hallucination rate
- Consistency across runs
- Inference speed (secondary)

## Test Methodology

### 1. Setup Phase

For each model:
- Load model with appropriate quantization (Q4_K_M or Q5_K_M recommended)
- Configure temperature: 0.1-0.3 (low for consistency)
- Set max_tokens: 2048 (sufficient for plan generation)
- Use same system prompt template across all models

### 2. Execution Phase

Run each test prompt **5 times** per model to measure consistency:
- Same prompt, same parameters
- Record all outputs
- Measure JSON validity
- Score instruction following
- Check for hallucinations

### 3. Analysis Phase

Compare results across models:
- Aggregate scores
- Identify patterns (what each model does well/poorly)
- Document edge cases
- Note inference speed (tokens/second)

## Success Criteria (Minimum Viable)

A model **PASSES** if it meets ALL of the following:

1. **JSON Validity**: ≥90% of outputs are valid JSON
2. **Instruction Following**: ≥80% of required fields present and correct
3. **Task Decomposition**: ≥70% of steps are logical and executable
4. **Tool Hallucination**: ≤10% of tool calls reference non-existent tools
5. **Consistency**: ≥60% similarity across 5 runs of the same prompt
6. **Hardware Fit**: Runs comfortably within 12GB VRAM

## Evaluation Timeline

- **Week 1**: Setup infrastructure, test first 2 models
- **Week 2**: Test remaining models, document results
- **Week 3**: Comparative analysis, final recommendation

## Output

At the end of Phase 1, we will produce:
1. **Model Comparison Matrix** (scores for each criterion)
2. **Detailed Results** (per model, per test)
3. **Recommendation** (selected model + rationale)
4. **Known Limitations** (what the selected model struggles with)

## Next Steps After Selection

Once a model is selected:
- Document optimal inference parameters
- Create baseline performance benchmarks
- Define fallback strategies for edge cases
- Begin Phase 2: Planner implementation

---

**Important**: This phase is **experimental only**. No production code will be written. All evaluation is done manually or with simple scripts for data collection.

