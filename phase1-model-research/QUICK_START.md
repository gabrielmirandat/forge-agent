# Phase 1: Quick Start Guide

Get started with model evaluation in 5 minutes using **Docker containers** (no host installation).

## Prerequisites

- Docker and Docker Compose installed
- NVIDIA Container Toolkit installed (for GPU support)
- See SETUP_GUIDE.md for installation instructions

## Step 1: Start Ollama Container (30 seconds)

```bash
cd /home/gabriel-miranda/repos/forge-agent/phase1-model-research

# Start Ollama service
docker-compose up -d ollama

# Verify it's running
docker-compose ps
```

## Step 2: Pull Your First Model (1-5 minutes, depends on internet)

```bash
# Pull model (stored in Docker volume)
docker exec phase1-ollama ollama pull llama3.1:8b-instruct

# Verify model is available
docker exec phase1-ollama ollama list
```

## Step 3: Build Evaluator Container (1 minute)

```bash
# Build the evaluator container with Python dependencies
docker-compose --profile evaluator build evaluator

# Start evaluator container
docker-compose --profile evaluator up -d evaluator
```

## Step 4: Run Your First Test (30 seconds)

```bash
# Run test 1 (Simple File Operation) 5 times
docker exec phase1-evaluator python evaluate_model.py --model llama3.1:8b-instruct --test 1
```

This will:
- Run test 1 (Simple File Operation) 5 times
- Save results to `results/llama3.1-8b-instruct/` (mounted volume)
- Show inference speed and success rate

## Step 5: Validate Results (10 seconds)

```bash
# Validate results
docker exec phase1-evaluator python validate_output.py results/llama3.1-8b-instruct/ --test 1
```

This shows:
- JSON validity rate
- Structure compliance
- Tool accuracy
- Hallucination rate
- Consistency across runs

## Next Steps

1. **Run all tests for this model**:
   ```bash
   docker exec phase1-evaluator python evaluate_model.py --model llama3.1:8b-instruct --all
   ```

2. **Validate all results**:
   ```bash
   docker exec phase1-evaluator python validate_output.py results/llama3.1-8b-instruct/ --all
   ```

3. **Document results** in `RESULTS_TEMPLATE.md`

4. **Test another model**:
   ```bash
   docker exec phase1-ollama ollama pull qwen2.5-coder:7b
   docker exec phase1-evaluator python evaluate_model.py --model qwen2.5-coder:7b --all
   ```

5. **Compare models** in `COMPARISON_MATRIX.md`

## Tips

- **Check GPU usage**: `watch -n 1 nvidia-smi`
- **Monitor Ollama**: `docker exec phase1-ollama ollama ps`
- **View container logs**: `docker-compose logs -f ollama`
- **Adjust temperature**: `--temperature 0.1` (lower = more consistent)
- **More runs**: `--runs 10` (for better consistency measurement)
- **Access results from host**: Results are in `./results/` directory (mounted volume)

## Troubleshooting

**"Connection refused" error**:
- Make sure Ollama container is running: `docker-compose ps`
- Check logs: `docker-compose logs ollama`
- Verify port mapping: `docker-compose port ollama 11434`

**Model not found**:
- Check available models: `docker exec phase1-ollama ollama list`
- Pull the model: `docker exec phase1-ollama ollama pull <model-name>`

**Slow inference**:
- Check GPU: `nvidia-smi` (should show GPU usage)
- Verify container sees GPU: `docker exec phase1-ollama nvidia-smi`
- Check container logs for CUDA errors

**Container won't start**:
- Check Docker daemon: `sudo systemctl status docker`
- Verify NVIDIA Container Toolkit: `docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu24.04 nvidia-smi`

**JSON validation errors**:
- This is expected! Document the error rate
- Some models need better prompting (we'll optimize in Phase 2)

**Access results from host**:
- Results are saved in `./results/` directory (mounted as volume)
- You can view/edit them directly from the host machine

---

**Ready?** Start with Step 1 and you'll have your first results in under 10 minutes!

