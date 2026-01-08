# Phase 1: Setup Guide

This guide helps you set up the evaluation environment for Phase 1 model testing using **Docker containers** (no host installation required).

## Prerequisites

- Ubuntu 24.04
- NVIDIA RTX 5070 (12GB VRAM)
- 32GB RAM
- Docker Engine 24.0+
- Docker Compose 2.20+
- NVIDIA Container Toolkit (for GPU support)

## Install Docker & NVIDIA Container Toolkit

```bash
# Install Docker (if not already installed)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify GPU access in Docker
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu24.04 nvidia-smi
```

## Option 1: Using Ollama in Docker (Recommended)

Ollama is the easiest way to test multiple models locally, and it runs in a container.

### Start Ollama Container

```bash
cd /home/gabriel-miranda/repos/forge-agent/phase1-model-research

# Start Ollama service
docker-compose up -d ollama

# Check logs
docker-compose logs -f ollama
```

### Pull Models

```bash
# Pull models to evaluate (models are stored in Docker volume)
docker exec phase1-ollama ollama pull llama3.1:8b-instruct
docker exec phase1-ollama ollama pull qwen2.5-coder:7b
docker exec phase1-ollama ollama pull deepseek-coder:6.7b

# Verify models are available
docker exec phase1-ollama ollama list
```

### Test a Model

```bash
# Quick test
docker exec phase1-ollama ollama run llama3.1:8b-instruct "Hello, can you output JSON?"
```

### Using Ollama API

Ollama exposes a REST API on `http://localhost:11434` (from host) or `http://ollama:11434` (from other containers).

---

## Option 2: Using LocalAI in Docker

LocalAI is more flexible and also runs in a container.

### Start LocalAI Container

```bash
# Start LocalAI service (with profile)
docker-compose --profile localai up -d localai

# Check logs
docker-compose logs -f localai
```

### Download Models

You'll need to download GGUF quantized models and place them in the `models/` directory (mounted as volume).

### Configuration

Create model configuration files in `configs/` directory.

---

## Running Evaluation Scripts

### Option A: Run Scripts in Container (Recommended)

```bash
# Build and start evaluator container
docker-compose --profile evaluator up -d evaluator

# Run evaluation inside container
docker exec phase1-evaluator python evaluate_model.py --model llama3.1:8b-instruct --test 1

# Validate results
docker exec phase1-evaluator python validate_output.py results/llama3.1-8b-instruct/ --test 1
```

### Option B: Run Scripts on Host (Requires Python)

If you prefer to run scripts on the host:

```bash
# Install Python dependencies (only if running on host)
pip install httpx pydantic

# Run evaluation (connects to containerized Ollama)
python evaluate_model.py --model llama3.1:8b-instruct --test 1
```

---

## Verify GPU Access in Containers

```bash
# Check NVIDIA driver on host
nvidia-smi

# Test GPU access in Docker
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu24.04 nvidia-smi

# Check if Ollama container sees GPU
docker exec phase1-ollama nvidia-smi
```

---

## Directory Structure

After setup, your structure should look like:

```
phase1-model-research/
├── README.md
├── EVALUATION_PLAN.md
├── TEST_PROMPTS.md
├── EVALUATION_CRITERIA.md
├── RESULTS_TEMPLATE.md
├── COMPARISON_MATRIX.md
├── SETUP_GUIDE.md (this file)
├── QUICK_START.md
├── docker-compose.yml
├── Dockerfile.evaluator
├── evaluate_model.py (evaluation script)
├── validate_output.py (JSON validator)
├── results/ (created automatically)
│   ├── llama3.1-8b-instruct/
│   │   ├── test1_run1.json
│   │   ├── test1_run2.json
│   │   └── ...
│   └── ...
└── configs/ (optional, for LocalAI)
    └── model-config.yaml
```

---

## Quick Start Checklist

- [ ] Install Docker and NVIDIA Container Toolkit
- [ ] Verify GPU access in Docker
- [ ] Start Ollama container: `docker-compose up -d ollama`
- [ ] Pull at least one model: `docker exec phase1-ollama ollama pull llama3.1:8b-instruct`
- [ ] Test model: `docker exec phase1-ollama ollama run llama3.1:8b-instruct "test"`
- [ ] Build evaluator container: `docker-compose --profile evaluator build evaluator`
- [ ] Run first evaluation test (see QUICK_START.md)

---

## Next Steps

1. **Start Ollama container**: `docker-compose up -d ollama`
2. **Pull your first model**: `docker exec phase1-ollama ollama pull llama3.1:8b-instruct`
3. **Run evaluation**: See QUICK_START.md for detailed steps
4. **Execute your first test**: 
   ```bash
   docker exec phase1-evaluator python evaluate_model.py --model llama3.1:8b-instruct --test 1
   ```

---

## Container Management

### Start Services

```bash
# Start Ollama only
docker-compose up -d ollama

# Start Ollama + LocalAI
docker-compose --profile localai up -d

# Start all services including evaluator
docker-compose --profile evaluator up -d
```

### Stop Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (deletes downloaded models!)
docker-compose down -v
```

### View Logs

```bash
# Ollama logs
docker-compose logs -f ollama

# All services
docker-compose logs -f
```

### Check Container Status

```bash
docker-compose ps
```

---

## Troubleshooting

### GPU not accessible in container
- Verify NVIDIA Container Toolkit is installed
- Check: `docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu24.04 nvidia-smi`
- Restart Docker: `sudo systemctl restart docker`

### Model won't load / OOM errors
- Try a smaller quantization (Q4_K_M instead of Q5_K_M)
- Reduce context length in model config
- Check GPU memory: `nvidia-smi`
- Stop other GPU containers

### Slow inference
- Check GPU utilization: `nvidia-smi`
- Verify container is using GPU: `docker exec phase1-ollama nvidia-smi`
- Check container logs for errors

### Connection refused errors
- Verify container is running: `docker-compose ps`
- Check port mapping: `docker-compose port ollama 11434`
- From host, use `localhost:11434`
- From container, use `ollama:11434`

### JSON parsing errors
- Use `validate_output.py` to debug
- Check model temperature (should be low: 0.1-0.3)
- Verify system prompt is being used correctly

---

## Resources

- [Ollama Documentation](https://github.com/ollama/ollama)
- [LocalAI Documentation](https://localai.io/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)

