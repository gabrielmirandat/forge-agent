.PHONY: help setup start stop pull-models test validate clean logs

help:
	@echo "Phase 1 Model Research - Docker Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make setup          - Install Docker and NVIDIA Container Toolkit (run once)"
	@echo "  make start          - Start Ollama container"
	@echo "  make stop           - Stop all containers"
	@echo ""
	@echo "Models:"
	@echo "  make pull-models    - Pull all evaluation models"
	@echo "  make list-models    - List available models"
	@echo ""
	@echo "Evaluation:"
	@echo "  make build-eval     - Build evaluator container"
	@echo "  make test MODEL=llama3.1:8b-instruct TEST=1"
	@echo "  make test-all MODEL=llama3.1:8b-instruct"
	@echo "  make validate MODEL=llama3.1-8b-instruct TEST=1"
	@echo ""
	@echo "Utilities:"
	@echo "  make logs           - View Ollama logs"
	@echo "  make clean          - Stop and remove containers (keeps volumes)"
	@echo "  make clean-all      - Stop and remove everything including volumes"

setup:
	@echo "Installing Docker..."
	@curl -fsSL https://get.docker.com -o get-docker.sh
	@sudo sh get-docker.sh
	@echo "Installing NVIDIA Container Toolkit..."
	@distribution=$$(. /etc/os-release;echo $$ID$$VERSION_ID) && \
	curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - && \
	curl -s -L https://nvidia.github.io/nvidia-docker/$$distribution/nvidia-docker.list | \
		sudo tee /etc/apt/sources.list.d/nvidia-docker.list && \
	sudo apt-get update && \
	sudo apt-get install -y nvidia-container-toolkit && \
	sudo systemctl restart docker
	@echo "Verifying GPU access..."
	@docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu24.04 nvidia-smi

start:
	docker-compose up -d ollama
	@echo "Ollama started. Waiting for service..."
	@sleep 3
	@docker-compose ps

stop:
	docker-compose down

pull-models:
	@echo "Pulling evaluation models..."
	docker exec ollama ollama pull llama3.1:8b-instruct
	docker exec ollama ollama pull qwen2.5-coder:7b
	docker exec ollama ollama pull deepseek-coder:6.7b
	@echo "Done. Available models:"
	@docker exec ollama ollama list

list-models:
	docker exec ollama ollama list

build-eval:
	docker-compose --profile evaluator build evaluator
	docker-compose --profile evaluator up -d evaluator

test:
	@if [ -z "$(MODEL)" ] || [ -z "$(TEST)" ]; then \
		echo "Usage: make test MODEL=llama3.1:8b-instruct TEST=1"; \
		exit 1; \
	fi
	docker exec evaluator python evaluate_model.py --model $(MODEL) --test $(TEST)

test-all:
	@if [ -z "$(MODEL)" ]; then \
		echo "Usage: make test-all MODEL=llama3.1:8b-instruct"; \
		exit 1; \
	fi
	docker exec evaluator python evaluate_model.py --model $(MODEL) --all

validate:
	@if [ -z "$(MODEL)" ]; then \
		echo "Usage: make validate MODEL=llama3.1-8b-instruct [TEST=1]"; \
		exit 1; \
	fi
	@if [ -z "$(TEST)" ]; then \
		docker exec evaluator python validate_output.py results/$(MODEL)/ --all; \
	else \
		docker exec evaluator python validate_output.py results/$(MODEL)/ --test $(TEST); \
	fi

logs:
	docker-compose logs -f ollama

clean:
	docker-compose down

clean-all:
	docker-compose down -v
	@echo "WARNING: This removed all volumes including downloaded models!"

