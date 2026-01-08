# Project Structure

## Overview

O projeto está organizado em fases, cada uma com suas próprias pastas de testes e outputs para manter o código principal limpo.

## Estrutura de Diretórios

```
forge-agent/
├── agent/                      # Código principal do agente
│   ├── config/                 # Gerenciamento de configuração
│   ├── llm/                    # Providers de LLM
│   ├── runtime/                # Planner e Executor
│   └── tools/                  # Implementações de tools
│
├── phase1-model-research/      # Fase 1: Avaliação de modelos
│   ├── tests/                  # Scripts de avaliação
│   │   ├── evaluate_model.py
│   │   └── validate_output.py
│   ├── outputs/                # Resultados das avaliações
│   │   ├── llama3.1-8b/
│   │   ├── qwen2.5-coder-7b/
│   │   └── deepseek-coder-6.7b/
│   ├── COMPARISON_MATRIX.md    # Matriz de comparação
│   └── docker-compose.yml      # Setup Docker
│
├── phase2-planner/             # Fase 2: Implementação do Planner
│   ├── tests/                  # Testes do Planner
│   │   └── test_phase2.py
│   ├── outputs/                # Outputs dos testes
│   ├── PHASE2_IMPLEMENTATION.md # Documentação
│   └── README.md
│
├── phase3-executor/            # Fase 3: Executor (futuro)
│   ├── tests/                 # Testes do Executor
│   └── outputs/               # Outputs dos testes
│
├── config/                     # Arquivos de configuração
├── .venv/                      # Ambiente virtual
└── requirements.txt            # Dependências Python
```

## Convenções

### Pastas por Fase

Cada fase tem:
- `tests/` - Scripts e testes específicos da fase
- `outputs/` - Resultados, logs e artefatos gerados
- `README.md` - Documentação da fase

### Código Principal

O código de produção fica em `agent/` e não deve ser poluído com:
- Testes específicos de fase
- Outputs temporários
- Scripts de avaliação
- Artefatos de build

### Gitignore

O `.gitignore` está configurado para ignorar:
- `phase*/outputs/` - Todos os outputs das fases
- `phase*/tests/__pycache__/` - Cache Python dos testes
- `phase*/tests/*.log` - Logs de teste

## Executando Testes

```bash
# Fase 2
source .venv/bin/activate
python3 phase2-planner/tests/test_phase2.py

# Fase 1 (requer Docker)
cd phase1-model-research
docker-compose up -d ollama
python3 tests/evaluate_model.py --model qwen2.5-coder:7b --all
```

## Adicionando Nova Fase

Para adicionar uma nova fase:

1. Criar pasta `phaseN-name/`
2. Criar subpastas `tests/` e `outputs/`
3. Adicionar `README.md` explicando a fase
4. Atualizar este documento
