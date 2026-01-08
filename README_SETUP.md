# Setup Instructions

## ⚠️ Requisito: python3-venv

Para criar o ambiente virtual, você precisa instalar o pacote `python3-venv`:

```bash
sudo apt install python3.12-venv
```

## Setup Completo

Após instalar o `python3-venv`, execute:

```bash
# Criar e configurar venv
./setup_venv.sh

# Ativar o venv
source .venv/bin/activate

# Rodar testes
python3 test_phase2.py
```

## Arquivos Criados

- `setup_venv.sh` - Script de setup automático
- `test_phase2.py` - Script de testes da Fase 2
- `SETUP.md` - Documentação de setup
- `requirements.txt` - Dependências do projeto (já existia)

## Testes Disponíveis

O script `test_phase2.py` testa:

1. ✅ Schema validation (Plan, PlanStep)
2. ✅ Tool/operation validation
3. ✅ OllamaProvider initialization
4. ✅ Planner imports

**Nota**: Testes de integração completa requerem:
- Ollama rodando (`docker-compose up ollama`)
- Modelo `qwen2.5-coder:7b` baixado
- Arquivo de config válido (`config/agent.yaml`)

