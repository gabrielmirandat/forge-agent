# Forge Agent — Model Configuration

## Active Models (3 router tiers)

All models are served via Ollama (Docker, port 11434). Model selection is automatic via `config/router.yaml`.

| Tier   | Model             | Size    | VRAM    | Use case |
|--------|-------------------|---------|---------|----------|
| fast   | qwen3:8b          | 5.2 GB  | ~5 GB   | Simple queries: git, list files, quick answers |
| smart  | qwen3:14b         | 9.3 GB  | ~9 GB   | Most dev tasks: implement, debug, refactor (DEFAULT) |
| max    | qwen3:30b-a3b     | ~19 GB  | GPU+RAM | Complex tasks: full architecture, lengthy docs |

**qwen3:30b-a3b** is a Mixture-of-Experts model (30B total / 3.3B active params).
GPU handles active compute (~3.3B), RAM stores inactive weights.

## Pull commands

```bash
ollama pull qwen3:8b        # fast tier (already installed)
ollama pull qwen3:14b       # smart tier (already installed)
ollama pull qwen3:30b-a3b   # max tier (~19GB, Q4_K_M)
```

## Router configuration

Edit `config/router.yaml` to adjust:
- Which keywords route to which tier
- Fallback model order per tier (`preferred_models`)
- Enable/disable automatic routing (`enabled: true/false`)

## Checking available models

```bash
# Via API:
curl http://localhost:11434/api/tags | python3 -m json.tool

# Via forge-agent API:
curl http://localhost:8000/api/v1/config/models/available
curl http://localhost:8000/api/v1/config/models/tiers
curl http://localhost:8000/api/v1/config/router
```
