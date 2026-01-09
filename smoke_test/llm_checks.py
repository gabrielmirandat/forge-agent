"""LLM provider availability checks."""

import httpx

from smoke_test.config import BACKEND_URL


def check_ollama_available() -> bool:
    """Check if Ollama is available.

    Returns:
        True if Ollama is running and accessible
    """
    try:
        response = httpx.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print("[LLM Check] ✓ Ollama is available")
            return True
        else:
            print(f"[LLM Check] Ollama returned HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"[LLM Check] Ollama not available: {e}")
        print("[LLM Check]   Start Ollama with: ollama serve")
        return False


def check_model_available(model: str = "qwen2.5-coder:7b") -> bool:
    """Check if required model is available in Ollama.

    Args:
        model: Model name to check

    Returns:
        True if model is available
    """
    try:
        response = httpx.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code != 200:
            return False

        data = response.json()
        models = [m.get("name", "") for m in data.get("models", [])]

        if model in models:
            print(f"[LLM Check] ✓ Model {model} is available")
            return True
        else:
            print(f"[LLM Check] Model {model} not found")
            print(f"[LLM Check]   Available models: {', '.join(models[:5])}")
            print(f"[LLM Check]   Pull model with: ollama pull {model}")
            return False
    except Exception as e:
        print(f"[LLM Check] Failed to check model availability: {e}")
        return False

