# Supported Models for Tool Calling

This document lists the models that support tool calling in Ollama and how to configure them.

## Models with Tool Calling Support

According to [Ollama's tool calling documentation](https://ollama.com/search?c=tools), the following models support tool calling:

### Recommended Models

1. **hhao/qwen2.5-coder-tools** (default)
   - ✅ Tested and working
   - Excellent tool-calling support, optimized for coding
   - Config: `agent.ollama.yaml` or `agent.ollama.qwen.yaml`
   - To download: `ollama pull hhao/qwen2.5-coder-tools`

2. **qwen2.5:14b**
   - Larger Qwen model with excellent tool-calling support
   - Config: `agent.ollama.qwen14b.yaml`
   - To download: `ollama pull qwen2.5:14b`

### Other Models with Tool Calling Support

- **devstral** (24b) - Best open source model for coding agents
- **granite4** (350m, 1b, 3b) - Improved tool-calling capabilities
- **command-r** (35b) - Optimized for conversational interaction and tool calling
- **mistral-small3.2** (24b) - Improved function calling
- **llama4** (16x17b, 128x17b) - Latest Llama with tool calling

### Models with Limited or No Tool Calling Support

- **llama3.1** - ❌ Removed: Tool calling does not work reliably with this model

## How to Use

1. **Download the model** (if not already available):
   ```bash
   ollama pull <model_name>
   ```

2. **Check available models**:
   ```bash
   ollama list
   ```

3. **Use the appropriate config file**:
   - For hhao/qwen2.5-coder-tools: `config/agent.ollama.yaml` or `config/agent.ollama.qwen.yaml`
   - For qwen3:8b: Use `config/agent.ollama.yaml` and change model to `qwen3:8b`
   - For qwen2.5:14b: `config/agent.ollama.qwen14b.yaml`

4. **Start the application** with the desired config:
   ```bash
   # Using default (hhao/qwen2.5-coder-tools)
   python -m api.app
   
   # Or specify a different config
   CONFIG_PATH=config/agent.ollama.qwen14b.yaml python -m api.app
   ```

## Model Name Variations

Model names in Ollama may vary. Common variations:

- `qwen3:14b` vs `qwen2.5:14b` vs `qwen:14b`

Always verify the exact model name with `ollama list` before configuring.

## Testing Tool Calling

To test if a model supports tool calling, use the test script:

```bash
python scripts/test_qwen3_tool_calling.py  # For hhao/qwen2.5-coder-tools or qwen3:8b
python scripts/test_qwen3_tool_calling.py  # For Qwen models
```

## References

- [Ollama Tool Calling Documentation](https://docs.ollama.com/capabilities/tool-calling#python)
- [Ollama Models with Tool Support](https://ollama.com/search?c=tools)
- [LangChain Tool Calling Documentation](https://docs.langchain.com/oss/python/langchain/models#example-nested-structures)
