from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# Use open-source model with Ollama
model = ChatOllama(
    model="qwen3:8b",
    base_url="http://localhost:11434",
    temperature=0.0,
)

# Stream response and adapt to content_blocks format
# Ollama returns AIMessageChunk, we adapt it to content_blocks format
for chunk in model.stream([HumanMessage(content="What color is the sky?")]):
    # Adapt Ollama streaming format to content_blocks format
    # Create content_blocks list from chunk attributes and attach to chunk
    content_blocks = []
    
    # Check if chunk has content (text)
    if hasattr(chunk, "content") and chunk.content:
        content_blocks.append({
            "type": "text",
            "text": chunk.content
        })
    
    # Check if chunk has tool_calls
    if hasattr(chunk, "tool_calls") and chunk.tool_calls:
        for tool_call in chunk.tool_calls:
            content_blocks.append({
                "type": "tool_call_chunk",
                "tool_call": tool_call
            })
    
    # Check for reasoning in response_metadata (if available)
    if hasattr(chunk, "response_metadata"):
        metadata = chunk.response_metadata
        if metadata and isinstance(metadata, dict) and "reasoning" in metadata:
            content_blocks.append({
                "type": "reasoning",
                "reasoning": metadata["reasoning"]
            })
    
    # Process each block according to the original logic
    # Note: We process content_blocks directly instead of attaching to chunk
    # because chunk is a Pydantic model that doesn't allow setting content_blocks
    for block in content_blocks:
        if block["type"] == "reasoning" and (reasoning := block.get("reasoning")):
            print(f"Reasoning: {reasoning}")
        elif block["type"] == "tool_call_chunk":
            print(f"Tool call chunk: {block}")
        elif block["type"] == "text":
            print(block["text"], end="", flush=True)
        else:
            print(block)