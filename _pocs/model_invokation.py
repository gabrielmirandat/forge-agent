# Tool calling - calling external tools (like databases queries or API calls) and use results in their responses.
# Structured output - where the model’s response is constrained to follow a defined format.
# Multimodality - process and return data other than text, such as images, audio, and video.
# Reasoning - models perform multi-step reasoning to arrive at a conclusion.

from langchain_ollama import ChatOllama

model = ChatOllama(model="qwen3:8b", base_url="http://localhost:11434", temperature=0.0)

result = model.invoke("What is the weather in Tokyo?")

print(result)