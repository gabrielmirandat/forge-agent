# ProviderStrategy uses the model provider’s native structured output generation. 
# This is more reliable but only works with providers that support native structured output:

# As of langchain 1.0, simply passing a schema (e.g., response_format=ContactInfo) will default to ProviderStrategy 
# if the model supports native structured output. It will fall back to ToolStrategy otherwise.

from langchain.agents.structured_output import ProviderStrategy

agent = create_agent(
    model="gpt-4o",
    response_format=ProviderStrategy(ContactInfo)
)