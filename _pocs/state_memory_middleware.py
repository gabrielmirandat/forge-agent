# Agents maintain conversation history automatically through the message state. 
# You can also configure the agent to use a custom state schema to remember additional information during the conversation.

# Information stored in the state can be thought of as the short-term memory of the agent:
# Custom state schemas must extend AgentState as a TypedDict.
# There are two ways to define custom state:
# - Via middleware (preferred)
# - Via state_schema on create_agent

# efining state via middleware
# Use middleware to define custom state when your custom state needs to be accessed by specific middleware hooks and tools attached to said middleware.

# As of langchain 1.0, custom state schemas must be TypedDict types. Pydantic models and dataclasses are no longer supported. See the v1 migration guide for more details.

# Defining custom state via middleware is preferred over defining it via state_schema on create_agent because it allows you to keep state extensions 
# conceptually scoped to the relevant middleware and tools. 
# state_schema is still supported for backwards compatibility on create_agent.

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from typing import Any


class CustomState(AgentState):
    user_preferences: dict

class CustomMiddleware(AgentMiddleware):
    state_schema = CustomState
    tools = [tool1, tool2]

    def before_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
        ...

agent = create_agent(
    model,
    tools=tools,
    middleware=[CustomMiddleware()]
)

# The agent can now track additional state beyond messages
result = agent.invoke({
    "messages": [{"role": "user", "content": "I prefer technical explanations"}],
    "user_preferences": {"style": "technical", "verbosity": "detailed"},
})