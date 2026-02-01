# Many models are capable of performing multi-step reasoning to arrive at a conclusion. 
# This involves breaking down complex problems into smaller, more manageable steps.
# If supported by the underlying model, you can surface this reasoning process to better 
# understand how the model arrived at its final answer.

# Stream reasoning output

# Complete reasoning output
for chunk in model.stream("Why do parrots have colorful feathers?"):
    reasoning_steps = [r for r in chunk.content_blocks if r["type"] == "reasoning"]
    print(reasoning_steps if reasoning_steps else chunk.text)

# Depending on the model, you can sometimes specify the level of effort it should put into reasoning. Similarly, you can request that the model turn off reasoning entirely. This may take the form of categorical “tiers” of reasoning (e.g., 'low' or 'high') or integer token budgets.
# For details, see the integrations page or reference for your respective chat model.