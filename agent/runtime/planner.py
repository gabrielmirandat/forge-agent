"""Planning module - converts goals into executable plans.

The Planner is a deterministic component that uses an LLM as a reasoning engine
to propose execution steps. It does NOT execute actions and does NOT have direct
access to tools. It only produces structured, machine-readable plans.
"""

import json
import re
import uuid
from typing import Any, Dict, List, Optional

from agent.config.loader import AgentConfig
from agent.llm.base import LLMProvider
from agent.runtime.schema import (
    ALLOWED_OPERATIONS,
    Plan,
    PlanningError,
    InvalidPlanError,
    ToolName,
)


class Planner:
    """Converts high-level goals into structured execution plans.

    The Planner uses an LLM strictly as a reasoning engine to propose execution
    steps. It validates tool availability and safety constraints, but never
    executes actions itself.
    """

    def __init__(self, config: AgentConfig, llm: LLMProvider):
        """Initialize planner.

        Args:
            config: Agent configuration
            llm: LLM provider instance (used as reasoning engine only)
        """
        self.config = config
        self.llm = llm

    def _build_system_prompt(self) -> str:
        """Build system prompt for planning.

        Returns:
            System prompt string
        """
        # Build tool list
        tool_descriptions = []
        for tool_name, operations in ALLOWED_OPERATIONS.items():
            ops_str = ", ".join(operations)
            tool_descriptions.append(f"- {tool_name.value}: {ops_str}")

        tools_text = "\n".join(tool_descriptions)

        return f"""You are a planning assistant for an autonomous code agent. Your role is to break down high-level goals into structured execution plans.

You have access to the following tools:
{tools_text}

CRITICAL CONSTRAINTS:
1. You can ONLY use the tools and operations listed above
2. You MUST NOT invent or hallucinate tools that don't exist
3. You MUST output ONLY valid JSON - no markdown, no explanations outside JSON
4. If you are unsure about a tool or operation, return an empty plan with a clear reason
5. You cannot execute code directly
6. You cannot access the filesystem directly
7. You must propose steps, not execute them

OUTPUT FORMAT (valid JSON only):
{{
  "plan_id": "unique-identifier",
  "objective": "clear description of the goal",
  "steps": [
    {{
      "step_id": 1,
      "tool": "filesystem",
      "operation": "read_file",
      "arguments": {{"path": "src/main.py"}},
      "rationale": "Read the file to understand its contents"
    }}
  ],
  "estimated_time_seconds": 60,
  "notes": "optional notes or constraints"
}}

EXAMPLE:
{{
  "plan_id": "backup-main-file",
  "objective": "Create a backup of src/main.py",
  "steps": [
    {{
      "step_id": 1,
      "tool": "filesystem",
      "operation": "read_file",
      "arguments": {{"path": "src/main.py"}},
      "rationale": "Read the source file"
    }},
    {{
      "step_id": 2,
      "tool": "filesystem",
      "operation": "write_file",
      "arguments": {{"path": "src/main.py.backup", "content": "{{content_from_step_1}}"}},
      "rationale": "Write backup file"
    }}
  ],
  "estimated_time_seconds": 10
}}

Remember: Output ONLY valid JSON. No markdown code blocks, no explanations."""

    def _build_user_prompt(self, goal: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Build user prompt with goal and context.

        Args:
            goal: High-level goal description
            context: Optional context information

        Returns:
            User prompt string
        """
        prompt = f"Goal: {goal}\n\n"

        if context:
            prompt += "Context:\n"
            for key, value in context.items():
                prompt += f"- {key}: {value}\n"
            prompt += "\n"

        prompt += "Generate a plan to accomplish this goal. Output ONLY valid JSON."

        return prompt

    def _extract_json(self, text: str) -> str:
        """Extract JSON from LLM output.

        Handles cases where JSON is wrapped in markdown code blocks.

        Args:
            text: Raw LLM output

        Returns:
            Extracted JSON string

        Raises:
            InvalidPlanError: If no valid JSON found
        """
        # Try to find JSON in markdown code blocks
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()

        # Try to find JSON object directly
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            return json_match.group(0).strip()

        # If no JSON found, raise error
        raise InvalidPlanError(
            "No valid JSON found in LLM output. "
            "The model must output valid JSON only.",
            validation_errors=[f"Raw output: {text[:200]}..."]
        )

    def _parse_and_validate_plan(self, json_str: str) -> Plan:
        """Parse and validate plan from JSON string.

        Args:
            json_str: JSON string to parse

        Returns:
            Validated Plan object

        Raises:
            InvalidPlanError: If parsing or validation fails
        """
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise InvalidPlanError(
                f"Invalid JSON format: {e}",
                validation_errors=[f"JSON error: {str(e)}"]
            ) from e

        try:
            plan = Plan(**data)
            return plan
        except Exception as e:
            # Collect validation errors
            errors = []
            if hasattr(e, "errors"):
                errors = [str(err) for err in e.errors()]
            else:
                errors = [str(e)]

            raise InvalidPlanError(
                f"Plan validation failed: {e}",
                validation_errors=errors
            ) from e

    async def plan(
        self, goal: str, context: Optional[Dict[str, Any]] = None, retry: bool = True
    ) -> Plan:
        """Generate execution plan from goal.

        Uses the LLM as a reasoning engine to propose execution steps.
        Does NOT execute actions. Does NOT have direct access to tools.

        Args:
            goal: High-level goal description
            context: Optional context information
            retry: Whether to retry once on failure (default: True)

        Returns:
            Validated Plan object

        Raises:
            PlanningError: If planning fails after retries
            InvalidPlanError: If plan validation fails
        """
        # Build prompts
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(goal, context)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Get LLM config for temperature
        temperature = self.config.llm.temperature
        max_tokens = self.config.llm.max_tokens

        # Call LLM
        try:
            response = await self.llm.chat(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            if retry:
                # Retry once
                try:
                    response = await self.llm.chat(
                        messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                except Exception as retry_error:
                    raise PlanningError(
                        f"LLM request failed after retry: {retry_error}"
                    ) from retry_error
            else:
                raise PlanningError(f"LLM request failed: {e}") from e

        # Extract and validate JSON
        try:
            json_str = self._extract_json(response)
            plan = self._parse_and_validate_plan(json_str)
            return plan
        except InvalidPlanError as e:
            if retry:
                # Retry once with clearer error message
                retry_messages = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": user_prompt
                        + f"\n\nPrevious attempt failed: {e}. Please output ONLY valid JSON.",
                    },
                ]

                try:
                    retry_response = await self.llm.chat(
                        retry_messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    json_str = self._extract_json(retry_response)
                    plan = self._parse_and_validate_plan(json_str)
                    return plan
                except Exception as retry_error:
                    raise InvalidPlanError(
                        f"Plan validation failed after retry: {retry_error}",
                        validation_errors=getattr(retry_error, "validation_errors", [])
                    ) from retry_error
            else:
                raise
