"""Planning module - converts goals into executable plans.

The Planner is a deterministic component that uses an LLM as a reasoning engine
to propose execution steps. It does NOT execute actions and does NOT have direct
access to tools. It only produces structured, machine-readable plans.
"""

import hashlib
import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from agent.config.loader import AgentConfig
from agent.llm.base import LLMProvider
from agent.observability import (
    get_logger,
    log_event,
    planner_duration_seconds,
    planner_requests_total,
    planner_validation_errors_total,
    set_plan_id,
    trace_span,
)
from agent.runtime.schema import (
    ALLOWED_OPERATIONS,
    Plan,
    PlanResult,
    PlannerDiagnostics,
    PlanningError,
    LLMCommunicationError,
    JSONExtractionError,
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
        # Get model name from LLM config if available
        self.model_name = getattr(llm, "model", config.llm.model)
        self.logger = get_logger("planner", "planner")

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

        # Get filesystem security constraints from config
        filesystem_constraints = ""
        if hasattr(self.config, "tools") and hasattr(self.config.tools, "filesystem"):
            fs_config = self.config.tools.filesystem
            allowed_paths = fs_config.get("allowed_paths", [])
            if allowed_paths:
                # Expand ~ to show actual paths
                from pathlib import Path
                expanded_paths = [str(Path(p).expanduser().resolve()) for p in allowed_paths]
                filesystem_constraints = f"\n\nFILESYSTEM TOOL SECURITY CONSTRAINTS:\n"
                filesystem_constraints += f"- Allowed paths: {', '.join(expanded_paths)}\n"
                filesystem_constraints += f"- Path format: Use absolute paths (e.g., {expanded_paths[0]}/...) or paths relative to allowed directories\n"
                filesystem_constraints += f"- Tilde (~) expansion: The system will automatically expand ~ to the home directory\n"
                filesystem_constraints += f"- IMPORTANT: All filesystem operations MUST use paths within the allowed directories listed above\n"

        return f"""You are a planning assistant for an autonomous code agent. Your role is to break down high-level goals into structured execution plans.

You have access to the following tools:
{tools_text}
{filesystem_constraints}

TOOL PRIORITY RULES (CRITICAL):
- Prefer structured tools (filesystem, system, git) over execution tools (shell)
- Use filesystem tool for file I/O (read_file, list_directory, write_file)
- Use system tool for system introspection (get_info, get_status)
- Use git tool for Git operations (create_branch, commit, push)
- Use shell tool ONLY when execution semantics are required (running scripts, programs)
- NEVER use shell tool for operations that structured tools can handle

CRITICAL CONSTRAINTS:
1. You can ONLY use the tools and operations listed above
2. You MUST NOT invent or hallucinate tools that don't exist
3. You MUST output ONLY valid JSON - no markdown, no explanations outside JSON
4. You MUST output EXACTLY ONE JSON object - no multiple JSON objects
5. If you are unsure about a tool or operation, return an EMPTY PLAN (steps: []) with notes explaining why
6. Empty plans are valid when no action is needed or the goal cannot be accomplished
7. You cannot execute code directly
8. You cannot access the filesystem directly
9. You must propose steps, not execute them
10. For filesystem operations: Use absolute paths or paths relative to allowed directories. The system will expand ~ automatically.
11. Tool selection priority: filesystem > system > git > shell (use shell only when execution is required)

OUTPUT FORMAT (valid JSON only, EXACTLY ONE JSON object):
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

EXAMPLE - Regular Plan:
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

EXAMPLE - Empty Plan (when no action needed):
{{
  "plan_id": "no-action-needed",
  "objective": "Check if file exists",
  "steps": [],
  "notes": "The requested file already exists and is up to date. No action required."
}}

Remember: Output EXACTLY ONE valid JSON object. No markdown code blocks, no explanations, no multiple JSON objects."""

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
        """Extract JSON from LLM output with strict rules.

        Rules:
        - If NO JSON object is found, this is an error
        - If MORE THAN ONE JSON object is found, this is an error
        - If JSON is wrapped in markdown code blocks, extract it safely
        - The raw LLM response must always be preserved in diagnostics

        Args:
            text: Raw LLM output

        Returns:
            Extracted JSON string (exactly one JSON object)

        Raises:
            JSONExtractionError: If no JSON found or multiple JSON objects found
        """
        # First, try to find JSON in markdown code blocks
        code_block_matches = list(re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL))
        
        if code_block_matches:
            if len(code_block_matches) > 1:
                raise JSONExtractionError(
                    f"Multiple JSON objects found in markdown code blocks ({len(code_block_matches)} found). "
                    "The model must output exactly one JSON object.",
                    raw_response=text
                )
            return code_block_matches[0].group(1).strip()

        # If no code blocks, try to find JSON objects directly
        # Use a more precise pattern to find complete JSON objects
        json_objects = []
        brace_count = 0
        start_pos = -1
        
        for i, char in enumerate(text):
            if char == '{':
                if brace_count == 0:
                    start_pos = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_pos != -1:
                    json_objects.append((start_pos, i + 1))
                    start_pos = -1

        if len(json_objects) == 0:
            raise JSONExtractionError(
                "No JSON object found in LLM output. "
                "The model must output exactly one valid JSON object.",
                raw_response=text
            )
        
        if len(json_objects) > 1:
            raise JSONExtractionError(
                f"Multiple JSON objects found ({len(json_objects)} found). "
                "The model must output exactly one JSON object.",
                raw_response=text
            )

        # Extract the single JSON object
        start, end = json_objects[0]
        return text[start:end].strip()

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

    def _generate_plan_id(self, goal: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate deterministic plan ID.

        Plan ID is based on:
        - objective (normalized)
        - context (normalized and sorted)
        - timestamp bucket (rounded to nearest minute for stability)

        This ensures:
        - Same goal + context in same minute = same ID (stable across retries)
        - Different goals or contexts = different IDs
        - Unique enough for tracing and logging

        Args:
            goal: High-level goal description
            context: Optional context information

        Returns:
            Deterministic plan ID (hex string)
        """
        # Normalize goal (lowercase, strip whitespace)
        normalized_goal = goal.lower().strip()

        # Normalize context (sort keys, convert to string)
        if context:
            normalized_context = json.dumps(context, sort_keys=True)
        else:
            normalized_context = ""

        # Timestamp bucket (round to nearest minute for stability across retries)
        timestamp_bucket = int(time.time() // 60)

        # Create hash input
        hash_input = f"{normalized_goal}|{normalized_context}|{timestamp_bucket}"

        # Generate deterministic hash
        plan_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

        return f"plan-{plan_hash}"

    async def plan(
        self, goal: str, context: Optional[Dict[str, Any]] = None, retry: bool = True
    ) -> PlanResult:
        """Generate execution plan from goal.

        Uses the LLM as a reasoning engine to propose execution steps.
        Does NOT execute actions. Does NOT have direct access to tools.

        Args:
            goal: High-level goal description
            context: Optional context information
            retry: Whether to retry once on failure (default: True)

        Returns:
            PlanResult containing validated Plan and PlannerDiagnostics

        Raises:
            LLMCommunicationError: If LLM communication fails after retries
            JSONExtractionError: If JSON extraction fails after retries
            InvalidPlanError: If plan validation fails after retries
        """
        # Generate deterministic plan ID
        plan_id = self._generate_plan_id(goal, context)
        set_plan_id(plan_id)

        start_time = time.time()
        log_event(self.logger, "planner.plan.started", goal=goal[:100])

        # Build prompts
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(goal, context)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Get LLM config
        temperature = self.config.llm.temperature
        max_tokens = self.config.llm.max_tokens

        # Track diagnostics
        retries_used = 0
        raw_response = ""
        extracted_json = None
        validation_errors = None

        # Attempt planning (with retry if enabled)
        for attempt in range(2 if retry else 1):
            try:
                # Call LLM
                try:
                    raw_response = await self.llm.chat(
                        messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                except Exception as e:
                    if attempt == 0 and retry:
                        retries_used += 1
                        continue  # Retry on first attempt
                    else:
                        # Create diagnostics before raising
                        diagnostics = PlannerDiagnostics(
                            model_name=self.model_name,
                            temperature=temperature,
                            retries_used=retries_used,
                            raw_llm_response=raw_response or str(e),
                            extracted_json=None,
                            validation_errors=[f"LLM communication error: {e}"],
                        )
                        duration = time.time() - start_time
                        planner_requests_total.labels(status="llm_error").inc()
                        planner_duration_seconds.observe(duration)

                        log_event(
                            self.logger,
                            "planner.plan.failed",
                            level="ERROR",
                            plan_id=plan_id,
                            duration_ms=duration * 1000,
                            retries_used=retries_used,
                            error=str(e),
                        )

                        raise LLMCommunicationError(
                            f"LLM request failed after {retries_used} retries: {e}",
                            diagnostics=diagnostics
                        ) from e

                # Extract JSON
                try:
                    extracted_json = self._extract_json(raw_response)
                except JSONExtractionError as e:
                    if attempt == 0 and retry:
                        retries_used += 1
                        # Update prompt with clearer instructions
                        messages[-1]["content"] = (
                            user_prompt
                            + f"\n\nPrevious attempt failed: {e}. Please output EXACTLY ONE valid JSON object."
                        )
                        continue  # Retry
                    else:
                        # Create diagnostics before raising
                        diagnostics = PlannerDiagnostics(
                            model_name=self.model_name,
                            temperature=temperature,
                            retries_used=retries_used,
                            raw_llm_response=raw_response,
                            extracted_json=None,
                            validation_errors=[str(e)],
                        )
                        duration = time.time() - start_time
                        planner_requests_total.labels(status="json_error").inc()
                        planner_duration_seconds.observe(duration)

                        log_event(
                            self.logger,
                            "planner.plan.failed",
                            level="ERROR",
                            plan_id=plan_id,
                            duration_ms=duration * 1000,
                            retries_used=retries_used,
                            error=str(e),
                        )

                        # Re-raise with diagnostics attached
                        raise JSONExtractionError(
                            str(e),
                            raw_response=raw_response,
                            diagnostics=diagnostics
                        ) from e

                # Parse and validate plan
                try:
                    plan = self._parse_and_validate_plan(extracted_json)
                    # Override plan_id with deterministic one
                    plan.plan_id = plan_id

                    # Create successful diagnostics
                    diagnostics = PlannerDiagnostics(
                        model_name=self.model_name,
                        temperature=temperature,
                        retries_used=retries_used,
                        raw_llm_response=raw_response,
                        extracted_json=extracted_json,
                        validation_errors=None,
                    )

                    duration = time.time() - start_time
                    planner_duration_seconds.observe(duration)
                    planner_requests_total.labels(status="success").inc()

                    log_event(
                        self.logger,
                        "planner.plan.completed",
                        plan_id=plan_id,
                        steps_count=len(plan.steps),
                        duration_ms=duration * 1000,
                        retries_used=retries_used,
                    )

                    return PlanResult(plan=plan, diagnostics=diagnostics)

                except InvalidPlanError as e:
                    if attempt == 0 and retry:
                        retries_used += 1
                        validation_errors = e.validation_errors or [str(e)]
                        # Update prompt with validation errors
                        errors_str = "; ".join(validation_errors[:3])  # Limit to first 3
                        messages[-1]["content"] = (
                            user_prompt
                            + f"\n\nPrevious attempt failed validation: {errors_str}. Please output valid JSON matching the schema."
                        )
                        continue  # Retry
                    else:
                        # Create diagnostics before raising
                        diagnostics = PlannerDiagnostics(
                            model_name=self.model_name,
                            temperature=temperature,
                            retries_used=retries_used,
                            raw_llm_response=raw_response,
                            extracted_json=extracted_json,
                            validation_errors=e.validation_errors or [str(e)],
                        )

                        duration = time.time() - start_time
                        planner_requests_total.labels(status="validation_error").inc()
                        planner_validation_errors_total.inc()
                        planner_duration_seconds.observe(duration)

                        log_event(
                            self.logger,
                            "planner.validation.failed",
                            level="ERROR",
                            plan_id=plan_id,
                            duration_ms=duration * 1000,
                            retries_used=retries_used,
                            validation_errors=e.validation_errors or [str(e)],
                        )

                        # Re-raise with diagnostics attached
                        raise InvalidPlanError(
                            str(e),
                            validation_errors=e.validation_errors,
                            diagnostics=diagnostics
                        ) from e

            except (LLMCommunicationError, JSONExtractionError, InvalidPlanError):
                # Re-raise specific errors (diagnostics already created)
                raise
            except Exception as e:
                # Unexpected error
                diagnostics = PlannerDiagnostics(
                    model_name=self.model_name,
                    temperature=temperature,
                    retries_used=retries_used,
                    raw_llm_response=raw_response,
                    extracted_json=extracted_json,
                    validation_errors=[f"Unexpected error: {e}"],
                )
                raise PlanningError(f"Unexpected planning error: {e}") from e

        # Should never reach here, but just in case
        raise PlanningError("Planning failed after all retries")
