"""Execution module - executes plans and manages tool calls.

The Executor is the ONLY component allowed to invoke tools. It executes plans
deterministically, supports controlled retries and rollback via ExecutionPolicy,
and produces fully auditable results.
"""

import asyncio
import time
from typing import Any, Dict

from agent.config.loader import AgentConfig
from agent.observability import (
    execution_duration_seconds,
    execution_runs_total,
    execution_step_duration_seconds,
    execution_steps_total,
    get_logger,
    log_event,
    set_plan_id,
    set_run_id,
    set_step_id,
    trace_span,
)
from agent.runtime.schema import (
    ExecutionError,
    ExecutionPolicy,
    ExecutionResult,
    OperationNotSupportedError,
    Plan,
    RollbackStepResult,
    StepExecutionResult,
    ToolNotFoundError,
)
from agent.tools.base import ToolRegistry


class Executor:
    """Executes plans and manages tool call lifecycle.

    The Executor is the ONLY component allowed to invoke tools. It executes
    plans deterministically, supports controlled retries and rollback via
    ExecutionPolicy, and produces fully auditable results.

    Core principles:
    - Executor is dumb and literal
    - Retries and rollback are policy-driven, not intelligent
    - No parallelism
    - No LLM calls
    - All behavior is explicit and auditable
    """

    def __init__(
        self,
        config: AgentConfig,
        tool_registry: ToolRegistry,
        execution_policy: ExecutionPolicy | None = None,
    ):
        """Initialize executor.

        Args:
            config: Agent configuration
            tool_registry: Registry of available tools
            execution_policy: Execution policy for retries and rollback.
                If None, defaults to no retries and no rollback (Phase 3 behavior).
        """
        self.config = config
        self.tool_registry = tool_registry
        self.policy = execution_policy or ExecutionPolicy()  # Default: no retries, no rollback
        self.logger = get_logger("executor", "executor")

    async def execute(self, plan: Plan) -> ExecutionResult:
        """Execute a plan deterministically.

        This is the ONLY component that can invoke tools. Executes steps sequentially,
        stops immediately on first failure, and returns a fully auditable result.

        Args:
            plan: Validated Plan to execute

        Returns:
            ExecutionResult with complete execution details

        Note:
            Execution always returns a result, even on failure. Failures are
            recorded in the ExecutionResult, not raised as exceptions.
        """
        execution_start = time.time()
        set_plan_id(plan.plan_id)

        log_event(
            self.logger,
            "executor.execution.started",
            plan_id=plan.plan_id,
            steps_count=len(plan.steps),
        )

        # Handle empty plans
        if not plan.steps:
            execution_end = time.time()
            duration = execution_end - execution_start
            execution_runs_total.labels(status="success").inc()
            execution_duration_seconds.observe(duration)

            log_event(
                self.logger,
                "executor.execution.stopped",
                plan_id=plan.plan_id,
                success=True,
                duration_ms=duration * 1000,
                reason="empty_plan",
            )

            return ExecutionResult(
                plan_id=plan.plan_id,
                objective=plan.objective,
                steps=[],
                success=True,
                stopped_at_step=None,
                started_at=execution_start,
                finished_at=execution_end,
            )

        step_results: list[StepExecutionResult] = []
        stopped_at_step: int | None = None

        # Execute steps sequentially
        for step in plan.steps:
            set_step_id(step.step_id)
            step_result = await self._execute_step_with_retries(step)
            step_results.append(step_result)

            # Stop immediately on first failure (after retries)
            if not step_result.success:
                stopped_at_step = step.step_id
                break

        execution_end = time.time()
        success = stopped_at_step is None
        duration = execution_end - execution_start

        # Perform rollback if enabled and execution failed
        rollback_attempted = False
        rollback_success: bool | None = None
        rollback_steps: list[RollbackStepResult] = []

        if not success and self.policy.rollback_on_failure:
            rollback_attempted = True
            rollback_steps, rollback_success = await self._rollback_steps(step_results)

        # Emit metrics and logs
        status_label = "success" if success else "failure"
        execution_runs_total.labels(status=status_label).inc()
        execution_duration_seconds.observe(duration)

        log_event(
            self.logger,
            "executor.execution.stopped",
            plan_id=plan.plan_id,
            success=success,
            duration_ms=duration * 1000,
            stopped_at_step=stopped_at_step,
            steps_executed=len(step_results),
            rollback_attempted=rollback_attempted,
            rollback_success=rollback_success,
        )

        return ExecutionResult(
            plan_id=plan.plan_id,
            objective=plan.objective,
            steps=step_results,
            success=success,
            stopped_at_step=stopped_at_step,
            rollback_attempted=rollback_attempted,
            rollback_success=rollback_success,
            rollback_steps=rollback_steps,
            started_at=execution_start,
            finished_at=execution_end,
        )

    async def _execute_step_with_retries(self, step) -> StepExecutionResult:
        """Execute a single plan step with retries if configured.

        Args:
            step: PlanStep to execute

        Returns:
            StepExecutionResult with execution details including retry count
        """
        retries_attempted = 0
        step_start = time.time()

        # Try execution with retries
        for attempt in range(self.policy.max_retries_per_step + 1):
            step_result = await self._execute_step(step, step_start)

            # If successful, return immediately
            if step_result.success:
                step_result.retries_attempted = retries_attempted
                return step_result

            # If this was not the last attempt, wait and retry
            if attempt < self.policy.max_retries_per_step:
                retries_attempted += 1
                if self.policy.retry_delay_seconds > 0:
                    await asyncio.sleep(self.policy.retry_delay_seconds)
            else:
                # Last attempt failed, return with retry count
                step_result.retries_attempted = retries_attempted
                return step_result

        # Should never reach here, but just in case
        step_result.retries_attempted = retries_attempted
        return step_result

    async def _execute_step(self, step, step_start: float | None = None) -> StepExecutionResult:
        """Execute a single plan step.

        Args:
            step: PlanStep to execute

        Returns:
            StepExecutionResult with execution details

        Args:
            step: PlanStep to execute
            step_start: Optional start time (if None, uses current time)

        Returns:
            StepExecutionResult with execution details

        Note:
            This method catches all exceptions and converts them to StepExecutionResult
            with success=False. The Executor does NOT raise exceptions - all failures
            are recorded in the result.
        """
        if step_start is None:
            step_start = time.time()
        tool_name = step.tool.value if hasattr(step.tool, "value") else str(step.tool)
        operation = step.operation
        arguments = step.arguments

        log_event(
            self.logger,
            "executor.step.started",
            step_id=step.step_id,
            tool=tool_name,
            operation=operation,
        )

        try:
            # Validate tool exists
            tool = self.tool_registry.validate_tool(tool_name)

            # Execute tool operation
            tool_result = await tool.execute(operation=operation, arguments=arguments)

            step_end = time.time()
            duration = step_end - step_start

            status_label = "success" if tool_result.success else "failure"
            execution_steps_total.labels(tool=tool_name, operation=operation, status=status_label).inc()
            execution_step_duration_seconds.labels(tool=tool_name, operation=operation).observe(duration)

            log_event(
                self.logger,
                "executor.step.completed",
                step_id=step.step_id,
                tool=tool_name,
                operation=operation,
                success=tool_result.success,
                duration_ms=duration * 1000,
            )

            return StepExecutionResult(
                step_id=step.step_id,
                tool=tool_name,
                operation=operation,
                arguments=arguments,
                success=tool_result.success,
                output=tool_result.output if tool_result.success else None,
                error=tool_result.error if not tool_result.success else None,
                retries_attempted=0,  # Will be set by _execute_step_with_retries
                started_at=step_start,
                finished_at=step_end,
            )

        except ToolNotFoundError as e:
            step_end = time.time()
            duration = step_end - step_start
            execution_steps_total.labels(tool=tool_name, operation=operation, status="error").inc()
            execution_step_duration_seconds.labels(tool=tool_name, operation=operation).observe(duration)

            log_event(
                self.logger,
                "executor.step.failed",
                level="ERROR",
                step_id=step.step_id,
                tool=tool_name,
                operation=operation,
                error=f"Tool not found: {e.tool_name}",
                duration_ms=duration * 1000,
            )

            return StepExecutionResult(
                step_id=step.step_id,
                tool=tool_name,
                operation=operation,
                arguments=arguments,
                success=False,
                output=None,
                error=f"Tool not found: {e.tool_name}",
                retries_attempted=0,  # Will be set by _execute_step_with_retries
                started_at=step_start,
                finished_at=step_end,
            )

        except OperationNotSupportedError as e:
            step_end = time.time()
            duration = step_end - step_start
            execution_steps_total.labels(tool=tool_name, operation=operation, status="error").inc()
            execution_step_duration_seconds.labels(tool=tool_name, operation=operation).observe(duration)

            log_event(
                self.logger,
                "executor.step.failed",
                level="ERROR",
                step_id=step.step_id,
                tool=tool_name,
                operation=operation,
                error=f"Operation not supported: {e}",
                duration_ms=duration * 1000,
            )

            return StepExecutionResult(
                step_id=step.step_id,
                tool=tool_name,
                operation=operation,
                arguments=arguments,
                success=False,
                output=None,
                error=f"Operation '{e.operation}' not supported by tool '{e.tool_name}'",
                retries_attempted=0,  # Will be set by _execute_step_with_retries
                started_at=step_start,
                finished_at=step_end,
            )

        except Exception as e:
            # Catch any unexpected exceptions
            step_end = time.time()
            duration = step_end - step_start
            execution_steps_total.labels(tool=tool_name, operation=operation, status="error").inc()
            execution_step_duration_seconds.labels(tool=tool_name, operation=operation).observe(duration)

            log_event(
                self.logger,
                "executor.step.failed",
                level="ERROR",
                step_id=step.step_id,
                tool=tool_name,
                operation=operation,
                error=f"Unexpected error: {str(e)}",
                duration_ms=duration * 1000,
            )

            return StepExecutionResult(
                step_id=step.step_id,
                tool=tool_name,
                operation=operation,
                arguments=arguments,
                success=False,
                output=None,
                error=f"Unexpected error: {str(e)}",
                retries_attempted=0,  # Will be set by _execute_step_with_retries
                started_at=step_start,
                finished_at=step_end,
            )

    async def _rollback_steps(
        self, step_results: list[StepExecutionResult]
    ) -> tuple[list[RollbackStepResult], bool]:
        """Rollback previously successful steps in reverse order.

        Args:
            step_results: List of step execution results (only successful steps are rolled back)

        Returns:
            Tuple of (rollback_step_results, rollback_success)
            rollback_success is True only if ALL rollback attempts succeeded
        """
        rollback_results: list[RollbackStepResult] = []
        rollback_success = True

        # Filter to only successful steps and reverse order
        successful_steps = [sr for sr in step_results if sr.success]
        steps_to_rollback = list(reversed(successful_steps))

        for step_result in steps_to_rollback:
            rollback_start = time.time()

            try:
                tool = self.tool_registry.get(step_result.tool)
                if tool is None:
                    # Tool not found - cannot rollback
                    rollback_end = time.time()
                    rollback_results.append(
                        RollbackStepResult(
                            step_id=step_result.step_id,
                            tool=step_result.tool,
                            operation=step_result.operation,
                            success=False,
                            error=f"Tool not found: {step_result.tool}",
                            started_at=rollback_start,
                            finished_at=rollback_end,
                        )
                    )
                    rollback_success = False
                    continue

                # Attempt rollback
                try:
                    rollback_result = await tool.rollback(
                        operation=step_result.operation,
                        arguments=step_result.arguments,
                        execution_output=step_result.output,
                    )
                    rollback_end = time.time()

                    rollback_results.append(
                        RollbackStepResult(
                            step_id=step_result.step_id,
                            tool=step_result.tool,
                            operation=step_result.operation,
                            success=rollback_result.success,
                            error=rollback_result.error if not rollback_result.success else None,
                            started_at=rollback_start,
                            finished_at=rollback_end,
                        )
                    )

                    if not rollback_result.success:
                        rollback_success = False

                except NotImplementedError:
                    # Tool does not support rollback - skip and record
                    rollback_end = time.time()
                    rollback_results.append(
                        RollbackStepResult(
                            step_id=step_result.step_id,
                            tool=step_result.tool,
                            operation=step_result.operation,
                            success=False,
                            error=f"Tool '{step_result.tool}' does not support rollback",
                            started_at=rollback_start,
                            finished_at=rollback_end,
                        )
                    )
                    # Rollback skip is not considered a failure for rollback_success
                    # (tool simply doesn't support it)

            except Exception as e:
                # Unexpected error during rollback
                rollback_end = time.time()
                rollback_results.append(
                    RollbackStepResult(
                        step_id=step_result.step_id,
                        tool=step_result.tool,
                        operation=step_result.operation,
                        success=False,
                        error=f"Unexpected rollback error: {str(e)}",
                        started_at=rollback_start,
                        finished_at=rollback_end,
                    )
                )
                rollback_success = False

        return rollback_results, rollback_success

