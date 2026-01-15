"""Planner output schema definitions.

This module defines the structured output format for the Planner component.
All plans must conform to this schema for validation and execution.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ToolName(str, Enum):
    """Allowed tool names."""

    FILESYSTEM = "filesystem"
    GIT = "git"
    GITHUB = "github"
    SHELL = "shell"
    SYSTEM = "system"


class FilesystemOperation(str, Enum):
    """Allowed filesystem operations."""

    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    LIST_DIRECTORY = "list_directory"
    CREATE_FILE = "create_file"
    DELETE_FILE = "delete_file"
    CHANGE_DIRECTORY = "change_directory"


class GitOperation(str, Enum):
    """Allowed Git operations."""

    INIT = "init"
    CLONE = "clone"
    ADD = "add"
    ADD_ALL = "add_all"
    CREATE_BRANCH = "create_branch"
    CHECKOUT = "checkout"
    COMMIT = "commit"
    PUSH = "push"
    PULL = "pull"
    FETCH = "fetch"
    MERGE = "merge"
    STATUS = "status"
    DIFF = "diff"
    LOG = "log"
    TAG = "tag"
    ADD_REMOTE = "add_remote"


class GitHubOperation(str, Enum):
    """Allowed GitHub operations."""

    CREATE_REPOSITORY = "create_repository"
    CREATE_REPOSITORY_WITH_CLI = "create_repository_with_cli"
    CREATE_PR = "create_pr"
    LIST_PRS = "list_prs"
    COMMENT_PR = "comment_pr"


class ShellOperation(str, Enum):
    """Allowed shell operations."""

    EXECUTE_COMMAND = "execute_command"


class SystemOperation(str, Enum):
    """Allowed system operations."""

    GET_OS_INFO = "get_os_info"
    GET_RUNTIME_INFO = "get_runtime_info"
    GET_AGENT_STATUS = "get_agent_status"
    GET_WORKSPACE_INFO = "get_workspace_info"


# Mapping of tools to their allowed operations
ALLOWED_OPERATIONS: Dict[ToolName, List[str]] = {
    ToolName.FILESYSTEM: [op.value for op in FilesystemOperation],
    ToolName.GIT: [op.value for op in GitOperation],
    ToolName.GITHUB: [op.value for op in GitHubOperation],
    ToolName.SHELL: [op.value for op in ShellOperation],
    ToolName.SYSTEM: [op.value for op in SystemOperation],
}

# Operations that require approval (CREATE/UPDATE/DELETE operations)
# READ operations (read_file, list_directory, get_os_info, etc.) do NOT require approval
APPROVAL_REQUIRED_OPERATIONS: Dict[str, List[str]] = {
    "filesystem": ["write_file", "delete_file", "create_file"],  # read_file, list_directory, change_directory are read-only, no approval needed
    "git": ["init", "commit", "push", "add", "add_all", "checkout", "merge"],  # status, diff, log, etc. are read-only, no approval needed
    "github": ["create_repository", "create_repository_with_cli", "create_pr"],  # list_prs, comment_pr might need approval depending on context
    "shell": [],  # Shell commands are checked dynamically - read-only commands (ls, cat, head, tail, grep, etc.) don't need approval
}


# Read-only shell commands that don't require approval
READ_ONLY_SHELL_COMMANDS = {
    "ls", "cat", "head", "tail", "grep", "find", "wc", "sort", "uniq", 
    "cut", "sed", "awk", "echo", "pwd", "whoami", "date", "git",  # git status, diff, log, etc. are read-only
    "python", "python3",  # Can be used for read-only scripts
    "npm",  # Can be used for read-only operations
}


def is_read_only_shell_command(command: str) -> bool:
    """Check if a shell command is read-only (doesn't modify anything).
    
    Args:
        command: Command string to check
        
    Returns:
        True if command is read-only, False otherwise
    """
    if not command:
        return False
    
    # Extract base command (first word, ignoring shell operators)
    import re
    parts = re.split(r'\s*(?:&&|\|\||\||;)\s*', command)
    if not parts:
        return False
    
    cmd_base = parts[0].strip().split()[0] if parts[0] else ""
    
    # Check if it's a read-only command
    if cmd_base in READ_ONLY_SHELL_COMMANDS:
        # Additional check: git commands that modify state
        if cmd_base == "git" and len(parts[0].split()) > 1:
            git_op = parts[0].split()[1]
            # These git operations modify state and need approval
            modifying_git_ops = {"add", "commit", "push", "init", "checkout", "merge", "reset", "rebase"}
            if git_op in modifying_git_ops:
                return False
        return True
    
    return False


def requires_approval(tool: str, operation: str, arguments: dict | None = None) -> bool:
    """Check if an operation requires approval.

    Args:
        tool: Tool name
        operation: Operation name
        arguments: Optional operation arguments (for shell commands to check if read-only)

    Returns:
        True if operation requires approval, False otherwise
    """
    # For shell commands, check if it's read-only
    if tool == "shell" and operation == "execute_command":
        if arguments:
            command = arguments.get("command", "")
            if is_read_only_shell_command(command):
                return False  # Read-only commands don't need approval
            else:
                return True  # Non-read-only shell commands need approval
    
    # Check the approval list for other tools
    return operation in APPROVAL_REQUIRED_OPERATIONS.get(tool, [])


def get_operations_requiring_approval(plan: "Plan") -> List["PlanStep"]:
    """Get list of steps that require approval.

    Args:
        plan: Plan to check

    Returns:
        List of steps that require approval
    """
    steps_requiring_approval = []
    for step in plan.steps:
        if requires_approval(step.tool.value, step.operation, step.arguments):
            steps_requiring_approval.append(step)
    return steps_requiring_approval


def is_restricted_command(command: str, restricted_commands: set[str]) -> bool:
    """Check if a shell command is restricted.
    
    Args:
        command: Command string to check
        restricted_commands: Set of restricted command names
        
    Returns:
        True if command is restricted, False otherwise
    """
    if not command:
        return False
    cmd_base = command.split()[0] if command else ""
    return cmd_base in restricted_commands


def get_restricted_steps(plan: "Plan", restricted_commands: set[str]) -> List["PlanStep"]:
    """Get list of steps that use restricted commands.
    
    Args:
        plan: Plan to check
        restricted_commands: Set of restricted command names
        
    Returns:
        List of steps that use restricted commands
    """
    restricted_steps = []
    for step in plan.steps:
        if step.tool.value == "shell" and step.operation == "execute_command":
            command = step.arguments.get("command", "")
            if is_restricted_command(command, restricted_commands):
                restricted_steps.append(step)
    return restricted_steps


class PlanStep(BaseModel):
    """A single step in an execution plan."""

    step_id: int = Field(..., description="Sequential step identifier", gt=0)
    tool: ToolName = Field(..., description="Tool name to use")
    operation: str = Field(..., description="Operation to perform")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Operation arguments")
    rationale: str = Field(..., min_length=1, description="Brief explanation of this step")

    @field_validator("operation", mode="after")
    @classmethod
    def validate_operation(cls, v: str, info) -> str:
        """Validate that operation is allowed for the specified tool."""
        # Get tool from the model instance (info.data is the model instance)
        tool = info.data.get("tool") if hasattr(info, "data") else None
        if tool is None:
            # If tool is not set yet, skip validation (will be validated when tool is set)
            return v

        # Handle both enum and string tool values
        if isinstance(tool, ToolName):
            tool_name = tool
        elif isinstance(tool, str):
            try:
                tool_name = ToolName(tool)
            except ValueError:
                # Tool validation will fail separately, just return operation
                return v
        else:
            return v

        allowed_ops = ALLOWED_OPERATIONS.get(tool_name, [])

        if v not in allowed_ops:
            allowed_str = ", ".join(allowed_ops)
            raise ValueError(
                f"Operation '{v}' is not allowed for tool '{tool_name.value}'. "
                f"Allowed operations: {allowed_str}"
            )

        return v


class Plan(BaseModel):
    """Complete execution plan generated by the Planner.

    Supports both regular plans (with steps) and empty plans (no steps).
    Empty plans are valid when the Planner determines no action is needed.
    """

    plan_id: str = Field(..., min_length=1, description="Unique plan identifier")
    objective: str = Field(..., min_length=1, description="High-level goal description")
    steps: List[PlanStep] = Field(
        default_factory=list, description="Ordered list of execution steps (empty for empty plans)"
    )
    estimated_time_seconds: Optional[int] = Field(
        default=None, ge=0, description="Estimated execution time"
    )
    notes: Optional[str] = Field(
        default=None, description="Additional notes or constraints (required for empty plans)"
    )

    @field_validator("steps")
    @classmethod
    def validate_step_ids(cls, v: List[PlanStep]) -> List[PlanStep]:
        """Validate that step IDs are sequential and unique."""
        if not v:
            # Empty plan is valid - no step ID validation needed
            return v

        step_ids = [step.step_id for step in v]
        expected_ids = list(range(1, len(v) + 1))

        if step_ids != expected_ids:
            raise ValueError(
                f"Step IDs must be sequential starting from 1. "
                f"Got: {step_ids}, expected: {expected_ids}"
            )

        return v


class PlannerDiagnostics(BaseModel):
    """Diagnostics information from planning process.

    Captures raw LLM responses, retry attempts, and validation errors
    for debugging and observability.
    """

    model_name: str = Field(..., description="LLM model name used")
    temperature: float = Field(..., description="Temperature setting used")
    retries_used: int = Field(..., ge=0, description="Number of retries used")
    raw_llm_response: str = Field(..., description="Raw LLM response text")
    extracted_json: Optional[str] = Field(
        default=None, description="Extracted JSON from LLM response (if available)"
    )
    validation_errors: Optional[List[str]] = Field(
        default=None, description="List of validation errors (if any)"
    )


class PlanResult(BaseModel):
    """Complete planning result.

    Contains the validated plan and diagnostics information.
    """

    plan: Plan = Field(..., description="Validated execution plan")
    diagnostics: PlannerDiagnostics = Field(..., description="Planning diagnostics")


# Execution result schemas


class StepExecutionResult(BaseModel):
    """Result of executing a single plan step."""

    step_id: int = Field(..., description="Step identifier")
    tool: str = Field(..., description="Tool name used")
    operation: str = Field(..., description="Operation performed")
    arguments: Dict[str, Any] = Field(..., description="Arguments used")
    success: bool = Field(..., description="Whether step succeeded")
    output: Optional[Any] = Field(default=None, description="Step output (if successful)")
    error: Optional[str] = Field(default=None, description="Error message (if failed)")
    retries_attempted: int = Field(..., ge=0, description="Number of retries attempted")
    started_at: float = Field(..., description="Unix timestamp when step started")
    finished_at: float = Field(..., description="Unix timestamp when step finished")


class RollbackStepResult(BaseModel):
    """Result of a rollback step."""

    step_id: int = Field(..., description="Step identifier that was rolled back")
    tool: str = Field(..., description="Tool name used for rollback")
    operation: str = Field(..., description="Rollback operation performed")
    success: bool = Field(..., description="Whether rollback succeeded")
    error: Optional[str] = Field(default=None, description="Error message (if rollback failed)")
    started_at: float = Field(..., description="Unix timestamp when rollback started")
    finished_at: float = Field(..., description="Unix timestamp when rollback finished")


class ExecutionResult(BaseModel):
    """Complete execution result.

    Contains results for all executed steps, success status, and rollback information.
    """

    plan_id: str = Field(..., description="Plan identifier")
    objective: str = Field(..., description="Plan objective")
    steps: List[StepExecutionResult] = Field(..., description="Step execution results")
    success: bool = Field(..., description="Whether execution succeeded")
    stopped_at_step: Optional[int] = Field(
        default=None, description="Step ID where execution stopped (if failed)"
    )
    rollback_attempted: bool = Field(..., description="Whether rollback was attempted")
    rollback_success: Optional[bool] = Field(
        default=None, description="Whether rollback succeeded (if attempted)"
    )
    rollback_steps: List[RollbackStepResult] = Field(
        default_factory=list, description="Rollback step results"
    )
    started_at: float = Field(..., description="Unix timestamp when execution started")
    finished_at: float = Field(..., description="Unix timestamp when execution finished")


# Error schemas


class PlanningError(Exception):
    """Base exception for planning errors."""

    pass


class LLMCommunicationError(PlanningError):
    """Raised when LLM communication fails."""

    def __init__(self, message: str, diagnostics: Optional[PlannerDiagnostics] = None):
        super().__init__(message)
        self.diagnostics = diagnostics


class JSONExtractionError(PlanningError):
    """Raised when JSON extraction from LLM response fails."""

    def __init__(self, message: str, diagnostics: Optional[PlannerDiagnostics] = None):
        super().__init__(message)
        self.diagnostics = diagnostics


class InvalidPlanError(PlanningError):
    """Raised when plan validation fails."""

    def __init__(self, message: str, diagnostics: Optional[PlannerDiagnostics] = None):
        super().__init__(message)
        self.diagnostics = diagnostics


class OperationNotSupportedError(Exception):
    """Raised when a tool operation is not supported."""

    def __init__(self, tool_name: str, operation: str):
        self.tool_name = tool_name
        self.operation = operation
        super().__init__(f"Operation '{operation}' is not supported by tool '{tool_name}'")


class ToolNotFoundError(Exception):
    """Raised when a tool is not found in the registry."""

    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' not found in registry")


class ExecutionError(Exception):
    """Base exception for execution errors."""

    pass


class ExecutionPolicy(BaseModel):
    """Execution policy for controlling retries and rollback behavior."""

    max_retries_per_step: int = Field(default=0, ge=0, description="Maximum retries per step")
    retry_delay_seconds: float = Field(
        default=0.0, ge=0.0, description="Delay between retries in seconds"
    )
    rollback_on_failure: bool = Field(
        default=False, description="Whether to attempt rollback on failure"
    )
