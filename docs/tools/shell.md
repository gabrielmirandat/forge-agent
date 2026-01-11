# Shell Tool Contract

## Purpose

The `shell` tool provides **command execution semantics** for running external programs and scripts. It is designed for **execution intent** where the goal is to run a command, not just read data.

## Allowed Responsibilities

✅ **Command Execution:**
- Execute shell commands with explicit execution semantics
- Run external programs and scripts
- Capture stdout, stderr, and return codes

✅ **Execution Context:**
- Set working directory (`cwd`)
- Set environment variables (future)
- Set execution timeout

✅ **Process Management:**
- Kill processes on timeout
- Capture process output
- Report execution status

## Forbidden Responsibilities

❌ **Information-Only Operations:**
- Must NOT be used for operations that `system` or `filesystem` can handle
- Must NOT be used to read files (use `filesystem.read_file`)
- Must NOT be used to list directories (use `filesystem.list_directory`)
- Must NOT be used to get system info (use `system.get_info`)

❌ **Unrestricted Execution:**
- Must NEVER execute commands not in `allowed_commands`
- Must NEVER execute commands in `restricted_commands`
- Must NEVER bypass command validation

❌ **Security Violations:**
- Must NEVER execute with elevated privileges
- Must NEVER execute destructive commands without explicit intent
- Must NEVER execute commands that could escape sandbox

## Security Invariants

1. **Command Validation is Mandatory:**
   - Every command MUST be validated against `allowed_commands` and `restricted_commands`
   - Only the base command (first word) is checked
   - Arguments are not validated (by design - command is trusted if base is allowed)

2. **Execution Intent Required:**
   - Shell tool MUST only be used when execution semantics are required
   - If the goal can be achieved with `filesystem` or `system`, those tools MUST be preferred
   - No silent fallback to shell for read-only operations

3. **No Silent Failures:**
   - Command rejections MUST fail with explicit error messages
   - Timeouts MUST be reported explicitly
   - Process failures MUST surface return codes and stderr

## Contract Enforcement

### Command Validation Rules

```python
# ✅ VALID: Allowed command
command = "ls -la ~/repos"
# Base command: "ls"
# Allowed if: "ls" is in allowed_commands

# ❌ INVALID: Restricted command
command = "rm -rf /"
# Base command: "rm"
# Rejected: "rm" is in restricted_commands

# ❌ INVALID: Not in allowed list
command = "curl https://example.com"
# Base command: "curl"
# Rejected: "curl" not in allowed_commands
```

### Operation Examples

**✅ Valid Use Cases:**
```json
// Executing a build script
{
  "tool": "shell",
  "operation": "execute_command",
  "arguments": {
    "command": "npm run build",
    "cwd": "~/repos/forge-agent/frontend"
  }
}

// Running a test suite
{
  "tool": "shell",
  "operation": "execute_command",
  "arguments": {
    "command": "python -m pytest tests/",
    "timeout": 60
  }
}
```

**❌ Invalid Use Cases:**
```json
// ❌ Reading a file - use filesystem tool
{
  "tool": "shell",
  "operation": "execute_command",
  "arguments": {
    "command": "cat src/main.py"
  }
}

// ❌ Listing directory - use filesystem tool
{
  "tool": "shell",
  "operation": "execute_command",
  "arguments": {
    "command": "ls -la"
  }
}

// ❌ System info - use system tool
{
  "tool": "shell",
  "operation": "execute_command",
  "arguments": {
    "command": "uname -a"
  }
}

// ❌ Restricted command
{
  "tool": "shell",
  "operation": "execute_command",
  "arguments": {
    "command": "sudo rm -rf /"
  }
}
```

## Overlap with Other Tools

### vs. Filesystem Tool

**Shell:** Command execution with side effects
- Use for: running scripts, executing programs, process management
- Execution semantics: commands run, processes spawn, side effects occur

**Filesystem:** Structured file operations
- Use for: reading files, listing directories, creating files
- No execution: pure I/O operations

**Rule:** If you need to **execute** something (script, program, command), use `shell`. If you need to **read/write** files without execution, use `filesystem`.

**Priority:** Prefer `filesystem` for read-only operations. Only use `shell` when execution semantics are required.

### vs. System Tool

**Shell:** Command execution
- Use for: running external programs, scripts, commands
- Returns: stdout, stderr, return code

**System:** System introspection (side-effect free)
- Use for: platform info, Python version, system status
- Returns: structured system metadata

**Rule:** If you need to **execute** a command, use `shell`. If you need **system information** without execution, use `system`.

**Priority:** Prefer `system` for introspection. Only use `shell` when you need to run a command.

## Rationale

The shell tool exists to provide **explicit execution semantics** for running external programs. By separating execution from file I/O and system introspection, we:

1. **Enable explicit intent:** Execution is intentional, not accidental
2. **Improve security:** Command validation prevents unauthorized execution
3. **Enable auditability:** All executions are explicit and traceable
4. **Reduce misuse:** Clear boundaries prevent using shell for read-only operations

## Implementation Notes

- Command validation checks only the base command (first word)
- Arguments are not validated (trusted if base command is allowed)
- Working directory (`cwd`) is validated separately (future: should use filesystem tool's path validation)
- Timeout defaults to 30 seconds, configurable per command
- Process output is captured as text (stdout/stderr decoded as UTF-8)

## Internal Assertions

The following assertions MUST hold:

1. **Execution Intent Assertion:**
   ```python
   # Shell tool MUST only be used when execution semantics are required
   assert operation == "execute_command", "Shell tool only supports execute_command"
   ```

2. **Command Validation Assertion:**
   ```python
   # Commands MUST be validated before execution
   assert self._check_command(command), f"Command not allowed: {command}"
   ```

3. **No Silent Fallback:**
   ```python
   # Shell tool MUST NOT silently fallback to other tools
   # If command is invalid, fail explicitly
   ```
