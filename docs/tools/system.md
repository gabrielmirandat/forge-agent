# System Tool Contract

## Purpose

The `system` tool provides **side-effect-free system introspection** for querying platform information, Python environment, and system status. It is designed for **read-only metadata queries** without execution semantics.

## Allowed Responsibilities

✅ **System Introspection:**
- Query platform information (OS, architecture, Python version)
- Get system status (operational state, health)
- Retrieve environment metadata (no mutation)

✅ **Side-Effect-Free Operations:**
- All operations MUST be read-only
- No file system changes
- No process execution
- No network I/O

## Forbidden Responsibilities

❌ **Execution Semantics:**
- Must NEVER execute commands or scripts
- Must NEVER invoke shell interpreters
- Must NEVER run external programs
- Must NEVER spawn processes

❌ **File Operations:**
- Must NOT read or write files (use `filesystem` tool)
- Must NOT list directories (use `filesystem` tool)
- Must NOT access file contents

❌ **State Mutation:**
- Must NEVER modify system state
- Must NEVER change environment variables
- Must NEVER alter configuration

❌ **Network Operations:**
- Must NOT perform network I/O
- Must NOT make HTTP requests
- Must NOT access remote resources

## Security Invariants

1. **Side-Effect-Free Guarantee:**
   - All operations MUST be read-only
   - No state changes are allowed
   - No external processes are spawned

2. **No Execution Intent:**
   - System tool MUST reject any request that implies execution
   - Operations that require execution MUST fail explicitly
   - No silent fallback to shell tool

3. **Explicit Operation Validation:**
   - Only operations in `allowed_operations` are permitted
   - Unknown operations MUST raise `OperationNotSupportedError`
   - No implicit operation mapping

## Contract Enforcement

### Operation Validation Rules

```python
# ✅ VALID: System introspection
operation = "get_info"
# Returns: platform, Python version, architecture

operation = "get_status"
# Returns: system status, operational state

# ❌ INVALID: Execution attempt
operation = "execute_command"
# Rejected: System tool does not support execution

# ❌ INVALID: File operation
operation = "read_file"
# Rejected: Use filesystem tool
```

### Operation Examples

**✅ Valid Use Cases:**
```json
// Get system information
{
  "tool": "system",
  "operation": "get_info",
  "arguments": {}
}

// Get system status
{
  "tool": "system",
  "operation": "get_status",
  "arguments": {}
}
```

**❌ Invalid Use Cases:**
```json
// ❌ Execution attempt - use shell tool
{
  "tool": "system",
  "operation": "execute_command",
  "arguments": {"command": "uname -a"}
}

// ❌ File operation - use filesystem tool
{
  "tool": "system",
  "operation": "read_file",
  "arguments": {"path": "/etc/os-release"}
}

// ❌ Unknown operation
{
  "tool": "system",
  "operation": "get_memory_info",
  "arguments": {}
}
```

## Overlap with Other Tools

### vs. Shell Tool

**System:** Side-effect-free introspection
- Use for: platform info, Python version, system status
- No execution: pure metadata queries

**Shell:** Command execution with side effects
- Use for: running commands, executing programs, process management
- Execution semantics: commands run, processes spawn

**Rule:** If you need **system information** without execution, use `system`. If you need to **execute a command** to get information, use `shell`.

**Priority:** Prefer `system` for introspection. Only use `shell` when you need to run a command.

**Example Overlap:**
- `system.get_info()` vs `shell.execute_command("uname -a")`
- **Prefer:** `system.get_info()` (structured, side-effect-free)
- **Use shell only if:** You need command-specific output format or execution semantics

### vs. Filesystem Tool

**System:** System metadata
- Use for: platform info, Python version, architecture

**Filesystem:** File and directory operations
- Use for: reading files, listing directories, file I/O

**Rule:** If you need **system metadata**, use `system`. If you need **file contents**, use `filesystem`.

**Example Overlap:**
- `system.get_info()` vs `filesystem.read_file("/etc/os-release")`
- **Prefer:** `system.get_info()` (structured, tool-specific)
- **Use filesystem only if:** You need raw file contents or file-specific operations

## Rationale

The system tool exists to provide **safe, side-effect-free system introspection** without execution semantics. By separating introspection from execution and file I/O, we:

1. **Prevent accidental execution:** No commands can be run through system tool
2. **Enable structured queries:** System info is returned as structured data, not text
3. **Improve reliability:** Side-effect-free operations are more predictable
4. **Reduce attack surface:** No execution means no code injection vectors

## Implementation Notes

- All operations are synchronous and side-effect-free
- Operations return structured dictionaries, not raw text
- No external processes are spawned
- No file I/O is performed (all data comes from Python's `platform` and `sys` modules)

## Internal Assertions

The following assertions MUST hold:

1. **Side-Effect-Free Assertion:**
   ```python
   # System tool MUST be side-effect free
   # No file I/O, no process execution, no state mutation
   assert operation in ["get_info", "get_status"], \
       "System tool only supports introspection operations"
   ```

2. **No Execution Intent:**
   ```python
   # System tool MUST reject execution attempts
   if "execute" in operation.lower() or "command" in operation.lower():
       raise OperationNotSupportedError(
           self.name, operation,
           "System tool does not support execution. Use shell tool for commands."
       )
   ```

3. **Structured Output Assertion:**
   ```python
   # System tool MUST return structured data, not raw text
   assert isinstance(output, dict), "System tool must return structured output"
   ```
