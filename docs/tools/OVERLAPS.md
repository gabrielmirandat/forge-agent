# Tool Overlaps and Priority Rules

This document explains intentional overlaps between tools and the priority rules that govern when to use each tool.

## Core Principle

**Prefer structured tools over execution tools. Use execution tools only when execution semantics are required.**

## Overlap Matrix

### System Tool vs Shell Tool

**Overlap:** Getting system information

**System Tool (`system.get_info`):**
- ✅ Side-effect-free
- ✅ Structured output (dict)
- ✅ Fast (no process spawn)
- ✅ Deterministic

**Shell Tool (`shell.execute_command("uname -a")`):**
- ❌ Spawns process
- ❌ Text output (requires parsing)
- ❌ Slower
- ❌ Non-deterministic (depends on system state)

**Priority Rule:** **ALWAYS prefer `system` tool for system introspection.**

**Example:**
```json
// ✅ CORRECT: Use system tool
{
  "tool": "system",
  "operation": "get_info",
  "arguments": {}
}

// ❌ WRONG: Using shell for system info
{
  "tool": "shell",
  "operation": "execute_command",
  "arguments": {"command": "uname -a"}
}
```

### Filesystem Tool vs Shell Tool

**Overlap:** Reading files and listing directories

**Filesystem Tool (`filesystem.read_file`, `filesystem.list_directory`):**
- ✅ Structured output
- ✅ Path validation (automatic)
- ✅ No process spawn
- ✅ Fast and deterministic

**Shell Tool (`shell.execute_command("cat file.txt")`, `shell.execute_command("ls -la")`):**
- ❌ Spawns process
- ❌ Text output (requires parsing)
- ❌ No automatic path validation
- ❌ Slower and non-deterministic

**Priority Rule:** **ALWAYS prefer `filesystem` tool for file I/O operations.**

**Example:**
```json
// ✅ CORRECT: Use filesystem tool
{
  "tool": "filesystem",
  "operation": "read_file",
  "arguments": {"path": "src/main.py"}
}

// ❌ WRONG: Using shell for file read
{
  "tool": "shell",
  "operation": "execute_command",
  "arguments": {"command": "cat src/main.py"}
}
```

### Git Tool vs Shell Tool

**Overlap:** Git operations

**Git Tool (`git.create_branch`, `git.commit`, `git.push`):**
- ✅ Structured API
- ✅ Explicit operations
- ✅ Auditable
- ✅ Validation built-in

**Shell Tool (`shell.execute_command("git branch")`, `shell.execute_command("git commit")`):**
- ❌ Arbitrary command execution
- ❌ Text output (requires parsing)
- ❌ No structured validation
- ❌ Harder to audit

**Priority Rule:** **ALWAYS prefer `git` tool for Git operations.**

**Example:**
```json
// ✅ CORRECT: Use git tool
{
  "tool": "git",
  "operation": "create_branch",
  "arguments": {
    "repo_path": "~/repos/forge-agent",
    "branch_name": "feature/new-feature"
  }
}

// ❌ WRONG: Using shell for Git operations
{
  "tool": "shell",
  "operation": "execute_command",
  "arguments": {"command": "git checkout -b feature/new-feature"}
}
```

## When to Use Shell Tool

The shell tool should **ONLY** be used when:

1. **Execution semantics are required:**
   - Running build scripts (`npm run build`)
   - Executing test suites (`pytest tests/`)
   - Running programs that have side effects

2. **Structured tools don't support the operation:**
   - Custom scripts that don't have structured tool equivalents
   - One-off commands that aren't worth creating a structured tool for

3. **The operation requires process execution:**
   - Programs that need to run as separate processes
   - Commands that interact with external services

## Enforcement

These priority rules are enforced through:

1. **Contract Documentation:** Each tool's contract explicitly states when to use it and when not to
2. **Code Assertions:** Tools reject operations that violate their contract
3. **Planner Instructions:** The Planner's system prompt instructs it to prefer structured tools
4. **Validation:** Tools fail explicitly when misused

## Rationale

By enforcing these priority rules, we:

1. **Improve Security:** Structured tools have explicit validation and boundaries
2. **Enable Auditability:** Structured operations are easier to trace and audit
3. **Reduce Attack Surface:** Fewer process spawns means fewer attack vectors
4. **Improve Reliability:** Structured operations are more predictable than command execution
5. **Enable Better Error Handling:** Structured tools return structured errors, not raw text
