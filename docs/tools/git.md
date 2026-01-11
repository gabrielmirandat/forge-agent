# Git Tool Contract

## Purpose

The `git` tool provides **structured Git repository operations** for version control tasks. It is designed for **explicit Git operations** with clear boundaries and safety constraints.

## Allowed Responsibilities

✅ **Repository Operations:**
- Create and manage branches
- Create commits with explicit messages
- Push branches to remotes
- Get repository status and diffs

✅ **Structured Git Commands:**
- All operations use Git's structured API
- Operations are explicit and auditable
- No arbitrary command execution

## Forbidden Responsibilities

❌ **Arbitrary Git Commands:**
- Must NOT execute arbitrary `git` commands via shell
- Must NOT bypass Git's structured API
- Must NOT use shell tool to run git commands

❌ **File Operations:**
- Must NOT read or write files directly (use `filesystem` tool)
- Must NOT modify files outside Git's control
- Must NOT perform file I/O operations

❌ **Execution Semantics:**
- Must NOT execute scripts or programs
- Must NOT invoke external tools
- Must NOT spawn processes (except Git itself)

## Security Invariants

1. **Structured Operations Only:**
   - All Git operations MUST use the tool's structured API
   - No arbitrary command execution
   - No shell tool fallback for Git operations

2. **Explicit Intent:**
   - Branch names MUST follow naming conventions (if configured)
   - Commit messages MUST be explicit
   - Push operations MUST specify branch and remote

3. **No Silent Failures:**
   - Invalid repository paths MUST fail explicitly
   - Git errors MUST be surfaced, not hidden
   - Operation failures MUST return explicit error messages

## Contract Enforcement

### Operation Examples

**✅ Valid Use Cases:**
```json
// Create a branch
{
  "tool": "git",
  "operation": "create_branch",
  "arguments": {
    "repo_path": "~/repos/forge-agent",
    "branch_name": "feature/new-feature"
  }
}

// Create a commit
{
  "tool": "git",
  "operation": "commit",
  "arguments": {
    "repo_path": "~/repos/forge-agent",
    "message": "Add new feature",
    "files": ["src/main.py", "tests/test_main.py"]
  }
}
```

**❌ Invalid Use Cases:**
```json
// ❌ Arbitrary git command - use structured operations
{
  "tool": "git",
  "operation": "execute_command",
  "arguments": {
    "command": "git log --oneline"
  }
}

// ❌ File operation - use filesystem tool
{
  "tool": "git",
  "operation": "read_file",
  "arguments": {
    "path": ".git/config"
  }
}
```

## Overlap with Other Tools

### vs. Shell Tool

**Git:** Structured Git operations
- Use for: branches, commits, pushes, status, diffs
- Structured API: explicit operations with validation

**Shell:** Command execution
- Use for: running arbitrary commands, scripts, programs
- Execution semantics: commands run, processes spawn

**Rule:** If you need **Git operations**, use `git` tool. If you need to **execute arbitrary commands**, use `shell` tool.

**Priority:** Prefer `git` tool for all Git operations. Only use `shell` for Git commands if the structured API doesn't support the operation.

### vs. Filesystem Tool

**Git:** Version control operations
- Use for: branches, commits, pushes, repository state

**Filesystem:** File and directory operations
- Use for: reading files, listing directories, file I/O

**Rule:** If you need **Git operations**, use `git` tool. If you need **file I/O**, use `filesystem` tool.

## Rationale

The Git tool exists to provide **structured, auditable Git operations** with explicit boundaries. By separating Git operations from file I/O and command execution, we:

1. **Enable explicit intent:** Git operations are intentional and traceable
2. **Improve security:** Structured API prevents arbitrary command execution
3. **Enable auditability:** All Git operations are explicit and logged
4. **Reduce misuse:** Clear boundaries prevent using shell for Git operations

## Implementation Notes

- All operations use Git's Python API (via `subprocess` or `GitPython` if available)
- Branch names can be prefixed automatically (if `branch_prefix` is configured)
- Commit messages can use templates (if `commit_message_template` is configured)
- Auto-commit can be enabled/disabled per operation
