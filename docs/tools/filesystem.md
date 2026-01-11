# Filesystem Tool Contract

## Purpose

The `filesystem` tool provides **structured, read-only and write operations** on the local filesystem. It is designed for **deterministic file and directory operations** with explicit path validation and security boundaries.

## Allowed Responsibilities

✅ **Read Operations:**
- Read file contents as text or binary
- List directory contents with structured metadata
- Check file/directory existence and properties

✅ **Write Operations:**
- Create files and directories
- Write file contents (overwrite or append)
- Delete files and directories

✅ **Path Resolution:**
- Expand `~` to home directory
- Resolve relative paths to absolute paths
- Validate paths against security constraints

## Forbidden Responsibilities

❌ **Execution Semantics:**
- Must NEVER execute commands or scripts
- Must NEVER invoke shell interpreters
- Must NEVER run external programs

❌ **System Introspection:**
- Must NOT provide system information (use `system` tool)
- Must NOT check process status
- Must NOT query environment variables

❌ **Network Operations:**
- Must NOT perform network I/O
- Must NOT access remote filesystems (unless explicitly mounted)

❌ **Security Violations:**
- Must NEVER access paths outside `allowed_paths`
- Must NEVER follow symlinks outside allowed boundaries
- Must NEVER bypass path validation

## Security Invariants

1. **Path Validation is Mandatory:**
   - Every path MUST be validated against `allowed_paths` and `restricted_paths`
   - Paths are resolved and normalized before validation
   - Symlinks are resolved, but final path must still be allowed

2. **No Silent Failures:**
   - Path violations MUST fail with explicit error messages
   - Permission errors MUST be surfaced, not hidden
   - Invalid operations MUST raise `OperationNotSupportedError`

3. **Side-Effect Transparency:**
   - All mutations (write, create, delete) are explicit
   - No hidden state changes
   - No caching or optimization that hides operations

## Contract Enforcement

### Path Validation Rules

```python
# ✅ VALID: Path within allowed directory
path = "~/repos/forge-agent/src/main.py"
# Resolves to: /home/user/repos/forge-agent/src/main.py
# Allowed if: /home/user/repos is in allowed_paths

# ❌ INVALID: Path outside allowed directory
path = "/etc/passwd"
# Rejected: Not in allowed_paths

# ❌ INVALID: Symlink escape attempt
path = "~/repos/symlink_to_etc"
# Rejected if symlink resolves outside allowed_paths
```

### Operation Examples

**✅ Valid Use Cases:**
```json
{
  "tool": "filesystem",
  "operation": "list_directory",
  "arguments": {"path": "~/repos/forge-agent"}
}

{
  "tool": "filesystem",
  "operation": "read_file",
  "arguments": {"path": "src/main.py"}
}
```

**❌ Invalid Use Cases:**
```json
// ❌ Execution attempt - use shell tool
{
  "tool": "filesystem",
  "operation": "execute",
  "arguments": {"path": "script.sh"}
}

// ❌ System info - use system tool
{
  "tool": "filesystem",
  "operation": "get_system_info",
  "arguments": {}
}

// ❌ Path outside allowed boundaries
{
  "tool": "filesystem",
  "operation": "read_file",
  "arguments": {"path": "/etc/passwd"}
}
```

## Overlap with Other Tools

### vs. Shell Tool

**Filesystem:** Structured file operations with explicit validation
- Use for: reading files, listing directories, creating files
- Path validation is automatic and enforced

**Shell:** Command execution with execution semantics
- Use for: running scripts, executing commands, process management
- Path validation must be done manually via `cwd` argument

**Rule:** If you need to **execute** something, use `shell`. If you need to **read/write** files, use `filesystem`.

### vs. System Tool

**Filesystem:** File and directory operations
- Use for: file I/O, directory traversal

**System:** System introspection and status
- Use for: platform info, Python version, system status

**Rule:** If you need **file contents**, use `filesystem`. If you need **system metadata**, use `system`.

## Rationale

The filesystem tool exists to provide **safe, structured file operations** with explicit security boundaries. By separating file I/O from command execution, we:

1. **Prevent privilege escalation:** Path validation prevents access to sensitive areas
2. **Enable auditability:** All file operations are explicit and traceable
3. **Reduce attack surface:** No execution semantics means no code injection vectors
4. **Improve reliability:** Structured operations are more predictable than shell commands

## Implementation Notes

- Path resolution uses `Path.expanduser().resolve()` to normalize paths
- `allowed_paths` take priority over `restricted_paths` (explicit allow overrides implicit deny)
- Symlinks are resolved, but the final resolved path must still pass validation
- All operations are synchronous (no async I/O yet, but interface is async-ready)
