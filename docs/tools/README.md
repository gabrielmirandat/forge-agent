# Tool Contracts and Hardening Documentation

This directory contains explicit contracts, security boundaries, and hardening documentation for all tools in the forge-agent system.

## Purpose

These documents make tool responsibilities, boundaries, and constraints **explicit and enforced**, preventing:
- Semantic overlap confusion
- Future misuse (including by better LLMs)
- Accidental privilege creep
- Silent failures

## Documents

### Tool Contracts

- **[filesystem.md](./filesystem.md)** - Structured file I/O with path validation
- **[shell.md](./shell.md)** - Command execution with validation
- **[system.md](./system.md)** - Side-effect-free system introspection
- **[git.md](./git.md)** - Structured Git operations
- **[github.md](./github.md)** - GitHub API operations

### Overlaps and Priority Rules

- **[OVERLAPS.md](./OVERLAPS.md)** - Tool overlaps, priority rules, and when to use each tool

## Core Principles

1. **Explicit Contracts:** Each tool has a documented contract stating what it can and cannot do
2. **Fail Fast:** Tools reject invalid operations with explicit, structured errors
3. **No Silent Fallbacks:** Tools never silently fallback to other tools
4. **Priority Rules:** Structured tools (filesystem, system, git) are preferred over execution tools (shell)
5. **Security Invariants:** Each tool enforces security boundaries that cannot be bypassed

## Quick Reference

### When to Use Each Tool

| Operation | Tool | Why |
|-----------|------|-----|
| Read a file | `filesystem` | Structured, validated, fast |
| List directory | `filesystem` | Structured, validated, fast |
| Execute a script | `shell` | Execution semantics required |
| Get system info | `system` | Side-effect-free, structured |
| Git operations | `git` | Structured API, auditable |
| GitHub API | `github` | Structured API, authenticated |

### Priority Rules

1. **Prefer structured tools over execution tools**
2. **Use shell tool ONLY when execution semantics are required**
3. **Never use shell for operations that structured tools can handle**

## Contract Enforcement

Contracts are enforced through:

1. **Code Assertions:** Tools reject operations that violate their contract
2. **Validation:** Path validation, command validation, operation validation
3. **Explicit Errors:** Failures are explicit and structured, not silent
4. **Planner Instructions:** The Planner is instructed to prefer structured tools

## Security Invariants

Each tool enforces security invariants:

- **Filesystem:** Path validation is mandatory, no execution, no system introspection
- **Shell:** Command validation is mandatory, execution intent required
- **System:** Side-effect-free guarantee, no execution, no file I/O
- **Git:** Structured API only, no arbitrary commands
- **GitHub:** Authentication required, structured API only

## For Developers

When adding a new tool or modifying an existing one:

1. **Document the contract** in `docs/tools/<tool>.md`
2. **Enforce the contract** in code with explicit validations
3. **Add internal assertions** to catch violations early
4. **Update agent.yaml** with comments explaining the tool's purpose
5. **Document overlaps** if the tool overlaps with existing tools

## For LLMs (Planner)

The Planner receives explicit instructions about:
- Tool priorities (prefer structured tools)
- When to use each tool
- What operations are forbidden
- Security constraints (paths, commands, operations)

These instructions are in the Planner's system prompt and are enforced by tool contracts.
