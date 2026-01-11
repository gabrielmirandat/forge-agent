# GitHub Tool Contract

## Purpose

The `github` tool provides **structured GitHub API operations** for interacting with GitHub repositories, pull requests, and issues. It is designed for **explicit GitHub API calls** with clear boundaries and authentication.

## Allowed Responsibilities

✅ **GitHub API Operations:**
- Create pull requests
- List pull requests
- Comment on pull requests
- Get pull request details

✅ **Structured API Calls:**
- All operations use GitHub's REST API
- Operations are explicit and auditable
- Authentication via `GITHUB_TOKEN` environment variable

## Forbidden Responsibilities

❌ **Arbitrary API Calls:**
- Must NOT make arbitrary HTTP requests
- Must NOT bypass GitHub's structured API
- Must NOT use shell tool to call GitHub API

❌ **File Operations:**
- Must NOT read or write files directly (use `filesystem` tool)
- Must NOT modify repository files
- Must NOT perform file I/O operations

❌ **Git Operations:**
- Must NOT perform Git operations (use `git` tool)
- Must NOT create branches or commits
- Must NOT push to repositories

❌ **Execution Semantics:**
- Must NOT execute scripts or programs
- Must NOT invoke external tools
- Must NOT spawn processes

## Security Invariants

1. **Authentication Required:**
   - All operations REQUIRE `GITHUB_TOKEN` environment variable
   - Missing token MUST fail explicitly
   - No silent fallback or anonymous access

2. **Structured Operations Only:**
   - All GitHub operations MUST use the tool's structured API
   - No arbitrary HTTP requests
   - No shell tool fallback for GitHub operations

3. **Explicit Intent:**
   - PR creation MUST specify repo, title, body, head, base
   - PR listing MUST specify repo and state
   - PR commenting MUST specify repo, PR number, body

## Contract Enforcement

### Operation Examples

**✅ Valid Use Cases:**
```json
// Create a pull request
{
  "tool": "github",
  "operation": "create_pr",
  "arguments": {
    "repo": "owner/repo",
    "title": "Add new feature",
    "body": "This PR adds a new feature",
    "head": "feature/new-feature",
    "base": "main"
  }
}

// List pull requests
{
  "tool": "github",
  "operation": "list_prs",
  "arguments": {
    "repo": "owner/repo",
    "state": "open"
  }
}
```

**❌ Invalid Use Cases:**
```json
// ❌ Arbitrary API call - use structured operations
{
  "tool": "github",
  "operation": "make_request",
  "arguments": {
    "url": "https://api.github.com/repos/owner/repo/issues"
  }
}

// ❌ File operation - use filesystem tool
{
  "tool": "github",
  "operation": "read_file",
  "arguments": {
    "path": ".github/workflows/ci.yml"
  }
}

// ❌ Git operation - use git tool
{
  "tool": "github",
  "operation": "create_branch",
  "arguments": {
    "repo": "owner/repo",
    "branch_name": "feature/new"
  }
}
```

## Overlap with Other Tools

### vs. Git Tool

**GitHub:** GitHub API operations
- Use for: PRs, issues, comments, GitHub-specific operations
- Remote operations: interacts with GitHub's API

**Git:** Local Git repository operations
- Use for: branches, commits, pushes, local repository state
- Local operations: works with local Git repositories

**Rule:** If you need **GitHub API operations** (PRs, issues), use `github` tool. If you need **local Git operations** (branches, commits), use `git` tool.

**Workflow:** Typically, you use `git` tool to create branches and commits locally, then use `github` tool to create PRs.

### vs. Filesystem Tool

**GitHub:** GitHub API operations
- Use for: PRs, issues, comments

**Filesystem:** File and directory operations
- Use for: reading files, listing directories, file I/O

**Rule:** If you need **GitHub API operations**, use `github` tool. If you need **file I/O**, use `filesystem` tool.

## Rationale

The GitHub tool exists to provide **structured, auditable GitHub API operations** with explicit boundaries. By separating GitHub operations from file I/O, Git operations, and command execution, we:

1. **Enable explicit intent:** GitHub operations are intentional and traceable
2. **Improve security:** Structured API prevents arbitrary HTTP requests
3. **Enable auditability:** All GitHub operations are explicit and logged
4. **Reduce misuse:** Clear boundaries prevent using shell or HTTP clients for GitHub operations

## Implementation Notes

- All operations use GitHub's REST API (via `httpx` or `requests`)
- Authentication via `GITHUB_TOKEN` environment variable
- PR titles and bodies can use templates (if configured)
- Auto-PR creation can be enabled/disabled (if configured)
