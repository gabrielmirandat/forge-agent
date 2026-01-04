# Forge Agent

A self-hosted, autonomous code agent that operates on your local repositories. Similar in spirit to Claude Code, but running entirely on your own machine with full control and privacy.

## Vision

Forge Agent is a production-quality, long-term infrastructure project for autonomous code operations. It provides:

- **Local-first**: Runs entirely on your machine, no cloud dependencies
- **Self-hosted**: Uses local LLMs (LocalAI initially, extensible to others)
- **Persistent workspace**: Operates on existing repositories without cloning/deleting
- **Git-native**: Creates branches, commits, and PRs as a first-class citizen
- **Modular & auditable**: Clean architecture with replaceable components
- **Config-driven**: Behavior controlled via YAML configuration files
- **API-first**: RESTful API for integration with web/mobile frontends

## Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (React)                      │
│              Web UI + Mobile-friendly                    │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP/REST
┌────────────────────▼────────────────────────────────────┐
│                  API Layer (FastAPI)                     │
│              /api/v1/goals, /api/v1/status               │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              Agent Runtime                               │
│  ┌──────────────┐         ┌──────────────┐             │
│  │   Planner    │────────▶│   Executor   │             │
│  │              │         │              │             │
│  └──────┬───────┘         └──────┬───────┘             │
│         │                        │                      │
└─────────┼────────────────────────┼──────────────────────┘
          │                        │
┌─────────▼────────────────────────▼──────────────────────┐
│                    Tool System                          │
│  ┌──────────┐ ┌──────┐ ┌────────┐ ┌──────┐ ┌────────┐ │
│  │Filesystem│ │ Git  │ │ GitHub │ │ Shell│ │ System │ │
│  └──────────┘ └──────┘ └────────┘ └──────┘ └────────┘ │
└─────────────────────────────────────────────────────────┘
          │
┌─────────▼──────────────────────────────────────────────┐
│              LLM Provider (LocalAI)                     │
│              Extensible to other providers              │
└─────────────────────────────────────────────────────────┘
```

### Component Details

#### 1. Agent Runtime (`agent/runtime/`)

- **Planner**: Converts high-level goals into structured execution plans
  - Uses LLM to break down goals into tool calls
  - Validates tool availability and safety constraints
  - Returns structured plan with steps

- **Executor**: Executes plans and manages tool call lifecycle
  - Iterates through plan steps
  - Calls appropriate tools
  - Handles errors and retries
  - Collects and aggregates results

#### 2. Tool System (`agent/tools/`)

Modular tools for different operations:

- **FilesystemTool**: Read, write, list, create, delete files/directories
  - Path validation with allowed/restricted paths
  - Safe file operations

- **GitTool**: Git repository operations
  - Create branches (with configurable prefix)
  - Commit changes
  - Push to remotes
  - Status and diff operations

- **GitHubTool**: GitHub API integration
  - Create pull requests
  - List PRs
  - Comment on PRs
  - Get PR details

- **ShellTool**: Safe shell command execution
  - Command whitelist/blacklist
  - Timeout handling
  - Output capture

- **SystemTool**: System information and status
  - Platform info
  - Agent status

All tools implement a common `Tool` interface for consistency and extensibility.

#### 3. LLM Provider (`agent/llm/`)

Abstracted LLM interface supporting multiple providers:

- **Base Provider**: `LLMProvider` abstract class
- **LocalAI Provider**: Implementation for LocalAI
- **Extensible**: Easy to add OpenAI, Anthropic, Ollama, etc.

#### 4. Configuration (`agent/config/`)

YAML-based configuration system:

- **ConfigLoader**: Loads and validates configuration
- **Structured schemas**: Pydantic models for type safety
- **Path expansion**: Handles `~` and relative paths
- **Environment variables**: Support for secrets (e.g., `GITHUB_TOKEN`)

#### 5. API Layer (`api/`)

FastAPI-based REST API:

- **Goal endpoint**: Submit goals and receive plans
- **Execution endpoint**: Execute plans and get results
- **Status endpoint**: Agent status and available tools
- **CORS enabled**: For frontend integration

#### 6. Frontend (`frontend/`)

React + TypeScript + Vite:

- **Modern stack**: React 18, TypeScript, Vite
- **API client**: Typed API client with axios
- **Mobile-friendly**: Responsive design
- **Production-ready**: Docker support

## Project Structure

```
agent/
├── README.md                 # This file
├── .gitignore
├── pyproject.toml            # Python dependencies
├── requirements.txt          # Python dependencies (alternative)
├── Dockerfile                # Backend container
├── docker-compose.yml        # Full stack orchestration
│
├── config/                   # Configuration files
│   ├── agent.yaml            # Main configuration
│   └── example.yaml          # Example configuration
│
├── agent/                    # Agent runtime (Python package)
│   ├── __init__.py
│   ├── config/               # Configuration loading
│   │   ├── __init__.py
│   │   └── loader.py
│   ├── runtime/              # Planning and execution
│   │   ├── __init__.py
│   │   ├── planner.py
│   │   └── executor.py
│   ├── tools/                # Tool implementations
│   │   ├── __init__.py
│   │   ├── base.py           # Tool interface
│   │   ├── filesystem.py
│   │   ├── git.py
│   │   ├── github.py
│   │   ├── shell.py
│   │   └── system.py
│   └── llm/                  # LLM providers
│       ├── __init__.py
│       ├── base.py           # LLM interface
│       └── localai.py
│
├── api/                      # FastAPI application
│   ├── __init__.py
│   ├── main.py               # FastAPI app
│   ├── models/               # Pydantic schemas
│   │   ├── __init__.py
│   │   └── schemas.py
│   └── routes/               # API routes
│       ├── __init__.py
│       └── agent.py
│
├── frontend/                 # React frontend
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── Dockerfile
│   ├── index.html
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── App.css
│       ├── index.css
│       ├── api/
│       │   └── client.ts
│       └── components/
│
└── tests/                    # Test suite
    ├── __init__.py
    ├── test_config.py
    ├── test_tools.py
    └── test_runtime.py
```

## Configuration

Configuration is managed via YAML files in `config/`. See `config/agent.yaml` for the full schema.

Key configuration areas:

- **Workspace**: Base path for repositories (`~/repos` by default)
- **LLM**: Provider, model, temperature, etc.
- **Runtime**: Max iterations, timeouts, safety checks
- **Tools**: Enable/disable tools, set allowed paths/commands
- **Security**: Sandbox settings, audit logging

Environment variables:
- `CONFIG_PATH`: Path to config file (default: `config/agent.yaml`)
- `GITHUB_TOKEN`: GitHub API token for PR operations
- `REPOS_BASE_PATH`: Base path for repositories (overrides config)

## Filesystem Philosophy

**Persistent Workspace**: The agent operates on existing repositories in `~/repos`. It does NOT:
- Clone repositories on each run
- Delete repositories after operations
- Create temporary workspaces

**Git Operations**: The agent:
- Creates branches with configurable prefix (default: `agent/`)
- Makes commits with descriptive messages
- Pushes branches to remotes
- Creates pull requests via GitHub API

**Path Safety**: All filesystem operations are validated against:
- Allowed paths (whitelist)
- Restricted paths (blacklist)

## Development

### Prerequisites

- Python 3.10+
- Node.js 20+ (for frontend)
- Docker & Docker Compose (optional, for full stack)
- LocalAI instance (or configure another LLM provider)

### Setup

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   # or
   pip install -e .
   ```

2. **Install frontend dependencies**:
   ```bash
   cd frontend
   npm install
   ```

3. **Configure**:
   - Copy `config/example.yaml` to `config/agent.yaml`
   - Edit `config/agent.yaml` with your settings
   - Set `GITHUB_TOKEN` environment variable if using GitHub features

4. **Run API**:
   ```bash
   uvicorn api.main:app --reload
   ```

5. **Run frontend**:
   ```bash
   cd frontend
   npm run dev
   ```

### Docker

Full stack via Docker Compose:

```bash
docker-compose up
```

This starts:
- `agent-api`: FastAPI backend (port 8000)
- `localai`: LocalAI LLM service (port 8080)
- `frontend`: React frontend (port 3000)

## Security Considerations

- **Path validation**: All filesystem operations check allowed/restricted paths
- **Command whitelist**: Shell commands are restricted to allowed list
- **No sudo**: Restricted commands prevent privilege escalation
- **Audit logging**: Configurable audit logs for all operations
- **Token management**: Secrets via environment variables, not config files

## Extensibility

### Adding a New Tool

1. Create a new file in `agent/tools/`
2. Inherit from `Tool` base class
3. Implement `name`, `description`, and `execute` methods
4. Register in tool registry

### Adding a New LLM Provider

1. Create a new file in `agent/llm/`
2. Inherit from `LLMProvider` base class
3. Implement `generate` and `chat` methods
4. Update config loader to support new provider

### Modifying API

1. Add new schemas in `api/models/schemas.py`
2. Add routes in `api/routes/`
3. Register routes in `api/main.py`

## Status

**Current State**: Initial structure and scaffolding complete. Core logic implementation pending.

**Next Steps**:
1. Implement tool execution logic
2. Implement planner and executor
3. Implement LLM provider integration
4. Complete API endpoints
5. Build frontend UI
6. Add comprehensive tests

## License

[Specify your license]

## Contributing

[Contributing guidelines if applicable]
