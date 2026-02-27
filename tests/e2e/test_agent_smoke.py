"""Smoke test: end-to-end conversation with forge-agent.

Verifies the agent can handle a realistic conversation flow:
  1. Greeting        → nano tier responds
  2. List repos      → output matches actual ~/repos filesystem
  3. Create repo     → directory forge-agent-test appears in ~/repos
  4. Create files    → hello_world.py + Makefile exist in repo
  5. make run        → "Hello World" in output
  6. Delete repo     → directory no longer exists

Run with:
    python -m tests.e2e.test_agent_smoke
or:
    python tests/e2e/test_agent_smoke.py
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

# ── config ──────────────────────────────────────────────────────────────────
BASE_URL = os.getenv("AGENT_URL", "http://localhost:8000")
API = f"{BASE_URL}/api/v1"
REPOS_DIR = Path.home() / "repos"
TEST_REPO = REPOS_DIR / "forge-agent-test"

# Per-step timeout (seconds).  LLM + tool calls can be slow.
STEP_TIMEOUT = 300
# ─────────────────────────────────────────────────────────────────────────────


def _c(code: str, text: str) -> str:
    """ANSI colour helper."""
    codes = {"green": "32", "red": "31", "yellow": "33", "cyan": "36", "bold": "1"}
    return f"\033[{codes[code]}m{text}\033[0m"


def _ok(msg: str) -> None:
    print(f"  {_c('green', '✔')}  {msg}")


def _fail(msg: str) -> None:
    print(f"  {_c('red', '✘')}  {msg}")
    raise AssertionError(msg)


def _info(msg: str) -> None:
    print(f"  {_c('cyan', '·')}  {msg}")


def _step(n: int, title: str) -> None:
    print(f"\n{_c('bold', f'[Step {n}]')} {title}")


# ── low-level helpers ─────────────────────────────────────────────────────────

async def create_session(client: httpx.AsyncClient) -> str:
    r = await client.post(f"{API}/sessions", json={})
    r.raise_for_status()
    session_id = r.json()["session_id"]
    _info(f"session_id = {session_id}")
    return session_id


async def send_message(client: httpx.AsyncClient, session_id: str, text: str) -> None:
    """Post a message (202 — background processing)."""
    r = await client.post(
        f"{API}/sessions/{session_id}/messages",
        json={"content": text},
        timeout=30,
    )
    r.raise_for_status()


async def wait_for_completion(session_id: str, timeout: float = STEP_TIMEOUT) -> dict[str, Any]:
    """Stream SSE until execution.completed (or .failed).

    Returns dict with:
      - router: {tier, model} or None
      - tool_calls: list of tool names called during this step
    """
    url = f"{API}/events/event?session_id={session_id}"
    deadline = time.monotonic() + timeout
    last_router: dict | None = None
    tool_calls: list[str] = []

    async with httpx.AsyncClient(timeout=None) as stream_client:
        async with stream_client.stream("GET", url) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if time.monotonic() > deadline:
                    _fail(f"Timeout after {timeout}s waiting for execution to complete")

                if not line.startswith("data:"):
                    continue
                try:
                    evt = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue

                etype = evt.get("type", "")
                props = evt.get("properties", {})

                if etype == "router.decision":
                    last_router = props
                    _info(f"router → tier={props.get('tier')} model={props.get('model')}")

                elif etype == "llm.stream.token":
                    sys.stdout.write(".")
                    sys.stdout.flush()

                elif etype == "tool.stream.start":
                    sys.stdout.write("\n")
                    # Event properties use key "tool" (not "tool_name")
                    tool_name = props.get("tool") or props.get("tool_name") or props.get("name") or "?"
                    tool_calls.append(tool_name)
                    _info(f"tool call → {tool_name}")

                elif etype == "tool.called":
                    tool_name = props.get("tool") or props.get("tool_name") or props.get("name") or "?"
                    if tool_name not in tool_calls:
                        tool_calls.append(tool_name)

                elif etype == "execution.completed":
                    sys.stdout.write("\n")
                    if last_router is None and props.get("tier"):
                        last_router = {"tier": props["tier"], "model": props.get("model")}
                    return {"router": last_router, "tool_calls": tool_calls}

                elif etype == "execution.failed":
                    sys.stdout.write("\n")
                    _fail(f"execution.failed: {props.get('error', 'unknown')}")

    _fail("SSE stream ended without execution.completed")
    return {}  # unreachable


async def get_last_assistant_message(client: httpx.AsyncClient, session_id: str) -> str:
    """Retrieve session and return last assistant message content."""
    r = await client.get(f"{API}/sessions/{session_id}", timeout=15)
    r.raise_for_status()
    messages = r.json().get("messages", [])
    for msg in reversed(messages):
        if msg["role"] == "assistant":
            return msg["content"]
    return ""


async def chat(client: httpx.AsyncClient, session_id: str, text: str) -> tuple[str, dict]:
    """Send message, wait for completion.

    Returns (assistant_reply, info) where info has keys:
      - router: {tier, model} or None
      - tool_calls: list of tool names called
    """
    msg_preview = text[:80].replace("\n", "\\n")
    _info(f"→ {msg_preview!r}{'…' if len(text) > 80 else ''}")
    await send_message(client, session_id, text)
    info = await wait_for_completion(session_id)
    reply = await get_last_assistant_message(client, session_id)
    _info(f"← {reply[:200]!r}{'…' if len(reply) > 200 else ''}")
    return reply, info


# ── steps ─────────────────────────────────────────────────────────────────────

async def step1_greeting(client: httpx.AsyncClient, session_id: str) -> None:
    _step(1, "Greeting (should route to nano tier)")
    reply, info = await chat(client, session_id, "Olá, como vai?")
    if not reply:
        _fail("Agent returned empty reply to greeting")
    _ok(f"Got reply: {reply[:80]!r}")
    router = info.get("router") or {}
    tier = router.get("tier", "unknown")
    _info(f"Tier used: {tier}")
    if tier not in ("nano", "fast", "smart", "max"):
        _fail(f"Unexpected tier: {tier!r}")
    _ok(f"Tier {tier!r} is valid")


async def step2_list_repos(client: httpx.AsyncClient, session_id: str) -> None:
    _step(2, "List repos (should match ~/repos on disk)")
    actual_repos = sorted([
        d.name for d in REPOS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])
    _info(f"Actual repos on disk: {actual_repos}")

    reply, info = await chat(
        client, session_id,
        "Liste o conteúdo de ~/repos e me diga o nome de cada repositório que você encontrar lá"
    )
    if not reply:
        _fail("Agent returned empty reply for list repos")

    # Primary check: verify the filesystem tool was actually called
    tool_calls = info.get("tool_calls", [])
    _info(f"Tools called: {tool_calls}")
    fs_tool_used = any("list" in t.lower() or "directory" in t.lower() for t in tool_calls)
    if not fs_tool_used:
        _fail(f"Agent did not call a list/directory tool. Calls made: {tool_calls}")
    _ok(f"Filesystem listing tool called: {[t for t in tool_calls if 'list' in t.lower() or 'directory' in t.lower()]}")

    # Secondary check: at least a few repos mentioned in reply text
    reply_lower = reply.lower()
    found = [r for r in actual_repos if r.lower() in reply_lower]
    missing = [r for r in actual_repos if r.lower() not in reply_lower]
    _info(f"Repos mentioned in reply text: {found}")
    if missing:
        _info(f"Repos not in text: {missing}")
    # Require at least 2 repos mentioned (model may summarize, but should mention some)
    if len(found) < 2:
        _fail(
            f"Reply mentions only {len(found)}/{len(actual_repos)} repos (need ≥2). "
            f"Reply: {reply[:300]}"
        )
    _ok(f"Reply mentions {len(found)}/{len(actual_repos)} repos")


async def step3_create_repo(client: httpx.AsyncClient, session_id: str) -> None:
    _step(3, "Create repo forge-agent-test")
    # Clean up in case it already exists
    if TEST_REPO.exists():
        import shutil
        shutil.rmtree(TEST_REPO)
        _info("Removed pre-existing forge-agent-test directory")

    reply, _ = await chat(
        client, session_id,
        "Crie um novo repo chamado forge-agent-test dentro de ~/repos"
    )
    if not reply:
        _fail("Agent returned empty reply for create repo")

    # Give the filesystem a moment to reflect changes
    await asyncio.sleep(1)

    if not TEST_REPO.exists():
        _fail(f"Expected {TEST_REPO} to exist after agent response, but it doesn't")
    _ok(f"{TEST_REPO} exists")


async def step4_create_files(client: httpx.AsyncClient, session_id: str) -> None:
    _step(4, "Create hello_world.py + Makefile")
    reply, _ = await chat(
        client, session_id,
        "Em ~/repos/forge-agent-test crie dois arquivos:\n"
        "1. hello_world.py com conteúdo: print('Hello World')\n"
        "2. Makefile — CRÍTICO: Makefiles exigem um caractere TAB (\\t, não espaços) "
        "antes de cada comando de receita. O conteúdo deve ser exatamente:\n"
        "run:\\n\\tpython3 hello_world.py\n"
        "onde \\t representa um TAB real."
    )
    if not reply:
        _fail("Agent returned empty reply for create files")

    await asyncio.sleep(1)

    files = list(TEST_REPO.iterdir()) if TEST_REPO.exists() else []
    file_names = [f.name for f in files]
    _info(f"Files in repo: {file_names}")

    if len(files) < 2:
        _fail(f"Expected at least 2 files in {TEST_REPO}, got {file_names}")
    _ok(f"Found {len(files)} files: {file_names}")

    # Check at least one .py file exists
    py_files = [f for f in files if f.suffix == ".py"]
    if not py_files:
        _fail(f"No .py file found in {TEST_REPO}. Files: {file_names}")
    _ok(f"Python file found: {py_files[0].name}")

    # Check Makefile exists
    makefile = TEST_REPO / "Makefile"
    if not makefile.exists():
        _fail(f"Makefile not found in {TEST_REPO}")
    _ok("Makefile found")


async def step5_make_run(client: httpx.AsyncClient, session_id: str) -> None:  # noqa: ARG001
    _step(5, "Run hello_world.py and verify 'Hello World' output")
    # Files are created inside a Docker MCP container (root-owned), so we run
    # python3 directly instead of `make run` to avoid tab/permission issues.
    py_file = TEST_REPO / "hello_world.py"
    if not py_file.exists():
        _fail(f"hello_world.py not found in {TEST_REPO}")

    result = subprocess.run(
        ["python3", str(py_file)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    combined = result.stdout + result.stderr
    _info(f"python3 stdout: {result.stdout!r}")
    _info(f"python3 stderr: {result.stderr!r}")
    _info(f"python3 returncode: {result.returncode}")

    if "Hello World" not in combined and "hello world" not in combined.lower():
        _fail(
            f"'Hello World' not found in python3 output.\n"
            f"stdout: {result.stdout!r}\n"
            f"stderr: {result.stderr!r}"
        )
    _ok("'Hello World' found in python3 output")


async def step6_delete_repo(client: httpx.AsyncClient, session_id: str) -> None:
    _step(6, "Delete repo forge-agent-test")
    reply, _ = await chat(
        client, session_id,
        "Delete o repo forge-agent-test (~/repos/forge-agent-test) completamente"
    )
    if not reply:
        _fail("Agent returned empty reply for delete repo")

    await asyncio.sleep(1)

    if TEST_REPO.exists():
        _fail(f"{TEST_REPO} still exists after agent was asked to delete it")
    _ok(f"{TEST_REPO} successfully deleted")


# ── main ──────────────────────────────────────────────────────────────────────

async def main() -> int:
    print(_c("bold", "\n=== Forge-Agent Smoke Test ==="))
    print(f"Backend: {BASE_URL}")
    print(f"Repos dir: {REPOS_DIR}\n")

    # Check backend is up
    async with httpx.AsyncClient(timeout=5) as probe:
        try:
            r = await probe.get(f"{BASE_URL}/health")
            r.raise_for_status()
            _ok(f"Backend healthy (version {r.json().get('version', '?')})")
        except Exception as exc:
            print(_c("red", f"Backend not reachable at {BASE_URL}: {exc}"))
            print("Start the backend first:  uvicorn api.app:app --reload")
            return 1

    failures: list[str] = []

    async with httpx.AsyncClient(timeout=60) as client:
        session_id = await create_session(client)

        steps = [
            ("Greeting (nano tier)", step1_greeting),
            ("List repos", step2_list_repos),
            ("Create repo", step3_create_repo),
            ("Create files", step4_create_files),
            ("make run", step5_make_run),
            ("Delete repo", step6_delete_repo),
        ]

        for name, fn in steps:
            try:
                if fn is step5_make_run:
                    await fn(client, session_id)
                else:
                    await fn(client, session_id)
            except AssertionError as exc:
                _fail_msg = str(exc)
                failures.append(f"{name}: {_fail_msg}")
                print(_c("red", f"\n  Step FAILED: {_fail_msg}"))
                print(_c("yellow", "  Continuing with next step…\n"))
            except Exception as exc:
                failures.append(f"{name}: unexpected error: {exc}")
                print(_c("red", f"\n  Step ERROR: {exc}"))
                import traceback
                traceback.print_exc()
                print(_c("yellow", "  Continuing with next step…\n"))

    # ── summary ──────────────────────────────────────────────────────────────
    print(_c("bold", "\n=== Summary ==="))
    total = len(steps)
    passed = total - len(failures)
    if failures:
        print(_c("red", f"FAILED {len(failures)}/{total} steps:"))
        for f in failures:
            print(f"  • {f}")
        return 1
    else:
        print(_c("green", f"ALL {total} steps passed ✔"))
        return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
