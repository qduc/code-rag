"""Interactive setup wizard for Code-RAG.

Goal: keep the base install lean while allowing users to opt into heavy
dependencies (local models vs cloud providers) via optional extras.

This wizard:
1) Asks the user which embedding backend they want (local/cloud)
2) Installs the corresponding optional extras into the active environment
3) Optionally writes a project-local .env copied from env.example

Note: This is intentionally stdlib-only (no rich/inquirer deps).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

PACKAGE_NAME = "code-rag-mcp"


def _print(title: str) -> None:
    sys.stdout.write(title + "\n")


def _ask_choice(prompt: str, choices: list[tuple[str, str]], default_key: str) -> str:
    """Prompt the user to choose one option.

    choices: list of (key, description)
    Returns: chosen key
    """
    keys = [k for k, _ in choices]
    if default_key not in keys:
        raise ValueError("default_key must be one of the choices")

    _print("\n" + prompt)
    for k, desc in choices:
        suffix = " (default)" if k == default_key else ""
        _print(f"  [{k}] {desc}{suffix}")

    while True:
        raw = input(f"Select [{'/'.join(keys)}] (default: {default_key}): ").strip()
        if not raw:
            return default_key
        if raw in keys:
            return raw
        _print(f"Invalid choice: {raw!r}")


def _ask_yes_no(prompt: str, default: bool = True) -> bool:
    default_str = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{prompt} [{default_str}]: ").strip().lower()
        if raw == "" and default is not None:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        _print("Please answer 'y' or 'n'.")


def _pip_install(requirements: Iterable[str]) -> None:
    reqs = [r for r in requirements if r]
    if not reqs:
        return

    cmd = [sys.executable, "-m", "pip", "install", *reqs]
    _print("\nInstalling dependencies:")
    _print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def _maybe_write_env(project_root: Path) -> None:
    env_example = project_root / "env.example"
    env_file = project_root / ".env"

    if env_file.exists():
        _print(f"\nFound existing {env_file}, leaving it unchanged.")
        return

    if not env_example.exists():
        _print("\nNo env.example found; skipping .env creation.")
        return

    if not _ask_yes_no("Create a local .env file from env.example?", default=True):
        return

    env_file.write_text(env_example.read_text(encoding="utf-8"), encoding="utf-8")
    _print(f"Wrote {env_file}")


def _detect_project_root() -> Path:
    # Prefer CWD if it looks like the repo root
    cwd = Path.cwd()
    if (cwd / "pyproject.toml").exists() and (cwd / "src").exists():
        return cwd
    # Fallback: locate from this file
    return Path(__file__).resolve().parents[3]


def main(argv: Optional[list[str]] = None) -> int:
    _print("Code-RAG setup wizard")
    _print("====================")

    project_root = _detect_project_root()
    _print(f"Project root: {project_root}")

    backend = _ask_choice(
        "Which embedding backend do you want?",
        choices=[
            (
                "local",
                "Local models (sentence-transformers; downloads model weights on first use)",
            ),
            ("cloud", "Cloud providers (OpenAI / Azure / Vertex / etc via LiteLLM)"),
        ],
        default_key="local",
    )

    extras: list[str] = []
    if backend == "local":
        extras.append("local")
    else:
        extras.append("cloud")

    if _ask_yes_no("Enable local reranking (cross-encoder)?", default=False):
        # If backend is local this is already covered, but harmless to include.
        extras.append("reranker")

    # Install extras into current environment
    extras = sorted(set(extras))
    requirement = f"{PACKAGE_NAME}[{','.join(extras)}]"

    _print("\nSelected components:")
    _print(f"  - extras: {', '.join(extras)}")

    if _ask_yes_no("Install/upgrade these dependencies now?", default=True):
        _pip_install(["--upgrade", requirement])

    _maybe_write_env(project_root)

    _print("\nSetup complete.")
    _print("Next steps:")
    _print("  - Run: code-rag (MCP server)")
    _print("  - Or:  code-rag-cli (CLI)")
    if backend == "cloud":
        _print(
            "  - For cloud providers, set the relevant API keys (e.g. OPENAI_API_KEY)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
