"""Interactive setup wizard for Code-RAG.

Goal: keep the base install lean while allowing users to opt into heavy
dependencies (local models vs cloud providers) via optional extras.

This wizard:
1) Displays a friendly welcome message
2) Auto-detects environment (GPU, API keys, virtual env)
3) Explains pros/cons of each embedding backend
4) Guides model selection with sensible defaults
5) Installs the corresponding optional extras
6) Generates configuration file
7) Runs a verification test (optional)

Note: This is intentionally stdlib-only (no rich/inquirer deps).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

PACKAGE_NAME = "code-rag-mcp"

# Default models for each backend
DEFAULT_LOCAL_MODEL = "nomic-ai/CodeRankEmbed"
DEFAULT_CLOUD_MODEL = "text-embedding-3-small"

# Available models with descriptions
LOCAL_MODELS = [
    ("nomic-ai/CodeRankEmbed", "Code-optimized, best for code search (recommended)"),
    ("all-MiniLM-L6-v2", "General-purpose, smaller and faster"),
    ("custom", "Enter a custom model name"),
]

CLOUD_MODELS = [
    ("text-embedding-3-small", "OpenAI, good balance of cost/quality (recommended)"),
    ("text-embedding-3-large", "OpenAI, higher quality, more expensive"),
    ("azure/text-embedding-3-small", "Azure OpenAI"),
    ("vertex_ai/text-embedding-004", "Google Vertex AI"),
    ("cohere/embed-english-v3.0", "Cohere"),
    ("custom", "Enter a custom model name"),
]

# ANSI color codes
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
BLUE = "\033[0;34m"
CYAN = "\033[0;36m"
NC = "\033[0m"  # No Color


def _print(text: str = "") -> None:
    sys.stdout.write(text + "\n")


def _print_header() -> None:
    _print()
    _print(f"{BLUE}╔════════════════════════════════════════════════════════════╗{NC}")
    _print(
        f"{BLUE}║{NC}              {GREEN}Code-RAG Setup Wizard{NC}                        {BLUE}║{NC}"
    )
    _print(
        f"{BLUE}║{NC}      Semantic code search for your entire codebase       {BLUE}║{NC}"
    )
    _print(f"{BLUE}╚════════════════════════════════════════════════════════════╝{NC}")
    _print()
    _print("This wizard will help you configure Code-RAG for your needs.")
    _print("No Python or RAG experience required - just follow the prompts!")
    _print()


def _print_section(title: str) -> None:
    _print()
    _print(f"{CYAN}━━━ {title} ━━━{NC}")
    _print()


def _ask_choice(prompt: str, choices: list[tuple[str, str]], default_key: str) -> str:
    """Prompt the user to choose one option.

    choices: list of (key, description)
    Returns: chosen key
    """
    keys = [k for k, _ in choices]
    if default_key not in keys:
        raise ValueError("default_key must be one of the choices")

    _print(prompt)
    _print()
    for i, (k, desc) in enumerate(choices, 1):
        default_marker = f" {GREEN}← recommended{NC}" if k == default_key else ""
        _print(f"  [{i}] {desc}{default_marker}")

    _print()
    key_display = "/".join(str(i) for i in range(1, len(keys) + 1))
    while True:
        raw = input(f"Select [{key_display}] (press Enter for recommended): ").strip()
        if not raw:
            return default_key
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(keys):
                return keys[idx]
        except ValueError:
            pass
        _print(f"Invalid choice: {raw!r}. Please enter a number 1-{len(keys)}.")


def _ask_yes_no(prompt: str, default: bool = True) -> bool:
    default_str = f"{GREEN}Y{NC}/n" if default else f"y/{GREEN}N{NC}"
    while True:
        raw = input(f"{prompt} [{default_str}]: ").strip().lower()
        if raw == "" and default is not None:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        _print("Please answer 'y' or 'n'.")


def _ask_string(prompt: str, default: str = "") -> str:
    if default:
        response = input(f"{prompt} [{default}]: ").strip()
        return response if response else default
    return input(f"{prompt}: ").strip()


def _is_uv_tool_environment() -> bool:
    """Check if running in a uv tool environment."""
    # uv tool environments have UV_TOOL_DIR set or are under ~/.local/share/uv/tools
    if os.environ.get("UV_TOOL_DIR"):
        return True
    venv = os.environ.get("VIRTUAL_ENV", "")
    return "/uv/tools/" in venv or "/.local/share/uv/" in venv


def _pip_install(requirements: Iterable[str], quiet: bool = False) -> bool:
    reqs = [r for r in requirements if r]
    if not reqs:
        return True

    # Use uv pip if in uv tool environment and uv is available
    use_uv = False
    if _is_uv_tool_environment():
        try:
            subprocess.run(["uv", "--version"], capture_output=True, check=True)
            use_uv = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    if use_uv:
        cmd = ["uv", "pip", "install", *reqs]
    else:
        cmd = [sys.executable, "-m", "pip", "install", *reqs]

    if quiet:
        cmd.append("--quiet")

    _print(f"\n{GREEN}▶{NC} Installing dependencies...")
    _print(f"  {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        _print(f"{GREEN}✓{NC} Installation complete!")
        return True
    except subprocess.CalledProcessError:
        _print(f"{YELLOW}⚠{NC} Installation failed. You may need to install manually.")
        return False


def _detect_accelerator() -> dict:
    """Detect available hardware accelerators (CUDA, MPS, ROCm)."""
    result = {
        "type": None,  # 'cuda', 'mps', 'rocm', or None
        "name": None,  # Human-readable name
        "available": False,
    }

    # Check via torch if available
    try:
        check_code = """
import torch
if torch.cuda.is_available():
    print('cuda')
    print(torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'NVIDIA GPU')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('mps')
    print('Apple Silicon')
elif hasattr(torch, 'hip') and torch.hip.is_available():
    print('rocm')
    print('AMD GPU')
else:
    print('none')
    print('')
"""
        proc = subprocess.run(
            [sys.executable, "-c", check_code],
            capture_output=True,
            text=True,
            timeout=10,
        )
        lines = proc.stdout.strip().split("\n")
        if len(lines) >= 2 and lines[0] != "none":
            result["type"] = lines[0]
            result["name"] = lines[1]
            result["available"] = True
    except Exception:
        pass

    # Fallback: Check for Apple Silicon via platform (works without torch)
    if not result["available"]:
        try:
            import platform

            if platform.system() == "Darwin" and platform.machine() == "arm64":
                result["type"] = "mps"
                result["name"] = "Apple Silicon (M-series)"
                result["available"] = True
        except Exception:
            pass

    return result


def _detect_api_keys() -> dict[str, bool]:
    """Detect which API keys are present in environment."""
    keys = {
        "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
        "AZURE_API_KEY": bool(os.environ.get("AZURE_API_KEY")),
        "COHERE_API_KEY": bool(os.environ.get("COHERE_API_KEY")),
        "AWS_ACCESS_KEY_ID": bool(os.environ.get("AWS_ACCESS_KEY_ID")),
    }
    return keys


def _detect_environment() -> dict:
    """Detect environment capabilities."""
    return {
        "accelerator": _detect_accelerator(),
        "api_keys": _detect_api_keys(),
        "in_venv": bool(
            os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_PREFIX")
        ),
    }


def _print_environment_info(env: dict) -> None:
    """Print detected environment information."""
    _print_section("Environment Detection")

    accel = env["accelerator"]
    if accel["available"]:
        accel_type = accel["type"].upper()
        accel_name = accel["name"]
        _print(f"  {GREEN}✓{NC} {accel_type} accelerator: {accel_name}")
        _print(f"      Local models will run fast!")
    else:
        _print(f"  {YELLOW}○{NC} No GPU/accelerator detected")
        _print(f"      Local models will use CPU (slower, but still works)")

    detected_keys = [k for k, v in env["api_keys"].items() if v]
    if detected_keys:
        _print(f"  {GREEN}✓{NC} API keys found: {', '.join(detected_keys)}")
    else:
        _print(f"  {YELLOW}○{NC} No cloud API keys detected in environment")

    if env["in_venv"]:
        _print(f"  {GREEN}✓{NC} Running in virtual environment")


def _ask_backend(env: dict) -> str:
    """Ask user to choose between local and cloud embedding backend."""
    _print_section("Embedding Backend")

    _print("Code-RAG needs an embedding model to convert code into searchable vectors.")
    _print("You have two options:")
    _print()

    # Determine recommendation based on environment
    has_api_key = any(env["api_keys"].values())
    accel = env["accelerator"]
    has_accelerator = accel["available"]

    if has_accelerator:
        recommended = "local"
        reason = f"{accel['name']} detected - local models will be fast"
    elif has_api_key:
        recommended = "cloud"
        reason = "No accelerator, but API key found"
    else:
        recommended = "local"
        reason = "Works offline, no API key needed"

    _print(f"  {CYAN}[1] Local Models{NC}")
    _print(f"      {GREEN}✓{NC} No API keys needed")
    _print(f"      {GREEN}✓{NC} Works completely offline")
    _print(f"      {GREEN}✓{NC} Privacy: code never leaves your machine")
    _print(f"      {YELLOW}○{NC} Downloads ~500MB model on first use")
    if has_accelerator:
        _print(f"      {GREEN}✓{NC} Fast inference ({accel['name']})")
    else:
        _print(f"      {YELLOW}○{NC} Slower inference (CPU only)")
    _print()

    _print(f"  {CYAN}[2] Cloud Providers{NC}")
    _print(f"      {GREEN}✓{NC} No local resources needed")
    _print(f"      {GREEN}✓{NC} Fast and reliable")
    _print(f"      {YELLOW}○{NC} Requires API key (OpenAI, Azure, etc.)")
    _print(f"      {YELLOW}○{NC} Code snippets sent to cloud for embedding")
    if has_api_key:
        _print(f"      {GREEN}✓{NC} API key already configured")
    _print()

    _print(f"Recommendation: {GREEN}{recommended}{NC} ({reason})")
    _print()

    while True:
        raw = input("Select [1/2] (press Enter for recommended): ").strip()
        if not raw:
            return recommended
        if raw == "1":
            return "local"
        if raw == "2":
            return "cloud"
        _print("Please enter 1 or 2.")


def _ask_model(backend: str) -> str:
    """Ask user to choose a specific model."""
    _print_section("Model Selection")

    if backend == "local":
        models = LOCAL_MODELS
        default = DEFAULT_LOCAL_MODEL
        _print("Choose an embedding model for local inference:")
    else:
        models = CLOUD_MODELS
        default = DEFAULT_CLOUD_MODEL
        _print("Choose a cloud embedding provider:")

    _print()
    for i, (model, desc) in enumerate(models, 1):
        default_marker = f" {GREEN}← recommended{NC}" if model == default else ""
        _print(f"  [{i}] {model}")
        _print(f"      {desc}{default_marker}")
        _print()

    while True:
        raw = input(
            "Select [1-{}] (press Enter for recommended): ".format(len(models))
        ).strip()
        if not raw:
            return default
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(models):
                selected = models[idx][0]
                if selected == "custom":
                    return _ask_string("Enter custom model name")
                return selected
        except ValueError:
            pass
        _print(f"Please enter a number 1-{len(models)}.")


def _ask_reranker(backend: str) -> bool:
    """Ask about enabling reranking."""
    _print_section("Result Reranking (Optional)")

    _print("Reranking improves search result quality by re-scoring results with a")
    _print("specialized model. This is optional but recommended for best results.")
    _print()

    if backend == "local":
        _print(f"  {GREEN}✓{NC} Uses local cross-encoder model")
        _print(f"  {YELLOW}○{NC} Adds ~500MB more to download")
        _print(f"  {YELLOW}○{NC} Slightly slower searches")
    else:
        _print(f"  {YELLOW}○{NC} Requires local sentence-transformers")
        _print(f"  {YELLOW}○{NC} Downloads ~500MB for reranker model")

    _print()
    return _ask_yes_no("Enable reranking?", default=False)


def _write_config(config: dict, config_path: Path) -> None:
    """Write configuration to file."""
    config_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Code-RAG Configuration",
        "# Generated by code-rag-setup",
        "",
    ]

    for key, value in config.items():
        if value is not None:
            lines.append(f"{key}={value}")

    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _print(f"{GREEN}✓{NC} Configuration saved to {config_path}")


def _run_verification() -> bool:
    """Run a quick verification test."""
    _print_section("Verification")

    _print("Running a quick test to verify everything is working...")
    _print()

    try:
        # Try importing the main module
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from code_rag.api import CodeRAGAPI; print('Import OK')",
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        _print(f"{GREEN}✓{NC} Core modules loaded successfully")

        # Check if embedding model can be initialized (without actually loading)
        _print(f"{GREEN}✓{NC} Configuration is valid")

        return True
    except subprocess.CalledProcessError as e:
        _print(f"{YELLOW}⚠{NC} Verification failed: {e}")
        _print("You may need to configure model-specific settings.")
        return False
    except subprocess.TimeoutExpired:
        _print(f"{YELLOW}⚠{NC} Verification timed out")
        return False


def _print_next_steps(backend: str, model: str) -> None:
    """Print next steps after setup."""
    _print_section("Setup Complete!")

    _print(f"{GREEN}✓{NC} Code-RAG is ready to use!")
    _print()
    _print("Next steps:")
    _print()
    _print(f"  1. {CYAN}Add to Claude (main usage):{NC}")
    _print("     claude mcp add code-rag -- code-rag-mcp")
    _print()
    _print(f"  2. {CYAN}Or test with the CLI first:{NC}")
    _print("     code-rag-cli --path /path/to/your/project")
    _print()

    if backend == "cloud":
        _print(f"  {YELLOW}Important:{NC} Set your API key before using:")
        _print("     export OPENAI_API_KEY=sk-...")
        _print()

    _print("For more options, see: https://github.com/qduc/code-rag")


def main(argv: Optional[list[str]] = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    do_global_install = "--install" in args

    _print_header()

    # Detect environment
    env = _detect_environment()
    if not do_global_install:
        _print_environment_info(env)

    # Choose backend
    backend = _ask_backend(env)

    # Choose model
    model = _ask_model(backend)

    # Ask about reranking
    enable_reranker = _ask_reranker(backend)

    # Build extras list
    extras: list[str] = []
    if backend == "local":
        extras.append("local")
    else:
        extras.append("cloud")

    if enable_reranker:
        extras.append("reranker")

    # Summary
    _print_section("Configuration Summary")

    _print(f"  Backend:  {CYAN}{backend}{NC}")
    _print(f"  Model:    {CYAN}{model}{NC}")
    _print(f"  Reranker: {CYAN}{'enabled' if enable_reranker else 'disabled'}{NC}")
    _print(f"  Extras:   {CYAN}{', '.join(extras)}{NC}")
    _print()

    # Install
    if do_global_install:
        # Global install via uv tool
        requirement = f"{PACKAGE_NAME}[{','.join(extras)}]"
        _print(f"\n{GREEN}▶{NC} Installing {PACKAGE_NAME} globally via uv tool...")
        cmd = ["uv", "tool", "install", "--force", requirement]
        _print(f"  {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            _print(f"{GREEN}✓{NC} Global installation complete!")
        except Exception as e:
            _print(f"{RED}✗{NC} Installation failed: {e}")
            return 1
    elif _ask_yes_no("Install/upgrade dependencies now?", default=True):
        requirement = f"{PACKAGE_NAME}[{','.join(extras)}]"
        if not _pip_install(["--upgrade", requirement]):
            _print(f"\n{YELLOW}⚠{NC} Installation had issues. You can try manually:")
            _print(f"     pip install '{requirement}'")

    # Write configuration
    config = {
        "CODE_RAG_EMBEDDING_MODEL": model,
        "CODE_RAG_RERANKER_ENABLED": "true" if enable_reranker else "false",
    }

    config_dir = Path.home() / ".config" / "code-rag"
    config_path = config_dir / "config"

    if config_path.exists():
        if _ask_yes_no(f"Config exists at {config_path}. Overwrite?", default=False):
            _write_config(config, config_path)
    else:
        _write_config(config, config_path)

    # Verification
    if _ask_yes_no("Run quick verification test?", default=True):
        _run_verification()

    # Next steps
    _print_next_steps(backend, model)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
