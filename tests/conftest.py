"""Pytest configuration and shared fixtures for the tests package."""

from pathlib import Path
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))  # Ensure src/ is importable when running tests


def pytest_configure(config):
    """Register custom markers used throughout the tests."""
    config.addinivalue_line("markers", "asyncio: mark test as async")


@pytest.fixture(scope="session")
def event_loop():
    """Provide a dedicated event loop for the entire pytest session."""
    import asyncio

    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
