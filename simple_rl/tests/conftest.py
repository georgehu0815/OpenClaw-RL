"""
Shared fixtures and markers for simple-rl tests.

Marks:
  @pytest.mark.unit        — no Docker or network needed
  @pytest.mark.integration — requires Docker daemon
"""
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: fast, no external deps")
    config.addinivalue_line("markers", "integration: requires Docker daemon")


# ── Minimal task fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def hello_task():
    return {
        "task_name": "hello_file",
        "instruction": "Create /tmp/hello.txt containing 'Hello World'",
        "grader": "[ -f /tmp/hello.txt ] && grep -qF 'Hello World' /tmp/hello.txt && echo 1.0 || echo 0.0",
        "docker_image": "ubuntu:22.04",
    }


@pytest.fixture
def trivial_task():
    """A task whose grader always returns 0.5 (for buffer tests)."""
    return {
        "task_name": "trivial",
        "instruction": "do anything",
        "grader": "echo 0.5",
        "docker_image": "ubuntu:22.04",
    }
