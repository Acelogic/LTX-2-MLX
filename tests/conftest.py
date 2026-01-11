"""Pytest configuration and shared fixtures for LTX-2-MLX tests.

This module provides shared test configuration, fixtures, and utilities
for all test files in the test suite.
"""

import os
import sys
from pathlib import Path
from typing import Generator

import pytest


# Add project root to path so tests can import LTX_2_MLX
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class VerboseTestLogger:
    """Logger for verbose test output."""

    def __init__(self, test_name: str):
        self.test_name = test_name
        self.start_time = None

    def log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp."""
        import time

        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] [{level}] {self.test_name}: {message}")

    def start(self):
        """Mark test start."""
        import time

        self.start_time = time.time()
        self.log("Test started", "INFO")

    def end(self):
        """Mark test end and print duration."""
        import time

        if self.start_time:
            duration = time.time() - self.start_time
            self.log(f"Test completed in {duration:.2f}s", "INFO")

    def log_step(self, message: str):
        """Log a step in the test process."""
        self.log(message, "STEP")

    def log_info(self, message: str):
        """Log an informational message."""
        self.log(message, "INFO")


@pytest.fixture
def test_logger(request) -> Generator[VerboseTestLogger, None, None]:
    """Fixture providing a verbose test logger."""
    logger = VerboseTestLogger(request.node.name)
    logger.start()
    yield logger
    logger.end()


@pytest.fixture
def temp_output_dir(tmp_path) -> Generator[Path, None, None]:
    """Fixture providing a temporary output directory."""
    output_dir = tmp_path / "test_outputs"
    output_dir.mkdir(exist_ok=True)
    yield output_dir


@pytest.fixture(scope="session")
def weights_dir() -> Path:
    """Fixture providing path to weights directory."""
    return Path(__file__).parent.parent / "weights"


@pytest.fixture(scope="session")
def examples_dir() -> Path:
    """Fixture providing path to examples directory."""
    return Path(__file__).parent.parent / "examples"


def pytest_configure(config):
    """Pytest configuration hook."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "requires_weights: marks tests that require model weights (deselect with '-m \"not requires_weights\"')",
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (deselect with '-m \"not integration\"')",
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Auto-mark unit tests
        if "test_scheduler" in item.nodeid or "test_conditioning" in item.nodeid or "test_upscalers" in item.nodeid:
            item.add_marker(pytest.mark.unit)

        # Auto-mark integration tests
        if "test_video_generation" in item.nodeid:
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.requires_weights)
            item.add_marker(pytest.mark.slow)
