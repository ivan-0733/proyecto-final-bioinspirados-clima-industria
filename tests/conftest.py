"""
Pytest configuration and shared fixtures.
"""
import pytest
import pandas as pd
from pathlib import Path


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for testing."""
    return pd.DataFrame({
        "age": [0, 1, 0, 1, 0],
        "gender": [1, 1, 0, 0, 1],
        "diabetes": [1, 1, 0, 1, 0]
    })


@pytest.fixture
def sample_supports():
    """Sample supports dictionary for testing."""
    return {
        "variables": {
            "age": {"0": 0.6, "1": 0.4},
            "gender": {"0": 0.4, "1": 0.6},
            "diabetes": {"0": 0.4, "1": 0.6}
        }
    }


@pytest.fixture
def sample_metadata():
    """Sample metadata for testing."""
    return {
        "feature_order": ["age", "gender", "diabetes"],
        "num_features": 3,
        "variables_info": [
            {"name": "age", "type": "categorical", "values": ["young", "old"]},
            {"name": "gender", "type": "categorical", "values": ["female", "male"]},
            {"name": "diabetes", "type": "categorical", "values": ["no", "yes"]}
        ]
    }


@pytest.fixture
def test_config_path():
    """Path to test configuration file."""
    return Path("config/escenario_1.json")


@pytest.fixture
def temp_results_dir(tmp_path):
    """Temporary results directory for testing."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    return results_dir


# Pytest markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


# Coverage thresholds
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Custom terminal summary."""
    try:
        if hasattr(config.option, 'cov') and config.option.cov:
            # Add custom summary if pytest-cov is installed
            terminalreporter.write_sep("=", "Coverage Summary")
            terminalreporter.write_line("Target: >90% overall coverage")
    except AttributeError:
        # pytest-cov not installed, skip
        pass
