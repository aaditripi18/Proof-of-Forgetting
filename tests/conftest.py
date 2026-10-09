import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true",
                     help="also run tests marked slow (minutes on CPU; the Kaggle run uses this)")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: takes minutes on CPU; run with --runslow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="slow: run with --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
