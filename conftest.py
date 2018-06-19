import pytest


def pytest_addoption(parser):
    try:
        parser.addoption("--runslow", action="store_true",
                         default=False, help="run slow tests")
    except ValueError:
        print('did not add attribute run slow')


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
