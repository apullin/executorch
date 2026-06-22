# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import random
import sys
from importlib import import_module
from typing import Any

import pytest

"""
This file contains the pytest hooks, fixtures etc. for the Arm test suite.
"""


# ==== Pytest hooks ====


def pytest_configure(config):
    pytest._test_options = {}  # type: ignore[attr-defined]

    if getattr(config.option, "llama_inputs", False) and config.option.llama_inputs:
        pytest._test_options["llama_inputs"] = config.option.llama_inputs  # type: ignore[attr-defined]

    logging.basicConfig(stream=sys.stdout)

    if os.environ.get("ARM_TEST_ENABLE_ARM_TOSA_ENGINE_PATCH", "0") in (
        "",
        "0",
        "false",
        "False",
    ):
        return

    engine_root = os.environ.get("ARM_TOSA_ENGINE_ROOT")
    if engine_root and engine_root not in sys.path:
        sys.path.insert(0, engine_root)

    executorch_src_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")
    )
    if executorch_src_root not in sys.path:
        sys.path.insert(0, executorch_src_root)

    integration_module_name = os.environ.get(
        "ARM_TOSA_ENGINE_IMPORT", "fast_preprocess"
    )

    try:
        integration_module = import_module(integration_module_name)
        install = getattr(
            integration_module,
            "install",
            getattr(integration_module, "patch_tosa_backend", None),
        )
        if not callable(install):
            raise AttributeError(
                f"{integration_module_name} does not expose install() or patch_tosa_backend()"
            )
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "ARM test lane requested arm-tosa-engine patching, but the "
            f"integration module '{integration_module_name}' could not be "
            "loaded. Set ARM_TOSA_ENGINE_ROOT or ARM_TOSA_ENGINE_IMPORT and "
            "install the arm-tosa-engine package in the active environment."
        ) from exc

    install()


def pytest_collection_modifyitems(config, items):
    pass


def pytest_addoption(parser):
    def try_addoption(*args, **kwargs):
        try:
            parser.addoption(*args, **kwargs)
        except Exception:  # nosec B110 - pytest redefines options, safe to ignore
            pass

    try_addoption("--arm_quantize_io", action="store_true", help="Deprecated.")
    try_addoption("--arm_run_corstoneFVP", action="store_true", help="Deprecated.")
    try_addoption(
        "--llama_inputs",
        nargs="+",
        help="List of two files. Firstly .pt file. Secondly .json",
    )


def pytest_sessionstart(session):
    pass


def pytest_sessionfinish(session, exitstatus):
    pass


# ==== End of Pytest hooks =====


# ==== Pytest fixtures =====


@pytest.fixture(autouse=True)
def set_random_seed():
    """Control random numbers in Arm test suite. Default behavior is to use a
    fixed seed (0), which ensures reproducible tests. Use the env variable
    ARM_TEST_SEED to set a custom seed, or set it to RANDOM for random seed
    behavior.

    Examples:
    As default use fixed seed (0) for reproducible tests
        pytest --config-file=/dev/null --verbose -s --color=yes  backends/arm/test/ops/test_avg_pool.py -k <TESTCASE>
    Use a random seed for each test
        ARM_TEST_SEED=RANDOM pytest --config-file=/dev/null --verbose -s --color=yes  backends/arm/test/ops/test_avg_pool.py -k <TESTCASE>
    Rerun with a specific seed
        ARM_TEST_SEED=3478246 pytest --config-file=/dev/null --verbose -s --color=yes  backends/arm/test/ops/test_avg_pool.py -k <TESTCASE>

    """
    import torch

    seed_env = os.environ.get("ARM_TEST_SEED", "0")
    if seed_env == "RANDOM":
        random.seed()  # reset seed, in case any other test has fiddled with it
        seed = random.randint(0, 2**32 - 1)  # nosec B311 - non-crypto seed for tests
        torch.manual_seed(seed)
    elif str.isdigit(seed_env):
        seed = int(seed_env)
        random.seed(seed)
        torch.manual_seed(seed)
    else:
        raise TypeError(
            "ARM_TEST_SEED env variable must be integers or the string RANDOM"
        )

    print(f" ARM_TEST_SEED={seed} ", end=" ")


# ==== End of Pytest fixtures =====


def is_option_enabled(option: str, fail_if_not_enabled: bool = False) -> bool:
    """Returns whether an option is successfully enabled, i.e. if the flag was
    given to pytest and the necessary requirements are available.

    The optional parameter 'fail_if_not_enabled' makes the function raise a
    RuntimeError instead of returning False.

    """

    if hasattr(pytest, "_test_options") and option in pytest._test_options and pytest._test_options[option]:  # type: ignore[attr-defined]
        return True
    else:
        if fail_if_not_enabled:
            raise RuntimeError(f"Required option '{option}' for test is not enabled")
        else:
            return False


def get_option(option: str) -> Any | None:
    """Returns the value of an pytest option if it is set, otherwise None.

    Args:
        option (str): The option to check for.

    """
    if option in pytest._test_options:  # type: ignore[attr-defined]
        return pytest._test_options[option]  # type: ignore[attr-defined]
    return None
