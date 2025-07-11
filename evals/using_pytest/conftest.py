"""
Configuration shared by all pytest scripts in this folder (and it's subfolders)
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Final

import dotenv
import openai
import pytest
from loguru import logger


# get test (experiment) parameters from the user
def pytest_addoption(parser):
    """
    User must supply test (experiment) parameters from the command line
    e.g. pytest tests/test_script_name.py --model gpt-4o --temp 0.5
    """
    parser.addoption(
        "--model",
        action="store",
        type=str,
        help="Large Language Model name",
    )
    parser.addoption(
        "--temp",
        action="store",
        type=float,
        help="Large Language Model temperature",
    )


# # r and log test (experiment) parameters
# def pytest_configure(config):
#     """
#     Code which runs before any tests start
#     """
#     llm_name: str = config.getoption("--model")
#     llm_temp: float = config.getoption("--temp")


@pytest.fixture(
    scope="package",
    autouse=True,
)
def test_logger(pytestconfig):
    """
    Configure logger once per test-session, before any tests run.
    """
    llm_name = pytestconfig.getoption("--model")
    llm_temp = pytestconfig.getoption("--temp")

    # build a timestamped path
    log_dir = Path(__file__).parent / "test_logs" / "llm_evals"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{datetime.now():%Y%m%d_%H%M%S}.log"

    logger.add(
        log_path,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:^7} | {message}",
        rotation=None,  # overwrite each run
        backtrace=True,  # include full tracebacks
        diagnose=False,  # no extra debug fluff
    )

    logger.info(
        "\n=== Started LLM Evals ===\n\n"
        f"Model Params:\n"
        f"    name:        {llm_name}\n"
        f"    temperature: {llm_temp}\n"
    )

    return logger


def pytest_sessionfinish(session, exitstatus):
    """
    This code runs after all tests have finished
    """
    logger.info(
        """
        === Finished LLM Evals ===
        """
    )


# Put model parameters into a fixture so multiple tests can refer to them #
@pytest.fixture(
    scope="package",  # teardown occurs only after all tests in the current test folder (and subfolders)
)
def llm_params(pytestconfig):
    """
    Large Language Model parameters
    """
    return {
        "llm_name": pytestconfig.getoption("--model"),
        "llm_temp": pytestconfig.getoption("--temp"),
    }


# get environment variables shared by all tests #
@pytest.fixture(
    scope="package",  # teardown occurs only after all tests in the current test folder (and subfolders)
    autouse=True,  # available to all tests without needed to explicitly declare they want it
)
def env_setup():
    REQUIRED_ENV_VARS: Final[tuple[str, ...]] = (
        "OPENAI_API_BASE_URL",
        "OPENAI_API_KEY",
    )
    if any(os.getenv(var) is None for var in REQUIRED_ENV_VARS):
        # if any environment variables missing, load from local ".env" file #
        dotenv.load_dotenv(".env")

    for var in REQUIRED_ENV_VARS:
        if os.getenv(var) is None:
            error_msg: str = (
                f"Could not find environment variable '{var}' in ENV or .env file"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)


# Large Language Model client which can be shared by all tests #
@pytest.fixture(
    scope="module",  # teardown occurs only after all tests in this script have finished
)
def llm_client():
    return openai.OpenAI(  # or use openai.AsyncOpenAI()
        base_url=os.environ["OPENAI_API_BASE_URL"],
        api_key=os.environ["OPENAI_API_KEY"],
    )
