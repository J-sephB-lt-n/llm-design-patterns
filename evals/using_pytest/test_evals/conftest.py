"""
Configuration shared by all pytest scripts in this folder (and it's subfolders)
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Final

import dotenv
import openai
import pytest


# get test (experiment) parameters from the user on the command line
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


# set up logger shared by all tests #
# (used to write test results to file)
@pytest.fixture(
    scope="package",  # teardown occurs only after all tests in the current test folder (and subfolders)
)
def test_logger(pytestconfig):
    """
    Configure logger once per test-session, before any tests run.
    """
    llm_name = pytestconfig.getoption("--model")
    llm_temp = pytestconfig.getoption("--temp")

    log_dir = Path(__file__).parent / "test_logs" / "llm_evals"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{datetime.now():%Y%m%d_%H%M%S}.log"

    logger = logging.getLogger("TEST_LOGGER")
    log_level = logging.INFO
    log_format = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    logger.setLevel(log_level)

    # Create a file handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(log_format)

    # Create a stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(log_format)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info(
        "\n=== Started LLM Evals ===\n\n"
        f"Model Params:\n"
        f"    name:        {llm_name}\n"
        f"    temperature: {llm_temp}\n"
    )

    yield logger

    logger.info("\n=== Finished LLM Evals ===\n")


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
            raise RuntimeError(error_msg)


# Large Language Model client which can be shared by all tests #
@pytest.fixture(
    scope="package",  # teardown occurs only after all tests in the current test folder (and subfolders)
)
def llm_client():
    return openai.OpenAI(  # or use openai.AsyncOpenAI()
        base_url=os.environ["OPENAI_API_BASE_URL"],
        api_key=os.environ["OPENAI_API_KEY"],
    )
