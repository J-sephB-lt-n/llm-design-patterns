"""
Lightweight model evaluations can live alongside the standard software tests in pytest

e.g.

tests/
├── end2end/
│   ├── end2end_test1_name.py
│   └── ...
├── integration/
│   ├── integration_test1_name.py
│   └── ...
├── llm_evals/                       < ---------- this one
│   ├── llm_eval1_name.py
│   └── ...
└── unit/
    ├── unit_test1_name.py
    └── ...

Run this example test script using:
    uv run pytest evals/test_basic_evals_using_pytest.py --model gpt-4o --temp 0

Notes:
    - pytest "fixtures" are resources which are created once and shared by multiple tests
        (e.g. environment variables, datasets, network connections etc.)
    - Where you have many evals, rather put them into multiple independent test_*.py scripts
        - They can share setup config by giving them a shared `conftest.py` script
            (e.g. all test scripts can use the same LLM config supplied once)
        - You can have multiple conftest.py files (each applies to the test folder it's in)
    - Not shown here, but pytest can be set up to write the results to an experiment tracking platform e.g. MLFlow or wandb
"""

import re

import openai
import pytest


# load data to use in the tests in this script
@pytest.fixture(
    scope="module",  # teardown occurs only after all tests in this script have finished
)
def eval_data(test_logger):
    """
    Use this function to load your evaluation data
    (loaded once at start and will be available to all tests)
    """
    # this example creates the data in place.
    # You'd like load from file or blob storage here in a real app
    examples: list[dict] = [
        {
            "input": "6 * 9",
            "correct_answer": "54",
        },
        {
            "input": "2387428734 * 69420",
            "correct_answer": "165735302714280",
        },
    ]
    test_logger.info(f"Loaded {len(examples):,} evaluation examples")

    return examples


def chat_completion_text(
    llm_client: openai.OpenAI,
    model_name: str,
    model_temperature: float,
    user_message: str,
) -> str:
    """
    Helper function which returns just the text response from the LLM chat completion
    """
    api_response = llm_client.chat.completions.create(
        model=model_name,
        temperature=model_temperature,
        messages=[
            {
                "role": "system",
                "content": "You never make a mistake in arithmetical calculations",
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
    )
    return api_response.choices[0].message.content


def test_evaluate_llm_arithmetical_accuracy(
    # specify which fixtures this test needs #
    eval_data,
    llm_client,
    llm_params,
    test_logger,
):
    answer_correct_count: list[bool] = []

    for example in eval_data:
        llm_response_text = chat_completion_text(
            llm_client=llm_client,
            model_name=llm_params["llm_name"],
            model_temperature=llm_params["llm_temp"],
            user_message=f"Please solve the following multiplication problem: {example['input']}",
        )
        llm_response_only_numbers: str = re.sub(
            r"[^\d]+",  # remove all characters in response other than numbers
            "",
            llm_response_text,
        )
        # assert (
        #     example["correct_answer"] in llm_response_only_numbers
        # ), f'Expected {example["correct_answer"]} in LLM response. LLM response:\n{llm_response_text}'
        if example["correct_answer"] in llm_response_only_numbers:
            answer_correct_count.append(True)
        else:
            answer_correct_count.append(False)

    accuracy_metric: float = sum(answer_correct_count) / len(answer_correct_count)

    test_logger.info(f"LLM ACCURACY METRIC: {accuracy_metric}")

    assert accuracy_metric >= 0.8, (
        f"Required LLM accuracy on multiplication is 0.5 - observed accuracy={accuracy_metric}"
    )
