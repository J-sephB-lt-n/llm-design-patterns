"""
Example LLM model evaluations using pytest
(uses some of the assets and configuration defined in conftest.py e.g. `test_logger`)
"""

import re
from typing import Final

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
    # You'd likely load from file or blob storage here in a real app
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
    llm_client,  # from conftest.py
    llm_params,  # from conftest.py
    test_logger,  # from conftest.py
):
    answer_correct_count: list[bool] = []

    for example in eval_data:
        test_logger.info("------ start of example ------")
        input_prompt: str = (
            f"Please solve the following multiplication problem: {example['input']}"
        )
        test_logger.info(f"INPUT PROMPT:\n{input_prompt}")
        llm_response_text = chat_completion_text(
            llm_client=llm_client,
            model_name=llm_params["llm_name"],
            model_temperature=llm_params["llm_temp"],
            user_message=input_prompt,
        )
        test_logger.info(f"LLM RESPONSE:\n{llm_response_text}")
        llm_response_only_numbers: str = re.sub(
            r"[^\d]+",  # remove all characters in response other than numbers
            "",
            llm_response_text,
        )
        if example["correct_answer"] in llm_response_only_numbers:
            test_logger.info("answer is CORRECT")
            answer_correct_count.append(True)
        else:
            test_logger.info(f"answer is WRONG. Expected {example['correct_answer']}")
            answer_correct_count.append(False)
        test_logger.info("------ end of example ------")

    accuracy_metric: float = sum(answer_correct_count) / len(answer_correct_count)

    test_logger.info(f"LLM ACCURACY METRIC: {accuracy_metric}")

    REQUIRED_ACCURACY: Final[float] = 0.8
    assert accuracy_metric >= REQUIRED_ACCURACY, (
        f"Required LLM accuracy on multiplication is {REQUIRED_ACCURACY} - observed accuracy={accuracy_metric}"
    )


# ... can add more tests here by defining more test_*() functions
