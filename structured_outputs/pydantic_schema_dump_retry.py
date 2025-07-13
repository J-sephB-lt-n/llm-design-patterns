"""
Example of enforcing output to adhere to a given JSON schema, in the style \
of https://github.com/567-labs/instructor 
"""

import copy
import json
import logging
import os
import re
from typing import Final

import dotenv
import openai
import pydantic
from loguru import logger
from pydantic import BaseModel, Field


class StructuredOutputError(Exception):
    """
    Error raised when language model produces output not adhering to the required JSON schema
    """

    pass


def inject_structured_output_prompt_instructions(
    messages: list[dict],
    response_model: BaseModel,
) -> None:
    """
    Appends structured output instructions instructions to the last message in `messages`
    (instructions on required output format)
    """
    SCHEMA_REQUIREMENT_INSTRUCTION: Final[
        str
    ] = f"""
Your response must include a single JSON markdown code block (containing valid JSON) whose \
contents adheres to the following constraints:
<non-negotiable-response-json-constraints>
```json
{json.dumps(response_model.model_json_schema(), indent=4)}
```
</non-negotiable-response-json-constraints>

The contents of your JSON response will be parsed using python pydantic with the command \
`{response_model.__name__}.model_validate_json()`, \
and doing so must not raise a pydantic.ValidationError.
        """
    msg_to_augment = messages[-1]
    match msg_to_augment["content"]:
        case str():  # text-only input
            msg_to_augment["content"] += "\n" + SCHEMA_REQUIREMENT_INSTRUCTION
        case list():  # multimodal input
            for msg in msg_to_augment["content"]:
                if msg["type"] == "text":
                    msg["text"] += "\n" + SCHEMA_REQUIREMENT_INSTRUCTION
                    return
            raise ValueError(
                "Could not find text content in last chat completion message"
            )
        case _:
            raise ValueError(
                'unexpected type: type(messages[-1]["content"]) is '
                + f"{type(messages[-1]['content'])}"
            )


def structured_output_chat_completion(
    response_model: pydantic.BaseModel,
    max_n_retries: int,
    messages: list[dict],
    llm_client: openai.OpenAI,
    chat_kwargs: dict,
    logger: logging.Logger,
) -> pydantic.BaseModel:
    """
    Generate a chat completion returning valid JSON with schema adhering to `response_model`

    Raises:
        StructuredOutputError
    """

    messages_history: list[dict] = copy.deepcopy(messages)
    inject_structured_output_prompt_instructions(
        response_model=response_model,
        messages=messages_history,
    )
    logger.debug("\n" + json.dumps(messages_history, indent=4))

    llm_response = llm_client.chat.completions.create(
        messages=messages_history,
        **chat_kwargs,
    )

    for _ in range(1, max_n_retries + 2):
        try:
            find_json = re.search(
                r"```json\s*(?P<json_content>.*?)```",
                llm_response.choices[0].message.content,
                re.DOTALL,
            )
            if not find_json:
                raise ValueError("No JSON markdown code block found.")

            return response_model.model_validate_json(
                find_json.group("json_content").strip(),
            )

        except pydantic.ValidationError as pydantic_error:
            format_pydantic_errors: list[dict] = [
                {
                    "field": err["loc"][0] if err.get("loc") else None,
                    "type": err["type"],
                    "message": err["msg"],
                }
                for err in pydantic_error.errors()
            ]
            error_string = json.dumps(format_pydantic_errors, indent=4)

        except ValueError as error:
            error_string = str(error)

        messages_history.append(
            {
                "role": "user",
                "content": error_string,
            },
        )
        llm_response = llm_client.chat.completions.create(
            messages=messages_history,
            **chat_kwargs,
        )

    raise StructuredOutputError(
        f"Language model failed to return required JSON schema in {max_n_retries + 1} attempts",
    )


if __name__ == "__main__":
    max_n_retries: int = 2
    user_query: str = input("Enter your question: ")

    class RequiredResponseSchema(BaseModel):
        question_category: str = Field(
            description="A label classifying the user query into a useful category for later retrieval",
        )
        tags: list[str] = Field(
            description="Topic tags relevant to the user query which will be used for later retrieval",
        )
        response: str = Field(
            description="The answer to the user's question",
        )

    dotenv.load_dotenv(".env")

    llm_client = openai.OpenAI(
        base_url=os.environ["OPENAI_API_BASE_URL"],
        api_key=os.environ["OPENAI_API_KEY"],
    )

    structured_llm_response = structured_output_chat_completion(
        response_model=RequiredResponseSchema,
        max_n_retries=2,
        llm_client=llm_client,
        chat_kwargs={
            "model": os.environ["DEFAULT_MODEL_NAME"],
            "temperature": float(os.environ["DEFAULT_MODEL_TEMPERATURE"]),
        },
        messages=[
            {
                "role": "system",
                "content": "You are a meticulous worker who follows instructions to the letter.",
            },
            {
                "role": "user",
                "content": user_query,
            },
        ],
        logger=logger,
    )

    print(
        json.dumps(
            structured_llm_response.model_dump(),
            indent=4,
        )
    )
