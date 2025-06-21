"""
Example of an LLM client class which logs all LLM token usage
"""

import json
import os
from collections import namedtuple

import dotenv
import openai

TokenUsageRecord = namedtuple(
    "TokenUsageRecord",
    ["model_name", "search_tags", "token_usage"],
)


class LLM:
    """
    Wrapper around openai.OpenAI() which logs all token usage
    """

    def __init__(self, api_base_url: str, api_key: str):
        self._llm_client = openai.OpenAI(
            base_url=api_base_url,
            api_key=api_key,
        )
        self.token_usage_history: list = []

    def chat(self, search_tags: tuple[str, ...], *args, **kwargs):
        """
        Wraps openai.OpenAI().chat.completions.create(), logging all token usage

        Args:
            search_tags (tuple[str, ...]): Tags for later search or aggregation of token history
            *args/**kwargs: Arguments passed to openai.OpenAI().chat.completions.create()
        """
        llm_response = self._llm_client.chat.completions.create(
            *args,
            **kwargs,
        )

        self.token_usage_history.append(
            TokenUsageRecord(
                model_name=llm_response.model,
                search_tags=search_tags,
                token_usage=llm_response.usage,
            )
        )

        return llm_response


if __name__ == "__main__":
    dotenv.load_dotenv(".env")

    llm = LLM(
        api_base_url=os.environ["OPENAI_API_BASE_URL"],
        api_key=os.environ["OPENAI_API_KEY"],
    )

    for topic in ("ignorance", "systems", "earwigs"):
        print(
            llm.chat(
                search_tags=("poem", "poetry"),
                model=os.environ["DEFAULT_MODEL_NAME"],
                temperature=1,
                messages=[
                    {
                        "role": "user",
                        "content": f"Write me a 4 line poem about {topic}",
                    },
                ],
            )
            .choices[0]
            .message.content
        )
    for entry in llm.token_usage_history:
        print(
            json.dumps(
                entry._asdict(),
                indent=4,
                default=lambda x: x.model_dump(),
            )
        )
