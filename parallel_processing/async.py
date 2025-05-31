"""
Example of parallel chat completions using async
(technically concurrent not parallel but the speed gain is still massive)
"""

import asyncio
import os
from typing import Final, Iterable, Optional

import dotenv
import openai

dotenv.load_dotenv(".env")

POEM_TOPICS: Final[tuple[str, ...]] = (
    "obsolescence",
    "ignorance",
    "daymares",
    "the minute before sleep",
    "tiny lies",
)

async_llm_client = openai.AsyncOpenAI(
    base_url=os.environ["OPENAI_API_BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"],
)


async def async_create_poem(
    poem_topic: str,
    llm_client: openai.AsyncOpenAI,
) -> tuple[str, str]:
    """
    Generate a poem about `poem_topic`

    Returns:
        tuple[
            str,    # poem topic
            str     # poem
        ]
    """
    llm_response = await llm_client.chat.completions.create(
        model="azure.gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"write me a poem about {poem_topic} in the style of E.E. Cummings",
            },
        ],
    )
    return (
        poem_topic,
        llm_response.choices[0].message.content,
    )


async def async_create_poems(
    poem_topics: Iterable[str],
    llm_client: openai.AsyncOpenAI,
) -> tuple[dict[str, str], list[BaseException]]:
    """
    Generate multiple poems

    Returns:
        tuple[
            dict,                   # successful poems as {poem_topic: poem}
            list[BaseException]     # failed poems (list of exception objects)
        ]
    """
    tasks = [async_create_poem(topic, llm_client) for topic in poem_topics]
    results: list[tuple[str, str] | BaseException] = await asyncio.gather(
        *tasks,
        return_exceptions=True,  # errors are returned as Exception objects in results
    )

    poems = {}
    errors = []

    for result in results:
        if isinstance(result, BaseException):
            errors.append(result)
        else:
            topic, poem = result
            poems[topic] = poem

    return poems, errors


if __name__ == "__main__":
    async_poems, async_errors = asyncio.run(
        async_create_poems(
            poem_topics=POEM_TOPICS,
            llm_client=async_llm_client,
        )
    )

    for poem_topic, poem in async_poems.items():
        print("---", poem_topic.upper(), "---")
        print(poem)
        print("\n\n")
