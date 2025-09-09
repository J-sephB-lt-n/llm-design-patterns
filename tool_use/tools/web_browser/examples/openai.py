"""
TODO.
"""

import asyncio
import json
import os
from collections.abc import Callable
from typing import Final

import dotenv
import openai

from tool_use.tools.web_browser import BrowserManager, WebBrowser, go_to_url
from utils import func_defn_as_json_schema

AGENT_TASKS: Final[list[str]] = [
    (
        "Go to https://en.wikipedia.org/wiki/List_of_serial_killers_by_number_of_victims, "
        "navigate to one of the URLs you see on that page and succinctly summarise the content "
        "of that page in a short bulleted list. Just choose a random one."
    ),
    "Go to hacker news, navigate to the 4th highest post and summarise it's content.",
]
AGENT_TOOLS: Final[dict[str, Callable]] = {
    "go_to_url": go_to_url,
}
MAX_N_AGENT_LOOPS: Final[int] = 5

print(
    f'Attempting to load credentials from .env file. success={dotenv.load_dotenv(".env")}'
)

llm_client = openai.AsyncOpenAI(
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"],
)


async def main():
    """Run the agent tasks."""
    browser_manager = BrowserManager()
    await browser_manager.start_browser(
        browser_args=["--ignore-certificate-errors"],
    )
    try:
        browser = WebBrowser(browser_manager)
        for agent_task in AGENT_TASKS:
            agent_failed: bool = False
            print(f"Started task '{agent_task}'")
            async with browser.isolated_browser_session():
                messages_history: list[dict] = [
                    {
                        "role": "system",
                        "content": """
You are a helpful assistant with access to the internet.

Functions available to you:
- For all page navigation, use the `go_to_url` tool.
                        """.strip(),
                    },
                    {"role": "user", "content": agent_task},
                ]
                for _ in range(MAX_N_AGENT_LOOPS):
                    llm_response = await llm_client.chat.completions.create(
                        model=os.environ["DEFAULT_MODEL"],
                        messages=messages_history,
                        tools=[
                            func_defn_as_json_schema(func)
                            for func in AGENT_TOOLS.values()
                        ],
                    )
                    messages_history.append(
                        {
                            "role": llm_response.choices[0].message.role,
                            "content": llm_response.choices[0].message.content,
                            "tool_calls": llm_response.choices[0].message.tool_calls,
                        },
                    )
                    tool_calls: list | None = llm_response.choices[0].message.tool_calls
                    if not tool_calls:
                        print("No tool calls - assumed finished.")
                        break
                    for tool_call in tool_calls:
                        try:
                            func_name: str = tool_call.function.name
                            func_kwargs: dict = json.loads(tool_call.function.arguments)
                            print(
                                f"Called tool [{func_name}] with kwargs {func_kwargs}"
                            )
                            func_result = await AGENT_TOOLS[func_name](**func_kwargs)
                            print("Tool used successfully")
                            messages_history.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": str(func_result),
                                }
                            )
                        except Exception as err:
                            error_type: str = type(err).__name__
                            error_message: str = str(err)
                            print(
                                f"Tool failed with error:\n{error_type}\n{error_message}"
                            )
                            error_info = {
                                "error_type": error_type,
                                "error_message": error_message,
                            }
                            messages_history.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": f"Tool call failed with error:\n{json.dumps(error_info, indent=4)}",
                                }
                            )
                else:
                    print(f"Exhausted {MAX_N_AGENT_LOOPS}.")
                    agent_failed = True

            print(
                f"Finished task '{agent_task}'",
                "(exhausted max n agent loops)" if agent_failed else "",
            )
            print(json.dumps(messages_history, indent=4, default=str))
            print("-" * 60)
            print()
    finally:
        await browser_manager.shutdown_browser()


if __name__ == "__main__":
    asyncio.run(
        main(),
    )
