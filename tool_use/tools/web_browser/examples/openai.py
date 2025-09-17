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

from tool_use.tools.web_browser import (
    BrowserManager,
    click_button,
    enter_text_into_textbox,
    go_to_url,
    # press_enter_key,
    view_page_screenshot,
    view_section,
    text_search,
    WebBrowser,
)
from tool_use.tools.web_browser.examples.web_agents import SYSTEM_PROMPT
from utils import func_defn_as_json_schema

AGENT_TASKS: Final[list[str]] = [
    "Find me the names of the team members of stubben edge labs UK using duckduckgo",
    # (
    #     "Please find me the company number of the company Stubben Edge UK. "
    #     "Navigate to a useful page using the web browsing tools provided to you, then find "
    #     "the answer by extensively exploring the page."
    # ),
    # (
    #     "Go to https://en.wikipedia.org/wiki/List_of_serial_killers_by_number_of_victims, "
    #     "navigate to one of the URLs you see on that page and succinctly summarise the content "
    #     "of that page in a short bulleted list. Just choose a random one."
    # ),
    # "Go to hacker news, navigate to the 4th highest post and summarise it's content.",
    # (
    #     "Go to hacker news, find me a post about memory (if there are multiple just choose one), "
    #     "give me the URL, go to the linked article "
    #     "and summarise the content for me. You may have to check the first few pages of hacker "
    #     "news."
    # ),
    # "Go to companies house UK and find me contact phone numbers for EPSILON HEAT TRANSFER",
]
AGENT_TOOLS: Final[dict[str, Callable]] = {
    "click_button": click_button,
    "enter_text_into_textbox": enter_text_into_textbox,
    "go_to_url": go_to_url,
    # "press_enter_key": press_enter_key,
    "text_search": text_search,
    "view_page_screenshot": view_page_screenshot,
    "view_section": view_section,
}
MAX_N_AGENT_LOOPS: Final[int] = 10

print(
    f'Attempting to load credentials from .env file. success={dotenv.load_dotenv(".env", override=True)}'
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
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": agent_task},
                ]
                for _ in range(MAX_N_AGENT_LOOPS):
                    await asyncio.sleep(5)
                    llm_response = await llm_client.chat.completions.create(
                        model=os.environ["DEFAULT_MODEL"],
                        messages=messages_history,
                        tools=[
                            func_defn_as_json_schema(func)
                            for func in AGENT_TOOLS.values()
                        ],
                    )
                    msg_role: str = llm_response.choices[0].message.role
                    msg_content: str = llm_response.choices[0].message.content
                    messages_history.append(
                        {
                            "role": msg_role,
                            "content": msg_content,
                            "tool_calls": llm_response.choices[0].message.tool_calls,
                        },
                    )
                    if msg_role == "assistant" and msg_content:
                        print(msg_content)

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
                            if func_name == "view_page_screenshot":
                                messages_history.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": tool_call.id,
                                        "content": [
                                            {
                                                "type": "image_url",
                                                "image_url": {
                                                    "url": f"data:image/png:base64,{func_result}"
                                                },
                                            }
                                        ],
                                    }
                                )
                            else:
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
