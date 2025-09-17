"""
Simple implementation of an agent using just the openai python API client
A for loop with tools and full chat history.
"""

import json
import os
from collections.abc import Callable
from typing import Final

import openai

from tool_use.tools.web_browser import (
    BrowserManager,
    click_button,
    enter_text_into_textbox,
    go_to_url,
    text_search,
    view_page_screenshot,
    view_section,
    WebBrowser,
)
from tool_use.tools.web_browser.examples.web_agents import SYSTEM_PROMPT
from utils import func_defn_as_json_schema


AGENT_TOOLS: Final[dict[str, Callable]] = {
    "click_button": click_button,
    "enter_text_into_textbox": enter_text_into_textbox,
    "go_to_url": go_to_url,
    # "press_enter_key": press_enter_key,
    "text_search": text_search,
    "view_page_screenshot": view_page_screenshot,
    "view_section": view_section,
}


async def simple_openai_agent(
    task: str,
    max_n_agent_loops: int = 20,
) -> None:
    """A for loop with tools that includes the full chat history in every chat completion."""
    print(f"SYSTEM PROMPT: \n{SYSTEM_PROMPT}")

    llm = openai.AsyncOpenAI(
        base_url=os.environ["OPENAI_BASE_URL"],
        api_key=os.environ["OPENAI_API_KEY"],
    )

    messages_history: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]

    browser_manager = BrowserManager()
    await browser_manager.start_browser()
    try:
        browser = WebBrowser(browser_manager)
        async with browser.isolated_browser_session():
            for loop_idx in range(max_n_agent_loops):
                is_finished: bool = await run_agent_loop(
                    llm=llm, messages_history=messages_history
                )
                if is_finished:
                    print(
                        f"SUCCESS: Agent completed task after {loop_idx + 1} iterations."
                    )
                    break
            else:
                print(
                    f"FAILURE: Agent did not reach a solution after {max_n_agent_loops} iterations."
                )

    finally:
        await browser_manager.shutdown_browser()


async def run_agent_loop(llm: openai.AsyncOpenAI, messages_history: list[dict]) -> bool:
    """
    Run a single iteration of the agent for loop.

    Returns:
        bool: `True` if the agent considers the task finished.
    """
    llm_response = await llm.chat.completions.create(
        model=os.environ["DEFAULT_MODEL"],
        messages=messages_history,
        tools=[func_defn_as_json_schema(func) for func in AGENT_TOOLS.values()],
    )

    msg_role: str = llm_response.choices[0].message.role
    msg_content: str = llm_response.choices[0].message.content
    tool_calls: list | None = llm_response.choices[0].message.tool_calls
    messages_history.append(
        {
            "role": msg_role,
            "content": msg_content,
            "tool_calls": tool_calls,
        },
    )

    if msg_role == "assistant" and msg_content:
        print(msg_content)

    if not tool_calls:
        print("No tool calls - assumed finished.")
        return True

    for tool_call in tool_calls:
        try:
            func_name: str = tool_call.function.name
            func_kwargs: dict = json.loads(tool_call.function.arguments)
            print(f"Called tool [{func_name}] with kwargs {func_kwargs}")
            func_result = await AGENT_TOOLS[func_name](**func_kwargs)
            print("Tool used successfully")
            if func_name == "view_page_screenshot":
                messages_history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": "Successfully captured page screenshot",
                    }
                )
                messages_history.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Here is the page screenshot.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{func_result}"
                                },
                            },
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
            print(f"Tool failed with error:\n{error_type}\n{error_message}")
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
