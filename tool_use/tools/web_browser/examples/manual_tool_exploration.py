"""
CLI interface for manually testing out the tools available to a web browsing agent.
"""

import asyncio
import inspect
import sys
from collections.abc import Callable
from typing import Final, get_type_hints

import questionary

from tool_use.tools.web_browser import (
    BrowserManager,
    enter_text_into_textbox,
    go_to_url,
    press_enter_key,
    text_search,
    view_section,
    WebBrowser,
)

AGENT_TOOLS: Final[dict[str, Callable]] = {
    "enter_text_into_textbox": enter_text_into_textbox,
    "go_to_url": go_to_url,
    "press_enter_key": press_enter_key,
    "text_search": text_search,
    "view_section": view_section,
}


def list_func_args(func: Callable) -> list[tuple[str, type]]:
    """List the arguments of `func` (and their types)."""
    sig = inspect.signature(func)
    hints = get_type_hints(func)
    args: list[tuple[str, type]] = []
    for name in sig.parameters:
        args.append(
            (name, hints[name]),
        )

    return args


async def main():
    """Run the CLI loop."""
    browser_manager = BrowserManager()
    await browser_manager.start_browser()
    # await browser_manager.start_browser(browser_args=["--ignore-certificate-errors"])
    try:
        browser = WebBrowser(browser_manager)
        async with browser.isolated_browser_session():
            while True:
                user_choice = await questionary.select(
                    "Please select a tool:",
                    choices=[*AGENT_TOOLS.keys(), "exit"],
                ).ask_async()
                if user_choice == "exit" or user_choice is None:
                    sys.exit(0)
                print("Please provide tool arguments:")
                func_kwargs = {}
                for arg_name, arg_type in list_func_args(AGENT_TOOLS[user_choice]):
                    func_kwargs[arg_name] = arg_type(
                        input(f"{arg_name} ({arg_type.__name__}): "),
                    )
                print(f"executing tool [{user_choice}]")
                func_result = await AGENT_TOOLS[user_choice](**func_kwargs)
                print(func_result)

    finally:
        await browser_manager.shutdown_browser()


if __name__ == "__main__":
    asyncio.run(
        main(),
    )
