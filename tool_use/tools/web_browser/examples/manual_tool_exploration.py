"""
CLI interface for manually testing out the tools available to a web browsing agent.
"""

import asyncio
from collections.abc import Callable
from typing import Final

from tool_use.tools.web_browser import (
    BrowserManager,
    go_to_url,
    view_section,
    WebBrowser,
)

AGENT_TOOLS: Final[dict[str, Callable]] = {
    "go_to_url": go_to_url,
    "view_section": view_section,
}


async def main():
    """Run the CLI loop."""
    browser_manager = BrowserManager()
    await browser_manager.start_browser()
    # await browser_manager.start_browser(browser_args=["--ignore-certificate-errors"])
    try:
        browser = WebBrowser(browser_manager)
        async with browser.isolated_browser_session():
            test = input("Please provide a URL: ")
            await go_to_url(test)
            print("going there")
            await asyncio.sleep(10)
    finally:
        await browser_manager.shutdown_browser()


if __name__ == "__main__":
    asyncio.run(
        main(),
    )
