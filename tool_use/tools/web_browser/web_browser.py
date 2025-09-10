"""
Tools which an AI agent can use to browse the web.
"""

import asyncio
import contextlib
from contextvars import ContextVar
from dataclasses import dataclass
from itertools import batched

import pydoll.browser.tab
from markdownify import markdownify as md
from pydoll.browser.chromium import Chrome
from pydoll.browser.options import ChromiumOptions


def split_text_into_pages(text: str, n_lines_per_page: int) -> list[str]:
    """Split `text` into substrings containing `n_lines_per_page` lines of text."""
    lines: list[str] = text.splitlines()
    pages: list[str] = []
    for batch in batched(lines, n=n_lines_per_page):
        pages.append("\n".join(batch))

    return pages


@dataclass
class BrowserSessionState:
    """State of a single isolated persistent browser session."""

    browser_tab: pydoll.browser.tab.Tab
    url: str | None = None
    html: str | None = None
    text: str | None = None
    text_paged: list[str] | None = None


current_browser_session: ContextVar[BrowserSessionState | None] = ContextVar(
    "current_browser_session", default=None
)


class BrowserManager:
    """A singleton manager for the global web browser instance."""

    def __init__(self):
        self._browser: Chrome | None = None
        self._async_lock = asyncio.Lock()

    async def start_browser(
        self,
        browser_args: list[str] | None = None,
    ) -> None:
        """
        Start the global web browser.

        Args:
            browser_args (list[str]): e.g. '--headless', '--ignore-certificate-errors'
        """
        if self._browser:
            raise RuntimeError("The global web browser has already started.")

        browser_args = browser_args or []

        async with self._async_lock:
            browser_options = ChromiumOptions()
            for arg in browser_args:
                browser_options.add_argument(arg)

            self._browser = Chrome(options=browser_options)
            await self._browser.start()

    async def shutdown_browser(
        self,
    ) -> None:
        """Stop the global web browser."""
        if not self._browser:
            raise RuntimeError("The global web browser was never started.")

        async with self._async_lock:
            await self._browser.stop()

    async def new_session(self) -> pydoll.browser.tab.Tab:
        """
        Start a new isolated browser session.
        (like a new incognito tab)
        """
        context_id = await self._browser.create_browser_context()
        new_tab = await self._browser.new_tab(browser_context_id=context_id)

        return new_tab


class WebBrowser:
    """Used for creating browser sessions."""

    def __init__(self, browser_manager: BrowserManager) -> None:
        self._browser_manager = browser_manager

    @contextlib.asynccontextmanager
    async def isolated_browser_session(self):
        """
        Async context manager providing an isolated and persistent browser session
        (i.e. not sharing cookies, local storage etc. with any other session)
        """
        token = current_browser_session.set(
            BrowserSessionState(
                browser_tab=await self._browser_manager.new_session(),
            ),
        )

        try:
            yield
        finally:
            current_browser_session.reset(token)


async def go_to_url(url: str) -> str:
    """
    Navigate to a URL in the current browser session.
    You must provide the full URL (including the protocol) e.g. 'https://www.example.com'

    Returns:
        str: A status message, including the text content of the first section.
    """
    current_session = current_browser_session.get()
    if not current_session:
        raise RuntimeError("No active browser session.")

    await current_session.browser_tab.go_to(url)
    body = await current_session.browser_tab.find(tag_name="body", timeout=30)
    await body.wait_until(is_visible=True)

    current_session.url = url
    current_session.html = await current_session.browser_tab.page_source
    current_session.text = md(current_session.html)
    current_session.text_paged = split_text_into_pages(
        current_session.text, n_lines_per_page=50
    )

    return f"""
Successfully navigated to web page {url}

For size reasons, the web page text has been split into multiple sections.
Here is the text content of section 1 of {len(current_session.text_paged)}:
```
{current_session.text_paged[0]}
```
    """.strip()


async def view_section(section_num: int) -> str:
    """
    For size reasons, the full web page text has been partitioned.
    Use this function to view a specific one of the partitions.

    Args:
        section_num (int): Number (identifier) of partition to view (the first partition is `1`).

    Returns:
        str: The text content of the partition.
    """
    current_session = current_browser_session.get()
    if not current_session:
        raise RuntimeError("No active browser session.")
    if current_session.url is None:
        return "Please navigate to a URL first."
    if section_num < 0 or section_num > len(current_session.text_paged):
        return (
            "Invalid section choice. "
            f"Please choose a value between 1 and {len(current_session.text_paged)}"
        )

    return f"""
Text content of section {section_num} of web page {current_session.url}:
```
{current_session.text_paged[section_num]}
```
""".strip()


async def text_search(regex_pattern: str) -> list[str]:
    """
    Finds all pages
    """


async def search_current_page_hyperlinks(link_text_contains: str) -> list[str]:
    raise NotImplementedError
