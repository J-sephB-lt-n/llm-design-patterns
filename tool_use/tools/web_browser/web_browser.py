"""
Tools which an AI agent can use to browse the web.
"""

import asyncio
import contextlib
from contextvars import ContextVar
from dataclasses import dataclass

import pydoll.browser.tab
from pydoll.browser.chromium import Chrome
from pydoll.browser.options import ChromiumOptions


@dataclass
class BrowserSessionState:
    """State of a single isolated persistent browser session."""

    browser_tab: pydoll.browser.tab.Tab
    current_url: str | None = None
    current_page_html: str | None = None
    current_page_img_b64: str | None = None


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
        str: base64-encoded image of page (after navigating to it and waiting for body to load)
    """
    current_session = current_browser_session.get()
    if not current_session:
        raise RuntimeError("No active browser session.")

    await current_session.browser_tab.go_to(url)
    body = await current_session.browser_tab.find(tag_name="body", timeout=30)
    await body.wait_until(is_visible=True)

    current_session.current_url = url
    current_session.current_page_html = await current_session.browser_tab.page_source
    current_session.current_page_img_b64 = (
        await current_session.browser_tab.take_screenshot(
            as_base64=True,
        )
    )
    return current_session.current_page_img_b64


async def scroll_down_on_current_page():
    raise NotImplementedError


async def search_current_page_hyperlinks(link_text_contains: str) -> list[str]:
    raise NotImplementedError
