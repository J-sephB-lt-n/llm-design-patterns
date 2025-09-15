"""
Tools which an AI agent can use to browse the web.
"""

import asyncio
import contextlib
import itertools
import re
from contextvars import ContextVar
from dataclasses import dataclass
from itertools import batched
from typing import Literal

import bs4
import pydoll.browser.tab
from markdownify import MarkdownConverter
from pydoll.browser.chromium import Chrome
from pydoll.browser.options import ChromiumOptions
from pydoll.exceptions import PydollException


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
    url: str | None = None  # current url
    html: str | None = None  # HTML of current page
    text: str | None = None  # text of current page
    text_paged: list[str] | None = None  # text of current page partitioned into chunks
    tag_counters: dict[str, itertools.count] | None = None
    tag_ids: list[str] | None = None


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


class CustomMarkdownConverter(MarkdownConverter):
    """
    Makes markdownify render specific HTML tags in a custom way.
    I'm using it to highlight interactable elements (e.g. buttons, text inputs) each with a
    unique ID which the agent can use to interact with them.
    """

    ATTRIBUTES_TO_RENDER = {
        "a": ["title", "target", "rel", "download"],
        "button": ["name", "value", "type", "disabled"],
        "input": [
            "type",
            "name",
            "value",
            "placeholder",
            "checked",
            "disabled",
            "readonly",
            "required",
        ],
        "textarea": [
            "name",
            "placeholder",
            "rows",
            "cols",
            "disabled",
            "readonly",
            "required",
        ],
        "select": ["name", "multiple", "disabled"],
        "option": ["value", "selected"],
        "img": ["alt", "src", "title"],
        "iframe": ["src", "title"],
        # global attributes (always included)
        "*": [
            "id",
            "title",
            "role",
            "aria-hidden",
            "aria-label",
            "aria-expanded",
            "aria-current",
            "aria-haspopup",
        ],
    }

    def __init__(self, **options) -> None:
        super().__init__(**options)

        self.tag_counters = options["tag_counters"]
        self.tag_ids = options["tag_ids"]

    def convert_button(self, el: bs4.Tag, text: str, parent_tags) -> str:
        """Custom markdown rendering of <button> tags."""
        new_tag_id: str = f'button_{next(self.tag_counters["button"])}'
        self.tag_ids.append(new_tag_id)
        attributes_str: str = self.generate_attributes_string(
            tag_type="button", soup_obj=el
        )

        min_text: str = re.sub(r"\s+", " ", text.strip())
        return f"<button tag_id={new_tag_id} text='{min_text}' {attributes_str}>"

    def convert_img(self, el: bs4.Tag, text, parent_tags) -> str:
        """Custom markdown rendering of <img> tags."""
        src = el.get("src", "")
        if src.startswith("data:image"):
            return ""

        return super().convert_img(el, text, parent_tags)

    def convert_input(self, el: bs4.Tag, text, parent_tags) -> str:
        """Custom markdown rendering of <input> tags."""
        new_tag_id: str = f'input_{next(self.tag_counters["input"])}'
        self.tag_ids.append(new_tag_id)
        attributes_str: str = self.generate_attributes_string(
            tag_type="input", soup_obj=el
        )

        min_text: str = re.sub(r"\s+", " ", text.strip())
        return f"<input tag_id={new_tag_id} text='{min_text}' {attributes_str}>"

    def convert_textarea(self, el: bs4.Tag, text, parent_tags) -> str:
        """Custom markdown rendering of <textarea> tags."""
        new_tag_id: str = f'textarea_{next(self.tag_counters["textarea"])}'
        self.tag_ids.append(new_tag_id)
        attributes_str: str = self.generate_attributes_string(
            tag_type="input", soup_obj=el
        )

        min_text: str = re.sub(r"\s+", " ", text.strip())
        return f"<textarea tag_id={new_tag_id} text='{min_text}' {attributes_str}>"

    def generate_attributes_string(
        self,
        tag_type: Literal["button", "img", "input", "textarea"],
        soup_obj: bs4.Tag,
    ) -> str:
        """
        Generate a string containing the attributes of object `soup_obj`.
        (This string is intended to be included in markdownified HTML).
        Only the attributes specified in `self.ATTRIBUTES_TO_RENDER` are included.
        """
        attributes_to_render = []
        allowed_attributes = [
            *self.ATTRIBUTES_TO_RENDER[tag_type],
            *self.ATTRIBUTES_TO_RENDER["*"],
        ]
        for attr, value in soup_obj.attrs.items():
            if attr in allowed_attributes:
                if isinstance(value, list):
                    value = " ".join(value)

                if value:
                    attributes_to_render.append(f'{attr}="{value}"')
                else:
                    attributes_to_render.append(attr)

        attributes_str = " ".join(attributes_to_render)
        return attributes_str


def markdownify_custom(html: str, **options) -> str:
    """
    Convert `html` to a markdown string using CustomMarkdownConverter.
    """
    return CustomMarkdownConverter(**options).convert(html)


async def refresh_page_view(current_session: BrowserSessionState) -> None:
    """
    Update the text representations of the current webpage to reflect the
    actual page state.
    """
    current_session.tag_counters = {
        "input": itertools.count(start=1),
        "textarea": itertools.count(start=1),
        "button": itertools.count(start=1),
    }
    current_session.tag_ids = []
    current_session.url = await current_session.browser_tab.current_url
    current_session.html = await current_session.browser_tab.page_source
    # replace multiple blank lines with a single blank line #
    current_session.text = re.sub(
        r"(\n\s*){2,}",
        r"\n\n",
        markdownify_custom(
            current_session.html,
            tag_counters=current_session.tag_counters,
            tag_ids=current_session.tag_ids,
        ).strip(),
    )
    current_session.text_paged = split_text_into_pages(
        current_session.text, n_lines_per_page=50
    )


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
    await asyncio.sleep(3)  # to cater for redirects
    await body.wait_until(is_visible=True)

    await refresh_page_view(current_session)

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
    Use this function to view a specific one of the partitions (sections).

    Args:
        section_num (int): Number (identifier) of partition to view (the first partition is `1`).

    Returns:
        str: The text content of the partition.
    """
    current_session = current_browser_session.get()

    await refresh_page_view(current_session)

    if not current_session:
        raise RuntimeError("No active browser session.")
    if current_session.url is None:
        return "Please navigate to a URL first."
    if section_num < 1 or section_num > len(current_session.text_paged):
        return (
            "Invalid section choice. "
            f"Please choose a value between 1 and {len(current_session.text_paged)}"
        )

    return f"""
Text content of section {section_num} (of {len(current_session.text_paged):,}) \
of web page {current_session.url}:
```
{current_session.text_paged[section_num-1]}
```
""".strip()


async def text_search(regex_pattern: str, ignore_case: bool = True) -> str:
    """
    Finds all sections (partitions) matching `regex_pattern`.

    Args:
        regex_pattern (str): The regex pattern to search for.
        ignore_case (bool): Whether to perform a case-insensitive search.
    """
    current_session = current_browser_session.get()
    if not current_session:
        raise RuntimeError("No active browser session.")
    if current_session.url is None:
        return "Please navigate to a URL first."

    await refresh_page_view(current_session)

    flags = re.DOTALL
    if ignore_case:
        flags |= re.IGNORECASE

    match_sections: list[int] = []
    for section_num, section_text in enumerate(current_session.text_paged, start=1):
        if re.search(regex_pattern, section_text, flags):
            match_sections.append(section_num)

    if not match_sections:
        return (
            f"No matches found for regex pattern '{regex_pattern}' in any section text."
        )

    return f"""
The following sections contained text matching regex pattern '{regex_pattern}':
{"\n".join(" - section " + str(section_num) for section_num in match_sections)}
    """.strip()


async def enter_text_into_textbox(tag_id: str, text_to_enter: str) -> str:
    """Enter `text_to_enter` into a textarea or input element identified by `tag_id`."""
    current_session = current_browser_session.get()
    if not current_session:
        raise RuntimeError("No active browser session.")
    if current_session.url is None:
        return "Please navigate to a URL first."

    if tag_id not in current_session.tag_ids:
        return f"""
Invalid tag_id '{ tag_id }'
Available tag_id values are:
{"\n".join("    - " + x for x in current_session.tag_ids)}
"""

    tag_type, tag_num = tag_id.split("_")
    tag_num = int(tag_num)

    try:
        elements = await current_session.browser_tab.find(
            tag_name=tag_type, find_all=True
        )

        element = elements[tag_num - 1]
        await element.wait_until(is_interactable=True, timeout=10)
        await element.type_text(text_to_enter, interval=0.3)
    except PydollException as pydoll_error:
        return f"""
Failed to enter text into `{tag_id}`. Error was:
```
{pydoll_error}
```
        """
    else:
        await refresh_page_view(current_session)
        return f"Successfully entered text '{text_to_enter}' into {tag_type} text input with ID '{tag_id}'"


async def click_button(tag_id: str) -> str:
    """Click button with tag ID `tag_id`."""
    current_session = current_browser_session.get()
    if not current_session:
        raise RuntimeError("No active browser session.")
    if current_session.url is None:
        return "Please navigate to a URL first."

    if tag_id not in current_session.tag_ids:
        return f"""
Invalid tag_id '{ tag_id }'
Available tag_id values are:
{"\n".join("    - " + x for x in current_session.tag_ids)}
"""

    tag_type, tag_num = tag_id.split("_")
    tag_num = int(tag_num)

    try:
        elements = await current_session.browser_tab.find(
            tag_name=tag_type, find_all=True
        )

        element = elements[tag_num - 1]
        await element.wait_until(is_interactable=True, timeout=10)
        await element.click()
    except PydollException as pydoll_error:
        return f"""
Failed to click button with ID `{tag_id}`. Error was:
```
{pydoll_error}
```
        """
    else:
        # wait for new stuff to load if there was a page navigation #
        body = await current_session.browser_tab.find(tag_name="body", timeout=30)
        await asyncio.sleep(3)  # to cater for redirects
        await body.wait_until(is_visible=True)

        message: str = f"Successfully clicked button with ID '{tag_id}'."
        new_html: str = await current_session.browser_tab.page_source
        if new_html == current_session.html:
            return (
                message
                + " Warning: this action did not result in any change in the page HTML."
            )
        else:
            await refresh_page_view(current_session)
            return message + (
                " This action resulted in a change in the page HTML. "
                "Use the view_section() tool to view the new page."
            )


# async def press_enter_key(tag_id: str) -> str:
#     """Press the enter key while focused on element with ID `tag_id`."""
#     current_session = current_browser_session.get()
#     if not current_session:
#         raise RuntimeError("No active browser session.")
#     if current_session.url is None:
#         return "Please navigate to a URL first."
#
#     if tag_id not in current_session.tag_ids:
#         return f"""
# Invalid tag_id '{ tag_id }'
# Available tag_id values are:
# {"\n".join("    - " + x for x in current_session.tag_ids)}
# """
#
#     tag_type, tag_num = tag_id.split("_")
#     tag_num = int(tag_num)
#
#     try:
#         elements = await current_session.browser_tab.find(
#             tag_name=tag_type, find_all=True
#         )
#
#         element = elements[tag_num - 1]
#         await element.wait_until(is_interactable=True, timeout=10)
#         await element.press_keyboard_key(Key.ENTER, interval=0.1)
#     except PydollException as pydoll_error:
#         return f"""
# Failed to enter text into `{tag_id}`. Error was:
# ```
# {pydoll_error}
# ```
#         """
#     else:
#         # wait for new stuff to load if there was a page navigation #
#         body = await current_session.browser_tab.find(tag_name="body", timeout=30)
#         await asyncio.sleep(3)  # to cater for redirects
#         await body.wait_until(is_visible=True)
#
#         message: str = f"Successfully pressed enter on element with tag_id='{tag_id}'."
#         new_html: str = await current_session.browser_tab.page_source
#         if new_html == current_session.html:
#             return (
#                 message
#                 + " Warning: this action did not result in any change in the page HTML."
#             )
#         current_url: str = await current_session.browser_tab.current_url
#         if current_url != current_session.url:
#             await refresh_page_view(current_session)
#             return (
#                 message
#                 + f"""
# This resulted in a page redirection, and the current URL is now {current_session.url}.
#
# For size reasons, the web page text has been split into multiple sections.
# Here is the text content of section 1 of {len(current_session.text_paged)}:
# ```
# {current_session.text_paged[0]}
# ```
# """.strip()
#             )
#         else:
#             await refresh_page_view(current_session)
#             return (
#                 message
#                 + "See the latest state of the page using the view_section() tool."
#             )


async def search_current_page_hyperlinks(link_text_contains: str) -> list[str]:
    raise NotImplementedError
