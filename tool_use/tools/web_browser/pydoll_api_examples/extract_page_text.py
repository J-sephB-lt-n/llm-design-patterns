"""
Example of extracting HTML (and text) from a webpage using pydoll.

Usage:
    $ python extract_page_text.py https://www.google.com
"""

import asyncio
import sys
import warnings

from markdownify import markdownify as md
from pydoll.browser.chromium import Chrome
from pydoll.browser.options import ChromiumOptions


async def go_to_url_and_get_page_text(
    url: str,
):
    """
    Navigate to `url` and extract the text from the page.
    """
    # warnings.warn("SSL is disabled")
    options = ChromiumOptions()
    # options.add_argument("--ignore-certificate-errors")

    async with Chrome(options=options) as browser:
        tab = await browser.start()

        await tab.go_to(url)

        body = await tab.find(tag_name="body", timeout=30)
        await body.wait_until(is_visible=True)

        output_path: str = "temp_page_content.md"
        page_html: str = await tab.page_source
        page_text: str = md(page_html)

        with open(output_path, "w", encoding="utf-8") as file:
            file.write(page_text)
        f"Wrote text of page '{url}' to {output_path}"


if __name__ == "__main__":
    asyncio.run(
        go_to_url_and_get_page_text(
            url=sys.argv[1],
        )
    )
