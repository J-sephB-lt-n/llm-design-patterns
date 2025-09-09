"""
Example of navigating to a URL and taking a screenshot of the page using [pydoll](https://github.com/autoscrape-labs/pydoll)
"""

import asyncio
import warnings

from pydoll.browser.chromium import Chrome
from pydoll.browser.options import ChromiumOptions


async def go_to_url_and_screenshot(
    url: str,
    output_path: str = "temp_page_screenshot.png",
):
    """
    Navigate to `url`, take a screenshot and save the image to local file `output_path`.
    """
    # warnings.warn("SSL is disabled")
    options = ChromiumOptions()
    # options.add_argument("--ignore-certificate-errors")

    async with Chrome(options=options) as browser:
        tab = await browser.start()

        await tab.go_to(url)

        body = await tab.find(tag_name="body", timeout=30)
        await body.wait_until(is_visible=True)
        # await asyncio.sleep(5)

        # you can get the screenshot as base64 #
        screenshot_b64: str = await tab.take_screenshot(
            as_base64=True,
            # beyond_viewport=True,  # include entire page
        )
        print("Screenshot as base64:", screenshot_b64[:100], "...")

        # or you can save the screenshot to a file #
        await tab.take_screenshot(
            path=output_path,
            # beyond_viewport=True,  # include entire page
        )
        print(f"Screenshot saved to: {output_path}")


if __name__ == "__main__":
    url: str = (
        "https://en.wikipedia.org/wiki/List_of_serial_killers_by_number_of_victims"
    )
    asyncio.run(
        go_to_url_and_screenshot(
            url=url,
        )
    )
