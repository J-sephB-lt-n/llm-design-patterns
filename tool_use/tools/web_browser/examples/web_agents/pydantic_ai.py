"""
TODO.

NOTES:
    - The way which I handling the browser state in this code is not actually required when using \
      the pydanticAI library since pydanticAI tools can access persistent state.
"""

import base64
import dataclasses
import json
import os
import time

import dotenv
from pydantic_ai import Agent
from pydantic_ai.messages import BinaryContent, ToolReturn
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


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

dotenv.load_dotenv(".env", override=True)


async def pydantic_ai_agent(
    task: str,
) -> None:
    """Perform `task` using a pydanticAI agent with access to web browser tools."""
    llm = OpenAIChatModel(
        os.environ["DEFAULT_MODEL"],
        provider=OpenAIProvider(
            base_url=os.environ["OPENAI_BASE_URL"],
            api_key=os.environ["OPENAI_API_KEY"],
        ),
    )
    agent = Agent(
        llm,
        system_prompt=SYSTEM_PROMPT,
        tools=[
            click_button,
            enter_text_into_textbox,
            go_to_url,
            text_search,
            view_section,
        ],
        output_type=str,
    )

    @agent.tool_plain
    async def page_screenshot() -> str:
        """Take a screenshot of the current web page (i.e. see the page as an image)."""
        page_image_b64: str = await view_page_screenshot()
        page_image_bytes: bytes = base64.b64decode(page_image_b64)

        return ToolReturn(
            return_value="Successfully captured screenshot of current web page.",
            content=[
                "Here is the screenshot: ",
                BinaryContent(data=page_image_bytes, media_type="image/png"),
            ],
            metadata={
                "timestamp": time.time(),
            },
        )

    browser_manager = BrowserManager()
    await browser_manager.start_browser()
    try:
        browser = WebBrowser(browser_manager)
        async with browser.isolated_browser_session():
            # result = await agent.run(task)
            # print(result.output)
            # print(result.all_messages)
            async with agent.iter(task) as agent_run:
                async for node in agent_run:
                    if dataclasses.is_dataclass(node) and not isinstance(node, type):
                        node_dict: dict = dataclasses.asdict(node)
                        print(json.dumps(node_dict, indent=4, default=str))
                    else:
                        print(node)
    finally:
        await browser_manager.shutdown_browser()
