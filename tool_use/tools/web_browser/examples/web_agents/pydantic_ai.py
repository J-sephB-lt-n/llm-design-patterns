"""
TODO.

NOTES:
    - The way which I handling the browser state in this code is not actually required when using \
      the pydanticAI library since pydanticAI tools can access persistent state.
"""

import dataclasses
import json
import os

import dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


from tool_use.tools.web_browser import (
    BrowserManager,
    click_button,
    enter_text_into_textbox,
    go_to_url,
    text_search,
    WebBrowser,
)

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
        system_prompt="You are a helpful assistant with access to web browser tools.",
        tools=[
            click_button,
            enter_text_into_textbox,
            go_to_url,
            text_search,
        ],
        output_type=str,
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
