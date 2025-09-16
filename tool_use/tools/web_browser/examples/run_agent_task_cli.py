"""
CLI interface to run a single web browser agent task.

Example usage:
$ uv run python -m tool_use.tools.web_browser.examples.run_agent_task_cli
"""

import asyncio
from collections.abc import Callable

import questionary

from tool_use.tools.web_browser.examples import web_agents

AVAILABLE_AGENTS: dict[str, Callable] = {
    "Simple OpenAI for loop": web_agents.simple_openai_agent,
    "Pydantic AI": web_agents.pydantic_ai_agent,
}


if __name__ == "__main__":
    agent_name: str = questionary.select(
        "Please select an agent: ",
        choices=AVAILABLE_AGENTS.keys(),
    ).ask()
    agent = AVAILABLE_AGENTS[agent_name]
    task: str = input("Please describe your task: ")

    print(f"agent is '{agent}'. Task is '{task}'.")
    asyncio.run(
        agent(task),
    )
