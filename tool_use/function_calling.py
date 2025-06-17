"""
TODO

Notes:
    - I have purposely kept the user instruction and function descriptions a bit vague to see \
how the Large Language Model copes
"""

import datetime
import json
import os
import random
from decimal import Decimal
from typing import Callable, Final, Literal, Optional

import dotenv
import openai
from loguru import logger

from utils import func_defn_as_json_schema

MAX_N_STEPS: Final[int] = 20  # max number of steps the LLM take
MAX_N_OUTPUT_TOKENS_PER_CHAT_COMPLETION: Final[int] = 999
USER_QUERY: Final[str] = (
    "Give me the share price of Amazon for the past 5 days, as well as \
the day-to-day percentage change."
)

dotenv.load_dotenv(".env")


# Convenience function for printing dicts and lists legibly
def ppd(x: list | dict):
    """Print dict or list in nicely indented human-readable format"""
    logger.info(
        "\n"
        + json.dumps(
            x,
            indent=4,
            default=str,
        )
    )


# Define some toy tools
# (would be real functions in a real application)
share_prices = {}  # to store previously simulated prices (for consistency)


def get_share_price(
    date: str,
    symbol: str,
    currency: Literal["GBP", "JPY", "USD"],
) -> Decimal:
    """
    Get share price of symbol `symbol` on historic date `date`

    Args:
        date (str): Date on which share price is required.
        symbol (str): Ticker symbol of stock e.g. 'MSFT'
        currency (Literal): Share price currency - one of ['GBP', 'JPY', 'USD']
    """
    parsed_date: datetime.date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
    if parsed_date > datetime.datetime.now().date():
        raise ValueError("Cannot give share price for a future date")
    if currency not in ("GBP", "JPY", "USD"):
        raise ValueError("Only currencies ['GBP', 'JPY', 'USD'] are available")
    key = (date, symbol, currency)
    if key not in share_prices:
        share_prices[key] = round(Decimal(random.uniform(100, 500)), 2)

    return share_prices[key]


def simple_calculator(
    num1: float,
    num2: float,
    operation: Literal["+", "-", "*", "/"],
) -> float:
    """
    Perform basic arithmetic `operation` on numbers `num1` and `num2`

    Args:
        num1 (float): The first number
        num2 (float): The second number
        operation (Literal): The mathematical operation to perform

    Returns:
        float: The result of the arithmetic calculation `num1` `operation` `num2`
    """
    if operation not in ["+", "-", "*", "/"]:
        raise ValueError("Only the operations ['+', '-', '*', '/'] are supported")
    return {
        "+": lambda x, y: x + y,
        "-": lambda x, y: x - y,
        "*": lambda x, y: x * y,
        "/": lambda x, y: x / y,
    }[operation](num1, num2)


messages_history: list[dict] = [
    {
        "role": "system",
        "content": f"""
You exclusively use the tools provided to you in order to to accomplish the tasks which are \
required of you.
Today's date is {datetime.datetime.now().date().strftime("%Y-%m-%d")} 
        """.strip(),
    },
    {"role": "user", "content": USER_QUERY},
]

available_funcs: dict[str, Callable] = {
    func.__name__: func for func in (get_share_price, simple_calculator)
}

print("-- TOOL DEFINITIONS --")
print("  (json schema)  ")
ppd(
    {
        func_name: func_defn_as_json_schema(func)
        for func_name, func in available_funcs.items()
    }
)

llm_client = openai.OpenAI(
    base_url=os.environ["OPENAI_API_BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"],
)

ppd(messages_history)
for step_num in range(1, MAX_N_STEPS + 1):
    llm_response = llm_client.chat.completions.create(
        model=os.environ["DEFAULT_MODEL_NAME"],
        messages=messages_history,
        tools=[func_defn_as_json_schema(func) for func in available_funcs.values()],
    )
    messages_history.append(
        {
            "role": llm_response.choices[0].message.role,
            "content": llm_response.choices[0].message.content,
            "tool_calls": llm_response.choices[0].message.tool_calls,
        },
    )
    tool_calls: Optional[list] = llm_response.choices[0].message.tool_calls
    if tool_calls:
        for tool_call in tool_calls:
            try:
                func_name: str = tool_call.function.name
                func_kwargs: dict = json.loads(tool_call.function.arguments)
                func_result = available_funcs[func_name](**func_kwargs)
                messages_history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(func_result),
                    }
                )
            except Exception as err:
                error_info = {
                    "error_type": type(err).__name__,
                    "error_message": str(err),
                }
                messages_history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"Tool call failed with error: \n{json.dumps(error_info, indent=4)}",
                    }
                )
        ppd(messages_history)
    else:
        ppd(messages_history)
        logger.info(f"Finished after {step_num} steps")
        break

logger.info(
    f"""
Final response:
(after {step_num} steps)

{messages_history[-1]["content"]}
"""
)
