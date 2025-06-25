"""
MCP tool for generating a profound poem
"""

import random

from custom_mcp_server.server_setup import mcp


@mcp.tool()
def roll_dice(n_sides: int, n_rolls: int) -> tuple[int, ...]:
    """
    Simulate `n_rolls` rolls of a `n_sides`-sided die

    Args:
        n_sides (int): The number of sides on the die
        n_rolls (int): Number of rolls to perform
    """
    return tuple(random.randint(1, n_sides) for _ in range(n_rolls))
