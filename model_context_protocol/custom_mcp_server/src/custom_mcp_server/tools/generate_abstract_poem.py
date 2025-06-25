"""
MCP tool for generating a profound poem
"""

import random

from custom_mcp_server.server_setup import mcp


@mcp.tool()
def generate_abstract_poem(n_words: int) -> str:
    """
    Generates an abstract poem of length `n_words`
    """
    return " ".join(
        random.choices(
            [
                "\n",
                "capricious",
                "dilettante",
                "ennui",
                "equivocate",
                "esoteric",
                "gregarious",
                "harbinger",
                "numinous",
                "perenigrate",
                "zephyr",
            ],
            k=n_words,
        )
    ).strip()
