from utils import func_defn_as_json_schema

from typing import Any


def example(
    text: str,
    chunk_len_nchars: int,
) -> list[str]:
    """
    Splits the input text into chunks of approximately `chunk_len_nchars` length.

    Args:
        text (str): The input string to be split.
        chunk_len_nchars (int): Approximate number of characters per chunk.

    Returns:
        list[str]: A list of string chunks, each approximately `chunk_len_nchars` characters long.
    """
    ...


import json

print(json.dumps(func_defn_as_json_schema(example), indent=4))
