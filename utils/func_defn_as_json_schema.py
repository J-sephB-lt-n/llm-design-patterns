import json
import inspect
import re
from typing import Any, Callable, Optional

import pydantic


def func_defn_as_json_schema(func: Callable) -> dict:
    """
    Convert function signature to JSON schema

    Notes:
        - Parameter descriptions will only be parsed from google-style docstrings
        - Got the idea to use pydantic.TypeAdapter from Amit Chaudhary
            (https://amitness.com/posts/function-calling-schema/#approach-2-pydantic)
    """
    func_desc: Optional[str] = None
    if func.__doc__ is not None:
        desc_pattern = re.compile(
            r"""
            \A              # start of string
            (?P<func_desc>.*?)           # grab everything (non-greedy)
            (?=             # up to but not including the first header (e.g. Notes: or Args:)
                ^[ \t]*                         # may start with some tabs
                [A-Za-z][A-Za-z0-9_ ]{3,20}     # rest of header name
                :\s*$                           # colon, optional whitespace, end of line
            )
            """,
            re.DOTALL  # make . also match newline characters
            | re.MULTILINE  # makes ^$ match LINE start/end (not whole string start/end)
            | re.VERBOSE,  # allow whitespace and comments in regex pattern
        )
        find_func_desc = desc_pattern.search(func.__doc__)
        if find_func_desc:
            func_desc = re.sub(r"\s+", " ", find_func_desc.group("func_desc").strip())

    docstring_arg_descriptions: dict[str, str] = {}
    func_param_names: tuple[str, ...] = tuple(inspect.signature(func).parameters.keys())
    if len(func_param_names) > 0 and func.__doc__ is not None:
        args_section_pattern = re.compile(
            r"""
            Args:\s*\n          # ensure we're after "Args:"
            (?P<args>.+?)               # grab everything (non-greedy)
            (?=                 # stop at
                (?:\r?\n\s*){2,}    # 2 blank lines..
                | \Z                 # ..or end of string 
            )
            """,
            re.DOTALL  # make . also include \n
            | re.VERBOSE,  # allows whitespace and comments in regex pattern
        )
        find_docstring_args_section = args_section_pattern.search(
            func.__doc__,
        )
        if find_docstring_args_section:
            docstring_args_section: str = find_docstring_args_section.group("args")
            arg_pattern = re.compile(
                r"""
                ^[ \t]*                         # any leading indentation
                ([a-z_]+)                       # 1: a snake_case variable name
                (?:\s*\([^)]*\))?               #    optional parameter (type)
                :\s+                            #    the colon and space that follows
                ([\s\S]*?)                      # 2: non-greedy capture of everything (including newlines)
                (?=                             #    until you peek ahead and see either
                    ^[ \t]*[a-z_]+(?:\s*\([^)]*\))?:  #      a new snake_case/optional-type line
                | \Z                            #      or the end of the string
                )
            """,
                re.MULTILINE  # makes ^$ match LINE start/end (not whole string start/end)
                | re.VERBOSE,  # allows whitespace and comments in regex pattern
            )
            for match in arg_pattern.finditer(docstring_args_section):
                param_name, param_desc = (
                    match.group(1),
                    re.sub(r"\s+", " ", match.group(2).strip()),
                )
                if param_name in func_param_names:
                    docstring_arg_descriptions[param_name] = param_desc

    json_schema: dict = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func_desc or func.__doc__,
            "parameters": pydantic.TypeAdapter(func).json_schema(),
        },
    }

    for param_name, param in json_schema["function"]["parameters"][
        "properties"
    ].items():
        if param_name in docstring_arg_descriptions:
            param["description"] = docstring_arg_descriptions[param_name]

    return json_schema


if __name__ == "__main__":

    def example_function(
        data_source: str,
        params: dict[str, Any],
        debug,
        timeout: float,
        undocumented_list: list[int],
        flag_enabled: bool,
        display_message: str,
        retry_count: int = 3,
    ):
        """
        Please give me an example python function which has all of the following:

        1. Argument with simple type annotation
        2. Argument with complex type annotation
        3. Argument with no type annotation
        4. Argument with default value
        5. Argument with no default value
        6. Google-style docstring
        7. argument which is missing from Args: in the docstring
        8. argument with a type annotation in the docstring
        9. argument with no type annotation in the docstring
        10. argument with multi-line Args: description in docstring
        11. argument with multi-line description in docstring, where description contains colons

        Use realistic and meaningful parameter names and docstring descriptions.

        You may omit the actual function body (I am only interested in parsing the function signature)

        Process records according to the given parameters.

            Args:
                data_source: Path to the input data file.
                params: Configuration parameters for processing. This includes multiple keys:
                    - max_items: maximum number of items to process
                    - mode: processing mode, e.g., 'fast' or 'safe'
                timeout: Timeout in seconds before giving up.
                flag_enabled (bool): Whether to enable the special processing flag.
                display_message: Custom message to display to the user.
                retry_count: Number of times to retry on failure.

            Returns:
                None
        """

    ...

    example_func_json_schema: dict = func_defn_as_json_schema(example_function)

    print(json.dumps(example_func_json_schema, indent=4))
