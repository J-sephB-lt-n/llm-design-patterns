"""
Entrypoint script of this package (see pyproject.toml)
The MCP server definition
"""

from custom_mcp_server.server_setup import mcp

from custom_mcp_server.tools import generate_abstract_poem, roll_dice


def main():
    mcp.run()


if __name__ == "__main__":
    main()
