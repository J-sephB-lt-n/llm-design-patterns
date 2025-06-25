
# Custom MCP Server

This folder shows how to host tools on a python [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction) server. i.e. it lets the agent in your IDE (or any other MCP-compatible client) use the python functions which you define.

You can experiment with the tools in a local web app by running:

```bash
cd llm-design-patterns/mcp/custom_mcp_server
uv run mcp dev src/custom_mcp_server/main.py
```

## Make this MCP server available to roo code in vs-code (or any other MCP-compatible IDE/platform):

```bash
# build the python MCP server into a python package:
cd llm-design-patterns/mcp/custom_mcp_server

# build this package into a binary python package file (.whl):
uv build

# (optional) copy the python wheel file to somewhere:
mkdir -p ~/.mcp_servers/custom_mcp_server/dist
cp ./dist/custom_mcp_server-x.x.x-py3-none-any.whl ~/.mcp_servers/custom_mcp_server/dist

# install the package:
cd ~/.mcp_servers/custom_mcp_server/
uv run --no-project --python 3.13 python -m venv .venv # create virtual environment pointing to a specific installed python (one of the ones managed by uv)
source .venv/bin/activate
pip install ./dist/custom_mcp_server-x.x.x-py3-none-any.whl
```

Add the server to your existing MCP config in roo code in vs-code (or other IDE or MCP client software):

```json
{
  "mcpServers": {
    "custom-mcp-server": {
      "command": "bash",    // on windows, can use "cmd.exe /c" instead of "bash -c"
      "args": [
        "-c",
        "cd ~/.mcp_servers/custom_mcp_server && source .venv/bin/activate && custom-mcp-server"
      ]
    }
  }
}
```
