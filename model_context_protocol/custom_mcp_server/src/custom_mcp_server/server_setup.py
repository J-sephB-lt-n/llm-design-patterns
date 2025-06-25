"""
Central definition of FastMCP() object, so that it can be shared by different modules
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="Custom MCP Server",
)
