"""Expose the analyst tools over MCP (stdio) so any MCP client (desktop assistants, IDEs) can drive the fusion system. python mcp_server.py"""

from mcp.server.fastmcp import FastMCP

from tools.core import TOOLS

mcp = FastMCP("sigint-fusion")
for f in TOOLS:
    mcp.tool()(f)

if __name__ == "__main__":
    mcp.run()
