import asyncio
from fastmcp import Client, FastMCP

# In-memory server (ideal for testing)
server = FastMCP("FEP-MCP")

# HTTP server
client = Client("http://localhost:8000/mcp")

# ANSI escape codes for colors and styles
COLOR_GREEN = "\033[1;32m"  # Bold Green
COLOR_CYAN = "\033[1;36m"   # Bold Cyan
COLOR_DIM = "\033[2m"       # Dim text
COLOR_RESET = "\033[0m"     # Reset to normal


async def main():
    async with client:
        # Basic server interaction
        await client.ping()

        # List available operations
        tools = await client.list_tools()

        # Basic beautified output
        print(f"{COLOR_GREEN}✅ Connected to MCP Server{COLOR_RESET}")
        print(f"\n{COLOR_CYAN}Available Tools ({len(tools)}):{COLOR_RESET}")

        for tool in tools:
            print(f"\n{COLOR_CYAN}- {tool.name}{COLOR_RESET}")
            print(f"  {COLOR_DIM}Description:{COLOR_RESET} {tool.description}")
            if tool.inputSchema and "properties" in tool.inputSchema:
                inputs = tool.inputSchema["properties"]
                if inputs:
                    print(f"  {COLOR_DIM}Inputs:{COLOR_RESET}")
                    for key, schema in inputs.items():
                        type_ = schema.get("type", "unknown")
                        print(f"    - {key} ({type_})")

asyncio.run(main())
