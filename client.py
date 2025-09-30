import asyncio
from fastmcp import Client, FastMCP

# In-memory server (ideal for testing)
server = FastMCP("FEP-MCP")

# HTTP server
client = Client("http://localhost:8000/mcp")


async def main():
    async with client:
        # Basic server interaction
        await client.ping()

        # List available operations
        tools = await client.list_tools()
        # print("Available tools:", tools)
        print("[bold green]✅ Connected to MCP Server[/bold green]")
        print(f"\n[bold cyan]Available Tools ({len(tools)}):[/bold cyan]")

        for tool in tools:
            print(f"\n[bold]- {tool.name}[/bold]")
            print(f"  [dim]Description:[/dim] {tool.description}")
            if tool.inputSchema and "properties" in tool.inputSchema:
                inputs = tool.inputSchema["properties"]
                if inputs:
                    print("  [dim]Inputs:[/dim]")
                    for key, schema in inputs.items():
                        type_ = schema.get("type", "unknown")
                        print(f"    - {key} ({type_})")

asyncio.run(main())
