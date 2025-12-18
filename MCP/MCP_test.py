"""
MCP Test Script - Test MCP Server and Client Functionality

This script demonstrates how to:
1. Connect to the MCP server
2. List available tools
3. Call tools directly
4. Process queries with LLM integration

Run the web app: uv run python MCP/app.py
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "mcp_server.py")


async def test_mcp_server():
    """Test the MCP server directly."""
    print("\n" + "="*60)
    print("MCP Server Test")
    print("="*60)

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[SERVER_SCRIPT]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 1. List available tools
            print("\n[1] Available MCP Tools:")
            print("-" * 40)
            tools_response = await session.list_tools()
            for tool in tools_response.tools:
                print(f"  - {tool.name}: {tool.description[:50]}...")

            # 2. Test search_jobs tool
            print("\n[2] Testing search_jobs (AI industry):")
            print("-" * 40)
            result = await session.call_tool("search_jobs", {"industry": "AI", "limit": 3})
            print(result.content[0].text[:500] + "...")

            # 3. Test get_salary_info tool
            print("\n[3] Testing get_salary_info (AI industry):")
            print("-" * 40)
            result = await session.call_tool("get_salary_info", {"industry": "AI"})
            print(result.content[0].text)

            # 4. Test get_skills_demand tool
            print("\n[4] Testing get_skills_demand (Blockchain):")
            print("-" * 40)
            result = await session.call_tool("get_skills_demand", {"industry": "Blockchain", "top_n": 5})
            print(result.content[0].text)

            # 5. Test compare_salaries tool
            print("\n[5] Testing compare_salaries (by location):")
            print("-" * 40)
            result = await session.call_tool("compare_salaries", {"compare_by": "location"})
            print(result.content[0].text[:600] + "...")

            # 6. Test get_remote_stats tool
            print("\n[6] Testing get_remote_stats:")
            print("-" * 40)
            result = await session.call_tool("get_remote_stats", {})
            print(result.content[0].text)

            print("\n" + "="*60)
            print("All tests completed!")
            print("="*60)


async def test_with_llm():
    """Test LLM integration (requires LM Studio running)."""
    import httpx

    print("\n" + "="*60)
    print("LLM Integration Test")
    print("="*60)

    LM_STUDIO_URL = "http://127.0.0.1:1234"

    # Check if LM Studio is running
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{LM_STUDIO_URL}/v1/models")
            if response.status_code == 200:
                print(f"LM Studio is running at {LM_STUDIO_URL}")
                models = response.json()
                print(f"Available models: {models}")
            else:
                print("LM Studio responded but with an error")
                return
    except httpx.ConnectError:
        print(f"LM Studio is NOT running at {LM_STUDIO_URL}")
        print("Start LM Studio to test LLM integration")
        return

    # Test a simple query
    print("\nTesting chat completion...")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{LM_STUDIO_URL}/v1/chat/completions",
                json={
                    "model": "google/gemma-3n-e4b",
                    "messages": [
                        {"role": "user", "content": "Say 'Hello from MCP test!' in exactly 5 words."}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 50
                }
            )

            if response.status_code == 200:
                data = response.json()
                print(f"LLM Response: {data['choices'][0]['message']['content']}")
            else:
                print(f"Error: {response.status_code}")
    except Exception as e:
        print(f"Error testing LLM: {e}")


def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("MCP (Model Context Protocol) Demo")
    print("="*60)
    print("""
This demo shows how MCP works:

1. MCP Server (mcp_server.py):
   - Exposes tools for querying job data
   - Tools: search_jobs, get_salary_info, get_skills_demand, etc.

2. MCP Client (this script):
   - Connects to the server via stdio
   - Lists and calls available tools

3. Web App (app.py):
   - Flask web interface
   - Integrates LM Studio for natural language queries
   - Run with: uv run python MCP/app.py

How MCP Works:
--------------
1. Server defines tools with schemas (name, description, input params)
2. Client connects and discovers available tools
3. Client calls tools with arguments
4. Server processes and returns results
5. LLM can use tool descriptions to decide which tool to call

""")

    # Run tests
    asyncio.run(test_mcp_server())
    asyncio.run(test_with_llm())

    print("\n" + "="*60)
    print("To run the web app with GUI:")
    print("  uv run python MCP/app.py")
    print("Then open http://127.0.0.1:5000 in your browser")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
