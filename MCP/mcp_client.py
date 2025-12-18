"""
MCP Client - Connects to MCP server and integrates with LM Studio

This client:
1. Connects to an MCP server
2. Retrieves available tools
3. Uses LM Studio's local LLM to decide which tools to call
4. Executes tools and returns results
"""

import asyncio
import json
import subprocess
import sys
from contextlib import asynccontextmanager

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    """MCP Client that connects to server and uses LM Studio for tool selection."""

    def __init__(self, lm_studio_url: str = "http://127.0.0.1:1234"):
        self.lm_studio_url = lm_studio_url
        self.session: ClientSession | None = None
        self.tools: list = []

    @asynccontextmanager
    async def connect(self, server_script: str):
        """Connect to MCP server."""
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[server_script]
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                self.session = session
                await session.initialize()

                # Get available tools
                tools_response = await session.list_tools()
                self.tools = tools_response.tools

                yield self

    def get_tools_description(self) -> str:
        """Format tools for LLM prompt."""
        tools_desc = []
        for tool in self.tools:
            tools_desc.append(f"""
Tool: {tool.name}
Description: {tool.description}
Parameters: {json.dumps(tool.inputSchema, indent=2)}
""")
        return "\n".join(tools_desc)

    async def query_llm(self, prompt: str, system_prompt: str = "") -> str:
        """Query LM Studio LLM."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.lm_studio_url}/v1/chat/completions",
                json={
                    "model": "google-gemma",  # LM Studio handles model routing
                    "messages": [
                        {"role": "system", "content": system_prompt} if system_prompt else None,
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1024
                }
            )

            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                return f"LLM Error: {response.status_code} - {response.text}"

    async def process_with_tools(self, user_query: str) -> dict:
        """Process user query, determine tools to use, and execute them."""

        # Step 1: Ask LLM to determine which tool to use
        system_prompt = f"""You are an AI assistant with access to the following tools:

{self.get_tools_description()}

When the user asks a question, determine if you need to use a tool.
If you need to use a tool, respond with EXACTLY this JSON format:
{{"use_tool": true, "tool_name": "tool_name_here", "arguments": {{"param": "value"}}}}

If no tool is needed, respond with:
{{"use_tool": false, "response": "your natural response here"}}

IMPORTANT: Only respond with valid JSON, nothing else."""

        llm_response = await self.query_llm(user_query, system_prompt)

        result = {
            "user_query": user_query,
            "llm_decision": llm_response,
            "tool_used": None,
            "tool_result": None,
            "final_response": None
        }

        try:
            # Parse LLM decision
            # Extract JSON from response (handle markdown code blocks)
            json_str = llm_response
            if "```json" in llm_response:
                json_str = llm_response.split("```json")[1].split("```")[0]
            elif "```" in llm_response:
                json_str = llm_response.split("```")[1].split("```")[0]

            decision = json.loads(json_str.strip())

            if decision.get("use_tool"):
                tool_name = decision["tool_name"]
                arguments = decision.get("arguments", {})

                result["tool_used"] = tool_name

                # Step 2: Call the tool
                if self.session:
                    tool_response = await self.session.call_tool(tool_name, arguments)
                    tool_result = tool_response.content[0].text if tool_response.content else "No result"
                    result["tool_result"] = tool_result

                    # Step 3: Get final response from LLM with tool result
                    final_prompt = f"""The user asked: {user_query}

I used the tool '{tool_name}' and got this result:
{tool_result}

Please provide a helpful, natural language response to the user based on this information."""

                    final_response = await self.query_llm(final_prompt)
                    result["final_response"] = final_response
            else:
                result["final_response"] = decision.get("response", llm_response)

        except json.JSONDecodeError:
            # LLM didn't return valid JSON, use raw response
            result["final_response"] = llm_response
        except Exception as e:
            result["final_response"] = f"Error processing request: {str(e)}"

        return result

    async def call_tool_directly(self, tool_name: str, arguments: dict) -> str:
        """Call a tool directly without LLM decision."""
        if self.session:
            response = await self.session.call_tool(tool_name, arguments)
            return response.content[0].text if response.content else "No result"
        return "Error: Not connected to server"


# Standalone functions for testing
async def test_client():
    """Test the MCP client."""
    import os

    server_script = os.path.join(os.path.dirname(__file__), "mcp_server.py")
    client = MCPClient()

    async with client.connect(server_script):
        print("Connected to MCP server!")
        print(f"\nAvailable tools: {[t.name for t in client.tools]}")

        # Test direct tool calls
        print("\n--- Testing Direct Tool Calls ---")

        calc_result = await client.call_tool_directly("calculator", {"expression": "sqrt(144) + 10"})
        print(f"Calculator: {calc_result}")

        weather_result = await client.call_tool_directly("get_weather", {"city": "Vilnius"})
        print(f"Weather: {weather_result}")

        time_result = await client.call_tool_directly("get_current_time", {})
        print(f"Time: {time_result}")

        # Test with LLM (if available)
        print("\n--- Testing with LLM ---")
        try:
            result = await client.process_with_tools("What's 25 times 17?")
            print(f"Query: {result['user_query']}")
            print(f"Tool used: {result['tool_used']}")
            print(f"Final response: {result['final_response']}")
        except Exception as e:
            print(f"LLM test skipped (LM Studio may not be running): {e}")


if __name__ == "__main__":
    asyncio.run(test_client())
