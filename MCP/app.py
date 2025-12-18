"""
MCP Web Application - Flask GUI for MCP Job Search with LM Studio Integration

This application provides:
1. Web interface for querying job data
2. Integration with LM Studio local LLM (Google Gemma)
3. MCP server tools for data analysis
4. Chat interface for natural language queries
"""

import asyncio
import json
import os
import sys
import subprocess
from flask import Flask, render_template, request, jsonify
import httpx
import pandas as pd
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

app = Flask(__name__)

# Configuration
LM_STUDIO_URL = "http://127.0.0.1:1234"
DATASET_PATH = os.path.join(os.path.dirname(__file__), "future_jobs_dataset.csv")
SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "mcp_server.py")

# Load dataset for direct queries
df = pd.read_csv(DATASET_PATH)


# ============== MCP Client Functions ==============

async def get_mcp_tools():
    """Get list of available MCP tools."""
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[SERVER_SCRIPT]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_response = await session.list_tools()
            return [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "schema": tool.inputSchema
                }
                for tool in tools_response.tools
            ]


async def call_mcp_tool(tool_name: str, arguments: dict):
    """Call an MCP tool and return result."""
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[SERVER_SCRIPT]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            response = await session.call_tool(tool_name, arguments)
            if response.content:
                return response.content[0].text
            return "No result"


# ============== LLM Integration ==============

async def query_llm(prompt: str, system_prompt: str = "") -> str:
    """Query LM Studio LLM."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{LM_STUDIO_URL}/v1/chat/completions",
                json={
                    "model": "google/gemma-3n-e4b",
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 2048
                }
            )

            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                return f"LLM Error: {response.status_code}"
    except httpx.ConnectError:
        return "Error: Cannot connect to LM Studio. Make sure it's running at http://127.0.0.1:1234"
    except Exception as e:
        return f"Error: {str(e)}"


async def process_with_llm_and_tools(user_query: str) -> dict:
    """Process user query using LLM to decide which tools to use."""

    # Get available tools
    tools = await get_mcp_tools()
    tools_description = "\n".join([
        f"- {t['name']}: {t['description']}"
        for t in tools
    ])

    # Step 1: Ask LLM to decide which tool to use
    system_prompt = f"""You are an AI assistant helping users query job market data. You have access to these tools:

{tools_description}

When the user asks about jobs, salaries, skills, or industries, determine which tool to use.
Respond with ONLY valid JSON in this format:
{{"tool": "tool_name", "arguments": {{"param": "value"}}}}

If no tool is needed, respond with:
{{"tool": null, "response": "your response here"}}

Examples:
- "Find AI jobs" -> {{"tool": "search_jobs", "arguments": {{"industry": "AI"}}}}
- "What skills are needed for blockchain?" -> {{"tool": "get_skills_demand", "arguments": {{"industry": "Blockchain"}}}}
- "Compare salaries by location" -> {{"tool": "compare_salaries", "arguments": {{"compare_by": "location"}}}}
- "Show remote job stats" -> {{"tool": "get_remote_stats", "arguments": {{}}}}"""

    llm_decision = await query_llm(user_query, system_prompt)

    result = {
        "user_query": user_query,
        "llm_decision": llm_decision,
        "tool_used": None,
        "tool_result": None,
        "final_response": None
    }

    try:
        # Extract JSON from response
        json_str = llm_decision
        if "```json" in llm_decision:
            json_str = llm_decision.split("```json")[1].split("```")[0]
        elif "```" in llm_decision:
            json_str = llm_decision.split("```")[1].split("```")[0]

        decision = json.loads(json_str.strip())

        if decision.get("tool"):
            tool_name = decision["tool"]
            arguments = decision.get("arguments", {})

            result["tool_used"] = tool_name

            # Call the MCP tool
            tool_result = await call_mcp_tool(tool_name, arguments)
            result["tool_result"] = tool_result

            # Get final response from LLM
            final_prompt = f"""User asked: {user_query}

I used the '{tool_name}' tool and got this data:
{tool_result}

Please provide a helpful, conversational response summarizing this information for the user.
Be concise but informative. Format numbers nicely and highlight key insights."""

            final_response = await query_llm(final_prompt)
            result["final_response"] = final_response
        else:
            result["final_response"] = decision.get("response", llm_decision)

    except json.JSONDecodeError:
        result["final_response"] = llm_decision
    except Exception as e:
        result["final_response"] = f"Error: {str(e)}"

    return result


# ============== Flask Routes ==============

@app.route("/")
def index():
    """Main page with chat interface."""
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """Process chat message with LLM and MCP tools."""
    data = request.json
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Run async function
    result = asyncio.run(process_with_llm_and_tools(user_message))

    return jsonify(result)


@app.route("/api/tools", methods=["GET"])
def get_tools():
    """Get list of available MCP tools."""
    tools = asyncio.run(get_mcp_tools())
    return jsonify(tools)


@app.route("/api/tool/<tool_name>", methods=["POST"])
def call_tool(tool_name):
    """Call a specific MCP tool directly."""
    arguments = request.json or {}
    result = asyncio.run(call_mcp_tool(tool_name, arguments))

    try:
        return jsonify(json.loads(result))
    except json.JSONDecodeError:
        return jsonify({"result": result})


@app.route("/api/stats")
def get_stats():
    """Get dataset statistics."""
    stats = {
        "total_jobs": len(df),
        "industries": df["industry"].nunique(),
        "locations": df["location"].nunique(),
        "avg_salary": f"${df['salary_usd'].mean():,.0f}",
        "remote_percentage": f"{(df['remote_option'] == 'Yes').mean() * 100:.1f}%",
        "industry_counts": df["industry"].value_counts().to_dict(),
        "location_counts": df["location"].value_counts().to_dict()
    }
    return jsonify(stats)


@app.route("/api/llm/status")
def llm_status():
    """Check if LM Studio is running."""
    try:
        import requests
        response = requests.get(f"{LM_STUDIO_URL}/v1/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            return jsonify({
                "status": "connected",
                "url": LM_STUDIO_URL,
                "models": models
            })
    except Exception as e:
        pass

    return jsonify({
        "status": "disconnected",
        "url": LM_STUDIO_URL,
        "message": "LM Studio is not running. Start it to enable AI features."
    })


if __name__ == "__main__":
    # Create templates directory if needed
    templates_dir = os.path.join(os.path.dirname(__file__), "templates")
    os.makedirs(templates_dir, exist_ok=True)

    print("\n" + "="*60)
    print("MCP Job Search Application")
    print("="*60)
    print(f"Dataset: {DATASET_PATH}")
    print(f"LM Studio URL: {LM_STUDIO_URL}")
    print(f"Web Interface: http://127.0.0.1:5000")
    print("="*60 + "\n")

    app.run(debug=True, port=5000)
