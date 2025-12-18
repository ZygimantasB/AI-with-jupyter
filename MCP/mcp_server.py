"""
MCP Server - Model Context Protocol Server with Job Dataset Tools

This server provides tools for querying and analyzing the future jobs dataset.
Tools available:
- search_jobs: Search jobs by various criteria
- get_job_stats: Get statistics about jobs
- get_salary_range: Get salary information
- get_skills_demand: Analyze skill demand
- get_industries: List all industries
- get_locations: List all locations
"""

import asyncio
import json
import os
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import pandas as pd

# Create server instance
server = Server("jobs-mcp-server")

# Load dataset
DATASET_PATH = os.path.join(os.path.dirname(__file__), "future_jobs_dataset.csv")
df = pd.read_csv(DATASET_PATH)


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List all available tools."""
    return [
        Tool(
            name="search_jobs",
            description="Search for jobs by title, industry, location, or skills. Returns matching jobs with details.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search term (job title, skill, industry, or location)"
                    },
                    "industry": {
                        "type": "string",
                        "description": "Filter by industry (AI, Blockchain, Green Tech, Quantum Computing)"
                    },
                    "location": {
                        "type": "string",
                        "description": "Filter by location (city name)"
                    },
                    "remote_only": {
                        "type": "boolean",
                        "description": "Show only remote jobs"
                    },
                    "min_salary": {
                        "type": "number",
                        "description": "Minimum salary in USD"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default 10)"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="get_job_stats",
            description="Get statistics about jobs: count by industry, location, company size, and salary ranges",
            inputSchema={
                "type": "object",
                "properties": {
                    "group_by": {
                        "type": "string",
                        "description": "Group statistics by: industry, location, company_size, job_title",
                        "enum": ["industry", "location", "company_size", "job_title"]
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="get_salary_info",
            description="Get salary information and ranges for specific job titles or industries",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_title": {
                        "type": "string",
                        "description": "Job title to analyze"
                    },
                    "industry": {
                        "type": "string",
                        "description": "Industry to analyze"
                    },
                    "location": {
                        "type": "string",
                        "description": "Location to analyze"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="get_skills_demand",
            description="Analyze which skills are most in demand across jobs",
            inputSchema={
                "type": "object",
                "properties": {
                    "industry": {
                        "type": "string",
                        "description": "Filter by industry"
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of top skills to return (default 10)"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="get_industries",
            description="List all available industries in the dataset",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_locations",
            description="List all available locations in the dataset",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="compare_salaries",
            description="Compare salaries between different jobs, industries, or locations",
            inputSchema={
                "type": "object",
                "properties": {
                    "compare_by": {
                        "type": "string",
                        "description": "Compare by: job_title, industry, location",
                        "enum": ["job_title", "industry", "location"]
                    }
                },
                "required": ["compare_by"]
            }
        ),
        Tool(
            name="get_remote_stats",
            description="Get statistics about remote work options",
            inputSchema={
                "type": "object",
                "properties": {
                    "industry": {
                        "type": "string",
                        "description": "Filter by industry"
                    }
                },
                "required": []
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""

    if name == "search_jobs":
        return await handle_search_jobs(arguments)
    elif name == "get_job_stats":
        return await handle_job_stats(arguments)
    elif name == "get_salary_info":
        return await handle_salary_info(arguments)
    elif name == "get_skills_demand":
        return await handle_skills_demand(arguments)
    elif name == "get_industries":
        return await handle_get_industries()
    elif name == "get_locations":
        return await handle_get_locations()
    elif name == "compare_salaries":
        return await handle_compare_salaries(arguments)
    elif name == "get_remote_stats":
        return await handle_remote_stats(arguments)
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def handle_search_jobs(args: dict) -> list[TextContent]:
    """Search for jobs."""
    filtered = df.copy()

    query = args.get("query", "").lower()
    industry = args.get("industry")
    location = args.get("location")
    remote_only = args.get("remote_only", False)
    min_salary = args.get("min_salary")
    limit = args.get("limit", 10)

    if query:
        mask = (
            filtered["job_title"].str.lower().str.contains(query, na=False) |
            filtered["skills_required"].str.lower().str.contains(query, na=False) |
            filtered["industry"].str.lower().str.contains(query, na=False) |
            filtered["location"].str.lower().str.contains(query, na=False)
        )
        filtered = filtered[mask]

    if industry:
        filtered = filtered[filtered["industry"].str.lower() == industry.lower()]

    if location:
        filtered = filtered[filtered["location"].str.lower().str.contains(location.lower(), na=False)]

    if remote_only:
        filtered = filtered[filtered["remote_option"] == "Yes"]

    if min_salary:
        filtered = filtered[filtered["salary_usd"] >= min_salary]

    # Sort by salary descending
    filtered = filtered.sort_values("salary_usd", ascending=False).head(limit)

    if len(filtered) == 0:
        return [TextContent(type="text", text="No jobs found matching your criteria.")]

    results = []
    for _, row in filtered.iterrows():
        results.append({
            "job_title": row["job_title"],
            "industry": row["industry"],
            "location": row["location"],
            "salary_usd": f"${row['salary_usd']:,.0f}",
            "skills": row["skills_required"],
            "remote": row["remote_option"],
            "company_size": row["company_size"]
        })

    return [TextContent(type="text", text=json.dumps(results, indent=2))]


async def handle_job_stats(args: dict) -> list[TextContent]:
    """Get job statistics."""
    group_by = args.get("group_by", "industry")

    stats = df.groupby(group_by).agg({
        "job_id": "count",
        "salary_usd": ["mean", "min", "max"]
    }).round(0)

    stats.columns = ["job_count", "avg_salary", "min_salary", "max_salary"]
    stats = stats.sort_values("job_count", ascending=False)

    result = {
        "grouped_by": group_by,
        "total_jobs": len(df),
        "statistics": stats.to_dict("index")
    }

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_salary_info(args: dict) -> list[TextContent]:
    """Get salary information."""
    filtered = df.copy()

    job_title = args.get("job_title")
    industry = args.get("industry")
    location = args.get("location")

    if job_title:
        filtered = filtered[filtered["job_title"].str.lower().str.contains(job_title.lower(), na=False)]
    if industry:
        filtered = filtered[filtered["industry"].str.lower() == industry.lower()]
    if location:
        filtered = filtered[filtered["location"].str.lower().str.contains(location.lower(), na=False)]

    if len(filtered) == 0:
        return [TextContent(type="text", text="No data found for the specified criteria.")]

    result = {
        "filters_applied": {
            "job_title": job_title,
            "industry": industry,
            "location": location
        },
        "sample_size": len(filtered),
        "salary_stats": {
            "average": f"${filtered['salary_usd'].mean():,.0f}",
            "median": f"${filtered['salary_usd'].median():,.0f}",
            "minimum": f"${filtered['salary_usd'].min():,.0f}",
            "maximum": f"${filtered['salary_usd'].max():,.0f}",
            "std_dev": f"${filtered['salary_usd'].std():,.0f}"
        }
    }

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_skills_demand(args: dict) -> list[TextContent]:
    """Analyze skill demand."""
    filtered = df.copy()
    industry = args.get("industry")
    top_n = args.get("top_n", 10)

    if industry:
        filtered = filtered[filtered["industry"].str.lower() == industry.lower()]

    # Parse skills
    all_skills = []
    for skills_str in filtered["skills_required"].dropna():
        skills = [s.strip() for s in skills_str.split(",")]
        all_skills.extend(skills)

    skill_counts = pd.Series(all_skills).value_counts().head(top_n)

    result = {
        "industry_filter": industry or "All industries",
        "total_job_postings": len(filtered),
        "top_skills": skill_counts.to_dict()
    }

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_get_industries() -> list[TextContent]:
    """Get all industries."""
    industries = df["industry"].unique().tolist()
    counts = df["industry"].value_counts().to_dict()

    result = {
        "industries": industries,
        "job_counts": counts
    }

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_get_locations() -> list[TextContent]:
    """Get all locations."""
    locations = df["location"].unique().tolist()
    counts = df["location"].value_counts().to_dict()

    result = {
        "locations": locations,
        "job_counts": counts
    }

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_compare_salaries(args: dict) -> list[TextContent]:
    """Compare salaries."""
    compare_by = args.get("compare_by", "industry")

    comparison = df.groupby(compare_by)["salary_usd"].agg(["mean", "median", "min", "max", "count"]).round(0)
    comparison = comparison.sort_values("mean", ascending=False)

    result = {
        "compared_by": compare_by,
        "comparison": comparison.to_dict("index")
    }

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_remote_stats(args: dict) -> list[TextContent]:
    """Get remote work statistics."""
    filtered = df.copy()
    industry = args.get("industry")

    if industry:
        filtered = filtered[filtered["industry"].str.lower() == industry.lower()]

    remote_counts = filtered["remote_option"].value_counts()
    remote_pct = (remote_counts / len(filtered) * 100).round(1)

    # Remote salary comparison
    remote_salary = filtered[filtered["remote_option"] == "Yes"]["salary_usd"].mean()
    onsite_salary = filtered[filtered["remote_option"] == "No"]["salary_usd"].mean()

    result = {
        "industry_filter": industry or "All industries",
        "total_jobs": len(filtered),
        "remote_breakdown": {
            "remote_jobs": int(remote_counts.get("Yes", 0)),
            "onsite_jobs": int(remote_counts.get("No", 0)),
            "remote_percentage": f"{remote_pct.get('Yes', 0)}%"
        },
        "salary_comparison": {
            "remote_avg_salary": f"${remote_salary:,.0f}" if not pd.isna(remote_salary) else "N/A",
            "onsite_avg_salary": f"${onsite_salary:,.0f}" if not pd.isna(onsite_salary) else "N/A"
        }
    }

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
