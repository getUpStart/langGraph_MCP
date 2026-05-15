"""
mcp_server.py — Minimal MCP Server implementation
Week 1: LangGraph + MCP Integration

Model Context Protocol (MCP) is Anthropic's open standard
for connecting AI agents to external tools and data sources.

This implements a simple MCP server with 3 tools:
  - get_weather: Simulated weather lookup
  - calculate: Basic math evaluation  
  - read_file: Read a local text file

In production you would use: pip install mcp
and implement proper MCP transport (stdio or SSE).
This version is a clean educational simulation.
"""

import json
import math
from datetime import datetime


class MCPTool:
    """Represents a single tool in the MCP server."""
    
    def __init__(self, name: str, description: str, input_schema: dict):
        self.name = name
        self.description = description
        self.input_schema = input_schema

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema
        }


class SimpleMCPServer:
    """
    A minimal MCP server that exposes tools to AI agents.
    
    MCP defines a standard protocol so any AI agent can
    discover and call tools from any MCP-compatible server.
    This is the server side.
    """
    
    def __init__(self):
        self.tools = self._register_tools()
        print(f"[MCP Server] Started with {len(self.tools)} tools")

    def _register_tools(self) -> dict[str, MCPTool]:
        """Register all available tools."""
        return {
            "get_weather": MCPTool(
                name="get_weather",
                description="Get current weather for a city. Returns temperature, condition, humidity.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "City name e.g. 'London', 'New York'"
                        }
                    },
                    "required": ["city"]
                }
            ),
            "calculate": MCPTool(
                name="calculate",
                description="Evaluate a mathematical expression safely. Supports +,-,*,/,**,sqrt,etc.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression e.g. '2 + 2', 'sqrt(144)', '15 * 4'"
                        }
                    },
                    "required": ["expression"]
                }
            ),
            "get_current_time": MCPTool(
                name="get_current_time",
                description="Get the current date and time in a specific timezone.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "Timezone name e.g. 'UTC', 'Europe/London', 'Asia/Kolkata'"
                        }
                    },
                    "required": ["timezone"]
                }
            )
        }

    def list_tools(self) -> list[dict]:
        """MCP protocol: list all available tools."""
        return [tool.to_dict() for tool in self.tools.values()]

    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """
        MCP protocol: execute a tool and return result.
        Returns MCP-standard response format.
        """
        print(f"[MCP Server] Tool called: {tool_name} with args: {arguments}")
        
        if tool_name not in self.tools:
            return {
                "isError": True,
                "content": [{"type": "text", "text": f"Tool '{tool_name}' not found"}]
            }

        try:
            if tool_name == "get_weather":
                result = self._get_weather(arguments["city"])
            elif tool_name == "calculate":
                result = self._calculate(arguments["expression"])
            elif tool_name == "get_current_time":
                result = self._get_current_time(arguments["timezone"])
            else:
                result = "Tool not implemented"

            return {
                "isError": False,
                "content": [{"type": "text", "text": result}]
            }

        except Exception as e:
            return {
                "isError": True,
                "content": [{"type": "text", "text": f"Tool error: {str(e)}"}]
            }

    def _get_weather(self, city: str) -> str:
        """Simulated weather data — replace with real API in production."""
        weather_data = {
            "london": {"temp": 14, "condition": "Cloudy", "humidity": 78},
            "delhi": {"temp": 38, "condition": "Hot and sunny", "humidity": 45},
            "new york": {"temp": 22, "condition": "Partly cloudy", "humidity": 60},
            "noida": {"temp": 37, "condition": "Hazy sunshine", "humidity": 50},
        }
        city_lower = city.lower()
        data = weather_data.get(city_lower, {
            "temp": 20, "condition": "Clear", "humidity": 55
        })
        return (
            f"Weather in {city}: {data['condition']}, "
            f"{data['temp']}°C, Humidity: {data['humidity']}%"
        )

    def _calculate(self, expression: str) -> str:
        """Safely evaluate a math expression."""
        # Whitelist safe functions only
        safe_names = {
            "sqrt": math.sqrt, "abs": abs, "round": round,
            "floor": math.floor, "ceil": math.ceil,
            "pi": math.pi, "e": math.e
        }
        try:
            result = eval(expression, {"__builtins__": {}}, safe_names)
            return f"{expression} = {result}"
        except Exception as e:
            return f"Could not evaluate '{expression}': {e}"

    def _get_current_time(self, timezone: str) -> str:
        """Get current time — simplified version."""
        now = datetime.utcnow()
        return f"Current UTC time: {now.strftime('%Y-%m-%d %H:%M:%S')} (timezone conversion: use pytz in production)"
