"""
test_agent.py — Tests for Week 1 LangGraph + MCP agent
Run with: python -m pytest tests/
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mcp_server import SimpleMCPServer


class TestMCPServer:
    """Tests for the MCP server tool implementations."""
    
    def setup_method(self):
        self.server = SimpleMCPServer()
    
    def test_list_tools_returns_all_tools(self):
        tools = self.server.list_tools()
        assert len(tools) == 3
        tool_names = [t["name"] for t in tools]
        assert "get_weather" in tool_names
        assert "calculate" in tool_names
        assert "get_current_time" in tool_names
    
    def test_weather_tool_returns_result(self):
        result = self.server.call_tool("get_weather", {"city": "London"})
        assert result["isError"] == False
        assert "London" in result["content"][0]["text"]
        assert "°C" in result["content"][0]["text"]
    
    def test_calculate_tool_basic_math(self):
        result = self.server.call_tool("calculate", {"expression": "2 + 2"})
        assert result["isError"] == False
        assert "4" in result["content"][0]["text"]
    
    def test_calculate_tool_complex(self):
        result = self.server.call_tool("calculate", {"expression": "sqrt(144)"})
        assert result["isError"] == False
        assert "12" in result["content"][0]["text"]
    
    def test_unknown_tool_returns_error(self):
        result = self.server.call_tool("nonexistent_tool", {})
        assert result["isError"] == True
    
    def test_tool_schema_is_valid(self):
        tools = self.server.list_tools()
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
            assert tool["inputSchema"]["type"] == "object"
    
    def test_weather_unknown_city(self):
        """Unknown city should return default weather, not error."""
        result = self.server.call_tool("get_weather", {"city": "Atlantis"})
        assert result["isError"] == False


if __name__ == "__main__":
    # Run tests without pytest
    tester = TestMCPServer()
    tester.setup_method()
    
    tests = [
        tester.test_list_tools_returns_all_tools,
        tester.test_weather_tool_returns_result,
        tester.test_calculate_tool_basic_math,
        tester.test_calculate_tool_complex,
        tester.test_unknown_tool_returns_error,
        tester.test_tool_schema_is_valid,
        tester.test_weather_unknown_city,
    ]
    
    passed = 0
    for test in tests:
        try:
            tester.setup_method()
            test()
            print(f"{test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"{test.__name__}: {e}")
    
    print(f"\n{passed}/{len(tests)} tests passed")
