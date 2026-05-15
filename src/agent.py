"""
agent.py — LangGraph agent with MCP tool integration
Week 1: LangGraph + MCP Integration

This implements a ReAct-style agent using LangGraph's StateGraph.
The agent can reason, call MCP tools, observe results, and respond.

Flow:
  START → llm_node → should_use_tool? → tool_node → llm_node → END
                              ↓ (no tool needed)
                             END
"""

import os
import json
from typing import Literal

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from state import AgentState
from mcp_server import SimpleMCPServer


# ─────────────────────────────────────────────
# Initialize MCP Server (would be remote in production)
# ─────────────────────────────────────────────
mcp_server = SimpleMCPServer()


def create_agent():
    """
    Build and return a compiled LangGraph agent.
    
    The graph has two main nodes:
    1. llm_node    — calls the LLM with current messages
    2. tool_node   — executes MCP tool calls from LLM response
    """
    
    # ─── Node 1: LLM Node ───────────────────────────────────────────
    def llm_node(state: AgentState) -> AgentState:
        """
        Call the LLM with current conversation state.
        The LLM decides whether to call a tool or give a final answer.
        """
        print("\n[Agent] LLM node — thinking...")
        
        # Build available tools description for the prompt
        available_tools = mcp_server.list_tools()
        tools_description = "\n".join([
            f"- {t['name']}: {t['description']}" 
            for t in available_tools
        ])
        
        system_prompt = f"""You are a helpful AI assistant with access to tools via MCP.

Available tools:
{tools_description}

To use a tool, respond ONLY with valid JSON in this exact format:
{{"tool_call": {{"name": "tool_name", "arguments": {{"param": "value"}}}}}}

If you don't need a tool, respond normally in plain text.
Always be concise and helpful."""

        # Simulate LLM response (replace with real LLM call)
        # In production: use ChatOpenAI or ChatAnthropic
        last_message = state["messages"][-1].content if state["messages"] else ""
        
        # Simple routing logic — in production the LLM decides this
        response_text = _simulate_llm_decision(last_message, available_tools)
        
        ai_message = AIMessage(content=response_text)
        
        return {
            "messages": [ai_message],
            "tool_calls_made": state.get("tool_calls_made", []),
            "final_answer": ""
        }

    # ─── Node 2: Tool Node ──────────────────────────────────────────
    def tool_node(state: AgentState) -> AgentState:
        """
        Execute MCP tool calls from the LLM's response.
        Returns tool results back into the message history.
        """
        print("\n[Agent] Tool node — executing MCP call...")
        
        last_ai_message = state["messages"][-1]
        
        try:
            tool_call = json.loads(last_ai_message.content)["tool_call"]
            tool_name = tool_call["name"]
            tool_args = tool_call["arguments"]
            
            # ← This is the MCP call ←
            result = mcp_server.call_tool(tool_name, tool_args)
            tool_result = result["content"][0]["text"]
            
            tool_message = ToolMessage(
                content=tool_result,
                tool_call_id=f"call_{tool_name}"
            )
            
            tools_used = state.get("tool_calls_made", []) + [tool_name]
            
            return {
                "messages": [tool_message],
                "tool_calls_made": tools_used,
                "final_answer": ""
            }
            
        except Exception as e:
            error_message = ToolMessage(
                content=f"Tool execution failed: {str(e)}",
                tool_call_id="error"
            )
            return {
                "messages": [error_message],
                "tool_calls_made": state.get("tool_calls_made", []),
                "final_answer": ""
            }

    # ─── Node 3: Final Response Node ────────────────────────────────
    def final_response_node(state: AgentState) -> AgentState:
        """Generate final response after tool use."""
        print("\n[Agent] Generating final response...")
        
        # Find the tool result in messages
        tool_result = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, ToolMessage):
                tool_result = msg.content
                break
        
        final = f"Based on the tool result: {tool_result}"
        
        return {
            "messages": [AIMessage(content=final)],
            "tool_calls_made": state.get("tool_calls_made", []),
            "final_answer": final
        }

    # ─── Conditional Edge: Should we call a tool? ───────────────────
    def should_use_tool(state: AgentState) -> Literal["tool_node", "end"]:
        """
        Routing function — decides next node based on LLM output.
        If LLM returned a tool_call JSON → go to tool_node
        Otherwise → end the graph
        """
        last_message = state["messages"][-1]
        
        if not isinstance(last_message, AIMessage):
            return "end"
        
        try:
            parsed = json.loads(last_message.content)
            if "tool_call" in parsed:
                print("[Router] Tool call detected → routing to tool_node")
                return "tool_node"
        except (json.JSONDecodeError, TypeError):
            pass
        
        print("[Router] No tool call → ending")
        return "end"

    # ─── Build the Graph ────────────────────────────────────────────
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("llm_node", llm_node)
    graph.add_node("tool_node", tool_node)
    graph.add_node("final_response_node", final_response_node)
    
    # Add edges
    graph.add_edge(START, "llm_node")
    graph.add_conditional_edges(
        "llm_node",
        should_use_tool,
        {
            "tool_node": "tool_node",
            "end": END
        }
    )
    graph.add_edge("tool_node", "final_response_node")
    graph.add_edge("final_response_node", END)
    
    print("[Agent] Graph compiled successfully")
    print("[Agent] Nodes:", ["llm_node", "tool_node", "final_response_node"])
    
    return graph.compile()


def _simulate_llm_decision(user_message: str, tools: list) -> str:
    """
    Simulates LLM decision making.
    In production: replace with ChatOpenAI(model="gpt-4o").invoke(messages)
    
    This lets you run the project without an API key for learning purposes.
    """
    message_lower = user_message.lower()
    
    if any(word in message_lower for word in ["weather", "temperature", "rain", "sunny"]):
        city = "London"
        for word in message_lower.split():
            if word.istitle() and len(word) > 2:
                city = word
                break
        return json.dumps({
            "tool_call": {
                "name": "get_weather",
                "arguments": {"city": city}
            }
        })
    
    elif any(word in message_lower for word in ["calculate", "compute", "math", "what is", "×", "+"]):
        # Extract a simple expression
        return json.dumps({
            "tool_call": {
                "name": "calculate",
                "arguments": {"expression": "15 * 4 + 7"}
            }
        })
    
    elif any(word in message_lower for word in ["time", "date", "clock"]):
        return json.dumps({
            "tool_call": {
                "name": "get_current_time",
                "arguments": {"timezone": "Europe/London"}
            }
        })
    
    else:
        return f"I understand your question about: '{user_message}'. I can help with weather, calculations, and time. What would you like to know?"
