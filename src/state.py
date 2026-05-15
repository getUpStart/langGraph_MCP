"""
state.py — Agent state definition for LangGraph
Week 1: LangGraph + MCP Integration

The state is the shared memory that flows through
every node in our LangGraph agent.
"""

from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    State shared across all nodes in the agent graph.
    
    messages: Conversation history (add_messages = append-only)
    tool_calls_made: Track which tools were called (for debugging)
    final_answer: The agent's final response
    """
    messages: Annotated[list[AnyMessage], add_messages]
    tool_calls_made: list[str]
    final_answer: str
