"""
main.py — Entry point for Week 1: LangGraph + MCP Integration
Run this file to see the agent in action.

Usage:
    python src/main.py

What happens:
    1. MCP Server starts and registers tools
    2. LangGraph agent is compiled (graph with nodes + edges)
    3. Three demo queries are run through the agent
    4. You can see the full agentic loop in action
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from langchain_core.messages import HumanMessage
from agent import create_agent


def run_demo():
    """Run the agent with three example queries."""
    
    print("=" * 60)
    print("  WEEK 1: LangGraph + MCP Integration Demo")
    print("  AI Learning Portfolio — Vishal Kumar Singh")
    print("=" * 60)
    
    # Compile the agent graph
    agent = create_agent()
    
    # Test queries
    test_queries = [
        "What's the weather like in London today?",
        "Can you calculate 15 multiplied by 4 plus 7?",
        "What time is it in London right now?",
        "Tell me something about Python programming"  # No tool needed
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'─' * 60}")
        print(f"Query {i}: {query}")
        print("─" * 60)
        
        # Run the agent
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "tool_calls_made": [],
            "final_answer": ""
        }
        
        result = agent.invoke(initial_state)
        
        # Show results
        print(f"\n✅ Tools used: {result['tool_calls_made'] or ['None']}")
        print(f"📝 Response: {result['messages'][-1].content}")
        print(f"📊 Total messages in state: {len(result['messages'])}")
    
    print(f"\n{'=' * 60}")
    print("Demo complete! Key concepts demonstrated:")
    print("  ✅ StateGraph with typed state (AgentState)")
    print("  ✅ LLM node → conditional routing → tool node")
    print("  ✅ MCP server with multiple tools")
    print("  ✅ MCP tool discovery and execution")
    print("  ✅ Message history management")
    print("  ✅ Graceful no-tool fallback")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
