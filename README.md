# LangGraph + MCP Integration
### Build a simple AI agent that uses Model Context Protocol to access tools

---

## 🧠 What I Learned

- How **LangGraph** creates stateful agent workflows as graphs
- What **MCP (Model Context Protocol)** is and why Anthropic created it
- How agents discover and call tools via MCP servers
- Difference between a simple LLM call vs a proper agentic loop

## 🏗️ What This Project Does

A LangGraph agent that:
1. Takes a user question
2. Decides which MCP tool to use (weather, calculator, or file reader)
3. Calls the tool via MCP protocol
4. Returns a grounded answer

```
User Query
    ↓
LangGraph Agent (StateGraph)
    ↓
Tool Selection Node
    ↓
MCP Tool Call (weather / calculator / file)
    ↓
Response Generation Node
    ↓
Final Answer
```

## 📦 Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add your OPENAI_API_KEY or ANTHROPIC_API_KEY to .env
python src/main.py
```

## 🔑 Key Concepts

| Concept | What it means |
|---------|--------------|
| StateGraph | LangGraph's way of defining agent flow as nodes + edges |
| MCP Server | A standardised server that exposes tools to AI agents |
| MCP Client | The agent side that discovers and calls MCP tools |
| Tool Node | A LangGraph node that executes tool calls |
| Conditional Edge | Routes the agent to different nodes based on state |

## 📝 Files

```
week1-langgraph-mcp/
├── src/
│   ├── main.py          # Entry point — run this
│   ├── agent.py         # LangGraph agent definition
│   ├── mcp_server.py    # Simple MCP server with tools
│   ├── tools.py         # Tool implementations
│   └── state.py         # Agent state definition
├── tests/
│   └── test_agent.py    # Basic tests
├── requirements.txt
├── .env.example
└── README.md
```
