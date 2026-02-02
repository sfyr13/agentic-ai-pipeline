"""
agentic_rag_pipeline.py

Agentic RAG demo using:
- LangGraph (orchestrator / loop)
- LangChain tools (retrieve tool)
- Chroma vector store (embeddings + similarity search)
- OpenAI (LLM + embeddings)

Run:
  1) set OPENAI_API_KEY env var
  2) python agentic_rag_pipeline.py
"""

from __future__ import annotations

import os
from typing import List

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState  # IMPORTANT: message-channel schema
from langgraph.prebuilt import ToolNode


# -----------------------------
# Config
# -----------------------------
PERSIST_DIR = "./chroma_rag_db"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
TOP_K = 3

MAX_TOOL_CALLS = 2  # Guardrail: max number of tool calls allowed overall


# -----------------------------
# Demo docs
# -----------------------------
TEXTS = [
    "Agentic AI pipeline: στόχος, απόφαση, χρήση εργαλείων, loop, και state/memory.",
    "RAG (Retrieval-Augmented Generation): retrieval με embeddings + similarity search και μετά generation με context.",
    "Agentic RAG: το LLM αποφασίζει πότε χρειάζεται retrieval tool πριν απαντήσει.",
    "LangGraph: ορίζεις agents/workflows ως γράφο με state και conditional routing.",
    "Orchestrator: συντονίζει τη ροή LLM → tools → LLM μέχρι να βγει τελικό αποτέλεσμα.",
]


# -----------------------------
# Vector store (Chroma)
# -----------------------------
def build_or_load_vectorstore() -> Chroma:
    """
    Creates or loads a persisted Chroma vector store.

    NOTE: Chroma.from_texts will add the TEXTS every run.
    For demo it's fine. For a cleaner demo, see the comment below.
    """
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vs = Chroma.from_texts(
        texts=TEXTS,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name="rag_demo",
    )
    return vs


vectorstore = build_or_load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})


# -----------------------------
# Tool: retrieve
# -----------------------------
@tool
def retrieve(query: str) -> str:
    """Retrieve top-k relevant chunks using embeddings similarity search."""
    docs = retriever.invoke(query)  # new API (no deprecation warning)
    if not docs:
        return "No relevant context found."

    return "\n\n".join(
        f"[Chunk {i + 1}]\n{d.page_content}" for i, d in enumerate(docs)
    )


tools = [retrieve]
tool_node = ToolNode(tools)


# -----------------------------
# State: extend MessagesState
# -----------------------------
class AgentState(MessagesState):
    tool_calls: int  # total number of tool calls used so far


# -----------------------------
# LLM node (agent brain)
# -----------------------------
system_msg = SystemMessage(
    content=(
        "You are an assistant that uses an agentic RAG approach.\n"
        "Rules:\n"
        "1) Decide whether you need external context.\n"
        "2) If needed, call the tool `retrieve` with a focused search query.\n"
        "3) If not needed, answer directly.\n"
        "4) If you called retrieve, use the returned context in your final answer.\n\n"
        "Output format (ALWAYS):\n"
        "1) Ορισμός (1-2 προτάσεις)\n"
        "2) Πότε κάνουμε Retrieval (bullets)\n"
        "3) Πώς βοηθά το LangGraph (bullets)\n"
        "4) Key takeaway (1 πρόταση)\n"
        "Keep it concise."
    )
)

llm = ChatOpenAI(model=LLM_MODEL, temperature=0).bind_tools(tools)


def llm_call(state: AgentState):
    """
    One step of the agent:
    - ensure system message is present
    - call the LLM (either final answer OR tool calls)
    """
    messages = state["messages"]
    if not messages or messages[0].type != "system":
        messages = [system_msg] + messages

    response = llm.invoke(messages)
    return {"messages": [response]}


# -----------------------------
# Guardrail helpers
# -----------------------------
def requested_tool_calls_in_last_ai_message(state: AgentState) -> int:
    """
    Returns how many tool calls the LLM requested in its last AI message.
    """
    last = state["messages"][-1]
    return len(getattr(last, "tool_calls", []) or [])


def route_after_llm(state: AgentState) -> str:
    """
    If LLM requested tool calls AND we have remaining budget -> go to tools.
    Otherwise -> end.

    Important: Prevent exceeding the budget in a single step.
    """
    requested = requested_tool_calls_in_last_ai_message(state)
    if requested == 0:
        return "end"

    used = state.get("tool_calls", 0)
    if used + requested <= MAX_TOOL_CALLS:
        return "tools"

    # Budget exceeded: do NOT run tools.
    # We end here; the model should ideally answer with what it has.
    return "end"


def add_requested_tool_calls_to_counter(state: AgentState):
    """
    Increment the counter by exactly how many tool calls the LLM requested
    (per-tool-call counting).
    NOTE: This must run right after llm_call (because tool_calls live on the AI message).
    """
    requested = requested_tool_calls_in_last_ai_message(state)
    return {"tool_calls": state.get("tool_calls", 0) + requested}


# -----------------------------
# Orchestrator graph
# -----------------------------
graph = StateGraph(AgentState)

graph.add_node("llm_call", llm_call)
graph.add_node("count_tool_calls", add_requested_tool_calls_to_counter)
graph.add_node("tool_node", tool_node)

graph.add_edge(START, "llm_call")

# Count how many tool calls were requested (per call) before deciding to execute them
graph.add_edge("llm_call", "count_tool_calls")

graph.add_conditional_edges(
    "count_tool_calls",
    route_after_llm,
    {
        "tools": "tool_node",
        "end": END,
    },
)

# After tools, loop back to LLM
graph.add_edge("tool_node", "llm_call")

agent = graph.compile()


# -----------------------------
# Demo run
# -----------------------------
def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Set it before running.")

    query = "Εξήγησέ μου τι είναι agentic RAG και πώς σχετίζεται με το LangGraph."
    result = agent.invoke({"messages": [HumanMessage(content=query)], "tool_calls": 0})

    print("\n--- Conversation ---\n")
    for m in result["messages"]:
        m.pretty_print()

    print(f"\n[Guardrail] tool calls used: {result.get('tool_calls', 0)} / {MAX_TOOL_CALLS}\n")


if __name__ == "__main__":
    main()