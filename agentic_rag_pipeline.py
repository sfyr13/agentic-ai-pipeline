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

from langchain_chroma import Chroma


import os
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

class AgentState(MessagesState):
    tool_calls: int  # count of tool calls made so far

#config

PERSIST_DIR = "./chroma_rag_db"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
TOP_K = 3
MAX_TOOL_CALLS = 3  # Guardrail: max number of tool calls allowed overaLL

#Build / load vector store

texts = [
    "Agentic AI pipeline: στόχος, απόφαση, χρήση εργαλείων, loop, και state/memory.",
    "RAG (Retrieval-Augmented Generation): retrieval με embeddings + similarity search και μετά generation με context.",
    "Agentic RAG: το LLM αποφασίζει πότε χρειάζεται retrieval tool πριν απαντήσει.",
    "LangGraph: ορίζεις agents/workflows ως γράφο με state και conditional routing.",
    "Orchestrator: συντονίζει τη ροή LLM → tools → LLM μέχρι να βγει τελικό αποτέλεσμα."
]
def build_or_load_vectorstore() -> Chroma:

    """
    Creates a Chroma vector store with embeddings and persists it locally.
    If the persist directory already exists, Chroma will load from it.
    """

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name="rag_demo",
    )
    return vectorstore

vectorstore = build_or_load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})    

#Define retrieve tool
@tool
def retrieve(query: str) -> str:
    """
    Retrieve top-k relevant chunks from the vector store using embeddings similarity
    Returns formatted context to inject into the conversation
    """

    docs = retriever.invoke(query)

    if not docs:
        return "No relevant context found."

    # docs are LangChain Documents, doc.page_content contains the text chunk
    context = "\n\n".join(
        f"[Chunk {i+1}]\n{d.page_content}"
        for i, d in enumerate(docs)
    )
    return context

tools = [retrieve]
tool_node = ToolNode(tools)

#LLM node(agent brain)

system_msg = SystemMessage(
    content=(
        "You are an assistant that uses an agentic RAG approach.\n"
        "Rules:\n"
        "1) Decide whether you need external context to answer.\n"
        "2) If needed, call the tool `retrieve` with a good search query.\n"
        "3) If not needed, answer directly.\n"
        "4) If you called retrieve, use the returned context in your final answer.\n"
        "Keep answers concise and correct."
    )
)

llm = ChatOpenAI(model=LLM_MODEL, temperature=0).bind_tools(tools)

def llm_call(state: MessagesState):
    """
    One step of the agent:
    - ensure system message is present
    - ask the LLM to respond (either a final answer OR a tool call)
    - append response to the message state
    """

    messages = state["messages"]

    if not messages or messages[0].type != "system":
        messages = [system_msg] + messages

    response = llm.invoke(messages)

    # count tool calls from this LLM response
    tool_calls = getattr(response, "tool_calls", None) or []
    new_count = state.get("tool_calls_count", 0) + len(tool_calls)
    return {
        "messages": [response], 
        "tool_calls_count": new_count
        }

def tools_or_end_with_cap(state: AgentState):
    # If we've already hit the cap, force END
    if state.get("tool_calls_count", 0) >= 3:
        return END
    # Otherwise, use normal tool routing
    return tools_condition(state)

#Orchestrator graph

graph = StateGraph(AgentState)
graph.add_node("llm_call", llm_call)
graph.add_node("tool_node", tool_node)

graph.add_edge(START, "llm_call")

# Route depending on whether the LLM asked for tools.
graph.add_conditional_edges(
    "llm_call",
    tools_or_end_with_cap,              # checks last message for tool calls
    {"tools": "tool_node", END: END},
)

# After tool execution, go back to LLM (loop)
graph.add_edge("tool_node", "llm_call")

#Compile graph into a runnable  agent
agent = graph.compile()

#Demo run

def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Set it in your environment before running."
        )

    query = "Εξήγησέ μου τι είναι agentic RAG και πώς σχετίζεται με το LangGraph."
    result = agent.invoke({"messages": [HumanMessage(content=query)],
                           "tool_calls_count": 0})

    print("\n--- Conversation ---\n")
    for m in result["messages"]:
        m.pretty_print()


if __name__ == "__main__":
    main()