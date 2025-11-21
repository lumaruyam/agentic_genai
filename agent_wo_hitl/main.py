#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import uuid
from typing import TypedDict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import(
    PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, ArxivLoader
)
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import StateGraph, END
from langgraph.graph import MessagesState
from langgraph.checkpoint.memory import InMemorySaver

from langfuse.openai import openai
from langfuse import get_client
from langfuse.langchain import CallbackHandler


# In[ ]:


import os
from dotenv import load_dotenv

load_dotenv()  # Loads .env file into environment
api_key = os.getenv("OPENAI_API_KEY")
langfuse_public = os.getenv("LANGFUSE_PUBLIC_KEY")
langfuse_secret = os.getenv("LANGFUSE_SECRET_KEY")

print("✅ API key loaded:", bool(api_key))
print("✅ Langfuse public API key loaded:", bool(langfuse_public))
print("✅ Langfuse secret API key loaded:", bool(langfuse_secret))


# In[ ]:


os.makedirs("./notebooks/uploads", exist_ok=True)
os.makedirs("./notebooks/vectorstore", exist_ok=True)


# In[ ]:


langfuse = get_client()
langfuse_handler = CallbackHandler()


# In[ ]:


# Chat Logging Setup
conversation_log = []

def log_message(role: str, content: str):
    """Log a chat message to the conversation history."""
    conversation_log.append({"role": role, "content": content})


# In[ ]:


orchestrator = ChatOpenAI(model="gpt-4o-mini", temperature=0)
plan_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
draft_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
critic_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
final_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)


# In[ ]:


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Chroma(persist_directory="./vectorstore", embedding_function=embeddings)


# In[ ]:


class AgentState(TypedDict):
    query: str
    intent: str
    context: str
    plan: str
    draft: str
    critique: str
    final: str
    research_mode: str 


# In[ ]:


# Guardrails
FORBIDDEN_TOPICS = ["politics", "religion", "violence", "illegal", "personal"]

def is_out_of_scope(query: str) -> bool:
    q = query.lower()
    return any(t in q for t in FORBIDDEN_TOPICS)

SYSTEM_ORCHESTRATOR = """
You are the Orchestrator Agent.
Classify the user's query as:
- 'general' → for simple factual questions.
- 'research' → for analytical or academic topics.
- 'blocked' → if unrelated to factual or academic work.
Respond ONLY with one word: general, research, or blocked.
"""

SYSTEM_ANALYZER = """
You are the Analyzer Agent.
Search for factual and academic information, summarize objectively,
and include short source indicators like (source: arxiv, local doc, or web).
"""

SYSTEM_PLANNER = """
You are the Plan Writer Agent.
Create a clear academic outline (Introduction, Body, Conclusion)
based on the given question and retrieved context.
Do NOT write the essay — only the plan.
"""

SYSTEM_WRITER = """
You are the Draft Writer Agent.
Expand the plan into a coherent, well-structured essay (400–600 words)
with academic tone, logical flow, and factual precision.
"""

SYSTEM_CRITIC = """
You are the Critic Agent.
Review the essay for clarity, coherence, structure, and evidence quality.
Offer concise suggestions for improvement (under 150 words).
"""

SYSTEM_FINALIZER = """
You are the Final Drafter Agent.
Polish the essay for grammar, tone, and academic conciseness.
Ensure clear formatting, strong argumentation, and no redundancy.
"""

def safe_invoke(llm, query: str, system_role: str, context: str = "") -> str:
    """Unified LLM call enforcing safety and academic tone."""
    system_map = {
        "orchestrator": SYSTEM_ORCHESTRATOR,
        "analyzer": SYSTEM_ANALYZER,
        "planner": SYSTEM_PLANNER,
        "writer": SYSTEM_WRITER,
        "critic": SYSTEM_CRITIC,
        "finalizer": SYSTEM_FINALIZER,
    }

    system_prompt = system_map.get(system_role.lower(), system_role)
    prompt = f"""
System Role:
{system_prompt}

Context:
{context}

User query:
{query}

Respond academically, factually, and concisely.
    """
    return llm.invoke(prompt).content


# In[ ]:


def load_uploaded_documents(upload_folder="./notebooks/uploads"):
    """Loads and embeds uploaded PDFs, TXTs, DOCXs."""
    print("📂 Loading uploaded documents...")
    files = [f for f in os.listdir(upload_folder) if f.lower().endswith((".pdf", ".txt", ".docx"))]
    docs = []
    for file in files:
        path = os.path.join(upload_folder, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif file.endswith(".txt"):
            loader = TextLoader(path)
        else:
            loader = UnstructuredWordDocumentLoader(path)
        loaded = loader.load()
        vectordb.add_documents(filter_complex_metadata(loaded))
        docs.extend(loaded)
    print(f"✅ {len(docs)} uploaded docs indexed.")
    return docs


def route_query(state: AgentState):
    query = state.get("query", "")
    if is_out_of_scope(query):
        state["intent"] = "blocked"
        state["final"] = "⚠️ This question is out of academic scope."
        return state

    state["intent"] = safe_invoke(orchestrator, query, "orchestrator").strip().lower()
    print(f"🧭 Intent classified as: {state['intent']}")
    return state



def general_answer(state: AgentState):
    query = state["query"]
    state["final"] = safe_invoke(plan_llm, query, "General academic explanation")
    print("💬 General answer complete.")
    return state


def analyzer_collect(state: AgentState):
    """Academic content from Arxiv (main)"""
    query = state["query"]
    print("🔍 Searching arXiv for relevant preprints...")
    try:
        loader = ArxivLoader(query=query, load_max_docs=3)
        docs = loader.load()
        if not docs:
            state["context"] = "No academic sources found."
            return state
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        vectordb.add_documents(chunks)
        retriever = vectordb.as_retriever(search_kwargs={"k": 5})
        top_docs = retriever.invoke(query)
        state["context"] = "\n\n".join([d.page_content for d in top_docs])
        print(f"📚 {len(chunks)} arXiv chunks added.")
    except Exception as e:
        state["context"] = f"Error loading Arxiv: {e}"
    return state

def local_doc_search(state: AgentState):
    query = state["query"]
    print("📁 Searching uploaded docs...")
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(query)
    state["context"] = "\n\n".join([d.page_content for d in docs])
    print(f"📘 Retrieved {len(docs)} local docs.")
    return state


def list_documents(state: AgentState):
    """Return a list of the most relevant document summaries."""
    query = state.get("query", "")
    print("📑 Retrieving top documents...")
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(query)

    if not docs:
        state["final"] = "⚠️ No relevant documents found."
        return state

    summaries = "\n\n".join([
        f"📄 {i+1}. {d.page_content[:300]}..."
        for i, d in enumerate(docs)
    ])
    state["final"] = f"Top 5 relevant documents:\n\n{summaries}"
    print("📄 Generated top 5 relevant document list.")
    return state

def plan_writer(state: AgentState):
    query, context = state["query"], state.get("context", "")
    state["plan"] = safe_invoke(plan_llm, query, "Academic plan generator", context)
    print("📝 Plan written.")
    return state


def draft_writer(state: AgentState):
    """Writes or rewrites a draft based on plan and feedback."""
    plan = state.get("plan", "")
    feedback = state.get("feedback", "")

    if feedback:
        query = f"Rewrite the draft considering this human feedback:\n{feedback}\n\nPlan:\n{plan}"
        print("✍️ Rewriting draft using human feedback...")
    else:
        query = plan
        print("✍️ Writing new draft...")

    state["draft"] = safe_invoke(draft_llm, query, "writer")
    # Clear feedback after using it to avoid recursive loops
    state.pop("feedback", None)
    return state

def critic_agent(state: AgentState):
    draft = state["draft"]
    state["critique"] = safe_invoke(critic_llm, draft, "Academic critic")
    print("🧾 Critique done.")
    return state


def final_drafter(state: AgentState):
    draft = state["draft"]
    state["final"] = safe_invoke(final_llm, draft, "Academic finalizer")
    vectordb.add_texts([state["final"]])
    print("✅ Final draft ready.")
    return state


# In[ ]:


# --- Graph Definition ---
graph = StateGraph(AgentState)

# Existing nodes
graph.add_node("route_query", route_query)
graph.add_node("general_answer", general_answer)
graph.add_node("analyzer_collect", analyzer_collect)
graph.add_node("local_doc_search", local_doc_search)
graph.add_node("list_documents", list_documents)
graph.add_node("plan_writer", plan_writer)
graph.add_node("draft_writer", draft_writer)
graph.add_node("critic_agent", critic_agent)
graph.add_node("final_drafter", final_drafter)

def route_decision(state):
    intent = state.get("intent", "")
    query = state.get("query", "").lower()

    if intent == "blocked":
        return END
    elif intent == "general":
        return "general_answer"
    elif "list" in query or "show" in query or "papers" in query:
        return "list_documents"
    elif os.listdir("./notebooks/uploads"):
        return "local_doc_search"
    else:
        return "analyzer_collect"

graph.add_conditional_edges(
    "route_query",
    route_decision,
    {
        "general_answer": "general_answer",
        "list_documents": "list_documents",
        "local_doc_search": "local_doc_search",
        "analyzer_collect": "analyzer_collect",
    },
)


graph.add_edge("local_doc_search", "list_documents")
graph.add_edge("analyzer_collect", "list_documents")
graph.add_edge("list_documents", "plan_writer")
graph.add_edge("plan_writer", "draft_writer")
graph.add_edge("draft_writer", "critic_agent")
graph.add_edge("critic_agent", "final_drafter")
graph.add_edge("final_drafter", END)
graph.add_edge("general_answer", END)

graph.set_entry_point("route_query")
memory = InMemorySaver()
app = graph.compile(checkpointer=memory)
app


# In[ ]:


def list_checkpoints(memory, thread_id=None):
    """Safely list checkpoints for LangGraph 1.x+ (and fallbacks for older versions)."""
    print("\n🧠 Saved checkpoints:")
    try:
        if hasattr(memory, "list"):
            try:
                # Modern API for LangGraph >= 1.0
                checkpoints = memory.list(config={"thread_id": thread_id})
            except TypeError:
                # Backward fallback for older versions
                checkpoints = memory.list()

            if not checkpoints:
                print("⚠️ No checkpoints found.")
            else:
                for c in checkpoints:
                    tid = getattr(c, "thread_id", None) or c.get("thread_id", "unknown")
                    cid = getattr(c, "checkpoint_id", None) or c.get("checkpoint_id", "unknown")
                    print(f"🧩 Thread ID: {tid}, Checkpoint ID: {cid}")
        else:
            print("⚠️ This memory saver does not support listing checkpoints.")
    except Exception as e:
        print(f"⚠️ Could not list checkpoints: {e}")


# In[ ]:


if __name__ == "__main__":
    import uuid
    from langgraph.checkpoint.memory import InMemorySaver
    import json

    # Initialize memory and compile app with checkpointing
    memory = InMemorySaver()
    app = graph.compile()

    # Load uploaded docs (optional)
    load_uploaded_documents("./notebooks/uploads")

    # Ask user input
    query = input("🔍 Enter your academic question: ")
    state = {"query": query}

    # Generate unique thread ID for session
    thread_id = f"thread-{uuid.uuid4()}"
    print(f"🧩 Using thread_id: {thread_id}")

    # Run agent (modern or legacy fallback)
    try:
        final_state = app.invoke(
    state,
    config={
        "configurable": {"thread_id": thread_id},
        "callbacks": [langfuse_handler],
    }
)
    except Exception:
        print("⚙️ Falling back to legacy invocation...")
        final_state = app.invoke(
    state,
    config={
        "configurable": {"thread_id": thread_id},
        "callbacks": [langfuse_handler],
    }
)

    # Save chat history log
    os.makedirs("./notebooks/logs", exist_ok=True)
    log_path = f"./notebooks/logs/chat_history_{thread_id}.json"
    with open(log_path, "w") as f:
        json.dump(final_state, f, indent=2)
    print(f"\n🧾 Chat history saved → {log_path}")

    # List checkpoints safely
    list_checkpoints(memory, thread_id=thread_id)

    # Show final result
    print("\n🎓 --- FINAL OUTPUT ---")
    print(final_state["final"])

def orchestrate_agent(query: str):
    """Simple wrapper to run the LangGraph pipeline and return the final state."""
    state = {"query": query}

    # Each call gets its own thread_id for clean checkpointing
    thread_id = f"thread-{uuid.uuid4()}"

    try:
        final_state = app.invoke(
    state,
    config={
        "configurable": {"thread_id": thread_id},
        "callbacks": [langfuse_handler],
    }
)
    except Exception:
        # fallback for older LangGraph versions
        final_state = app.invoke(
    state,
    config={
        "configurable": {"thread_id": thread_id},
        "callbacks": [langfuse_handler],
    }
)

    return final_state

def add_document_to_vectordb(path: str):
    """Embed and store a new uploaded file in the vector DB."""
    ext = path.lower()

    if ext.endswith(".pdf"):
        loader = PyPDFLoader(path)
    elif ext.endswith(".txt"):
        loader = TextLoader(path)
    else:
        loader = UnstructuredWordDocumentLoader(path)

    docs = loader.load()

    # Split for better retrieval
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    vectordb.add_documents(filter_complex_metadata(chunks))

    print(f"📚 Added {len(chunks)} chunks from {os.path.basename(path)}")