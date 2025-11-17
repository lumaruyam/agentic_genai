# back_end.py
from typing import TypedDict
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, ArxivLoader
)
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END

# --- Environment & directories ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
print("✅ API key loaded:", bool(api_key))

os.makedirs("./notebooks/uploads", exist_ok=True)
os.makedirs("./notebooks/vectorstore", exist_ok=True)

# --- LLMs & embeddings ---
orchestrator = ChatOpenAI(model="gpt-4o-mini", temperature=0)
plan_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
draft_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
critic_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
final_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Chroma(persist_directory="./vectorstore", embedding_function=embeddings)

# --- Agent State ---
class AgentState(TypedDict, total=False):
    query: str
    intent: str
    context: str
    plan: str
    draft: str
    critique: str
    final: str
    research_mode: str 

# --- Guardrails ---
FORBIDDEN_TOPICS = ["politics", "religion", "violence", "illegal", "personal"]
def is_out_of_scope(query: str) -> bool:
    return any(t in query.lower() for t in FORBIDDEN_TOPICS)

# --- System prompts ---
SYSTEM_ORCHESTRATOR = """
You are the Orchestrator Agent.
Classify the user's query as:
- 'general' → for simple factual questions.
- 'research' → for analytical or academic topics.
- 'blocked' → if unrelated to factual or academic work.
Respond ONLY with one word: general, research, or blocked.
"""

SYSTEM_ANALYZER = "You are the Analyzer Agent. Summarize factual and academic info with short source indicators."
SYSTEM_PLANNER = "You are the Plan Writer Agent. Create an academic outline; do NOT write the essay."
SYSTEM_WRITER = "You are the Draft Writer Agent. Expand plan into a coherent essay (400–600 words)."
SYSTEM_CRITIC = "You are the Critic Agent. Review draft for clarity, coherence, and evidence quality."
SYSTEM_FINALIZER = "You are the Final Drafter Agent. Polish essay for grammar, tone, and conciseness."

# --- Core functions ---
def safe_invoke(llm, query: str, system_role: str, context: str = "") -> str:
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

def load_uploaded_documents(upload_folder="./notebooks/uploads"):
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

# --- State functions ---
def route_query(state: AgentState):
    query = state.get("query", "")
    if is_out_of_scope(query):
        state["intent"] = "blocked"
        state["final"] = "⚠️ This question is out of academic scope."
    else:
        state["intent"] = safe_invoke(orchestrator, query, "orchestrator").strip().lower()
    return state

def general_answer(state: AgentState):
    state["final"] = safe_invoke(plan_llm, state["query"], "General academic explanation")
    return state

def analyzer_collect(state: AgentState):
    query = state["query"]
    try:
        loader = ArxivLoader(query=query, load_max_docs=3)
        docs = loader.load()
        if docs:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(docs)
            vectordb.add_documents(chunks)
            retriever = vectordb.as_retriever(search_kwargs={"k": 5})
            top_docs = retriever.invoke(query)
            state["context"] = "\n\n".join([d.page_content for d in top_docs])
    except Exception as e:
        state["context"] = f"Error loading Arxiv: {e}"
    return state

def local_doc_search(state: AgentState):
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(state["query"])
    state["context"] = "\n\n".join([d.page_content for d in docs])
    return state

def list_documents(state: AgentState):
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(state["query"])
    if docs:
        summaries = "\n\n".join([f"📄 {i+1}. {d.page_content[:300]}..." for i, d in enumerate(docs)])
        state["final"] = f"Top 5 relevant documents:\n\n{summaries}"
    else:
        state["final"] = "⚠️ No relevant documents found."
    return state

def plan_writer(state: AgentState):
    state["plan"] = safe_invoke(plan_llm, state["query"], "Academic plan generator", state.get("context", ""))
    return state

def draft_writer(state: AgentState):
    state["draft"] = safe_invoke(draft_llm, state["plan"], "Essay writer")
    return state

def critic_agent(state: AgentState):
    state["critique"] = safe_invoke(critic_llm, state["draft"], "Academic critic")
    return state

def final_drafter(state: AgentState):
    state["final"] = safe_invoke(final_llm, state["draft"], "Academic finalizer")
    vectordb.add_texts([state["final"]])
    return state

# --- Graph setup ---
graph = StateGraph(AgentState)
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
    elif any(word in query for word in ["list", "show", "papers"]):
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

app = graph.compile()

def orchestrate_agent(query: str) -> AgentState:
    """Convenience function to run a query through the full agent graph."""
    state: AgentState = {"query": query}
    final_state = app.invoke(state)
    return final_state

def add_document_to_vectordb(file_path: str):
    """Add a single local document (PDF, TXT, DOCX) to the vector store."""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        loader = UnstructuredWordDocumentLoader(file_path)
    docs = loader.load()
    vectordb.add_documents(filter_complex_metadata(docs))
    print(f"✅ Added {len(docs)} documents to vector DB.")