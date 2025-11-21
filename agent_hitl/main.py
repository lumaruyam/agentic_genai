# main.py
import os
import uuid
from typing import TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, ArxivLoader
)
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver


# ---------------------------------------------------------
# ENVIRONMENT
# ---------------------------------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
print("✅ API key loaded:", bool(api_key))

os.makedirs("./notebooks/uploads", exist_ok=True)
os.makedirs("./notebooks/vectorstore", exist_ok=True)


# ---------------------------------------------------------
# MODELS & VECTOR STORE
# ---------------------------------------------------------
orchestrator = ChatOpenAI(model="gpt-4o-mini", temperature=0)
plan_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
draft_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
critic_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
final_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Chroma(
    persist_directory="./notebooks/vectorstore",
    embedding_function=embeddings
)


# ---------------------------------------------------------
# STATE
# ---------------------------------------------------------
class AgentState(TypedDict, total=False):
    query: str
    intent: str
    context: str
    plan: str
    draft: str
    critique: str
    final: str
    feedback: str
    review_decision: str


# ---------------------------------------------------------
# GUARDRAILS
# ---------------------------------------------------------
FORBIDDEN_TOPICS = ["politics", "religion", "violence", "illegal", "personal"]


def is_out_of_scope(query: str) -> bool:
    return any(t in query.lower() for t in FORBIDDEN_TOPICS)


# ---------------------------------------------------------
# SYSTEM PROMPTS
# ---------------------------------------------------------
SYSTEM_ORCHESTRATOR = """
You are the Orchestrator Agent.
Classify the user's query as:
- 'general'
- 'research'
- 'blocked'
Return only one word.
"""

SYSTEM_ANALYZER = """
You are the Analyzer Agent.
Summarize academic facts with short source indicators.
"""

SYSTEM_PLANNER = """
You are the Plan Writer Agent.
Write a clean academic outline (not a full essay).
"""

SYSTEM_WRITER = """
You are the Draft Writer Agent.
Generate a 400–600 word academic essay.
"""

SYSTEM_CRITIC = """
You are the Critic Agent.
Analyze clarity, structure, and argumentation.
"""

SYSTEM_FINALIZER = """
You are the Final Drafter Agent.
Polish grammar, tone, and coherence.
"""


# ---------------------------------------------------------
# SAFE LLM INVOKER
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# DOCUMENT UPLOAD SUPPORT
# ---------------------------------------------------------
def load_uploaded_documents(folder="./notebooks/uploads"):
    """Called on app startup; loads any pre-existing docs."""
    print("📂 Loading uploaded documents...")

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".pdf", ".txt", ".docx"))
    ]

    docs = []
    for file in files:
        path = os.path.join(folder, file)

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


def add_document_to_vectordb(file_path: str):
    """Called from Streamlit when user uploads a file."""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        loader = UnstructuredWordDocumentLoader(file_path)

    docs = loader.load()
    vectordb.add_documents(filter_complex_metadata(docs))
    print(f"📥 Added {len(docs)} docs to vectorstore.")


# ---------------------------------------------------------
# AGENT NODE FUNCTIONS
# ---------------------------------------------------------
def route_query(state):
    query = state["query"]

    if is_out_of_scope(query):
        state["intent"] = "blocked"
        state["final"] = "⚠️ This topic is out of academic scope."
        return state

    state["intent"] = safe_invoke(orchestrator, query, "orchestrator").strip()
    return state


def general_answer(state):
    state["final"] = safe_invoke(plan_llm, state["query"], "General academic explanation")
    return state


def analyzer_collect(state):
    """Pull academic data from arXiv."""
    query = state["query"]
    try:
        loader = ArxivLoader(query=query, load_max_docs=3)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        vectordb.add_documents(chunks)

        retriever = vectordb.as_retriever(search_kwargs={"k": 5})
        top_docs = retriever.invoke(query)

        state["context"] = "\n\n".join(d.page_content for d in top_docs)
    except Exception as e:
        state["context"] = f"Error loading arXiv: {e}"
    return state


def local_doc_search(state):
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(state["query"])
    state["context"] = "\n\n".join(d.page_content for d in docs)
    return state


def list_documents(state):
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(state["query"])

    if not docs:
        state["final"] = "⚠️ No relevant documents found."
        return state

    summaries = "\n\n".join(
        f"{i+1}. {d.page_content[:300]}..." for i, d in enumerate(docs)
    )
    state["final"] = f"Top documents:\n\n{summaries}"
    return state


def plan_writer(state):
    state["plan"] = safe_invoke(
        plan_llm,
        state["query"],
        "planner",
        context=state.get("context", "")
    )
    return state


def draft_writer(state):
    if "feedback" in state:
        prompt = f"Rewrite using this feedback:\n{state['feedback']}\n\nPlan:\n{state['plan']}"
    else:
        prompt = state["plan"]

    state["draft"] = safe_invoke(draft_llm, prompt, "writer")
    state.pop("feedback", None)
    return state


def hitl_review(state):
    """Human-in-the-loop step for CLI only."""
    print("\n🔍 --- HUMAN REVIEW ---")
    print(state["draft"][:700])
    print("\nApprove (a) / Edit (e) / Reject (r)?")

    choice = input("> ").strip().lower()

    if choice.startswith("a"):
        state["review_decision"] = "approved"
    elif choice.startswith("e"):
        state["review_decision"] = "edit"
        state["feedback"] = input("Enter improvement suggestions: ")
    else:
        state["review_decision"] = "reject"
        state["feedback"] = input("Enter rewrite instructions: ")

    return state


def critic_agent(state):
    state["critique"] = safe_invoke(critic_llm, state["draft"], "critic")
    return state


def final_drafter(state):
    state["final"] = safe_invoke(final_llm, state["draft"], "finalizer")
    vectordb.add_texts([state["final"]])
    return state


# ---------------------------------------------------------
# GRAPH WIRES
# ---------------------------------------------------------
graph = StateGraph(AgentState)

graph.add_node("route_query", route_query)
graph.add_node("general_answer", general_answer)
graph.add_node("analyzer_collect", analyzer_collect)
graph.add_node("local_doc_search", local_doc_search)
graph.add_node("list_documents", list_documents)
graph.add_node("plan_writer", plan_writer)
graph.add_node("draft_writer", draft_writer)
graph.add_node("hitl_review", hitl_review)
graph.add_node("critic_agent", critic_agent)
graph.add_node("final_drafter", final_drafter)


def route_decision(state):
    intent = state["intent"]
    q = state["query"].lower()

    if intent == "blocked":
        return END
    if intent == "general":
        return "general_answer"
    if any(k in q for k in ["list", "show", "papers"]):
        return "list_documents"
    if os.listdir("./notebooks/uploads"):
        return "local_doc_search"
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
graph.add_edge("draft_writer", "hitl_review")


def hitl_decision(state):
    if state["review_decision"] in ("reject", "edit"):
        return "draft_writer"
    return "critic_agent"


graph.add_conditional_edges(
    "hitl_review",
    hitl_decision,
    {
        "draft_writer": "draft_writer",
        "critic_agent": "critic_agent"
    }
)

graph.add_edge("critic_agent", "final_drafter")
graph.add_edge("final_drafter", END)

graph.set_entry_point("route_query")


# ---------------------------------------------------------
# CHECKPOINTED APP
# ---------------------------------------------------------
memory = InMemorySaver()
app = graph.compile(checkpointer=memory)


# ---------------------------------------------------------
# PUBLIC API FOR STREAMLIT
# ---------------------------------------------------------

def orchestrate_agent(query: str):
    """Streamlit --> call this."""
    import uuid
    state = {"query": query}
    final_state = app.invoke(state, config={"thread_id": str(uuid.uuid4())})
    return final_state

# # ---------------------------------------------------------
# # SCRIPT EXECUTION
# # ---------------------------------------------------------
# if __name__ == "__main__":
#     load_uploaded_documents()

#     query = input("Enter academic question: ")
#     state = {"query": query}

#     thread = f"thread-{uuid.uuid4()}"
#     print("Thread:", thread)

#     out = app.invoke(state, config={"thread_id": thread})
#     print("\nFINAL OUTPUT:\n")
#     print(out["final"])
