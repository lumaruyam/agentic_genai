import streamlit as st
from backend_agent import orchestrate_agent, add_document_to_vectordb

# UX/UI design
# --- Streamlit Page Setup ---
st.set_page_config(page_title="🎓 Academic Research Assistant", layout="wide")

st.title("🎓 Academic Research Assistant")
st.write("Upload academic documents, ask questions, and get AI-guided answers with sources!")

# --- Initialize session state ---
if "history" not in st.session_state:
    st.session_state["history"] = []

if "uploaded_docs" not in st.session_state:
    st.session_state["uploaded_docs"] = []

# --- Sidebar: Document Upload + History ---
with st.sidebar:
    st.header("📂 Document Upload")
    uploaded_file = st.file_uploader("Upload PDF/TXT documents", type=["pdf", "txt"], accept_multiple_files=True)
    if uploaded_file:
        for file in uploaded_file:
            if file.name not in st.session_state.uploaded_docs:
                st.session_state.uploaded_docs.append(file.name)
                # Save file locally
                with open(f"data/{file.name}", "wb") as f:
                    f.write(file.getbuffer())
                # Add to vector DB (Hot RAG)
                add_document_to_vectordb(f"data/{file.name}")
                st.success(f"✅ Uploaded {file.name}")

    st.header("💬 Chat History")
    if st.session_state["history"]:
        for role, msg in st.session_state["history"]:
            st.chat_message(role).write(msg)
    else:
        st.write("No conversation yet.")

# --- Main Chat Interface ---
st.header("Ask a question about your research")
user_query = st.chat_input("Type your academic question here...")

if user_query:
    # Add user message to history
    st.session_state["history"].append(("user", user_query))
    st.chat_message("user").write(user_query)

    # Generate AI response
    with st.spinner("Thinking..."):
        response, context_docs = orchestrate_agent(user_query)  # backend agent handles RAG & LLM

    # Add AI message to history
    st.session_state["history"].append(("assistant", response))
    st.chat_message("assistant").write(response)

    # Show sources
    if context_docs:
        st.subheader("📄 Sources used")
        for i, doc in enumerate(context_docs, 1):
            st.write(f"**Source {i}:** {doc.metadata.get('source', 'N/A')}")
            st.write(doc.page_content[:300] + "...")