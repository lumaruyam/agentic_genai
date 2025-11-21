import streamlit as st
import os
from main import orchestrate_agent, add_document_to_vectordb

# --- Page Setup ---
st.set_page_config(page_title="🎓 Academic Research Assistant", layout="wide")
st.title("🎓 Academic Research Assistant")
st.write("Upload academic documents, ask questions, and get AI-guided answers with sources!")

# --- Session State Initialization ---
if "history" not in st.session_state:
    st.session_state["history"] = []

if "uploaded_docs" not in st.session_state:
    st.session_state["uploaded_docs"] = []

# --- Sidebar: Upload Documents ---
with st.sidebar:
    st.header("📂 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF/TXT documents",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.uploaded_docs:
                st.session_state.uploaded_docs.append(file.name)
                os.makedirs("./notebooks/uploads", exist_ok=True)
                file_path = f"./notebooks/uploads/{file.name}"
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                add_document_to_vectordb(file_path)
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
    st.session_state["history"].append(("user", user_query))
    st.chat_message("user").write(user_query)

    # --- Run full agent workflow (no HITL) ---
    with st.spinner("Thinking..."):
        final_state = orchestrate_agent(user_query)

    # --- Display Final Answer ---
    response = final_state.get("final", "⚠️ No response generated.")
    st.session_state["history"].append(("assistant", response))
    st.chat_message("assistant").write(response)

    # --- Show sources ---
    context_docs = final_state.get("context_docs", [])
    if context_docs:
        st.subheader("📄 Sources used")
        for i, doc in enumerate(context_docs, 1):
            st.write(f"**Source {i}:** {doc.metadata.get('source', 'N/A')}")
            st.write(doc.page_content[:300] + "...")