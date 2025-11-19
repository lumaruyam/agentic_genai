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

if "final_state" not in st.session_state:
    st.session_state["final_state"] = {}

if "current_action" not in st.session_state:
    st.session_state["current_action"] = None

# --- Sidebar: Upload Documents ---
with st.sidebar:
    st.header("📂 Document Upload")
    uploaded_files = st.file_uploader("Upload PDF/TXT documents", type=["pdf", "txt"], accept_multiple_files=True)
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

    # --- Generate draft ---
    with st.spinner("Thinking..."):
        st.session_state["final_state"] = orchestrate_agent(user_query)
        st.session_state["current_action"] = "HITL"  # mark workflow stage

# --- HITL Draft Preview ---
final_state = st.session_state.get("final_state", {})
if final_state.get("draft") and st.session_state.get("current_action") == "HITL":
    st.subheader("✍️ Draft Preview")
    st.text_area("Draft", final_state["draft"], height=300)

    action = st.radio("Human-in-the-loop action:", ["Approve", "Edit", "Reject"])

    if action == "Edit":
        feedback = st.text_area("Enter your edits/instructions:")
        if st.button("Apply Edits"):
            final_state["feedback"] = feedback
            with st.spinner("Regenerating draft..."):
                st.session_state["final_state"] = orchestrate_agent(user_query)
            st.session_state["current_action"] = "HITL"

    elif action == "Reject":
        reason = st.text_area("Reason for rewrite:")
        if st.button("Rewrite"):
            final_state["feedback"] = reason
            with st.spinner("Regenerating draft..."):
                st.session_state["final_state"] = orchestrate_agent(user_query)
            st.session_state["current_action"] = "HITL"

    elif action == "Approve":
        # Move draft forward to critic/finalizer
        with st.spinner("Finalizing..."):
            st.session_state["final_state"] = orchestrate_agent(user_query)
        st.session_state["current_action"] = "Final"

# --- Display Final AI Response ---
if final_state.get("final") and st.session_state.get("current_action") == "Final":
    response = final_state["final"]
    st.session_state["history"].append(("assistant", response))
    st.chat_message("assistant").write(response)

    # Show sources if any
    context_docs = final_state.get("context_docs", [])
    if context_docs:
        st.subheader("📄 Sources used")
        for i, doc in enumerate(context_docs, 1):
            st.write(f"**Source {i}:** {doc.metadata.get('source', 'N/A')}")
            st.write(doc.page_content[:300] + "...")


# import streamlit as st
# import os
# from main import orchestrate_agent, add_document_to_vectordb

# # --- Streamlit Page Setup ---
# st.set_page_config(page_title="🎓 Academic Research Assistant", layout="wide")
# st.title("🎓 Academic Research Assistant")
# st.write("Upload academic documents, ask questions, and get AI-guided answers with sources!")

# # --- Initialize session state ---
# if "history" not in st.session_state:
#     st.session_state["history"] = []

# if "uploaded_docs" not in st.session_state:
#     st.session_state["uploaded_docs"] = []

# # --- Sidebar: Document Upload ---
# with st.sidebar:
#     st.header("📂 Document Upload")
#     uploaded_files = st.file_uploader("Upload PDF/TXT documents", type=["pdf", "txt"], accept_multiple_files=True)
#     if uploaded_files:
#         for file in uploaded_files:
#             if file.name not in st.session_state.uploaded_docs:
#                 st.session_state.uploaded_docs.append(file.name)
#                 os.makedirs("./notebooks/uploads", exist_ok=True)
#                 file_path = f"./notebooks/uploads/{file.name}"
#                 with open(file_path, "wb") as f:
#                     f.write(file.getbuffer())
#                 add_document_to_vectordb(file_path)
#                 st.success(f"✅ Uploaded {file.name}")

#     st.header("💬 Chat History")
#     if st.session_state["history"]:
#         for role, msg in st.session_state["history"]:
#             st.chat_message(role).write(msg)
#     else:
#         st.write("No conversation yet.")

# # --- Main Chat Interface ---
# st.header("Ask a question about your research")
# user_query = st.chat_input("Type your academic question here...")  # assign first

# if user_query:  # now it's safe to check
#     st.session_state["history"].append(("user", user_query))
#     st.chat_message("user").write(user_query)

#     # Generate AI response (draft + HITL)
#     with st.spinner("Thinking..."):
#         final_state = orchestrate_agent(user_query)

#     # --- HITL Draft Preview ---
#     if "draft" in final_state:
#         st.subheader("✍️ Draft Preview")
#         st.text_area("Draft", final_state["draft"], height=300)
#         action = st.radio("Human-in-the-loop action:", ["Approve", "Edit", "Reject"])
    
#     if action == "Edit":
#         feedback = st.text_area("Enter your edits/instructions:")
#         if st.button("Apply Edits"):
#             final_state["feedback"] = feedback
#             final_state = orchestrate_agent(user_query)
#     elif action == "Reject":
#         reason = st.text_area("Reason for rewrite:")
#         if st.button("Rewrite"):
#             final_state["feedback"] = reason
#             final_state = orchestrate_agent(user_query)
#     elif action == "Approve":
#         final_state = orchestrate_agent(user_query)  # move draft forward

# if "draft" in final_state:
#     st.subheader("✍️ Draft Preview")
#     st.text_area("Draft", final_state["draft"], height=300)

#     action = st.radio("Human-in-the-loop action:", ["Approve", "Edit", "Reject"], index=0)

#     if action == "Edit":
#         feedback = st.text_area("Enter your edits/instructions:")
#         if st.button("Apply Edits"):
#             final_state["feedback"] = feedback
#             final_state = orchestrate_agent(user_query)
#     elif action == "Reject":
#         reason = st.text_area("Reason for rewrite:")
#         if st.button("Rewrite"):
#             final_state["feedback"] = reason
#             final_state = orchestrate_agent(user_query)
#     # Approve just continues to critic/final stage


#     # --- Add final AI message ---
#     response = final_state.get("final", "No response generated.")
#     context_docs = final_state.get("context_docs", [])
#     st.session_state["history"].append(("assistant", response))
#     st.chat_message("assistant").write(response)

#     # Show sources
#     if context_docs:
#         st.subheader("📄 Sources used")
#         for i, doc in enumerate(context_docs, 1):
#             st.write(f"**Source {i}:** {doc.metadata.get('source', 'N/A')}")
#             st.write(doc.page_content[:300] + "...")
