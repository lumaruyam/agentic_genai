# 🎓 Academic Research Assistant

An **agentic AI assistant** designed to help researchers, students, and academics analyze, retrieve, and synthesize information from large collections of academic documents. Built with **Streamlit**, **LangGraph**, and **OpenAI models**, it supports human-in-the-loop drafting, academic querying, and document-grounded answers.

---

## 👥 Contributors

* Wei-Ling Hung
* Luli Maruyama
* Yushu Gong

---

## 🧩 Problem Statement

Researchers face two persistent challenges:

1. **Information Overload** – Academic materials are scattered across PDFs, Word documents, text files, and online archives (e.g., arXiv), making efficient extraction slow and tedious.
2. **Structured Academic Writing** – Producing well‑structured essays or literature reviews requires careful synthesis across multiple sources.

Existing tools either:

* Provide raw search results with limited synthesis, **or**
* Generate unverified text without human review, risking inaccuracies.

This tool solves both.

---

## ⭐ Why This Matters

* **Efficiency** – Speed up literature review and note extraction.
* **Accuracy** – Incorporates human feedback loops to guide revision.
* **Traceability** – Every answer links back to specific retrieved sources.
* **Scalability** – Works with large local libraries and online retrieval.

---

## ⚙️ Features

* 📄 Upload academic documents (PDF, TXT, DOCX)
* 🔍 AI‑powered search over local and online (arXiv) documents
* 🧠 Structured drafting workflow: **planning → drafting → critique → finalization**
* ✍️ Human‑in‑the‑loop (HITL) revision system
* 🔗 Automatic source linking for transparency
* 🤖 Query orchestration and intent classification

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
export OPENAI_API_KEY="YOUR_KEY_HERE"
```

or create a `.env` file:

```
OPENAI_API_KEY=YOUR_KEY_HERE
```

### 3. Run the Streamlit app

```bash
streamlit run academic_assistant.py
```

### 4. Upload documents and begin querying

Use the sidebar interface to upload PDFs/TXT/DOCX files.

---

## 🧠 How It Works (Architecture)

### High‑Level Flow

1. **Query Routing** – Classifies user intent (`general`, `research`, or `blocked`).
2. **Document Retrieval** – Searches local vectorstore or arXiv.
3. **Planning** – Produces a clean academic outline.
4. **Drafting** – Generates a full academic draft.
5. **Human‑in‑the‑Loop Review** – User approves/edits/rejects.
6. **Critique** – AI evaluates clarity and structure.
7. **Finalization** – Polishes the text.
8. **Storage** – Final result stored back into vectorstore.

---

## 🔀 Mermaid Diagram: Node Flow

```mermaid
graph TD
    A[route_query] -->|blocked| Z[END]
    A -->|general| B[general_answer]
    A -->|list documents| C[list_documents]
    A -->|local docs available| D[local_doc_search]
    A -->|no local docs| E[analyzer_collect]

    D --> C
    E --> C
    C --> F[plan_writer]
    F --> G[draft_writer]
    G --> H[hitl_review]

    H -->|edit/reject| G
    H -->|approve| I[critic_agent]

    I --> J[final_drafter]
    J --> Z
```

---

## 📁 File Structure

```
.
├── app.py                     # Streamlit UI
├── main.py                    # Core agent pipeline and graph logic
├── academic_assistant.py      # Web interface logic
├── notebooks/
│   ├── uploads/               # User-uploaded documents
│   └── vectorstore/           # Persistent embeddings DB
├── requirements.txt
└── README.md
```

---

## 🛡 Guardrails

* Blocks topics: **politics, religion, violence, illegal activities, personal info**
* Ensures responses remain academic and source‑driven

---

## 🔮 Future Improvements

* Better citation formatting (APA, MLA)
* Support for PubMed, IEEE, Semantic Scholar
* Multi-document comparison & synthesis mode

---

## 📚 References

* [LangChain](https://www.langchain.com/)
* [Chroma Vector Database](https://www.trychroma.com/)
* [Streamlit Documentation](https://docs.streamlit.io/)
