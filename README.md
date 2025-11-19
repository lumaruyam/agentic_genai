# 🎓 Academic Research Assistant

An **agentic AI assistant** designed to help researchers, students, and academics navigate large volumes of scholarly content. Upload your documents, ask questions, and receive AI-guided answers with sources—all in one intuitive interface powered by Streamlit.

---

## 🧩 Problem Statement

Researchers often face two major challenges:

1. **Information Overload**: Academic content is scattered across PDFs, Word documents, and online sources like arXiv. Finding relevant information quickly is time-consuming.
2. **Structured Academic Writing**: Drafting essays, papers, or literature reviews requires synthesizing information from multiple sources while maintaining clarity, coherence, and academic rigor.

Current tools either provide raw search results or generate text without human-in-the-loop verification, which risks inaccuracies or loss of context.

---

## Why This Matters

- **Efficiency**: Saves hours of manual literature review and note-taking.
- **Accuracy**: Incorporates human-in-the-loop (HITL) feedback to ensure drafts meet academic standards.
- **Traceability**: Sources are surfaced for every generated answer, maintaining research integrity.
- **Scalability**: Supports large document libraries and real-time AI-guided research queries.

By solving this, researchers can focus on analysis, insights, and creativity, rather than repetitive search and drafting.

---

## ⚙️ Features

- Upload academic documents (PDF, TXT, DOCX) directly via Streamlit.
- Ask AI questions about your uploaded documents or general research topics.
- Receive structured drafts, critiques, and polished final outputs.
- Human-in-the-loop feedback for editing, approval, or rewrite.
- Sources are automatically linked for transparency.
- Automatic retrieval of related documents from local uploads or online repositories (like arXiv).

---

## 🚀 How to Run

1. **Install dependencies**:

```bash
pip install -r requirements.txt
````

2. **Set your OpenAI API key**:

```bash
export OPENAI_API_KEY="YOUR_KEY_HERE"
```

or use a `.env` file in the project root:

```
OPENAI_API_KEY=YOUR_KEY_HERE
```

3. **Run the Streamlit app**:

```bash
streamlit run app.py
```

4. **Upload documents and start asking questions** via the interface.

---

## 🛠 How It Works

1. **Orchestration**: The agent classifies queries (`general`, `research`, `blocked`) and decides the workflow.
2. **Document Search**: Retrieves relevant information from uploaded documents or arXiv.
3. **Planning & Drafting**: Generates a structured plan and initial draft.
4. **Human-in-the-Loop (HITL)**: Users review drafts to approve, edit, or request rewrites.
5. **Critique & Finalization**: AI refines drafts, ensuring clarity, coherence, and correctness.
6. **Source Attribution**: Relevant documents are displayed for each answer.

---

## 🖇 File Structure

```
.
├── app.py                  # Streamlit interface
├── main.py                 # Core agent orchestration & vectorstore handling
├── notebooks/
│   ├── uploads/            # Uploaded documents
│   └── vectorstore/        # Persistent vector database
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Guardrails

* Topics like politics, religion, violence, illegal activities, or personal queries are blocked.
* AI responses are academic-focused; general conversational queries may be limited.

---

## 💡 Future Improvements

* Advanced citation formatting (APA, MLA)
* Integration with more scholarly databases (PubMed, IEEE, etc.)

---

## 📝 References

* [LangChain](https://www.langchain.com/)
* [Chroma Vector Database](https://www.trychroma.com/)
* [Streamlit Documentation](https://docs.streamlit.io/)

