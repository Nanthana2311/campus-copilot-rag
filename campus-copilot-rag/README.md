# 🎓 CampusCopilot AI

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Protocol: SSE](https://img.shields.io/badge/Protocol-Server--Sent_Events-orange.svg)](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)
[![RAG: Agentic](https://img.shields.io/badge/RAG-Agentic_Retrieval-purple.svg)](https://github.com/Nanthana2311/AI-Student-Support-Assistant)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Full-Stack Agentic RAG Platform**: An enterprise-grade AI Student Support Copilot delivering **token-by-token streaming (SSE)**, **split-screen document citation grounding**, and **anti-hallucination guardrails** for university academic regulations.

---

## ⚡ The Problem & The Solution

**The Problem:** Students constantly struggle to find authoritative answers to university policies (attendance requirements, medical condonation fees, revaluation procedures, scholarship eligibility, and placement criteria) buried inside 100-page PDF handbooks. Generic LLMs hallucinate rules, while simple Streamlit chatbots lack real-time streaming, document transparency, and responsive UI.

**The Solution:** CampusCopilot bridges the gap:
1. **Agentic Retrieval:** Evaluates user intent and executes semantic vector similarity search against official university bylaws.
2. **Strict Anti-Hallucination Guardrail:** If an inquiry cannot be substantiated by verified university documents, the agent honestly declines to answer and cites the official academic authority.
3. **Sub-Millisecond Token Streaming (SSE):** Answers stream character-by-character over HTTP using **Server-Sent Events** (`text/event-stream`).
4. **Split-Screen Citation Grounding:** Every claim carries an interactive source badge showing the document name and percentage match. Clicking the badge opens the side-by-side reader highlighting the exact passage.

---

## 🏗️ System Architecture

```
[ Student Web Interface ]
       │
       ▼ 1. Query: "What happens if attendance is 68%?"
[ FastAPI Endpoint (/api/chat/stream) ]
       │
       ▼ 2. Semantic Search (Cosine Similarity over Vector Index)
[ Top-K Document Chunks Retrieved ]
       │
       ├─► 3. Anti-Hallucination Evaluator (Similarity >= Threshold?)
       │
       ▼ 4. Token-by-Token Streaming (Server-Sent Events)
[ UI Stream Reader (EventSource / TextDecoder) ]
       │
       └─► 5. Interactive Citation Badge ──► Opens Split-Screen Document Reader!
```

---

## ✨ Key Features

- ⚡ **Real-Time Token Streaming:** Zero perceived latency using Server-Sent Events (`text/event-stream`).
- 📖 **Split-Screen Document Grounding:** View the official document side-by-side with the answer.
- 🛡️ **Anti-Hallucination Guardrail:** Strict fallback prevents inventing university policies.
- 📚 **Dynamic Document Ingestion:** Drag-and-drop new PDF or TXT policies to auto-chunk and re-index.
- 💼 **Pre-loaded Knowledge Base:**
  - Attendance Regulations & Detention Thresholds (75% / 65% rules).
  - Examination Bylaws, Letter Grading (10-point scale), & Revaluation Fees.
  - Academic Merit & First Generation Graduate Scholarships.
  - Campus Placement Minimum CGPA & Dream Company Offer Policy.

---

## 🚀 Quickstart Guide

### 1. Clone the Repository

```bash
git clone https://github.com/Nanthana2311/campus-copilot-rag.git
cd campus-copilot-rag
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser and navigate to:
```
http://localhost:8000
```

---

## 👩‍💻 Author

**Nanthana Sengather**
- GitHub: [@Nanthana2311](https://github.com/Nanthana2311)
- LinkedIn: [Nanthana Sengather](https://www.linkedin.com/in/nanthana-sengather-122a1b2a5)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
