import os
import json
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel

try:
    from backend.rag_engine import RAGEngine
except ImportError:
    from rag_engine import RAGEngine

DOCS_DIR = os.path.join(os.path.dirname(__file__), "documents")
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

rag_engine: RAGEngine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_engine
    print("🚀 [CampusCopilot] Initializing Agentic RAG engine...")
    os.makedirs(DOCS_DIR, exist_ok=True)
    rag_engine = RAGEngine(docs_dir=DOCS_DIR)
    yield
    print("🛑 [CampusCopilot] Shutting down RAG service.")

app = FastAPI(
    title="CampusCopilot Agentic RAG API",
    description="Full-Stack Agentic AI Student Support Assistant with SSE Streaming & Split-Screen Grounding",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return "<h1>CampusCopilot API Online. Frontend loading...</h1>"

@app.get("/api/health")
async def health():
    return {
        "status": "online",
        "service": "CampusCopilot Agentic RAG",
        "indexed_chunks": len(rag_engine.chunks) if rag_engine else 0
    }

@app.get("/api/documents")
async def get_documents():
    """List all indexed documents in the campus knowledge base."""
    files = []
    if os.path.exists(DOCS_DIR):
        for fname in os.listdir(DOCS_DIR):
            fpath = os.path.join(DOCS_DIR, fname)
            if os.path.isfile(fpath):
                files.append({
                    "name": fname,
                    "size_bytes": os.path.getsize(fpath),
                    "chunks": sum(1 for c in rag_engine.chunks if c.doc_name == fname)
                })
    return {"documents": files, "total_chunks": len(rag_engine.chunks)}

@app.get("/api/document/{doc_name}")
async def get_document_content(doc_name: str):
    """Fetch raw document content for the split-screen document viewer."""
    # Sanitize doc_name to prevent directory traversal
    clean_name = os.path.basename(doc_name)
    fpath = os.path.join(DOCS_DIR, clean_name)
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail="Document not found.")

    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    return {"doc_name": clean_name, "content": content}

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a new campus policy or course document into the knowledge base."""
    clean_name = os.path.basename(file.filename)
    dest_path = os.path.join(DOCS_DIR, clean_name)

    content = await file.read()
    with open(dest_path, "wb") as f:
        f.write(content)

    # Trigger re-indexing
    rag_engine.reload_documents()

    return {
        "status": "success",
        "message": f"Document '{clean_name}' uploaded and indexed.",
        "new_total_chunks": len(rag_engine.chunks)
    }

@app.post("/api/chat/stream")
async def stream_chat(req: QueryRequest):
    """Server-Sent Events (SSE) streaming endpoint."""
    async def event_generator():
        try:
            async for event in rag_engine.generate_streaming_response(req.query):
                # SSE format: data: <json>\n\n
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
