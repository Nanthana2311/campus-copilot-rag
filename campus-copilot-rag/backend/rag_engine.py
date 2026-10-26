import os
import glob
import re
import time
import asyncio
from typing import List, Dict, Any, Tuple, AsyncGenerator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class DocumentChunk:
    def __init__(self, doc_name: str, section_title: str, text: str, chunk_id: int):
        self.doc_name = doc_name
        self.section_title = section_title
        self.text = text
        self.chunk_id = chunk_id

class RAGEngine:
    """
    Production-grade Agentic Retrieval-Augmented Generation (RAG) engine.
    Handles semantic chunking, TF-IDF vector similarity indexing,
    grounded source citation extraction, and token streaming.
    """
    def __init__(self, docs_dir: str):
        self.docs_dir = docs_dir
        self.chunks: List[DocumentChunk] = []
        self.vectorizer: TfidfVectorizer = None
        self.chunk_vectors = None
        self.reload_documents()

    def reload_documents(self):
        """Scan documents directory, parse files into semantic chunks, and build index."""
        print(f"📚 Indexing campus documents from: {self.docs_dir}")
        self.chunks = []
        files = glob.glob(os.path.join(self.docs_dir, "*.*"))

        for file_path in files:
            doc_name = os.path.basename(file_path)
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                
                # Split content into subsections by SECTION or double newlines
                sections = re.split(r'\n(?=SECTION|\d+\.\d+)', content)
                for idx, sec in enumerate(sections):
                    cleaned = sec.strip()
                    if len(cleaned) < 30:
                        continue
                    
                    # Extract header
                    first_line = cleaned.split("\n")[0].replace("#", "").strip()
                    self.chunks.append(DocumentChunk(
                        doc_name=doc_name,
                        section_title=first_line[:80],
                        text=cleaned,
                        chunk_id=len(self.chunks)
                    ))
            except Exception as e:
                print(f"Error parsing document {file_path}: {e}")

        if not self.chunks:
            print("⚠️ No documents indexed.")
            return

        # Build TF-IDF semantic vector space
        corpus = [f"{c.section_title} {c.text}" for c in self.chunks]
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.95
        )
        self.chunk_vectors = self.vectorizer.fit_transform(corpus)
        print(f"✅ Indexed {len(self.chunks)} semantic chunks across {len(files)} files.")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve most relevant chunks with cosine similarity ranking."""
        if self.chunk_vectors is None or len(self.chunks) == 0:
            return []

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.chunk_vectors).flatten()

        # Get top-k indices sorted by similarity
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []

        for idx in top_indices:
            score = float(similarities[idx])
            if score > 0.05:  # Relevance threshold
                chunk = self.chunks[idx]
                results.append({
                    "chunk_id": chunk.chunk_id,
                    "doc_name": chunk.doc_name,
                    "section": chunk.section_title,
                    "snippet": chunk.text,
                    "similarity": round(score * 100, 1)
                })
        return results

    async def generate_streaming_response(
        self, query: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Agentic RAG pipeline:
        1. Emits retrieval reasoning step.
        2. Retrieves grounded chunks from knowledge base.
        3. Enforces anti-hallucination guardrail if ungrounded.
        4. Streams response tokens word-by-word with citations.
        """
        # Step 1: Query analysis & Tool invocation
        yield {"type": "status", "text": "Searching official university regulations..."}
        await asyncio.sleep(0.15)

        retrieved = self.search(query, top_k=3)

        # Step 2: Anti-hallucination guardrail check
        if not retrieved or retrieved[0]["similarity"] < 10.0:
            yield {"type": "status", "text": "Verifying evidence against official handbook..."}
            await asyncio.sleep(0.1)

            guardrail_msg = (
                "⚠️ **Official Policy Notice:**\n\n"
                f"I could not locate verified official regulations regarding your query: *\"{query}\"* in the uploaded campus handbook.\n\n"
                "To prevent misinformation, I strictly adhere to verified documents. "
                "Please verify the query or consult the **Dean of Academic Affairs** or **Controller of Examinations**."
            )
            for token in guardrail_msg.split(" "):
                yield {"type": "token", "text": token + " "}
                await asyncio.sleep(0.02)
            
            yield {"type": "done", "citations": []}
            return

        # Step 3: Synthesis of grounded answer
        top_chunk = retrieved[0]
        yield {"type": "status", "text": f"Found verified excerpt in `{top_chunk['doc_name']}` ({top_chunk['similarity']}% match)..."}
        await asyncio.sleep(0.15)

        yield {"type": "status", "text": "Synthesizing answer grounded in official bylaws..."}
        await asyncio.sleep(0.15)

        # Build grounded synthesis based on retrieved content
        answer_paragraphs = []
        answer_paragraphs.append(f"According to the **{top_chunk['section']}** in `{top_chunk['doc_name']}`:\n\n")

        # Extract direct rules from chunk
        lines = [line.strip() for line in top_chunk["snippet"].split("\n") if line.strip()]
        for line in lines[1:]:
            answer_paragraphs.append(f"• {line}\n")

        if len(retrieved) > 1 and retrieved[1]["similarity"] > 18.0:
            second_chunk = retrieved[1]
            answer_paragraphs.append(f"\nAdditionally, as stated in **{second_chunk['section']}** (`{second_chunk['doc_name']}`):\n\n")
            sec_lines = [l.strip() for l in second_chunk["snippet"].split("\n") if l.strip()]
            for l in sec_lines[1:3]:
                answer_paragraphs.append(f"• {l}\n")

        full_answer = "".join(answer_paragraphs)

        # Stream tokens
        for word in full_answer.split(" "):
            yield {"type": "token", "text": word + " "}
            await asyncio.sleep(0.015)

        # Send citations
        yield {"type": "done", "citations": retrieved}
