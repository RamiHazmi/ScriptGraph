"""
FastAPI backend for GraphRAG system.
Endpoints: Upload PDF, Query, Neo4j access, Library management
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import time
import json
from pathlib import Path
from datetime import datetime
import io
import re
from pypdf import PdfReader
from PIL import Image
import fitz  # PyMuPDF for PDF rendering

# Import our modules
from llm import LLMClient
from storage import GraphStore, VectorStore
from extraction import extract_pdf_text, chunk_text
from pipeline import build_knowledge_graph, ask_graphrag
import config

# #region agent log
def _agent_debug_log(location: str, message: str, data: dict, hypothesis_id: str = ""):
    try:
        payload = {
            "sessionId": "14d924",
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
            "hypothesisId": hypothesis_id,
        }
        with open("debug-14d924.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass
# #endregion


# Initialize FastAPI
app = FastAPI(title="GraphRAG API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
llm_client = None
graph_store = None
vector_store = None
current_document = None  # Changed from current_pdf
indexing_in_progress = False
indexing_status = {"status": "idle", "progress": 0, "message": ""}

# Library file path
LIBRARY_FILE = Path("library.json")


# Pydantic models
class QueryRequest(BaseModel):
    question: str


class LoadDocumentRequest(BaseModel):
    document_id: str


# ============================================================
# LIBRARY MANAGEMENT
# ============================================================

def load_library():
    """Load library from JSON file."""
    if LIBRARY_FILE.exists():
        try:
            with open(LIBRARY_FILE, "r", encoding="utf-8") as f:
                library = json.load(f)
                return library
        except json.JSONDecodeError as e:
            print(f"Warning: library.json is corrupted, creating backup and starting fresh")
            # Backup corrupted file
            backup_file = LIBRARY_FILE.with_suffix('.backup')
            LIBRARY_FILE.rename(backup_file)
            return []
        except Exception as e:
            print(f"Error loading library: {e}")
            return []
    return []


def save_library(library):
    """Save library to JSON file."""
    try:
        # Write to temporary file first
        temp_file = LIBRARY_FILE.with_suffix('.tmp')
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(library, f, indent=2, ensure_ascii=False)
        
        # Replace original file with temp file (atomic operation)
        temp_file.replace(LIBRARY_FILE)
    except Exception as e:
        print(f"Error saving library: {e}")
        # Try to remove temp file if it exists
        if temp_file.exists():
            temp_file.unlink()
        raise


def add_to_library(filename, title, chunks_count, entities_count=0):
    """Add document to library."""
    library = load_library()
    
    doc_id = filename.replace(".pdf", "").replace(" ", "_")
    
    # Ensure title is a string (handle PyPDF IndirectObject)
    if not isinstance(title, str):
        title = str(title) if title else filename
    
    # Clean up title
    title = title.strip()
    if not title or title.startswith("IndirectObject"):
        title = filename.replace(".pdf", "").replace("_", " ").title()
    
    # Check if already exists
    existing = next((doc for doc in library if doc["id"] == doc_id), None)
    
    if existing:
        # Update existing
        existing["last_accessed"] = datetime.now().isoformat()
        existing["chunks"] = chunks_count
        existing["title"] = title  # Update title too
        # Keep existing chat_history
        if "chat_history" not in existing:
            existing["chat_history"] = []
    else:
        # Add new
        library.append({
            "id": doc_id,
            "filename": filename,
            "title": title,
            "chunks": chunks_count,
            "entities": entities_count,
            "uploaded_at": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat(),
            "chat_history": []  # Initialize empty chat history
        })
    
    save_library(library)
    return doc_id


def save_chat_message(doc_id, role, message, graph_data=None):
    """Save a chat message to document's history."""
    library = load_library()
    doc = next((d for d in library if d["id"] == doc_id), None)
    
    if doc:
        if "chat_history" not in doc:
            doc["chat_history"] = []
        
        doc["chat_history"].append({
            "role": role,
            "message": message,
            "graph_data": graph_data,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 50 messages to avoid huge files
        if len(doc["chat_history"]) > 50:
            doc["chat_history"] = doc["chat_history"][-50:]
        
        save_library(library)


def get_chat_history(doc_id):
    """Get chat history for a document."""
    library = load_library()
    doc = next((d for d in library if d["id"] == doc_id), None)
    
    if doc and "chat_history" in doc:
        return doc["chat_history"]
    return []


# Pydantic models
class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: dict
    metrics: dict


class StatusResponse(BaseModel):
    indexed: bool
    pdf_name: str = None
    entities_count: int = 0
    chunks_count: int = 0


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML interface."""
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/status")
async def get_status():
    """Get current system status."""
    global current_document, vector_store, indexing_in_progress, indexing_status
    
    if indexing_in_progress:
        return {
            "indexed": False,
            "indexing": True,
            "status": indexing_status.get("status"),
            "progress": indexing_status.get("progress"),
            "message": indexing_status.get("message")
        }
    
    return {
        "indexed": current_document is not None,
        "indexing": False,
        "document": current_document,
        "chunks_count": vector_store.collection.count() if vector_store and vector_store.collection_exists() else 0,
        "neo4j_uri": config.NEO4J_URI
    }


@app.get("/api/library")
async def get_library():
    """Get list of all documents in library."""
    try:
        print("Loading library...")
        library = load_library()
        print(f"Loaded {len(library)} documents")
        
        # Ensure chat_history field exists for each doc (for backwards compatibility)
        for doc in library:
            if "chat_history" not in doc:
                doc["chat_history"] = []
        
        print("Returning library response")
        return {"documents": library}
    except Exception as e:
        print(f"Error loading library: {e}")
        import traceback
        traceback.print_exc()
        return {"documents": []}


@app.post("/api/chat-history/{doc_id}")
async def save_chat_message_endpoint(doc_id: str, request: dict):
    """Save a chat message to document's history."""
    try:
        save_chat_message(
            doc_id,
            request.get("role"),
            request.get("message"),
            request.get("graph_data")
        )
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chat-history/{doc_id}")
async def get_chat_history_endpoint(doc_id: str):
    """Get chat history for a document."""
    try:
        history = get_chat_history(doc_id)
        return {"history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


CHUNKS_CACHE_DIR = Path("chunks_cache")
CHUNKS_CACHE_DIR.mkdir(exist_ok=True)


def save_chunks_cache(doc_id: str, chunks: list):
    """Save chunks to JSON cache on disk."""
    try:
        cache_file = CHUNKS_CACHE_DIR / f"{doc_id}.json"
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ Failed to save chunks cache for {doc_id}: {e}")


def load_chunks_cache(doc_id: str):
    """Load chunks from JSON cache on disk if available."""
    try:
        cache_file = CHUNKS_CACHE_DIR / f"{doc_id}.json"
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"⚠️ Failed to load chunks cache for {doc_id}: {e}")
    return None



def enrich_chunk_with_header(chunk: dict) -> dict:
    """Parse INT./EXT. scene header from chunk text to fill location/int_ext/time when missing."""
    if chunk.get("location") and chunk["location"] != "Unknown Location":
        return chunk
    text = chunk.get("text", "")
    # Normalize smart/curly quotes and dashes to ASCII equivalents
    text_norm = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    text_norm = text_norm.replace("\u2013", "-").replace("\u2014", "-")

    # Scan first few lines for a scene heading (INT./EXT. pattern)
    # Pattern: (INT.|EXT.) LOCATION [- SUBLOCATION]* - TIME
    # Strategy: match greedily to end of line, then split on LAST " - TIME" token
    header_pat = re.compile(
        r"^(INT\.?/?EXT\.?|EXT\.?/?INT\.?|INT\.?|EXT\.?)\s*[.\s]*(.+?)\s*$",
        re.IGNORECASE
    )
    time_tokens = {"DAY", "NIGHT", "MORNING", "EVENING", "AFTERNOON", "LATER",
                   "CONTINUOUS", "DUSK", "DAWN", "MOMENTS LATER", "EARLIER"}

    for line in text_norm.strip().split("\n")[:5]:
        line = line.strip()
        m = header_pat.match(line)
        if m:
            int_ext_raw = m.group(1).upper().replace("/", "/")
            rest = m.group(2).strip()

            # Split on the last " - " to separate location from time
            parts = re.split(r"\s+-\s+", rest)
            if len(parts) >= 2 and parts[-1].upper() in time_tokens:
                time_str = parts[-1].strip().upper()
                location = " - ".join(parts[:-1]).strip().title()
            elif len(parts) >= 2:
                # Last part may still be a time-like token
                time_str = parts[-1].strip().upper()
                location = " - ".join(parts[:-1]).strip().title()
            else:
                location = rest.strip().title()
                time_str = "DAY"

            chunk = dict(chunk)
            chunk["int_ext"] = int_ext_raw.rstrip(".")
            chunk["location"] = location
            chunk["time"] = time_str
            chunk["scene_header"] = f"{chunk['int_ext']}. {location} - {time_str}"
            break

    return chunk

def get_chunks_from_chroma(v_store):
    """Recover chunks from ChromaDB collection."""
    try:
        data = v_store.collection.get()
        if data and data.get("documents"):
            chunks = []
            for i, doc_text in enumerate(data["documents"]):
                meta = data["metadatas"][i] if data.get("metadatas") else {}
                chunks.append({
                    "id": meta.get("chunk_id", i),
                    "text": doc_text,
                    "page": meta.get("page", 1),
                    "scene_number": meta.get("scene_number", 0)
                })
            chunks.sort(key=lambda c: c["id"])
            return chunks
    except Exception as e:
        print(f"⚠️ Could not load chunks from ChromaDB: {e}")
    return None



def rank_events_with_llm(document_id: str, doc_title: str) -> bool:
    """
    Ask the LLM to rank events by narrative importance (once per document).
    Stores importance_rank on Event nodes in Neo4j permanently.
    Returns True if ranking was performed, False if already done or failed.
    """
    global graph_store, llm_client

    if not graph_store or not llm_client:
        return False

    with graph_store.driver.session() as session:
        # Check if ranking already done for this document
        already_done = session.run("""
            MATCH (e:Event)
            WHERE e.document_id = $doc_id AND e.importance_rank IS NOT NULL
            RETURN count(e) as cnt LIMIT 1
        """, doc_id=document_id).single()

        if already_done and already_done["cnt"] > 0:
            print(f"[rank] Event ranking already stored for '{doc_title}' — skipping LLM call")
            return False

        # Fetch all events with name + description
        events_raw = session.run("""
            MATCH (e:Event)
            WHERE e.document_id = $doc_id
            RETURN id(e) as neo_id, e.name as name, e.description as description,
                   e.event_id as event_id
            ORDER BY e.chunk_id
        """, doc_id=document_id).data()

    if not events_raw:
        print(f"[rank] No events found for '{doc_title}' — skipping ranking")
        return False

    print(f"[rank] Ranking {len(events_raw)} events for '{doc_title}' via LLM...")

    # Build compact event list for LLM (name + short description)
    event_lines = []
    for i, ev in enumerate(events_raw):
        name = ev.get("name") or ev.get("event_id") or f"event_{i}"
        desc = (ev.get("description") or "")[:120]
        event_lines.append(f"{i}: {name}" + (f" — {desc}" if desc else ""))

    # Chunk into batches if > 100 events (LLM context)
    # For now send all names (just names, no descriptions) if too many
    if len(event_lines) > 80:
        event_lines = [f"{i}: {ev.get('name','?')}" for i, ev in enumerate(events_raw)]

    events_block = "\n".join(event_lines)

    prompt = f"""You are a professional script analyst. Below is a numbered list of ALL events extracted from the screenplay "{doc_title}".

Your task: identify the 15 MOST DRAMATICALLY SIGNIFICANT events — the ones that change the course of the story, create the highest tension, or represent major turning points. This works for ANY genre (thriller, romance, comedy, drama, sci-fi, etc.).

Consider:
- Character deaths, near-deaths, violence
- Major revelations that fundamentally change relationships or the plot
- Confrontations and irreversible decisions
- Emotional climaxes (breakdowns, confessions, declarations)
- Moments that split the story into "before" and "after"

Events list:
{events_block}

Return ONLY a JSON array of the top 15 event indices (numbers from the list above), ordered from MOST to LEAST important. Example:
[42, 179, 30, 126, 44, 72, 45, 113, 1, 8, 50, 160, 249, 95, 140]

Return ONLY the JSON array. No explanation."""

    try:
        response = llm_client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200
        )
        result = response.choices[0].message.content.strip()
        result = result.replace("```json", "").replace("```", "").strip()

        # Parse the returned indices
        import json as _json
        top_indices = _json.loads(result)
        if not isinstance(top_indices, list):
            raise ValueError("LLM did not return a list")

        top_indices = [int(i) for i in top_indices if isinstance(i, (int, float, str)) and str(i).isdigit()][:15]

        print(f"[rank] LLM selected indices: {top_indices}")

        # Store importance_rank on the selected Event nodes
        with graph_store.driver.session() as session:
            # Clear any stale ranks first
            session.run("""
                MATCH (e:Event) WHERE e.document_id = $doc_id
                REMOVE e.importance_rank
            """, doc_id=document_id)

            # Set ranks
            for rank, idx in enumerate(top_indices, start=1):
                if idx < len(events_raw):
                    neo_id = events_raw[idx]["neo_id"]
                    session.run("""
                        MATCH (e:Event) WHERE id(e) = $neo_id
                        SET e.importance_rank = $rank
                    """, neo_id=neo_id, rank=rank)

            print(f"[rank] ✅ Stored importance_rank 1-{len(top_indices)} on {len(top_indices)} Event nodes in Neo4j")

        # Print the selection for the user
        print("\n🎬 LLM-RANKED TOP EVENTS:")
        for rank, idx in enumerate(top_indices, 1):
            if idx < len(events_raw):
                print(f"  #{rank:2d} {events_raw[idx].get('name','?')}")
        print()
        return True

    except Exception as e:
        print(f"[rank] ⚠️ LLM ranking failed: {e} — falling back to formula scoring")
        return False


@app.post("/api/library/load")
async def load_from_library(request: LoadDocumentRequest):
    """Load a document from library (instantly reuses graph and vectors if they exist)."""
    global llm_client, graph_store, vector_store, current_document
    
    try:
        library = load_library()
        doc = next((d for d in library if d["id"] == request.document_id), None)
        
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found in library")
        
        # Check if PDF file exists
        file_path = Path("uploads") / doc["filename"]
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"PDF file not found: {doc['filename']}")
        
        # Initialize components if needed
        if not llm_client:
            llm_client = LLMClient()
        if not graph_store:
            graph_store = GraphStore()
        
        # Initialize document-specific vector store
        vector_store = VectorStore(document_id=request.document_id)
        vectors_exist = vector_store.collection_exists()
        
        # Check if this document already has its graph stored in Neo4j
        graph_exists = graph_store.document_exists_in_graph(request.document_id)
        
        # Try to load chunks from disk cache or ChromaDB
        chunks = load_chunks_cache(request.document_id)
        if not chunks and vectors_exist:
            chunks = get_chunks_from_chroma(vector_store)
            if chunks:
                save_chunks_cache(request.document_id, chunks)

        # CASE 1: Graph AND Vectors already exist -> INSTANT SWITCH!
        if graph_exists and vectors_exist:
            print(f"\n⚡ INSTANT LOAD: Document '{doc['title']}' already exists in Neo4j and ChromaDB!")
            print(f"   - Neo4j Graph: ✅ Found existing nodes & relationships (no extraction needed)")
            print(f"   - ChromaDB:    ✅ Found existing embeddings ({vector_store.collection.count()} chunks)")
            print(f"   - Chunks:      ✅ Reused ({len(chunks) if chunks else 'auto'} chunks)")
            
            # If chunks weren't in cache, re-extract chunks without running LLM graph extraction
            if not chunks:
                from extraction import extract_pdf_text, chunk_text
                pages, pdf_metadata = extract_pdf_text(str(file_path))
                chunks = chunk_text(pages)
                save_chunks_cache(request.document_id, chunks)

            graph_store.set_document_metadata(request.document_id, doc['title'])
            current_document = doc
            doc["last_accessed"] = datetime.now().isoformat()
            save_library(library)

            # Rank events with LLM (once per document, cached in Neo4j)
            try:
                rank_events_with_llm(request.document_id, doc['title'])
            except Exception as _re:
                print(f"[rank] Non-fatal ranking error: {_re}")

            return {
                "success": True,
                "document": doc,
                "message": f"Loaded '{doc['title']}' instantly from cache"
            }

        # CASE 2: Needs extraction or graph building
        print(f"\n🔄 Document '{doc['title']}' needs processing (graph_exists={graph_exists}, vectors_exist={vectors_exist})")
        
        from extraction import extract_pdf_text, chunk_text
        from pipeline import build_knowledge_graph
        
        if not chunks:
            pages, pdf_metadata = extract_pdf_text(str(file_path))
            chunks = chunk_text(pages)
            save_chunks_cache(request.document_id, chunks)
        
        if not graph_exists:
            print(f"   Building knowledge graph for document {request.document_id}...")
            build_knowledge_graph(chunks, llm_client, graph_store, vector_store, request.document_id)
        else:
            print(f"   ✅ Graph already in Neo4j - skipping build_knowledge_graph")
        
        graph_store.set_document_metadata(request.document_id, doc['title'])
        
        if not vectors_exist:
            print(f"   Building vector store...")
            vector_store.add_chunks(chunks)
        else:
            print(f"   ✅ Vectors already in ChromaDB - reusing")
            
        current_document = doc
        doc["last_accessed"] = datetime.now().isoformat()
        save_library(library)

        # Rank events with LLM (once per document, cached in Neo4j)
        try:
            rank_events_with_llm(request.document_id, doc['title'])
        except Exception as _re:
            print(f"[rank] Non-fatal ranking error: {_re}")

        print(f"✅ Loaded '{doc['title']}' successfully")

        return {
            "success": True,
            "document": doc,
            "message": f"Loaded '{doc['title']}' from library"
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and index a PDF file."""
    global llm_client, graph_store, vector_store, current_document, indexing_in_progress, indexing_status
    
    if indexing_in_progress:
        raise HTTPException(status_code=400, detail="Indexing already in progress")
    
    # Validate file
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        indexing_in_progress = True
        indexing_status = {"status": "uploading", "progress": 10, "message": "Uploading PDF..."}
        
        # Save uploaded file
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        file_path = upload_dir / file.filename
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        indexing_status = {"status": "reading", "progress": 20, "message": "Reading PDF..."}
        
        # Initialize components
        llm_client = LLMClient()
        graph_store = GraphStore()
        
        # Create document ID for collection naming
        doc_id = file.filename.replace(".pdf", "").replace(" ", "_").lower()
        vector_store = VectorStore(document_id=doc_id)
        
        # Clear ONLY this document's previous graph if re-uploading (preserves other documents!)
        indexing_status = {"status": "clearing", "progress": 25, "message": "Preparing document graph..."}
        graph_store.clear_document(doc_id)
        
        # Extract text
        indexing_status = {"status": "extracting", "progress": 30, "message": "Extracting text..."}
        pages, pdf_metadata = extract_pdf_text(str(file_path))
        
        # Chunk text
        indexing_status = {"status": "chunking", "progress": 40, "message": "Creating chunks..."}
        chunks = chunk_text(pages)
        save_chunks_cache(doc_id, chunks)
        
        # Build knowledge graph
        indexing_status = {"status": "building_graph", "progress": 50, "message": f"Building graph ({len(chunks)} chunks)..."}
        build_knowledge_graph(chunks, llm_client, graph_store, vector_store, doc_id)
        
        # Build vector store
        indexing_status = {"status": "building_vectors", "progress": 90, "message": "Building vector store..."}
        vector_store.add_chunks(chunks)
        
        # Add to library
        doc_id = add_to_library(file.filename, pdf_metadata.get("title", file.filename), len(chunks))
        
        # Mark this document as loaded in Neo4j
        graph_store.set_document_metadata(doc_id, pdf_metadata.get("title", file.filename))
        
        current_document = {
            "id": doc_id,
            "filename": file.filename,
            "title": pdf_metadata.get("title", "Unknown"),
            "chunks": len(chunks)
        }
        
        indexing_status = {"status": "complete", "progress": 100, "message": "Indexing complete!"}
        
        return {
            "success": True,
            "document_id": doc_id,
            "filename": file.filename,
            "title": pdf_metadata.get("title", "Unknown"),
            "chunks": len(chunks),
            "message": "PDF indexed and added to library!"
        }
    
    except Exception as e:
        indexing_status = {"status": "error", "progress": 0, "message": str(e)}
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        indexing_in_progress = False


@app.post("/api/query")
async def query(request: QueryRequest):
    """Query the indexed PDF."""
    global llm_client, graph_store, vector_store, current_document
    
    if not current_document:
        raise HTTPException(status_code=400, detail="No document loaded. Please upload or load from library first.")
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        start_time = time.time()
        
        # Import here to capture metrics
        from retrieval import vector_search, graph_search, build_context, generate_answer
        
        # Vector search
        vector_start = time.time()
        vector_results = vector_search(request.question, vector_store, top_k=5)
        vector_time = time.time() - vector_start
        
        # Graph search
        graph_start = time.time()
        graph_results = graph_search(request.question, graph_store, llm_client, current_document["id"])
        graph_time = time.time() - graph_start
        
        # Build context
        context = build_context(vector_results, graph_results)
        
        # Generate answer and get which paths were actually used
        answer, used_graph_data, gen_metrics = generate_answer(request.question, context, llm_client, graph_results)
        
        total_time = time.time() - start_time

        # Collect fact_ids and path_ids for evaluation / ground-truth tracking
        retrieved_path_ids = [r["path_id"] for r in graph_results if "path_id" in r]
        retrieved_fact_ids = list(dict.fromkeys([fid for r in graph_results for fid in r.get("fact_ids", [])]))
        used_path_ids = [r["path_id"] for r in used_graph_data if "path_id" in r]
        used_fact_ids = list(dict.fromkeys([fid for r in used_graph_data for fid in r.get("fact_ids", [])]))
        
        return {
            "answer": answer,
            "graph_data": used_graph_data,  # Contains path_id, fact_ids, nodes, relationships
            "sources": {
                "vector_chunks": len(vector_results),
                "graph_paths": len(graph_results),
                "used_paths": len(used_graph_data),
                "retrieved_path_ids": retrieved_path_ids,
                "retrieved_fact_ids": retrieved_fact_ids,
                "used_path_ids": used_path_ids,
                "used_fact_ids": used_fact_ids,
                "top_chunks": [
                    {
                        "chunk_id": r["chunk_id"],
                        "page": r["page"],
                        "similarity": round(r["score"], 4),
                        "preview": r["text"][:200] + "..."
                    }
                    for r in vector_results[:3]
                ]
            },
            "metrics": {
                "total_time": round(total_time, 2),
                "vector_search_time": round(vector_time, 2),
                "graph_search_time": round(graph_time, 2),
                "generation_time": round(gen_metrics["generation_time"], 2),
                "ttft": round(gen_metrics["ttft"], 2),
                "tokens_per_second": round(gen_metrics["tokens_per_second"], 1),
                "total_tokens": gen_metrics["total_tokens"]
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/facts")
async def get_facts():
    """Get all facts (edges with fact_ids) for current document for ground truth dataset evaluation."""
    global graph_store, current_document
    if not current_document:
        raise HTTPException(status_code=400, detail="No document loaded")
    if not graph_store:
        raise HTTPException(status_code=500, detail="Graph store not initialized")
    
    try:
        facts = graph_store.get_all_facts(current_document["id"])
        return {
            "document_id": current_document["id"],
            "total_facts": len(facts),
            "facts": facts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def extract_dialogue_from_text(text: str):
    """Extract character dialogue lines from screenplay formatted text."""
    if not text:
        return []
    dialogue = []
    lines = text.split("\n")
    current_char = None
    current_speech = []
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_char and current_speech:
                dialogue.append({"character": current_char, "text": " ".join(current_speech)})
                current_char = None
                current_speech = []
            continue
            
        # Check if line is a character cue (UPPERCASE, short, not scene header or transition)
        if stripped.isupper() and len(stripped) < 35 and not stripped.startswith(("INT.", "EXT.", "SCENE", "FADE", "CUT TO", "SMASH CUT")):
            if current_char and current_speech:
                dialogue.append({"character": current_char, "text": " ".join(current_speech)})
            current_char = re.sub(r'\s*\([^)]*\)', '', stripped).strip()
            current_speech = []
        elif current_char:
            if not (stripped.startswith("(") and stripped.endswith(")")):
                current_speech.append(stripped)
                
    if current_char and current_speech:
        dialogue.append({"character": current_char, "text": " ".join(current_speech)})
        
    return dialogue


def build_hierarchical_graph_data(document_id: str, level: str = "1", expand_node_id: str = None):
    """
    Build a 3-level hierarchical graph representation:
    - Level 1: Story Overview (Acts, Major Plot Events, Principal Characters, Major Conflicts)
    - Level 2: Character & Event Subgraph (Secondary Characters, Connected Events, Locations, Objects)
    - Level 3: Scene Detail (Scene Nodes, Granular Actions, Minor Characters, Supporting Facts)
    """
    global graph_store
    
    # Load scene chunks from cache or chroma
    chunks = load_chunks_cache(document_id)
    if not chunks:
        # Try ChromaDB
        v_store = VectorStore(document_id=document_id)
        chunks = get_chunks_from_chroma(v_store)
        if chunks:
            save_chunks_cache(document_id, chunks)
            
    chunks = chunks or []
    chunk_nums = [c.get("scene_number") or c.get("id") or 1 for c in chunks if isinstance(c, dict)]
    total_chunks = max([int(x) for x in chunk_nums if isinstance(x, (int, float))] + [1])
    
    scene_map = {}
    for c in chunks:
        if isinstance(c, dict):
            s_num = c.get("scene_number") or c.get("id") or 1
            if s_num not in scene_map:
                scene_map[s_num] = c

    with graph_store.driver.session() as session:
        # Check and migrate legacy nodes without document_id if any
        session.run("""
            MATCH (n) WHERE n.document_id IS NULL
            SET n.document_id = $document_id
        """, document_id=document_id)
        
        graph_store.ensure_fact_ids(document_id)
        
        nodes_q = session.run("""
            MATCH (n) WHERE n.document_id = $document_id
            RETURN id(n) as id, n.name as label, labels(n)[0] as node_label,
                   n.type as entity_type, n.event_id as event_id,
                   n.chunk_id as chunk_id, n.description as description,
                   n.participants as participants, n.location as location,
                   n.importance_rank as importance_rank,
                   size([(n)--() | 1]) as deg
        """, document_id=document_id)
        raw_nodes = [dict(r) for r in nodes_q]
        
        edges_q = session.run("""
            MATCH (n)-[r]->(m) WHERE n.document_id = $document_id AND m.document_id = $document_id
            RETURN id(n) as from_id, id(m) as to_id, n.name as from_name, m.name as to_name,
                   type(r) as label, type(r) as type, coalesce(r.fact_id, '') as fact_id
        """, document_id=document_id)
        raw_edges = [dict(r) for r in edges_q]

    # Build adjacency and lookup maps
    node_by_id = {str(n["id"]): n for n in raw_nodes}
    adj = {str(n["id"]): set() for n in raw_nodes}
    for e in raw_edges:
        f_id, t_id = str(e["from_id"]), str(e["to_id"])
        if f_id in adj and t_id in adj:
            adj[f_id].add(t_id)
            adj[t_id].add(f_id)

    # 1. Classify Characters (Strictly Principal vs Secondary vs Minor)
    chars = []
    for n in raw_nodes:
        typ = str(n.get("entity_type") or "").strip().lower()
        lbl = str(n.get("node_label") or "").strip().lower()
        if typ == "character" or lbl == "character":
            chars.append(n)
            
    chars.sort(key=lambda x: int(x.get("deg") or 0), reverse=True)

    # Principal characters: top 4 to 6 central characters
    principal_nodes = chars[:5] if len(chars) >= 5 else chars
    principal_names = {str(c.get("label") or c.get("name") or "").strip().upper() for c in principal_nodes if (c.get("label") or c.get("name"))}
    principal_ids = {str(c["id"]) for c in principal_nodes}

    # 2. Classify Events (Strictly Major Plot Turning Points vs Scene Details)
    events = [n for n in raw_nodes if n.get("node_label") == "Event"]
    # Tension keywords that signal high-stakes plot moments
    TENSION_KEYWORDS = {
        "KILL", "DEAD", "DEATH", "DIES", "MURDER", "REVEAL", "REVEALS", "BETRAYAL",
        "CONFRONTATION", "CONFESSION", "ARRESTED", "ESCAPE", "SHOT", "FIRE",
        "FIGHT", "ATTACK", "CRISIS", "BREAKS", "DISCOVERS", "LEARNS", "REALIZED",
        "REALIZES", "EXPOSES", "ADMITS", "SACRIFICE", "TRAPPED", "AMBUSH", "CRASH"
    }

    for e in events:
        p_names = {str(p).strip().upper() for p in (e.get("participants") or [])}
        p_count = len(p_names.intersection(principal_names))
        e["p_count"] = p_count
        deg = int(e.get("deg") or 0)
        label_upper = str(e.get("label") or "").upper()
        # Tension keyword bonus
        tension_bonus = sum(10 for kw in TENSION_KEYWORDS if kw in label_upper)
        # Base score: principal participation + degree + tension keywords
        base_score = (p_count * 15) + deg + tension_bonus
        # Override with LLM importance rank if stored (rank 1 = most important → highest score)
        importance_rank = e.get("importance_rank")
        if importance_rank is not None:
            # rank 1 → score 10000, rank 15 → score 9860, unranked → base_score
            e["score"] = 10000 - (int(importance_rank) * 10)
            e["llm_ranked"] = True
        else:
            e["score"] = base_score
            e["llm_ranked"] = False
        c_num = int(e.get("chunk_id") or 0)
        if c_num <= total_chunks * 0.25:
            e["act"] = "Act I (Setup)"
            e["act_id"] = "act_1"
        elif c_num <= total_chunks * 0.75:
            e["act"] = "Act II (Confrontation)"
            e["act_id"] = "act_2"
        else:
            e["act"] = "Act III (Resolution)"
            e["act_id"] = "act_3"

    events.sort(key=lambda x: x["score"], reverse=True)

    # DEBUG: print ALL event scores so user can tune selection
    selected_ids = set()
    print("\n🎬 EVENT SCORES — ALL (✅ = selected for L1):")
    # We need to know which will be selected; pre-compute
    _tmp_buckets = {"act_1": [], "act_2": [], "act_3": []}
    for _e in events:
        _tmp_buckets[_e.get("act_id", "act_2")].append(_e)
    _tmp_selected = []
    for _ak in ["act_1", "act_2", "act_3"]:
        _tmp_selected.extend(_tmp_buckets[_ak][:2])
    _tmp_sel_ids = {str(_e["id"]) for _e in _tmp_selected}
    for _e in events:
        if len(_tmp_sel_ids) < 10 and str(_e["id"]) not in _tmp_sel_ids:
            _tmp_sel_ids.add(str(_e["id"]))
    for i, ev in enumerate(events):
        label = str(ev.get("label") or ev.get("name") or ev.get("id") or "?")
        mark = "✅" if str(ev["id"]) in _tmp_sel_ids else "  "
        llm_tag = " 🤖" if ev.get("llm_ranked") else ""
        rank_tag = f" rank={ev.get('importance_rank','—'):>2}" if ev.get("llm_ranked") else ""
        print(f"  {mark} #{i+1:2d} [score={ev['score']:5d} pc={ev['p_count']} deg={int(ev.get('deg') or 0):3d} act={ev.get('act_id','?')}{rank_tag}{llm_tag}] {label[:50]}")
    print()

    # Pick top events but ensure spread across acts (at least 2 per act if possible)
    act_buckets = {"act_1": [], "act_2": [], "act_3": []}
    for e in events:
        act_buckets[e.get("act_id", "act_2")].append(e)

    major_event_nodes = []
    # Guarantee at least 2 from each act, fill remaining with highest scores
    for act_key in ["act_1", "act_2", "act_3"]:
        major_event_nodes.extend(act_buckets[act_key][:2])
    # Top up to 10 with remaining highest-scored events not yet included
    included_ids = {str(e["id"]) for e in major_event_nodes}
    for e in events:
        if len(major_event_nodes) >= 10:
            break
        if str(e["id"]) not in included_ids:
            major_event_nodes.append(e)
            included_ids.add(str(e["id"]))

    major_event_ids = {str(e["id"]) for e in major_event_nodes}
    major_event_names = {str(e.get("label") or "").strip().upper() for e in major_event_nodes}

    # Secondary Characters: connected to principal characters
    secondary_nodes = [c for c in chars if str(c["id"]) not in principal_ids][:6]
    secondary_ids = {str(c["id"]) for c in secondary_nodes}

    # 3. Virtual Act Milestone Nodes for Level 1 Spine
    act_nodes = [
        {
            "id": "act_1",
            "label": "Act I: Setup",
            "type": "Act",
            "level": 1,
            "act": "Act I",
            "degree": 4,
            "is_act": True
        },
        {
            "id": "act_2",
            "label": "Act II: Confrontation",
            "type": "Act",
            "level": 1,
            "act": "Act II",
            "degree": 6,
            "is_act": True
        },
        {
            "id": "act_3",
            "label": "Act III: Resolution",
            "type": "Act",
            "level": 1,
            "act": "Act III",
            "degree": 4,
            "is_act": True
        }
    ]

    act_progression_edges = [
        {
            "id": "act_edge_1_2",
            "from": "act_1",
            "to": "act_2",
            "label": "PROGRESSES_TO",
            "type": "PROGRESSES_TO",
            "fact_id": "act_spine_1_2",
            "level": 1
        },
        {
            "id": "act_edge_2_3",
            "from": "act_2",
            "to": "act_3",
            "label": "PROGRESSES_TO",
            "type": "PROGRESSES_TO",
            "fact_id": "act_spine_2_3",
            "level": 1
        }
    ]

    # 4. Classify all Nodes into Levels
    classified_nodes = {}
    for n in raw_nodes:
        nid = str(n["id"])
        nlbl = n.get("label") or n.get("name") or ""
        nlbl_clean = str(nlbl).strip().upper()
        ntyp = n.get("entity_type") or n.get("node_label") or "Entity"

        node_level = 2
        is_princ = False
        is_maj = False

        if nlbl_clean in principal_names or nid in principal_ids:
            node_level = 1
            is_princ = True
        elif nlbl_clean in major_event_names or nid in major_event_ids:
            node_level = 1
            is_maj = True
        elif nid in secondary_ids:
            node_level = 2
        else:
            node_level = 2

        c_num = int(n.get("chunk_id") or 0)
        act = "Act I (Setup)" if c_num <= total_chunks * 0.25 else ("Act II (Confrontation)" if c_num <= total_chunks * 0.75 else "Act III (Resolution)")

        classified_nodes[nid] = {
            "id": nid,
            "label": nlbl or nid,
            "type": ntyp,
            "level": node_level,
            "act": act,
            "is_principal": is_princ,
            "is_major": is_maj,
            "chunk_id": n.get("chunk_id"),
            "event_id": n.get("event_id"),
            "description": n.get("description"),
            "participants": n.get("participants") or [],
            "location": n.get("location"),
            "degree": int(n.get("deg") or 0),
            "importance_rank": n.get("importance_rank"),
            "connected_ids": list(adj.get(nid, set()))
        }

    # Hidden connection count for Level 1 nodes
    l1_ids = {nid for nid, n in classified_nodes.items() if n["level"] == 1} | {"act_1", "act_2", "act_3"}
    for nid, n in classified_nodes.items():
        if n["level"] == 1:
            hidden = sum(1 for neighbor_id in adj.get(nid, set()) if neighbor_id not in l1_ids)
            n["hidden_connections"] = hidden
            n["expandable"] = hidden > 0
        else:
            n["hidden_connections"] = 0
            n["expandable"] = False

    # 5. Classify Edges into Levels
    classified_edges = []
    for idx, e in enumerate(raw_edges):
        f_id, t_id = str(e["from_id"]), str(e["to_id"])
        src_node = classified_nodes.get(f_id)
        tgt_node = classified_nodes.get(t_id)
        if not src_node or not tgt_node:
            continue
            
        edge_level = max(src_node["level"], tgt_node["level"])
        classified_edges.append({
            "id": f"edge_{idx}",
            "from": f_id,
            "to": t_id,
            "label": e["label"],
            "type": e["type"],
            "fact_id": e.get("fact_id", ""),
            "level": edge_level,
            "source_name": e.get("from_name"),
            "target_name": e.get("to_name")
        })

    # Add Act membership links for Major Events in Level 1
    for e in major_event_nodes:
        eid = str(e["id"])
        act_id = e.get("act_id", "act_1")
        classified_edges.append({
            "id": f"act_link_{eid}",
            "from": eid,
            "to": act_id,
            "label": "PART_OF_ACT",
            "type": "PART_OF_ACT",
            "fact_id": f"fact_act_{eid}",
            "level": 1
        })

    # Add chronological Act spine edges
    classified_edges.extend(act_progression_edges)

    # 6. Build Filtered Views (Strictly Bounded to Prevent Browser Lag)
    target_level = str(level).lower().strip()
    
    if target_level == "1":
        # LEVEL 1: ONLY Major Characters, Major Events, and 3 Acts (~15-18 nodes total)
        active_node_ids = {nid for nid, n in classified_nodes.items() if n["level"] == 1} | {"act_1", "act_2", "act_3"}
        
        # If user explicitly expanded a node in Level 1
        if expand_node_id:
            exp_id = str(expand_node_id)
            if exp_id in classified_nodes:
                # Add up to 8 top neighbors
                neighbors = list(adj.get(exp_id, set()))[:8]
                active_node_ids.update(neighbors)
            
        visible_nodes = [n for nid, n in classified_nodes.items() if nid in active_node_ids]
        visible_nodes.extend([a for a in act_nodes if a["id"] in active_node_ids])
        visible_edges = [e for e in classified_edges if e["from"] in active_node_ids and e["to"] in active_node_ids]
        
    elif target_level == "2":
        # LEVEL 2: Major Characters + Secondary Characters + Top 15 Connected Events (~25-30 nodes total)
        level2_node_ids = {nid for nid, n in classified_nodes.items() if n["level"] == 1 or nid in secondary_ids}
        # Add top connected events
        connected_events = [nid for nid, n in classified_nodes.items() if n.get("type") == "Event" and any(pid in adj.get(nid, set()) for pid in principal_ids)][:15]
        level2_node_ids.update(connected_events)
        level2_node_ids.update({"act_1", "act_2", "act_3"})
        
        visible_nodes = [n for nid, n in classified_nodes.items() if nid in level2_node_ids] + act_nodes
        visible_edges = [e for e in classified_edges if e["from"] in level2_node_ids and e["to"] in level2_node_ids]
        
    else:
        # Default Level 1
        active_node_ids = {nid for nid, n in classified_nodes.items() if n["level"] == 1} | {"act_1", "act_2", "act_3"}
        visible_nodes = [n for nid, n in classified_nodes.items() if nid in active_node_ids] + act_nodes
        visible_edges = [e for e in classified_edges if e["from"] in active_node_ids and e["to"] in active_node_ids]

    return {
        "nodes": visible_nodes,
        "edges": visible_edges,
        "level": target_level,
        "hierarchy": {
            "level1_count": len([n for n in classified_nodes.values() if n["level"] == 1]) + len(act_nodes),
            "level2_count": len(visible_nodes),
            "principal_characters": list(principal_names),
            "major_events": [e["label"] for e in major_event_nodes],
            "acts": ["Act I: Setup", "Act II: Confrontation", "Act III: Resolution"]
        },
        "metadata": {
            "total_nodes": len(raw_nodes) + len(act_nodes),
            "total_edges": len(raw_edges) + len(act_progression_edges),
            "showing_nodes": len(visible_nodes),
            "showing_edges": len(visible_edges),
            "level": target_level
        }
    }


@app.get("/api/graph-data")
async def get_graph_data(
    importance_filter: str = "medium",
    level: str = "1",
    expand_node: str = None
):
    """
    Get multi-level hierarchical knowledge graph data:
    - Level 1: Story Overview (Major Characters & Major Events only, ~15 nodes)
    - Level 2: Character and Event Subgraph (Selected character/event connections)
    """
    global graph_store, current_document
    
    if not graph_store:
        graph_store = GraphStore()
        
    if not current_document:
        doc_id = graph_store.get_current_document()
        if doc_id:
            lib = load_library()
            doc = next((d for d in lib if d["id"] == doc_id), None)
            if doc:
                current_document = doc
                
    if not current_document:
        raise HTTPException(status_code=400, detail="No document loaded")
    
    try:
        data = build_hierarchical_graph_data(
            document_id=current_document["id"],
            level=level,
            expand_node_id=expand_node
        )
        return data
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))




@app.get("/api/act-timeline")
async def get_act_timeline(act: str = "act_1"):
    """
    Returns all events for a given act (act_1/act_2/act_3) ordered by chunk_id.
    Each event includes scene number, location, participants, and description.
    Used by the Narrative Timeline view.
    """
    global graph_store, current_document

    if not graph_store:
        graph_store = GraphStore()
    if not current_document:
        doc_id = graph_store.get_current_document()
        if doc_id:
            lib = load_library()
            current_document = next((d for d in lib if d["id"] == doc_id), None)

    if not current_document:
        raise HTTPException(status_code=400, detail="No document loaded")

    doc_id = current_document["id"]

    # Determine chunk_id range based on act
    chunks = load_chunks_cache(doc_id) or []
    total_chunks = len(chunks) if chunks else 125
    if act == "act_1":
        chunk_min, chunk_max = 0, int(total_chunks * 0.25)
        act_label = "Act I: Setup"
        act_color = "#0e7490"
    elif act == "act_3":
        chunk_min, chunk_max = int(total_chunks * 0.75), total_chunks + 1
        act_label = "Act III: Resolution"
        act_color = "#9a3412"
    elif act == "all":
        chunk_min, chunk_max = 0, total_chunks + 9999
        act_label = "All Acts (Full Story)"
        act_color = "#4f46e5"
    else:
        chunk_min, chunk_max = int(total_chunks * 0.25), int(total_chunks * 0.75)
        act_label = "Act II: Confrontation"
        act_color = "#581c87"

    # Build chunk location lookup
    chunk_info = {}
    for c in chunks:
        ec = enrich_chunk_with_header(c)
        cid = c.get("scene_number") or c.get("id")
        if cid is not None and ec.get("location") and ec["location"] != "Unknown Location":
            if cid not in chunk_info:
                chunk_info[cid] = {
                    "location": ec.get("location", "Unknown"),
                    "int_ext": ec.get("int_ext", "INT"),
                    "time": ec.get("time", "DAY"),
                    "page": ec.get("page", 1),
                    "scene_header": ec.get("scene_header", "")
                }

    try:
        with graph_store.driver.session() as session:
            evts_q = session.run("""
                MATCH (e:Event)
                WHERE e.document_id = $doc_id
                  AND toInteger(coalesce(e.chunk_id, 0)) >= $chunk_min
                  AND toInteger(coalesce(e.chunk_id, 0)) <= $chunk_max
                RETURN id(e) as id, e.name as name, e.description as description,
                       e.chunk_id as chunk_id, e.participants as participants,
                       e.importance_rank as importance_rank,
                       coalesce(e.sentiment, 'neutral') as sentiment
                ORDER BY toInteger(coalesce(e.chunk_id, 0)) ASC
            """, doc_id=doc_id, chunk_min=chunk_min, chunk_max=chunk_max)

            events = []
            for r in evts_q:
                cid = r["chunk_id"]
                info = chunk_info.get(cid, {})
                events.append({
                    "id": str(r["id"]),
                    "name": r["name"] or "",
                    "description": r["description"] or "",
                    "chunk_id": cid,
                    "participants": list(r["participants"] or []),
                    "importance_rank": r["importance_rank"],
                    "sentiment": r["sentiment"] or "neutral",
                    "location": info.get("location", "Unknown"),
                    "int_ext": info.get("int_ext", "INT"),
                    "time": info.get("time", "DAY"),
                    "page": info.get("page", 1),
                    "scene_header": info.get("scene_header", ""),
                    "is_major": r["importance_rank"] is not None and int(r["importance_rank"] or 99) <= 8
                })

        return {
            "act": act,
            "act_label": act_label,
            "act_color": act_color,
            "total_scenes": chunk_max - chunk_min,
            "events": events
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/all-characters")
async def get_all_characters():
    """Return ALL character nodes + inter-character edges for the Character Web view."""
    global graph_store, current_document

    if not graph_store:
        graph_store = GraphStore()
    if not current_document:
        doc_id = graph_store.get_current_document()
        if doc_id:
            lib = load_library()
            current_document = next((d for d in lib if d["id"] == doc_id), None)

    if not current_document:
        raise HTTPException(status_code=400, detail="No document loaded")

    doc_id = current_document["id"]
    JUNK = {"NO","YES","OK","OH","AH","VO","OS","BEAT","NONE","CONTINUED","EXT","INT","CUT","MAN","WOMAN","BOY","GIRL"}

    try:
        with graph_store.driver.session() as session:
            # All character-type entity nodes for this document
            chars_q = session.run("""
                MATCH (e:Entity)
                WHERE e.document_id = $doc_id AND e.type = 'Character'
                RETURN id(e) as id, e.name as label, 'Character' as type,
                       e.chunk_id as chunk_id, e.description as description
            """, doc_id=doc_id)
            
            raw_chars = []
            char_ids = set()
            for r in chars_q:
                lbl = (r["label"] or "").strip().upper()
                if lbl in JUNK or len(lbl) <= 1 or lbl.isdigit():
                    continue
                nid = str(r["id"])
                char_ids.add(nid)
                raw_chars.append({
                    "id": nid,
                    "label": r["label"],
                    "type": "Character",
                    "node_label": "Entity",
                    "chunk_id": r.get("chunk_id"),
                    "description": r.get("description"),
                    "level": 3  # will be styled as minor unless overridden
                })

            # Inter-character edges (Entity → Entity where both are characters)
            edges_q = session.run("""
                MATCH (a:Entity)-[r]->(b:Entity)
                WHERE a.document_id = $doc_id AND b.document_id = $doc_id
                  AND a.type = 'Character' AND b.type = 'Character'
                RETURN id(a) as from_id, id(b) as to_id, type(r) as rel_label,
                       coalesce(r.fact_id, '') as fact_id
                LIMIT 2000
            """, doc_id=doc_id)

            raw_edges = []
            seen_e = set()
            for r in edges_q:
                f = str(r["from_id"])
                t = str(r["to_id"])
                if f not in char_ids or t not in char_ids:
                    continue
                eid = f"{f}_{t}_{r['rel_label']}"
                if eid in seen_e:
                    continue
                seen_e.add(eid)
                raw_edges.append({
                    "id": f"cw_{eid}",
                    "from": f,
                    "to": t,
                    "label": r["rel_label"],
                    "type": r["rel_label"],
                    "fact_id": r.get("fact_id", ""),
                    "level": 2
                })

        return {"nodes": raw_chars, "edges": raw_edges}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/graph-expand")
async def graph_expand(node_id: str):
    """
    Expand a clicked node's immediate connections:
    - If Character: returns their direct connected events & relationships (up to 8 events).
    - If Event: returns its Scene node and characters in that scene.
    """
    global graph_store, current_document
    
    if not graph_store:
        graph_store = GraphStore()
    if not current_document:
        doc_id = graph_store.get_current_document()
        if doc_id:
            lib = load_library()
            current_document = next((d for d in lib if d["id"] == doc_id), None)
            
    if not current_document:
        raise HTTPException(status_code=400, detail="No document loaded")
        
    doc_id = current_document["id"]
    
    try:
        with graph_store.driver.session() as session:
            # 1. Look up the clicked node
            node_res = session.run("""
                MATCH (n)
                WHERE (id(n) = toInteger($node_id) OR n.name = $node_id)
                  AND n.document_id = $doc_id
                RETURN id(n) as id, n.name as label, labels(n)[0] as node_label,
                       n.type as entity_type, n.chunk_id as chunk_id, n.description as description
                LIMIT 1
            """, node_id=node_id, doc_id=doc_id).single()
            
            if not node_res:
                return {"parent_id": node_id, "nodes": [], "edges": []}
                
            clicked_node = dict(node_res)
            target_id = clicked_node["id"]
            node_type = clicked_node.get("entity_type") or clicked_node.get("node_label")
            
            new_nodes = []
            new_edges = []
            
            # CASE A: Character clicked -> Find ALL direct 1-hop events & relationships
            if node_type == "Character" or clicked_node.get("node_label") == "Entity":
                neighbors_q = session.run("""
                    MATCH (c)-[r]-(m)
                    WHERE id(c) = $target_id AND c.document_id = $doc_id
                    RETURN id(m) as id, m.name as label, labels(m)[0] as node_label,
                           m.type as entity_type, m.chunk_id as chunk_id, m.description as description,
                           type(r) as rel_label, coalesce(r.fact_id, '') as fact_id,
                           startNode(r) = c as is_outgoing
                """, target_id=target_id, doc_id=doc_id)
                
                for r in neighbors_q:
                    n_id = str(r["id"])
                    n_type = r.get("entity_type") or r.get("node_label")
                    new_nodes.append({
                        "id": n_id,
                        "label": r["label"],
                        "type": n_type,
                        "chunk_id": r.get("chunk_id"),
                        "description": r.get("description"),
                        "level": 2
                    })
                    
                    from_id = str(target_id) if r["is_outgoing"] else n_id
                    to_id = n_id if r["is_outgoing"] else str(target_id)
                    new_edges.append({
                        "id": f"expand_edge_{from_id}_{to_id}_{r['rel_label']}",
                        "from": from_id,
                        "to": to_id,
                        "label": r["rel_label"],
                        "type": r["rel_label"],
                        "fact_id": r.get("fact_id", ""),
                        "level": 2
                    })

            # CASE B: Event clicked -> Find its source Scene node and participating characters
            elif clicked_node.get("node_label") == "Event":
                chunk_id = clicked_node.get("chunk_id")
                if chunk_id is not None:
                    # Load scene chunk
                    chunks = load_chunks_cache(doc_id) or []
                    matched = next((c for c in chunks if c.get("scene_number") == chunk_id or c.get("id") == chunk_id), None)
                    sc_location = matched.get("location", "Scene") if matched else "Scene"
                    sc_page = matched.get("page", 1) if matched else 1
                    
                    scene_node_id = f"scene_{chunk_id}"
                    new_nodes.append({
                        "id": scene_node_id,
                        "label": f"🎬 Scene {chunk_id}: {sc_location}",
                        "type": "Scene",
                        "scene_number": chunk_id,
                        "page": sc_page,
                        "location": sc_location,
                        "level": 2,
                        "is_scene": True
                    })
                    
                    new_edges.append({
                        "id": f"link_evt_scene_{target_id}_{scene_node_id}",
                        "from": str(target_id),
                        "to": scene_node_id,
                        "label": "OCCURS_IN",
                        "type": "OCCURS_IN",
                        "fact_id": f"fact_scene_{target_id}_{chunk_id}",
                        "level": 2
                    })

            return {
                "parent_id": str(target_id),
                "nodes": new_nodes,
                "edges": new_edges
            }
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/scene-detail")
async def get_scene_detail(
    scene_number: int = None,
    chunk_id: int = None,
    event_name: str = None
):
    """
    Level 3 Scene Detail endpoint:
    Returns the source scene excerpt, character dialogue, minor characters,
    individual actions, and supporting facts with Fact IDs.
    """
    global graph_store, current_document
    
    if not current_document:
        raise HTTPException(status_code=400, detail="No document loaded")
        
    doc_id = current_document["id"]
    chunks = load_chunks_cache(doc_id)
    if not chunks:
        v_store = VectorStore(document_id=doc_id)
        chunks = get_chunks_from_chroma(v_store)
        if chunks:
            save_chunks_cache(doc_id, chunks)
    chunks = chunks or []

    # If event_name was passed, look up its chunk_id in Neo4j
    target_chunk_id = chunk_id
    target_event_info = None
    
    if event_name:
        with graph_store.driver.session() as session:
            evt_res = session.run("""
                MATCH (e:Event)
                WHERE e.document_id = $doc_id AND e.name = $event_name
                RETURN e.chunk_id as chunk_id, e.description as description,
                       e.participants as participants, e.location as location,
                       e.event_id as event_id
                LIMIT 1
            """, doc_id=doc_id, event_name=event_name).single()
            if evt_res:
                target_chunk_id = evt_res["chunk_id"]
                target_event_info = dict(evt_res)

    if scene_number is not None:
        target_chunk_id = scene_number

    # Find the corresponding chunk
    matched_chunk = None
    if target_chunk_id is not None:
        for c in chunks:
            if c.get("scene_number") == target_chunk_id or c.get("id") == target_chunk_id:
                matched_chunk = c
                break

    if not matched_chunk and chunks:
        matched_chunk = chunks[0]

    if not matched_chunk:
        raise HTTPException(status_code=404, detail="Scene not found")

    scene_num = matched_chunk.get("scene_number", matched_chunk.get("id", 1))
    script_text = matched_chunk.get("text", "")
    dialogue = extract_dialogue_from_text(script_text)

    # Query Neo4j for events in this scene
    events_in_scene = []
    with graph_store.driver.session() as session:
        evts_q = session.run("""
            MATCH (e:Event)
            WHERE e.document_id = $doc_id AND e.chunk_id = $chunk_id
            RETURN e.name as name, e.description as description,
                   e.participants as participants, e.location as location,
                   e.event_id as event_id
        """, doc_id=doc_id, chunk_id=scene_num)
        events_in_scene = [dict(r) for r in evts_q]

        # Query Neo4j for supporting facts (relationships) associated with this scene
        facts_q = session.run("""
            MATCH (a)-[r]->(b)
            WHERE (r.document_id = $doc_id OR a.document_id = $doc_id)
              AND (r.chunk_id = $chunk_id OR a.chunk_id = $chunk_id OR b.chunk_id = $chunk_id)
            RETURN a.name as source, type(r) as relation, b.name as target,
                   coalesce(r.fact_id, '') as fact_id
            LIMIT 20
        """, doc_id=doc_id, chunk_id=scene_num)
        facts = [dict(r) for r in facts_q]

    # If no facts with chunk_id directly, find facts connecting participants in this scene
    if not facts and matched_chunk.get("characters"):
        scene_chars = matched_chunk.get("characters", [])
        with graph_store.driver.session() as session:
            facts_q = session.run("""
                MATCH (a)-[r]->(b)
                WHERE (r.document_id = $doc_id OR a.document_id = $doc_id)
                  AND (a.name IN $chars OR b.name IN $chars)
                RETURN a.name as source, type(r) as relation, b.name as target,
                       coalesce(r.fact_id, '') as fact_id
                LIMIT 15
            """, doc_id=doc_id, chars=scene_chars)
            facts = [dict(r) for r in facts_q]

    # Try to get location from the matched_chunk itself, or fall back to the
    # first chunk of the same scene_number (handles split scenes)
    _ec = enrich_chunk_with_header(matched_chunk)
    if _ec.get("location") in (None, "Unknown Location"):
        target_scene = matched_chunk.get("scene_number")
        if target_scene is not None:
            for c in chunks:
                if c.get("scene_number") == target_scene:
                    candidate = enrich_chunk_with_header(c)
                    if candidate.get("location") and candidate["location"] != "Unknown Location":
                        _ec = {**_ec, **{k: candidate[k] for k in ("location","int_ext","time","scene_header") if k in candidate}}
                    break  # first chunk of that scene

    return {
        "scene_number": scene_num,
        "page": _ec.get("page", 1),
        "scene_header": _ec.get("scene_header", f"SCENE {scene_num}"),
        "location": _ec.get("location", "Unknown Location"),
        "time": _ec.get("time", "DAY"),
        "int_ext": _ec.get("int_ext", "INT."),
        "script_text": script_text,
        "dialogue": dialogue,
        "events": events_in_scene,
        "participants": list({
            p for ev in events_in_scene
            for p in (ev.get("participants") or [])
            if p and len(str(p)) > 1
        } | set(matched_chunk.get("characters", []) or [])),
        "facts": [
            f for f in facts
            if f.get("source") and str(f.get("source","")).upper()
            not in {"NO","YES","OK","OH","AH","WHY","WHAT","WELL","VO","OS","BEAT","NONE","CONTINUED"}
            and len(str(f.get("source",""))) > 1
        ],
        "selected_event": target_event_info
    }



@app.get("/api/neo4j-info")
async def neo4j_info():
    """Get Neo4j connection information."""
    return {
        "uri": config.NEO4J_URI,
        "username": config.NEO4J_USERNAME,
        "browser_url": config.NEO4J_URI.replace("bolt://", "http://").replace(":7687", ":7474")
    }


@app.get("/api/characters")
async def get_characters():
    """Get all characters from the loaded document."""
    global graph_store, current_document
    
    if not current_document:
        raise HTTPException(status_code=400, detail="No document loaded")
    
    if not graph_store:
        raise HTTPException(status_code=500, detail="Graph store not initialized")
    
    try:
        characters = graph_store.get_all_characters(current_document["id"])
        return {"characters": characters}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/character-arc/{character_name}")
async def get_character_arc(character_name: str):
    """Get character arc data including emotional journey and metrics."""
    global graph_store, vector_store, llm_client, current_document
    
    if not current_document:
        raise HTTPException(status_code=400, detail="No document loaded")
    
    if not graph_store:
        raise HTTPException(status_code=500, detail="Graph store not initialized")
    
    try:
        # Get base arc data from graph
        arc_data = graph_store.get_character_arc_data(character_name, current_document["id"])
        # #region agent log
        _agent_debug_log(
            "main.py:get_character_arc:entry",
            "arc request",
            {
                "character": character_name,
                "document_id": current_document["id"],
                "graph_scenes": len(arc_data.get("scenes", [])),
                "fixed_emotion_keys": ["tension", "confidence", "hope", "determination"],
            },
            "H1",
        )
        # #endregion
        
        # Enrich with emotional analysis using LLM and screenplay chunks
        emotional_journey = []
        scene_chunk_fingerprints = []
        
        for scene in arc_data["scenes"]:
            scene_number = scene["scene_number"]
            location = scene["location"]
            events = scene["events"]
            
            # Get the actual screenplay text for this scene from vector store
            # Use more specific query including scene number, location, and events
            chunk_query = f"scene {scene_number} {location} {character_name} {' '.join(events[:2])}"
            chunks = vector_store.search(chunk_query, top_k=2)
            
            # Combine up to 2 chunks for more context
            scene_text = "\n\n".join([chunk["text"] for chunk in chunks]) if chunks else ""
            chunk_ids = [c.get("chunk_id") or c.get("id") for c in chunks] if chunks else []
            scene_chunk_fingerprints.append(
                {"scene_number": scene_number, "chunk_ids": chunk_ids, "text_len": len(scene_text)}
            )
            
            # Extract emotional dimensions from the scene text using LLM
            if scene_text and llm_client:
                emotion_prompt = f"""
Analyze the emotional state of {character_name} specifically in Scene {scene_number} at {location}.

Scene {scene_number} context:
Location: {location}
Events happening: {', '.join(events)}

Scene text:
{scene_text}

Focus ONLY on Scene {scene_number}. Analyze how {character_name} is feeling based on the specific events and dialogue in THIS scene.

Return ONLY valid JSON with emotional dimensions on a scale of 0-10:

{{
    "tension": <0-10>,
    "confidence": <0-10>,
    "hope": <0-10>,
    "determination": <0-10>,
    "supporting_text": "<brief quote from THIS SPECIFIC SCENE supporting this analysis>"
}}

If the character is not present in this scene or emotions cannot be determined, return null values.
Be sure to analyze THIS SPECIFIC SCENE (Scene {scene_number}), not other scenes.
"""
                
                try:
                    response = llm_client.chat(
                        messages=[{"role": "user", "content": emotion_prompt}],
                        temperature=0,
                        max_tokens=300
                    )
                    
                    result = response.choices[0].message.content
                    result = result.replace("```json", "").replace("```", "").strip()
                    
                    emotions = json.loads(result)
                    # #region agent log
                    if len(emotional_journey) < 2:
                        _agent_debug_log(
                            "main.py:get_character_arc:llm_emotions",
                            "LLM emotion JSON for scene",
                            {
                                "scene_number": scene_number,
                                "chunk_ids": chunk_ids,
                                "emotions": {
                                    k: emotions.get(k)
                                    for k in ("tension", "confidence", "hope", "determination")
                                },
                                "temperature": 0,
                            },
                            "H3,H4",
                        )
                    # #endregion
                    
                    emotional_journey.append({
                        "scene_number": scene_number,
                        "location": scene["location"],
                        "tension": emotions.get("tension"),
                        "confidence": emotions.get("confidence"),
                        "hope": emotions.get("hope"),
                        "determination": emotions.get("determination"),
                        "supporting_text": emotions.get("supporting_text", ""),
                        "degree_centrality": scene["degree_centrality"],
                        "interaction_frequency": scene["interaction_frequency"]
                    })
                    
                except Exception as e:
                    print(f"Error analyzing emotions for scene {scene_number}: {e}")
                    # Add scene without emotion data
                    emotional_journey.append({
                        "scene_number": scene_number,
                        "location": scene["location"],
                        "tension": None,
                        "confidence": None,
                        "hope": None,
                        "determination": None,
                        "supporting_text": "",
                        "degree_centrality": scene["degree_centrality"],
                        "interaction_frequency": scene["interaction_frequency"]
                    })
        
        # #region agent log
        score_tuples = [
            (s.get("tension"), s.get("confidence"), s.get("hope"), s.get("determination"))
            for s in emotional_journey
        ]
        unique_fingerprints = len(
            {json.dumps(f, sort_keys=True) for f in scene_chunk_fingerprints}
        )
        unique_scores = len(set(score_tuples))
        _agent_debug_log(
            "main.py:get_character_arc:summary",
            "journey summary",
            {
                "journey_len": len(emotional_journey),
                "unique_chunk_fingerprints": unique_fingerprints,
                "unique_score_tuples": unique_scores,
                "score_tuples_sample": score_tuples[:6],
                "chunk_fingerprints_sample": scene_chunk_fingerprints[:4],
            },
            "H2,H3",
        )
        # #endregion
        
        # Extract key motivations from the character's journey
        motivations = []
        if emotional_journey and llm_client:
            # Get all events for context
            all_events = []
            for scene in arc_data["scenes"]:
                all_events.extend(scene["events"])
            
            # Ask LLM to identify key motivations
            motivation_prompt = f"""
Analyze {character_name}'s key motivations throughout the story based on these events:

Events: {', '.join(all_events[:15])}  

Identify 2-3 PRIMARY motivations that drive this character's actions. For each motivation:

Return ONLY valid JSON array:
[
    {{
        "text": "<concise motivation description>",
        "type": "primary",
        "icon": "<emoji that represents this motivation>"
    }}
]

Focus on core drives like: survival, revenge, love, duty, redemption, power, freedom, etc.
Keep descriptions under 50 characters.
"""
            
            try:
                response = llm_client.chat(
                    messages=[{"role": "user", "content": motivation_prompt}],
                    temperature=0.3,
                    max_tokens=400
                )
                
                result = response.choices[0].message.content
                result = result.replace("```json", "").replace("```", "").strip()
                motivations = json.loads(result)
                
            except Exception as e:
                print(f"Error extracting motivations: {e}")
                # Fallback motivations
                motivations = [
                    {"text": "Survive and return home", "type": "primary", "icon": "🏠"},
                    {"text": "Protect loved ones", "type": "primary", "icon": "❤️"}
                ]
        
        return {
            "character_name": character_name,
            "emotional_journey": emotional_journey,
            "total_scenes": arc_data["total_scenes"],
            "total_events": arc_data["total_events"],
            "motivations": motivations
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/pdf-thumbnail/{filename}")
async def get_pdf_thumbnail(filename: str):
    """Generate and return first page of PDF as thumbnail image."""
    try:
        file_path = Path("uploads") / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="PDF not found")
        
        # Open PDF with PyMuPDF
        pdf_document = fitz.open(str(file_path))
        
        # Get first page
        first_page = pdf_document[0]
        
        # Render page to image (higher quality)
        pix = first_page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
        
        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Resize to thumbnail (maintain aspect ratio)
        img.thumbnail((400, 600), Image.Resampling.LANCZOS)
        
        # Save to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=85)
        img_byte_arr.seek(0)
        
        pdf_document.close()
        
        return StreamingResponse(img_byte_arr, media_type="image/jpeg")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/clear")
async def clear_graph():
    """Clear the Neo4j graph database and library."""
    global graph_store, vector_store, current_document
    
    try:
        if graph_store:
            graph_store.clear_graph()
        
        vector_store = VectorStore()
        current_document = None
        
        # Clear library
        save_library([])
        
        return {"success": True, "message": "Graph, vector store, and library cleared"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    # Create necessary directories
    Path("uploads").mkdir(exist_ok=True)
    Path("templates").mkdir(exist_ok=True)
    
    print("=" * 60)
    print("🚀 GraphRAG API Server")
    print("=" * 60)
    print(f"📍 API: http://localhost:8000")
    print(f"📍 Web UI: http://localhost:8000")
    print(f"📍 API Docs: http://localhost:8000/docs")
    print(f"🔗 Neo4j: {config.NEO4J_URI.replace('bolt://', 'http://').replace(':7687', ':7474')}")
    print("=" * 60)
    
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
