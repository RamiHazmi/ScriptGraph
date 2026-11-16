"""
Retrieval layer: Vector Search + Graph Search + Context Building
"""

import json
import time


# ============================================================
# QUERY ENTITY EXTRACTION
# ============================================================

def extract_query_entities(query, llm_client):
    """Extract entity names from user query using LLM."""
    print("\n" + "=" * 70)
    print("🧠 QUERY ENTITY EXTRACTION - Understanding Your Question")
    print("=" * 70)
    print(f"Query: \"{query}\"")
    print("Method: LLM entity extraction from natural language")

    prompt = f"""
Extract the important entity names from this user question.

Extract entity names EXACTLY as mentioned in the question.

Do not normalize or change the names.

Return ONLY valid JSON.

Format:

{{
    "entities": [
        "entity1",
        "entity2"
    ]
}}

Now extract from this question:

{query}
"""

    response = llm_client.chat(
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=500
    )

    result = response.choices[0].message.content
    if not result:
        print("\n⚠️  No entities extracted")
        return []

    result = result.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(result)
        entities = data.get("entities", [])
        
        if entities:
            print(f"\n📋 Extracted Entities: {entities}")
        else:
            print(f"\n⚠️  No entities found in query")
        print("=" * 70)

        return entities

    except json.JSONDecodeError:
        print("\n⚠️  Failed to parse entities")
        print("=" * 70)
        return []


# ============================================================
# VECTOR SEARCH
# ============================================================

def vector_search(query, vector_store, top_k=5):
    """Search vector store for semantically similar chunks."""
    print("\n" + "=" * 70)
    print("🔍 VECTOR SEARCH - Finding Semantically Similar Text")
    print("=" * 70)
    print(f"Query: \"{query}\"")
    print(f"Method: Embedding cosine similarity")
    
    # Get count from ChromaDB collection
    chunk_count = vector_store.collection.count() if hasattr(vector_store, 'collection') else 0
    print(f"Searching {chunk_count} chunks...")

    results = vector_store.search(query, top_k=top_k)

    print(f"\n📊 Top {top_k} Results from Vector Database:")
    print("-" * 70)

    for i, result in enumerate(results, 1):
        print(
            f"\n[{i}] Chunk {result['chunk_id']} "
            f"(Page {result['page']}) "
            f"| Similarity: {result['score']:.4f}"
        )
        print(f"└─ Preview: {result['text'][:200]}...")

    print("\n" + "=" * 70)
    print(f"✓ Vector search returned {len(results)} chunks")
    print("=" * 70)

    return results


# ============================================================
# GRAPH SEARCH
# ============================================================

def graph_search(query, graph_store, llm_client, document_id):
    """Search knowledge graph for relationship paths."""
    print("\n" + "=" * 70)
    print("🕸️  GRAPH SEARCH - Traversing Knowledge Graph")
    print("=" * 70)
    print(f"Query: \"{query}\"")
    print("Method: Neo4j Cypher queries (2-hop traversal)")

    entities = extract_query_entities(query, llm_client)

    if not entities:
        print("\n⚠️  No entities found - skipping graph search")
        print("=" * 70)
        return []

    print(f"\n🔎 Searching graph for entities: {entities}")
    results = graph_store.search_entities(entities, document_id)

    print(f"\n📊 Graph Paths Retrieved ({len(results)} unique):")
    print("-" * 70)

    if not results:
        print("⚠️  No graph paths found")
    else:
        for i, result in enumerate(results[:10], 1):
            path_id = result.get("path_id", f"path_{i}")
            fact_ids = result.get("fact_ids", [])
            print(f"\n[{i}] [{path_id}] ", end="")
            nodes = result["nodes"]
            relationships = result["relationships"]

            if len(nodes) < 2:
                continue

            for j in range(len(relationships)):
                fid_str = f" ({fact_ids[j]})" if j < len(fact_ids) and fact_ids[j] else ""
                print(f"{nodes[j]} --{relationships[j]}{fid_str}--> ", end="")
            print(nodes[-1])

        if len(results) > 10:
            print(f"\n... and {len(results) - 10} more paths")

    print("\n" + "=" * 70)
    print(f"✓ Graph search returned {len(results)} paths")
    print("=" * 70)

    return results


# ============================================================
# CONTEXT BUILDING
# ============================================================

def build_context(vector_results, graph_results):
    """Combine vector and graph results into unified context."""
    print("\n" + "=" * 70)
    print("🔗 CONTEXT FUSION - Combining Vector + Graph Results")
    print("=" * 70)

    context = ""

    # Text context from vector search
    print(f"\n📝 Adding {len(vector_results)} text chunks from vector search")

    context += """
==============================
TEXT CONTEXT (from Vector DB)
==============================
"""

    for i, result in enumerate(vector_results, 1):
        context += f"""

[Chunk {i} - Page {result['page']} - Similarity: {result['score']:.4f}]

{result['text']}
"""

    # Graph context from graph search
    print(f"🕸️  Adding {len(graph_results)} relationship paths from knowledge graph")

    context += """
==============================
KNOWLEDGE GRAPH CONTEXT
==============================
"""

    for result in graph_results:
        path_id = result.get("path_id", "")
        nodes = result["nodes"]
        relationships = result["relationships"]
        fact_ids = result.get("fact_ids", [])

        if len(nodes) < 2:
            continue

        path_header = f"[PATH: {path_id}] " if path_id else ""
        context += f"\n{path_header}"
        for i in range(len(relationships)):
            fid = fact_ids[i] if i < len(fact_ids) and fact_ids[i] else ""
            fid_str = f" [Fact ID: {fid}]" if fid else ""
            context += f"{nodes[i]} --{relationships[i]}--> {nodes[i+1]}{fid_str} "
        context += "\n"

    context_size = len(context)
    print(f"\n✓ Combined context size: {context_size} characters")
    print("=" * 70)

    return context


# ============================================================
# ANSWER GENERATION
# ============================================================

def generate_answer(query, context, llm_client, graph_results=None):
    """Generate final answer from combined context and identify which paths were used."""
    print("\n" + "=" * 70)
    print("🤖 LLM SYNTHESIS - Generating Final Answer")
    print("=" * 70)
    print("Method: GPT-4o-mini synthesis from combined context")
    print(f"Context provided: Vector chunks + Graph paths")

    prompt = f"""
You are a GraphRAG question-answering assistant.

Answer the user's question using ONLY the retrieved context.

USER QUESTION:

{query}

RETRIEVED CONTEXT:

{context}

Instructions:

1. Answer the question directly and concisely.
2. Use the retrieved text and knowledge graph relationships.
3. Do NOT invent information.
4. If there is not enough information, say that the context is insufficient.

After your answer, identify which knowledge graph paths you actually used by their PATH ID (e.g. "path_...") or path description, and which FACT IDs contributed.

Return your response in this JSON format:

{{
    "answer": "Your answer here",
    "used_paths": [
        "path_id_or_path_description"
    ],
    "used_facts": [
        "fact_id_1"
    ]
}}

IMPORTANT: Only list paths and facts you ACTUALLY used in your answer. Be selective - only include paths that directly contributed to answering the question.
"""

    generation_start = time.time()
    llm_start = time.time()
    
    response = llm_client.chat(
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1000
    )
    
    # Simulated TTFT (first chunk received) - in streaming this would be actual
    ttft = time.time() - llm_start
    
    result = response.choices[0].message.content
    generation_elapsed = time.time() - generation_start
    
    # Parse JSON response
    result = result.replace("```json", "").replace("```", "").strip()
    
    try:
        data = json.loads(result)
        answer = data.get("answer", result)
        used_path_strings = data.get("used_paths", [])
        used_fact_ids = data.get("used_facts", [])
        
        # Match used paths to original graph results
        used_graph_data = []
        if graph_results and (used_path_strings or used_fact_ids):
            print(f"\n🎯 LLM identified {len(used_path_strings)} relevant paths, {len(used_fact_ids)} facts:")
            seen_used_keys = set()

            for result_item in graph_results:
                pid = result_item.get("path_id", "")
                fids = set(result_item.get("fact_ids", []))
                nodes = result_item["nodes"]
                rels = result_item["relationships"]

                reconstructed = ""
                for i in range(len(rels)):
                    reconstructed += f"{nodes[i]} --{rels[i]}--> "
                reconstructed += nodes[-1] if nodes else ""

                matched = False
                for p_str in used_path_strings:
                    p_clean = str(p_str).strip()
                    if pid and (p_clean == pid or p_clean in pid):
                        matched = True
                        break
                    if reconstructed.upper() in p_clean.upper() or p_clean.upper() in reconstructed.upper():
                        matched = True
                        break

                if not matched and used_fact_ids:
                    for fid in used_fact_ids:
                        if fid in fids:
                            matched = True
                            break

                if matched and pid not in seen_used_keys:
                    seen_used_keys.add(pid)
                    used_graph_data.append({
                        "path_id": pid,
                        "fact_ids": result_item.get("fact_ids", []),
                        "nodes": nodes,
                        "relationships": rels
                    })
        
        if not used_graph_data and graph_results:
            # Fallback: if no paths identified, use top paths
            print(f"\n⚠️  No specific paths identified by LLM, using top {min(len(graph_results), 10)} paths")
            used_graph_data = [
                {
                    "path_id": r.get("path_id", ""),
                    "fact_ids": r.get("fact_ids", []),
                    "nodes": r["nodes"],
                    "relationships": r["relationships"]
                }
                for r in graph_results[:10]
            ]
                
    except json.JSONDecodeError:
        # Fallback to plain text answer
        print("\n⚠️  Could not parse JSON response, using plain text")
        answer = result
        used_graph_data = [
            {
                "path_id": r.get("path_id", ""),
                "fact_ids": r.get("fact_ids", []),
                "nodes": r["nodes"],
                "relationships": r["relationships"]
            }
            for r in (graph_results or [])[:10]
        ]

    
    # Token calculations
    prompt_tokens = len(prompt) // 4
    completion_tokens = len(answer) // 4
    total_tokens = prompt_tokens + completion_tokens
    
    # Tokens per second
    tokens_per_second = completion_tokens / generation_elapsed if generation_elapsed > 0 else 0
    
    print("\n✓ Answer generated successfully")
    print(f"✓ {len(used_graph_data)} relevant graph paths identified for highlighting")
    print(f"\n⏱️  GENERATION METRICS:")
    print(f"  Time to first token (TTFT): {ttft:.2f}s")
    print(f"  Total generation time: {generation_elapsed:.2f}s")
    print(f"  Tokens: {total_tokens:,} (prompt: {prompt_tokens:,}, completion: {completion_tokens:,})")
    print(f"  Token generation rate: {tokens_per_second:.1f} tokens/s")
    print("=" * 70)

    return answer, used_graph_data, {
        "ttft": ttft,
        "generation_time": generation_elapsed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "tokens_per_second": tokens_per_second
    }
