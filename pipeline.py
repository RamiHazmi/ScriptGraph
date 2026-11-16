"""
Pipeline orchestration: Indexing + Query pipelines
"""

import re
import time
from validation import LLMValidator


# ============================================================
# ENTITY RESOLUTION
# ============================================================

def normalize_name(name):
    """Normalize entity name for comparison."""
    if not name:
        return ""
    return name.upper().strip().replace("_", " ")


def compute_name_similarity(name1, name2):
    """Conservative name similarity to avoid over-merging."""
    n1 = normalize_name(name1)
    n2 = normalize_name(name2)
    
    if n1 == n2:
        return 1.0
    
    # Known distinct pairs
    distinct_pairs = []
    
    for pair in distinct_pairs:
        if (n1, n2) == pair or (n2, n1) == pair:
            return 0.0
    
    # Substring matching
    if n1 in n2 or n2 in n1:
        longer = max(len(n1), len(n2))
        shorter = min(len(n1), len(n2))
        ratio = shorter / longer
        
        if ratio > 0.95:
            if shorter < 3:
                return 0.0
            return 0.90
        else:
            return 0.0
    
    # Word overlap
    words1 = set(n1.split())
    words2 = set(n2.split())
    
    if not words1 or not words2:
        return 0.0
    
    generic_terms = {
        "THE", "A", "AN", "SON", "DAUGHTER", "MOTHER", "FATHER",
        "CHILD", "BOY", "GIRL", "MAN", "WOMAN", "PERSON"
    }
    
    if words1.issubset(generic_terms) or words2.issubset(generic_terms):
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    jaccard = len(intersection) / len(union)
    
    if jaccard > 0.8:
        return jaccard
    else:
        return 0.0


def resolve_events(extracted_events_by_chunk):
    """Resolve duplicate events across chunks."""
    print("\n==============================")
    print("EVENT RESOLUTION")
    print("==============================")
    
    all_events = []
    for events in extracted_events_by_chunk:
        all_events.extend(events)
    
    print(f"Total events before resolution: {len(all_events)}")
    
    if not all_events:
        return []
    
    # Deduplicate events by event_name
    unique_events = {}
    for event in all_events:
        event_name = event.get("event_name", "").upper().strip()
        if not event_name:
            continue
        
        if event_name not in unique_events:
            unique_events[event_name] = {
                "event_name": event_name,
                "description": event.get("description", ""),
                "participants": event.get("participants", []),
                "event_type": event.get("event_type", "action"),
                "chunk_id": event.get("chunk_id", 0)
            }
        else:
            # Merge participants from duplicate events
            existing_participants = set(unique_events[event_name].get("participants", []))
            new_participants = set(event.get("participants", []))
            unique_events[event_name]["participants"] = list(existing_participants.union(new_participants))
    
    resolved = list(unique_events.values())
    print(f"Events after resolution: {len(resolved)}")
    print(f"Duplicates removed: {len(all_events) - len(resolved)}")
    
    return resolved


def resolve_event_chains(extracted_event_chains_by_chunk, resolved_entities, resolved_events):
    """Map event chains to canonical event/entity names."""
    print("\n==============================")
    print("EVENT CHAIN RESOLUTION")
    print("==============================")
    
    # Build lookups
    alias_to_canonical = {}
    for entity in resolved_entities:
        canonical = entity["canonical_name"]
        for alias in entity.get("aliases", []):
            alias_normalized = normalize_name(alias)
            alias_to_canonical[alias_normalized] = canonical
    
    event_names = {event["event_name"].upper().strip() for event in resolved_events}
    
    all_chains = []
    for chains in extracted_event_chains_by_chunk:
        all_chains.extend(chains)
    
    print(f"Total event chains before resolution: {len(all_chains)}")
    
    resolved_chains = []
    
    for chain in all_chains:
        source = chain.get("source", "").upper().strip()
        target = chain.get("target", "").upper().strip()
        relation = chain.get("relation", "").upper().strip()
        evidence = chain.get("evidence", "")
        chunk_id = chain.get("chunk_id", 0)
        
        if not source or not target or not relation:
            continue
        
        # Source/target could be events or entities
        # Check if they're events first, otherwise try entity resolution
        canonical_source = source if source in event_names else alias_to_canonical.get(
            normalize_name(source),
            source.replace(" ", "_")
        )
        
        canonical_target = target if target in event_names else alias_to_canonical.get(
            normalize_name(target),
            target.replace(" ", "_")
        )
        
        resolved_chains.append({
            "source": canonical_source,
            "relation": relation.replace(" ", "_"),
            "target": canonical_target,
            "evidence": evidence,
            "chunk_id": chunk_id
        })
    
    # Deduplicate
    unique_chains = []
    seen = set()
    
    for chain in resolved_chains:
        key = (chain["source"], chain["relation"], chain["target"])
        if key not in seen:
            seen.add(key)
            unique_chains.append(chain)
    
    print(f"Event chains after resolution: {len(unique_chains)}")
    print(f"Duplicates removed: {len(resolved_chains) - len(unique_chains)}")
    
    return unique_chains


def resolve_entity_event_links(extracted_entity_event_links_by_chunk, resolved_entities, resolved_events):
    """Map entity-event links to canonical names."""
    print("\n==============================")
    print("ENTITY-EVENT LINK RESOLUTION")
    print("==============================")
    
    # Build lookups
    alias_to_canonical = {}
    for entity in resolved_entities:
        canonical = entity["canonical_name"]
        for alias in entity.get("aliases", []):
            alias_normalized = normalize_name(alias)
            alias_to_canonical[alias_normalized] = canonical
    
    event_names = {event["event_name"].upper().strip() for event in resolved_events}
    
    all_links = []
    for links in extracted_entity_event_links_by_chunk:
        all_links.extend(links)
    
    print(f"Total entity-event links before resolution: {len(all_links)}")
    
    resolved_links = []
    
    for link in all_links:
        entity = link.get("entity", "")
        event = link.get("event", "").upper().strip()
        relation = link.get("relation", "").upper().strip()
        chunk_id = link.get("chunk_id", 0)
        
        if not entity or not event or not relation:
            continue
        
        # Resolve entity
        canonical_entity = alias_to_canonical.get(
            normalize_name(entity),
            entity.upper().replace(" ", "_")
        )
        
        # Verify event exists
        if event not in event_names:
            continue
        
        resolved_links.append({
            "entity": canonical_entity,
            "relation": relation.replace(" ", "_"),
            "event": event,
            "chunk_id": chunk_id
        })
    
    # Deduplicate
    unique_links = []
    seen = set()
    
    for link in resolved_links:
        key = (link["entity"], link["relation"], link["event"])
        if key not in seen:
            seen.add(key)
            unique_links.append(link)
    
    print(f"Entity-event links after resolution: {len(unique_links)}")
    print(f"Duplicates removed: {len(resolved_links) - len(unique_links)}")
    
    return unique_links


def resolve_entities(extracted_entities_by_chunk):
    """Resolve duplicate entities across chunks."""
    print("\n==============================")
    print("ENTITY RESOLUTION")
    print("==============================")
    
    all_entities = []
    for entities in extracted_entities_by_chunk:
        all_entities.extend(entities)
    
    print(f"Total entities before resolution: {len(all_entities)}")
    
    # Clean aliases
    DESCRIPTIVE_PATTERNS = [
        "the ", "a ", "an ", "his ", "her ", "their ",
        " mother", " father", " son", " daughter",
        " woman", " man", " boy", " girl", " child",
        "young ", "old ", "little ",
    ]
    
    for entity in all_entities:
        if "aliases" not in entity:
            continue
        
        cleaned_aliases = []
        canonical = entity.get("canonical_name", "").upper()
        
        for alias in entity["aliases"]:
            alias_lower = alias.lower().strip()
            is_descriptive = False
            
            for pattern in DESCRIPTIVE_PATTERNS:
                if pattern in alias_lower:
                    words = alias_lower.split()
                    has_capitalized_word = any(w[0].isupper() for w in alias.split() if w)
                    
                    if not has_capitalized_word or alias_lower.startswith("the ") or alias_lower.startswith("a "):
                        is_descriptive = True
                        print(f"  🗑️  Removing descriptive alias: '{alias}' from {canonical}")
                        break
            
            if not is_descriptive:
                cleaned_aliases.append(alias)
        
        entity["aliases"] = cleaned_aliases
    
    # Group by type
    entities_by_type = {}
    for entity in all_entities:
        entity_type = entity.get("type", "Entity")
        if entity_type not in entities_by_type:
            entities_by_type[entity_type] = []
        entities_by_type[entity_type].append(entity)
    
    resolved = []
    
    # Resolve within each type
    for entity_type, entities in entities_by_type.items():
        merged = [False] * len(entities)
        
        for i in range(len(entities)):
            if merged[i]:
                continue
            
            canonical_name = entities[i].get("canonical_name") or entities[i].get("name")
            all_aliases = set(entities[i].get("aliases", []))
            all_aliases.add(canonical_name)
            
            for j in range(i + 1, len(entities)):
                if merged[j]:
                    continue
                
                name_j = entities[j].get("canonical_name") or entities[j].get("name")
                aliases_j = set(entities[j].get("aliases", []))
                aliases_j.add(name_j)
                
                similarity = compute_name_similarity(canonical_name, name_j)
                alias_overlap = all_aliases.intersection(aliases_j)
                
                should_merge = False
                merge_reason = None
                
                normalized_overlap = {normalize_name(a) for a in alias_overlap}
                generic = {
                    "THE", "A", "AN", "SON", "DAUGHTER", "MOTHER", "FATHER",
                    "CHILD", "BOY", "GIRL", "MAN", "WOMAN", "PERSON"
                }
                meaningful = normalized_overlap - generic
                
                if len(meaningful) > 0:
                    should_merge = True
                    merge_reason = f"exact alias match: {meaningful}"
                elif similarity >= 0.99:
                    should_merge = True
                    merge_reason = f"near-identical names (similarity: {similarity:.3f})"
                
                if should_merge:
                    all_aliases.update(aliases_j)
                    merged[j] = True
                    print(f"  ✓ Merging: {canonical_name} ← {name_j}")
                    print(f"    Reason: {merge_reason}")
                elif similarity > 0.5:
                    print(f"  ✗ NOT merging: {canonical_name} vs {name_j}")
                    print(f"    Similarity: {similarity:.3f} - keeping separate (conservative)")
            
            resolved.append({
                "canonical_name": canonical_name.upper().replace(" ", "_"),
                "type": entity_type,
                "aliases": sorted(list(all_aliases))
            })
    
    print(f"Entities after resolution: {len(resolved)}")
    print(f"Duplicates removed: {len(all_entities) - len(resolved)}")
    
    return resolved


def resolve_relationships(extracted_relationships_by_chunk, resolved_entities):
    """Map relationships to canonical entity names."""
    print("\n==============================")
    print("RELATIONSHIP RESOLUTION")
    print("==============================")
    
    alias_to_canonical = {}
    for entity in resolved_entities:
        canonical = entity["canonical_name"]
        for alias in entity.get("aliases", []):
            alias_normalized = normalize_name(alias)
            alias_to_canonical[alias_normalized] = canonical
    
    SYMMETRIC_RELATIONS = {
        "IS_SIBLING_OF", "TALKS_TO", "MEETS", "KNOWS", "YEARNS_FOR", "HOLDING",
    }
    
    all_relationships = []
    for relationships in extracted_relationships_by_chunk:
        all_relationships.extend(relationships)
    
    print(f"Total relationships before resolution: {len(all_relationships)}")
    
    resolved_relationships = []
    
    for rel in all_relationships:
        source = rel.get("source")
        target = rel.get("target")
        relation = rel.get("relation")
        chunk_id = rel.get("chunk_id", 0)
        
        if not source or not target or not relation:
            continue
        
        source_normalized = normalize_name(source)
        target_normalized = normalize_name(target)
        
        canonical_source = alias_to_canonical.get(
            source_normalized,
            source.upper().replace(" ", "_")
        )
        
        canonical_target = alias_to_canonical.get(
            target_normalized,
            target.upper().replace(" ", "_")
        )
        
        relation_normalized = relation.upper().replace(" ", "_")
        
        resolved_relationships.append({
            "source": canonical_source,
            "relation": relation_normalized,
            "target": canonical_target,
            "chunk_id": chunk_id
        })
    
    # Deduplicate
    unique_rels = []
    seen = set()
    seen_symmetric = set()
    
    for rel in resolved_relationships:
        relation = rel["relation"]
        source = rel["source"]
        target = rel["target"]
        
        if relation in SYMMETRIC_RELATIONS:
            if source < target:
                sym_key = (source, relation, target)
            else:
                sym_key = (target, relation, source)
            
            if sym_key in seen_symmetric:
                continue
            seen_symmetric.add(sym_key)
        
        key = (source, relation, target)
        if key not in seen:
            seen.add(key)
            unique_rels.append(rel)
    
    print(f"Relationships after resolution: {len(unique_rels)}")
    print(f"Duplicates removed: {len(resolved_relationships) - len(unique_rels)}")
    
    return unique_rels


# ============================================================
# INDEXING PIPELINE
# ============================================================

def build_knowledge_graph(chunks, llm_client, graph_store, vector_store, document_id):
    """Build knowledge graph from chunks (extraction → validation → storage)."""
    print("\n==============================")
    print(f"STEP 3 - BUILDING KNOWLEDGE GRAPH FOR DOCUMENT: {document_id}")
    print("==============================")

    graph_build_start = time.time()
    
    # Import here to avoid circular dependency
    from extraction import extract_entities_relationships
    
    # Phase 1: Extract from all chunks
    extraction_start = time.time()
    extracted_entities_by_chunk = []
    extracted_events_by_chunk = []
    extracted_entity_relationships_by_chunk = []
    extracted_event_chains_by_chunk = []
    extracted_entity_event_links_by_chunk = []
    
    total_extraction_time = 0
    total_llm_time = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for chunk in chunks:
        print(
            f"\nProcessing chunk {chunk['id']} "
            f"(Scene {chunk.get('scene_number', '?')}, page {chunk['page']})..."
        )

        extracted = extract_entities_relationships(chunk, llm_client)
        entities = extracted.get("entities", [])
        events = extracted.get("events", [])
        entity_relationships = extracted.get("entity_relationships", [])
        event_chains = extracted.get("event_chains", [])
        entity_event_links = extracted.get("entity_event_links", [])
        metrics = extracted.get("metrics", {})

        print(f"  Entities: {len(entities)}")
        print(f"  Events: {len(events)}")
        print(f"  Entity Relationships: {len(entity_relationships)}")
        print(f"  Event Chains: {len(event_chains)}")
        print(f"  Entity-Event Links: {len(entity_event_links)}")
        
        if metrics:
            print(f"  ⏱️  Extraction: {metrics.get('extraction_time', 0):.2f}s | LLM: {metrics.get('llm_time', 0):.2f}s")
            total_extraction_time += metrics.get('extraction_time', 0)
            total_llm_time += metrics.get('llm_time', 0)
            total_prompt_tokens += metrics.get('prompt_tokens', 0)
            total_completion_tokens += metrics.get('completion_tokens', 0)
        
        # Add chunk_id to all relationships
        for rel in entity_relationships:
            rel["chunk_id"] = chunk["id"]
        for rel in event_chains:
            rel["chunk_id"] = chunk["id"]
        for rel in entity_event_links:
            rel["chunk_id"] = chunk["id"]
        
        # Add chunk_id to events
        for event in events:
            event["chunk_id"] = chunk["id"]

        extracted_entities_by_chunk.append(entities)
        extracted_events_by_chunk.append(events)
        extracted_entity_relationships_by_chunk.append(entity_relationships)
        extracted_event_chains_by_chunk.append(event_chains)
        extracted_entity_event_links_by_chunk.append(entity_event_links)
    
    extraction_elapsed = time.time() - extraction_start
    
    print(f"\n📊 EXTRACTION METRICS:")
    print(f"  Total time: {extraction_elapsed:.2f}s")
    print(f"  LLM time: {total_llm_time:.2f}s ({total_llm_time/extraction_elapsed*100:.1f}%)")
    print(f"  Tokens: {total_prompt_tokens + total_completion_tokens:,} (prompt: {total_prompt_tokens:,}, completion: {total_completion_tokens:,})")
    print(f"  Avg per chunk: {extraction_elapsed/len(chunks):.2f}s")

    # Phase 2: Entity Resolution
    resolution_start = time.time()
    resolved_entities = resolve_entities(extracted_entities_by_chunk)
    resolution_elapsed = time.time() - resolution_start
    print(f"⏱️  Entity resolution time: {resolution_elapsed:.2f}s")
    
    # Phase 2.5: Event Resolution (deduplicate events across chunks)
    event_resolution_start = time.time()
    resolved_events = resolve_events(extracted_events_by_chunk)
    event_resolution_elapsed = time.time() - event_resolution_start
    print(f"⏱️  Event resolution time: {event_resolution_elapsed:.2f}s")
    
    # Phase 3: Relationship Resolution
    resolved_entity_relationships = resolve_relationships(
        extracted_entity_relationships_by_chunk,
        resolved_entities
    )
    
    resolved_event_chains = resolve_event_chains(
        extracted_event_chains_by_chunk,
        resolved_entities,
        resolved_events
    )
    
    resolved_entity_event_links = resolve_entity_event_links(
        extracted_entity_event_links_by_chunk,
        resolved_entities,
        resolved_events
    )
    
    # Phase 3.5: Validation
    validation_start = time.time()
    print("\n" + "=" * 80)
    print("VALIDATION LAYER - Verifying Relationships")
    print("=" * 80)
    
    entity_lookup = {}
    for entity in resolved_entities:
        entity_lookup[entity["canonical_name"]] = {
            "type": entity["type"],
            "aliases": entity.get("aliases", [])
        }
    
    def llm_callable_wrapper(prompt, max_tokens=800, temperature=0):
        """Wrapper for LLMValidator."""
        response = llm_client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    
    validator = LLMValidator(llm_callable_wrapper, model_name=llm_client.get_model_name())
    
    # Validate entity relationships
    print("\n🔍 Validating entity-to-entity relationships...")
    valid_entity_relationships, rejected_entity_relationships = validator.validate_batch(
        resolved_entity_relationships,
        entity_lookup,
        chunks
    )
    
    validator.print_validation_report(valid_entity_relationships, rejected_entity_relationships)
    resolved_entity_relationships = valid_entity_relationships
    
    # Validate event chains
    print("\n🔍 Validating event chains...")
    valid_event_chains, rejected_event_chains = validator.validate_batch(
        resolved_event_chains,
        entity_lookup,
        chunks
    )
    
    validator.print_validation_report(valid_event_chains, rejected_event_chains)
    resolved_event_chains = valid_event_chains
    
    validation_elapsed = time.time() - validation_start
    print(f"\n⏱️  VALIDATION METRICS:")
    print(f"  Total validation time: {validation_elapsed:.2f}s")
    total_validated = len(resolved_entity_relationships) + len(rejected_entity_relationships) + len(resolved_event_chains) + len(rejected_event_chains)
    print(f"  Relationships validated: {total_validated}")
    print(f"  Avg per relationship: {validation_elapsed/total_validated:.2f}s" if total_validated > 0 else "")

    # Phase 4: Store in Neo4j with document_id
    storage_start = time.time()
    print("\n==============================")
    print(f"STORING IN NEO4J (Document: {document_id})")
    print("==============================")

    graph_store.store_entities(resolved_entities, document_id)
    graph_store.store_events(resolved_events, document_id)
    graph_store.store_relationships(resolved_entity_relationships, document_id)
    graph_store.store_event_chains(resolved_event_chains, document_id)
    graph_store.store_entity_event_links(resolved_entity_event_links, document_id)
    graph_store.ensure_fact_ids(document_id)
    
    storage_elapsed = time.time() - storage_start
    graph_build_elapsed = time.time() - graph_build_start
    
    print(f"\n⏱️  STORAGE METRICS:")
    print(f"  Storage time: {storage_elapsed:.2f}s")
    
    print(f"\n📊 TOTAL GRAPH BUILD METRICS:")
    print(f"  Total time: {graph_build_elapsed:.2f}s")
    print(f"  - Extraction: {extraction_elapsed:.2f}s ({extraction_elapsed/graph_build_elapsed*100:.1f}%)")
    print(f"  - Resolution: {resolution_elapsed + event_resolution_elapsed:.2f}s ({(resolution_elapsed + event_resolution_elapsed)/graph_build_elapsed*100:.1f}%)")
    print(f"  - Validation: {validation_elapsed:.2f}s ({validation_elapsed/graph_build_elapsed*100:.1f}%)")
    print(f"  - Storage: {storage_elapsed:.2f}s ({storage_elapsed/graph_build_elapsed*100:.1f}%)")
    print(f"  Total tokens used: {total_prompt_tokens + total_completion_tokens:,}")


# ============================================================
# QUERY PIPELINE
# ============================================================

def ask_graphrag(query, llm_client, graph_store, vector_store, document_id):
    """Complete GraphRAG query pipeline with timing metrics."""
    from retrieval import vector_search, graph_search, build_context, generate_answer
    
    query_start = time.time()
    
    print("\n\n")
    print("█" * 75)
    print("█" + " " * 73 + "█")
    print("█" + "         GRAPHRAG QUERY - REASONING TRACE".center(73) + "█")
    print("█" + " " * 73 + "█")
    print("█" * 75)

    print(f"\n📝 USER QUESTION: {query}")

    # Vector retrieval
    vector_start = time.time()
    vector_results = vector_search(query, vector_store, top_k=5)
    vector_elapsed = time.time() - vector_start

    # Graph retrieval
    graph_start = time.time()
    graph_results = graph_search(query, graph_store, llm_client, document_id)
    graph_elapsed = time.time() - graph_start

    # Combine results
    context_start = time.time()
    context = build_context(vector_results, graph_results)
    context_elapsed = time.time() - context_start

    # Show reasoning summary
    print("\n" + "=" * 70)
    print("📊 REASONING SUMMARY - How the Agent Got This Answer")
    print("=" * 70)
    
    print(f"\n1️⃣  VECTOR DATABASE contributed:")
    print(f"   • {len(vector_results)} semantically similar text chunks")
    print(f"   • Using embedding cosine similarity")
    if vector_results:
        print(f"   • Best match score: {vector_results[0]['score']:.4f}")
    print(f"   ⏱️  Time: {vector_elapsed:.2f}s")
    
    print(f"\n2️⃣  KNOWLEDGE GRAPH contributed:")
    print(f"   • {len(graph_results)} relationship paths")
    print(f"   • Using Cypher 2-hop traversal")
    if graph_results:
        print(f"   • Entities connected to query entities")
    print(f"   ⏱️  Time: {graph_elapsed:.2f}s")
    
    print(f"\n3️⃣  CONTEXT FUSION:")
    print(f"   • Combined vector chunks + graph paths")
    print(f"   • Total context: {len(context)} characters")
    print(f"   ⏱️  Time: {context_elapsed:.2f}s")
    
    print(f"\n4️⃣  LLM SYNTHESIS:")
    print(f"   • Model: {llm_client.get_model_name()}")
    print(f"   • Task: Synthesize answer from combined context")
    print(f"   • Mode: Zero-shot, temperature=0")
    
    print("\n" + "=" * 70)

    # Generate answer
    answer, gen_metrics = generate_answer(query, context, llm_client)
    
    total_query_time = time.time() - query_start

    print("\n\n")
    print("█" * 75)
    print("█" + " " * 73 + "█")
    print("█" + "         FINAL ANSWER".center(73) + "█")
    print("█" + " " * 73 + "█")
    print("█" * 75)
    print()

    print(answer)
    
    print("\n" + "█" * 75)
    print()
    
    # Detailed timing breakdown
    print("📌 ANSWER SOURCES:")
    print(f"   Vector DB: {'✓ YES' if vector_results else '✗ NO'} ({len(vector_results)} chunks)")
    print(f"   Graph DB:  {'✓ YES' if graph_results else '✗ NO'} ({len(graph_results)} paths)")
    
    print(f"\n⏱️  QUERY TIMING BREAKDOWN:")
    print(f"   Vector search:    {vector_elapsed:>6.2f}s ({vector_elapsed/total_query_time*100:>5.1f}%)")
    print(f"   Graph search:     {graph_elapsed:>6.2f}s ({graph_elapsed/total_query_time*100:>5.1f}%)")
    print(f"   Context building: {context_elapsed:>6.2f}s ({context_elapsed/total_query_time*100:>5.1f}%)")
    print(f"   Answer generation:{gen_metrics['generation_time']:>6.2f}s ({gen_metrics['generation_time']/total_query_time*100:>5.1f}%)")
    print(f"   {'─' * 40}")
    print(f"   TOTAL:            {total_query_time:>6.2f}s")
    
    print(f"\n🤖 LLM PERFORMANCE:")
    print(f"   TTFT (Time to First Token): {gen_metrics['ttft']:.2f}s")
    print(f"   Token generation rate: {gen_metrics['tokens_per_second']:.1f} tokens/s")
    print(f"   Total tokens: {gen_metrics['total_tokens']:,} (prompt: {gen_metrics['prompt_tokens']:,}, completion: {gen_metrics['completion_tokens']:,})")
    print()
