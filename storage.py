"""
Storage layer: Graph Store (Neo4j) + Vector Store (in-memory)
"""

import re
import uuid
import numpy as np
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import config


def _clean_relation(relation):
    return re.sub(r"[^A-Z0-9_]", "_", str(relation).upper()).strip("_")


# ============================================================
# FACT & PATH ID GENERATION
# ============================================================

def generate_fact_id(document_id, source, relation, target):
    """Generate a deterministic, reproducible fact ID for a relationship triple."""
    doc = str(document_id or "default").strip()
    s = str(source or "").upper().strip()
    r = _clean_relation(relation)
    t = str(target or "").upper().strip()
    key = f"{doc}:{s}:{r}:{t}"
    return f"fact_{uuid.uuid5(uuid.NAMESPACE_DNS, key).hex[:12]}"


def generate_event_id(document_id, event_name):
    """Generate a deterministic event ID for an event node."""
    doc = str(document_id or "default").strip()
    e = str(event_name or "").upper().strip()
    key = f"{doc}:{e}"
    return f"evt_{uuid.uuid5(uuid.NAMESPACE_DNS, key).hex[:12]}"


def generate_path_id(document_id, fact_ids, nodes=None, rels=None):
    """Generate a deterministic path ID for a retrieved graph path."""
    doc = str(document_id or "default").strip()
    if fact_ids and all(fact_ids):
        key = f"{doc}:" + "->".join(str(f) for f in fact_ids)
    elif nodes and rels:
        key = f"{doc}:" + "->".join(f"{nodes[i]}:{rels[i]}:{nodes[i+1]}" for i in range(len(rels)))
    else:
        key = f"{doc}:{uuid.uuid4()}"
    return f"path_{uuid.uuid5(uuid.NAMESPACE_DNS, key).hex[:12]}"



# ============================================================
# GRAPH STORE (Neo4j)
# ============================================================

def _norm(expr):
    """Cypher: compare names ignoring case, spaces and underscores (PEN_ELOPE = PENELOPE)."""
    return f"replace(replace(toUpper({expr}), '_', ''), ' ', '')"


def _by_name(var, param):
    """Cypher fragment: match a node by name OR alias, tolerant to case/spaces/underscores."""
    return (f"({_norm(var + '.name')} = {_norm('$' + param)} OR "
            f"ANY(x IN coalesce({var}.aliases, []) WHERE {_norm('x')} = {_norm('$' + param)}))")



class GraphStore:
    """Neo4j knowledge graph storage."""

    def __init__(self):
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
        )

    def get_current_document(self):
        """Get the document_id of the currently loaded graph."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (m:Metadata {type: 'document'})
                RETURN m.document_id as doc_id
                LIMIT 1
            """)
            record = result.single()
            return record["doc_id"] if record else None
    
    def set_document_metadata(self, document_id, title):
        """Set metadata node to track which document is currently in the graph."""
        with self.driver.session() as session:
            session.run("""
                MERGE (m:Metadata {type: 'document'})
                SET m.document_id = $doc_id,
                    m.title = $title,
                    m.loaded_at = datetime()
            """, doc_id=document_id, title=title)

    def document_exists_in_graph(self, document_id):
        """Check if this document already has nodes stored in Neo4j."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n)
                WHERE n.document_id = $document_id
                RETURN count(n) as count
            """, document_id=document_id)
            record = result.single()
            return (record["count"] if record else 0) > 0

    def clear_document(self, document_id):
        """Clear nodes and relationships for ONLY this specific document from Neo4j."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n)
                WHERE n.document_id = $document_id
                DETACH DELETE n
                RETURN count(n) as count
            """, document_id=document_id)
            print(f"Cleared existing graph data for document '{document_id}'")

    def clear_graph(self):
        """Clear all nodes and relationships from Neo4j (all documents)."""
        print("\n" + "=" * 70)
        print("CLEARING OLD GRAPH DATA")
        print("=" * 70)
        print("Removing old nodes and relationships...")

        with self.driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) as count")
            old_count = result.single()["count"]

            if old_count > 0:
                print(f"Found {old_count} existing nodes - deleting...")
                session.run("MATCH (n) DETACH DELETE n")
                print("Old graph data cleared")
            else:
                print("Graph already empty")

        print("=" * 70)

    # ---------------- nodes ----------------

    def create_entity(self, tx, canonical_name, entity_type, document_id, aliases=None):
        """Create entity; aliases from several scenes are merged, not overwritten."""
        if aliases is None:
            aliases = []

        query = """
        MERGE (e:Entity {name: $canonical_name, document_id: $document_id})
        WITH e, coalesce(e.aliases, []) + $aliases AS merged
        SET e.type = $entity_type,
            e.aliases = reduce(acc = [], x IN merged |
                CASE WHEN x IS NULL OR x IN acc THEN acc ELSE acc + x END)
        """
        tx.run(query, canonical_name=canonical_name, entity_type=entity_type, 
               document_id=document_id, aliases=aliases)

    def store_entities(self, entities, document_id):
        """Store multiple entities."""
        with self.driver.session() as session:
            for entity in entities:
                session.execute_write(
                    self.create_entity,
                    entity["canonical_name"],
                    entity.get("type", "Entity"),
                    document_id,
                    entity.get("aliases", [])
                )

        print(f"Stored {len(entities)} entities for document {document_id}")

    def create_event(self, tx, event_name, description, event_type, participants, chunk_id, document_id, location=None):
        """Create event node with deterministic event_id."""
        event_id = generate_event_id(document_id, event_name)
        query = """
        MERGE (e:Event {name: $event_name, document_id: $document_id})
        SET e.event_id = $event_id,
            e.description = $description,
            e.event_type = $event_type,
            e.participants = $participants,
            e.chunk_id = $chunk_id,
            e.location = $location
        """
        tx.run(query, event_name=event_name, description=description,
               event_type=event_type, participants=participants, chunk_id=chunk_id,
               document_id=document_id, location=location, event_id=event_id)

    def store_events(self, events, document_id):
        """Store multiple events as nodes."""
        with self.driver.session() as session:
            for event in events:
                session.execute_write(
                    self.create_event,
                    event["event_name"],
                    event.get("description", ""),
                    event.get("event_type", "action"),
                    event.get("participants", []),
                    event.get("chunk_id", 0),
                    document_id,
                    event.get("location")
                )

        print(f"Stored {len(events)} events for document {document_id}")

    # ---------------- relationships (each returns how many endpoint pairs matched) ----------------

    def create_relationship(self, tx, source, relation, target, document_id):
        """Entity -> Entity. Returns 0 when an endpoint does not exist."""
        relation = _clean_relation(relation)
        if not relation:
            return 0

        fact_id = generate_fact_id(document_id, source, relation, target)
        query = f"""
        MATCH (a:Entity) WHERE a.document_id = $document_id AND {_by_name('a', 'source')}
        MATCH (b:Entity) WHERE b.document_id = $document_id AND a <> b AND {_by_name('b', 'target')}
        MERGE (a)-[r:{relation}]->(b)
        SET r.fact_id = $fact_id,
            r.document_id = $document_id,
            r.source = a.name,
            r.target = b.name
        RETURN count(*) AS n
        """
        record = tx.run(query, source=source, target=target, document_id=document_id, fact_id=fact_id).single()
        return record["n"] if record else 0

    def create_event_chain(self, tx, source, relation, target, document_id):
        """Event -> Event or Event -> Entity. Returns 0 when an endpoint does not exist."""
        relation = _clean_relation(relation)
        if not relation:
            return 0

        fact_id = generate_fact_id(document_id, source, relation, target)
        query = f"""
        MATCH (a:Event) WHERE a.document_id = $document_id AND {_norm('a.name')} = {_norm('$source')}
        MATCH (b) WHERE b.document_id = $document_id AND (b:Event OR b:Entity) AND a <> b AND {_by_name('b', 'target')}
        MERGE (a)-[r:{relation}]->(b)
        SET r.fact_id = $fact_id,
            r.document_id = $document_id,
            r.source = a.name,
            r.target = b.name
        RETURN count(*) AS n
        """
        record = tx.run(query, source=source, target=target, document_id=document_id, fact_id=fact_id).single()
        return record["n"] if record else 0

    def create_entity_event_link(self, tx, entity, relation, event, document_id):
        """Entity -> Event. Returns 0 when an endpoint does not exist."""
        relation = _clean_relation(relation)
        if not relation:
            return 0

        fact_id = generate_fact_id(document_id, entity, relation, event)
        query = f"""
        MATCH (a:Entity) WHERE a.document_id = $document_id AND {_by_name('a', 'entity')}
        MATCH (b:Event) WHERE b.document_id = $document_id AND {_norm('b.name')} = {_norm('$event')}
        MERGE (a)-[r:{relation}]->(b)
        SET r.fact_id = $fact_id,
            r.document_id = $document_id,
            r.source = a.name,
            r.target = b.name
        RETURN count(*) AS n
        """
        record = tx.run(query, entity=entity, event=event, document_id=document_id, fact_id=fact_id).single()
        return record["n"] if record else 0

    def _store_edges(self, label, items, writer, keys, document_id):
        """Write edges one by one and report every edge that was dropped."""
        stored, dropped = 0, []
        with self.driver.session() as session:
            for item in items:
                args = [item[k] for k in keys] + [document_id]
                try:
                    created = session.execute_write(writer, *args)
                except Exception as e:
                    created = 0
                    print(f"  ⚠️  {label} error for {args}: {e}")
                if created:
                    stored += 1
                else:
                    dropped.append(tuple(args))

        print(f"Stored {stored}/{len(items)} {label}")
        if dropped:
            print(f"  ⚠️  Dropped {len(dropped)} {label} (an endpoint was not found in the graph):")
            for d in dropped[:10]:
                print(f"     {d}")
            if len(dropped) > 10:
                print(f"     ... and {len(dropped) - 10} more")
        return stored

    def store_relationships(self, relationships, document_id):
        """Store entity-to-entity relationships."""
        self._store_edges("entity relationships", relationships,
                          self.create_relationship, ("source", "relation", "target"), document_id)
        self.ensure_fact_ids(document_id)

    def store_event_chains(self, event_chains, document_id):
        """Store event-to-event and event-to-entity relationships."""
        self._store_edges("event chains", event_chains,
                          self.create_event_chain, ("source", "relation", "target"), document_id)
        self.link_orphans(document_id)
        self.ensure_fact_ids(document_id)

    def store_entity_event_links(self, entity_event_links, document_id):
        """Store entity-to-event relationships (PERFORMS, EXPERIENCES, WITNESSES)."""
        self._store_edges("entity-event links", entity_event_links,
                          self.create_entity_event_link, ("entity", "relation", "event"), document_id)
        self.link_orphans(document_id)
        self.ensure_fact_ids(document_id)

    # ---------------- safety net for nodes without relationships ----------------

    def ensure_fact_ids(self, document_id=None):
        """Ensure all relationships in Neo4j have a deterministic fact_id, document_id, and endpoints."""
        with self.driver.session() as session:
            where_parts = []
            if document_id:
                where_parts.append("(r.document_id = $document_id OR a.document_id = $document_id)")
            where_parts.append("(r.fact_id IS NULL OR r.document_id IS NULL)")
            where_clause = "WHERE " + " AND ".join(where_parts)

            query = f"""
            MATCH (a)-[r]->(b)
            {where_clause}
            RETURN id(r) AS rel_id, a.name AS source, type(r) AS relation, b.name AS target, 
                   coalesce(r.document_id, a.document_id, $document_id) AS doc_id
            LIMIT 5000
            """
            records = session.run(query, document_id=document_id).data()
            if records:
                for rec in records:
                    doc_id = rec.get("doc_id") or document_id or "default"
                    fact_id = generate_fact_id(doc_id, rec["source"], rec["relation"], rec["target"])
                    session.run("""
                        MATCH ()-[r]->()
                        WHERE id(r) = $rel_id
                        SET r.fact_id = $fact_id,
                            r.document_id = $doc_id,
                            r.source = $source,
                            r.target = $target
                    """, rel_id=rec["rel_id"], fact_id=fact_id, doc_id=doc_id,
                         source=rec["source"], target=rec["target"])

            # Also ensure Event nodes have event_id
            event_where = "WHERE (e.document_id = $document_id AND e.event_id IS NULL)" if document_id else "WHERE e.event_id IS NULL"
            event_query = f"""
            MATCH (e:Event)
            {event_where}
            RETURN id(e) AS eid_internal, e.name AS name, e.document_id AS doc_id
            LIMIT 5000
            """
            event_records = session.run(event_query, document_id=document_id).data()
            if event_records:
                for erec in event_records:
                    doc_id = erec.get("doc_id") or document_id or "default"
                    eid = generate_event_id(doc_id, erec["name"])
                    session.run("""
                        MATCH (e:Event)
                        WHERE id(e) = $id
                        SET e.event_id = $eid
                    """, id=erec["eid_internal"], eid=eid)

    def link_orphans(self, document_id):
        """
        Deterministic safety net, safe to call many times (idempotent):
          1. event -> entities listed in its 'participants'      (PARTICIPATES_IN)
          2. event -> the location of its scene                   (OCCURS_AT)
          3. entity still unlinked -> events whose description names it (MENTIONED_IN)
        Then reports any node that still has no relationship.
        """
        participants_query = f"""
        MATCH (e:Event) WHERE e.document_id = $document_id AND e.participants IS NOT NULL
        UNWIND e.participants AS p
        MATCH (a:Entity)
        WHERE a.document_id = $document_id 
          AND ({_norm('a.name')} = {_norm('p')}
               OR ANY(x IN coalesce(a.aliases, []) WHERE {_norm('x')} = {_norm('p')}))
          AND NOT (a)--(e)
        MERGE (a)-[:PARTICIPATES_IN]->(e)
        RETURN count(*) AS n
        """
        location_query = f"""
        MATCH (e:Event) WHERE e.document_id = $document_id AND e.location IS NOT NULL AND e.location <> ''
        MATCH (l:Entity) WHERE l.document_id = $document_id AND l.type = 'Location' AND {_norm('l.name')} = {_norm('e.location')}
        MERGE (e)-[:OCCURS_AT]->(l)
        RETURN count(*) AS n
        """
        mention_query = """
        MATCH (a:Entity) WHERE a.document_id = $document_id AND NOT (a)--()
        MATCH (e:Event)
        WHERE e.document_id = $document_id
          AND e.description IS NOT NULL
          AND toUpper(e.description) CONTAINS toUpper(replace(a.name, '_', ' '))
        MERGE (a)-[:MENTIONED_IN]->(e)
        RETURN count(*) AS n
        """
        with self.driver.session() as session:
            counts = {}
            for label, query in (("PARTICIPATES_IN", participants_query),
                                 ("OCCURS_AT", location_query),
                                 ("MENTIONED_IN", mention_query)):
                record = session.run(query, document_id=document_id).single()
                counts[label] = record["n"] if record else 0

            orphans = session.run(
                "MATCH (n) WHERE n.document_id = $document_id AND NOT (n)--() RETURN labels(n)[0] AS label, n.name AS name",
                document_id=document_id
            ).data()

        added = {k: v for k, v in counts.items() if v}
        if added:
            print("Safety-net links: " + ", ".join(f"{k} x{v}" for k, v in added.items()))
        if orphans:
            print(f"  ⚠️  {len(orphans)} nodes still have no relationship: "
                  + ", ".join(f"{o['label']}:{o['name']}" for o in orphans[:15]))
        
        # Ensure all safety net links get deterministic fact_ids
        self.ensure_fact_ids(document_id)

    # ---------------- retrieval ----------------

    def search_entities(self, entity_names, document_id):
        """Search paths from given entities (up to 3 hops, through events too) with fact_ids and path_ids."""
        self.ensure_fact_ids(document_id)
        results = []

        with self.driver.session() as session:
            for entity in entity_names:
                cypher_query = f"""
                MATCH path = (start:Entity)-[*1..3]-(connected)
                WHERE start.document_id = $document_id
                  AND {_by_name('start', 'entity')}
                  AND (connected:Entity OR connected:Event)
                  AND connected.document_id = $document_id
                  AND start <> connected

                RETURN
                    [node IN nodes(path) | node.name] AS nodes,
                    [rel IN relationships(path) | type(rel)] AS rels,
                    [rel IN relationships(path) | coalesce(rel.fact_id, '')] AS fact_ids,
                    [rel IN relationships(path) | startNode(rel).name] AS rel_sources,
                    [rel IN relationships(path) | endNode(rel).name] AS rel_targets,
                    relationships(path) AS relObjs

                ORDER BY length(path)
                LIMIT 40
                """

                result = session.run(cypher_query, entity=entity, document_id=document_id)

                for record in result:
                    nodes = record["nodes"]
                    rels = record["rels"]
                    raw_fact_ids = record["fact_ids"]
                    rel_sources = record.get("rel_sources", [])
                    rel_targets = record.get("rel_targets", [])

                    # Construct clean fact_ids and fact detail objects
                    fact_ids = []
                    facts = []
                    for i in range(len(rels)):
                        src = rel_sources[i] if i < len(rel_sources) else nodes[i]
                        tgt = rel_targets[i] if i < len(rel_targets) else nodes[i+1]
                        fid = raw_fact_ids[i] if i < len(raw_fact_ids) and raw_fact_ids[i] else generate_fact_id(document_id, src, rels[i], tgt)
                        fact_ids.append(fid)
                        facts.append({
                            "fact_id": fid,
                            "source": src,
                            "relation": rels[i],
                            "target": tgt
                        })

                    path_id = generate_path_id(document_id, fact_ids, nodes, rels)

                    results.append({
                        "path_id": path_id,
                        "fact_ids": fact_ids,
                        "facts": facts,
                        "nodes": nodes,
                        "relationships": rels,
                        "rel_objects": record["relObjs"]
                    })

        # Remove duplicates
        unique_results = []
        seen = set()

        for result in results:
            key = result["path_id"]
            if key not in seen:
                seen.add(key)
                unique_results.append(result)

        return unique_results

    def get_all_facts(self, document_id):
        """Get all facts (edges with fact_id) in the graph for a document."""
        self.ensure_fact_ids(document_id)
        with self.driver.session() as session:
            result = session.run("""
                MATCH (a)-[r]->(b)
                WHERE r.document_id = $document_id OR a.document_id = $document_id
                RETURN coalesce(r.fact_id, '') AS fact_id,
                       a.name AS source,
                       type(r) AS relation,
                       b.name AS target,
                       labels(a)[0] AS source_type,
                       labels(b)[0] AS target_type
                ORDER BY r.fact_id
            """, document_id=document_id)
            facts = []
            for record in result:
                fid = record["fact_id"] or generate_fact_id(document_id, record["source"], record["relation"], record["target"])
                facts.append({
                    "fact_id": fid,
                    "source": record["source"],
                    "relation": record["relation"],
                    "target": record["target"],
                    "source_type": record["source_type"],
                    "target_type": record["target_type"]
                })
            return facts


    def get_all_characters(self, document_id):
        """Get all character entities from the graph."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n:Entity)
                WHERE n.document_id = $document_id AND n.type = 'Character'
                RETURN n.name as name
                ORDER BY n.name
            """, document_id=document_id)
            return [record["name"] for record in result]

    def get_character_arc_data(self, character_name, document_id):
        """
        Get character arc data for a specific character.
        Returns scene-by-scene analysis with emotional dimensions and graph metrics.
        """
        with self.driver.session() as session:
            # Get all events the character participates in, ordered by scene
            events_query = f"""
                MATCH (char)-[r]->(event:Event)
                WHERE char.document_id = $document_id
                  AND event.document_id = $document_id
                  AND {_by_name('char', 'character_name')}
                RETURN 
                    event.name as event_name,
                    event.description as description,
                    event.event_type as event_type,
                    coalesce(event.chunk_id, 0) as chunk_id,
                    event.location as location,
                    type(r) as relationship_type
                ORDER BY coalesce(event.chunk_id, 0)
            """
            
            events_result = session.run(events_query, character_name=character_name, document_id=document_id)
            events = [dict(record) for record in events_result]
            
            # Fallback: also try matching via participants list on Event node
            if not events:
                events_query2 = f"""
                    MATCH (event:Event)
                    WHERE event.document_id = $document_id
                      AND $character_name IN coalesce(event.participants, [])
                    RETURN
                        event.name as event_name,
                        event.description as description,
                        event.event_type as event_type,
                        coalesce(event.chunk_id, 0) as chunk_id,
                        event.location as location,
                        'PARTICIPANT' as relationship_type
                    ORDER BY coalesce(event.chunk_id, 0)
                """
                events_result2 = session.run(events_query2, character_name=character_name, document_id=document_id)
                events = [dict(record) for record in events_result2]
            
            print(f"[arc] {character_name}: found {len(events)} events")
            
            # Get scene-level graph metrics
            scenes_data = []
            
            if events:
                # Group events by chunk_id — use sequential index if chunk_id is 0/None
                scenes_by_chunk = {}
                for idx, event in enumerate(events):
                    chunk_id = event['chunk_id'] if event['chunk_id'] else (idx + 1)
                    event['chunk_id'] = chunk_id  # normalize
                    if chunk_id not in scenes_by_chunk:
                        scenes_by_chunk[chunk_id] = []
                    scenes_by_chunk[chunk_id].append(event)

                # Calculate metrics for each scene
                for chunk_id in sorted(scenes_by_chunk.keys()):
                    scene_events = scenes_by_chunk[chunk_id]

                    # Simple degree: total connections of this character in graph
                    degree_query = f"""
                        MATCH (char)-[r]-(connected)
                        WHERE char.document_id = $document_id
                          AND {_by_name('char', 'character_name')}
                        RETURN count(DISTINCT connected) as degree
                    """
                    degree_result = session.run(degree_query,
                                              character_name=character_name,
                                              document_id=document_id,
                                              chunk_id=chunk_id)
                    degree_record = degree_result.single()
                    degree = degree_record["degree"] if degree_record else 0
                    
                    # Count interactions in this scene
                    interaction_count = len(scene_events)
                    
                    # Get location for this scene
                    location = scene_events[0]['location'] if scene_events else 'Unknown'
                    
                    scenes_data.append({
                        "scene_number": chunk_id,
                        "location": location,
                        "event_count": len(scene_events),
                        "events": [e['event_name'] for e in scene_events],
                        "degree_centrality": degree,
                        "interaction_frequency": interaction_count
                    })
            
            return {
                "character_name": character_name,
                "scenes": scenes_data,
                "total_scenes": len(scenes_data),
                "total_events": len(events)
            }

    def close(self):
        """Close Neo4j driver."""
        self.driver.close()


# ============================================================
# VECTOR STORE (ChromaDB - Persistent)
# ============================================================

class VectorStore:
    """ChromaDB-backed persistent vector store."""
    
    def __init__(self, document_id=None):
        """
        Initialize ChromaDB client.
        
        Args:
            document_id: Optional document identifier for collection naming
        """
        import chromadb
        from chromadb.config import Settings
        
        # Create persistent client
        self.client = chromadb.PersistentClient(
            path=config.CHROMA_PERSIST_DIRECTORY,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Use document-specific collection if provided
        self.collection_name = f"screenplay_{document_id}" if document_id else "screenplay_default"
        
        # Get or create collection with embedding function
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        # Keep sentence transformer for consistency
        self.model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        
        print(f"📦 ChromaDB initialized: collection '{self.collection_name}'")
    
    def add_chunks(self, chunks):
        """Embed and store chunks in ChromaDB."""
        print("\n==============================")
        print("STEP 4 - BUILDING VECTOR STORE")
        print("==============================")
        
        if not chunks:
            print("⚠️  No chunks to add")
            return
        
        # Check if already stored
        existing_count = self.collection.count()
        if existing_count > 0:
            print(f"ℹ️  Collection already has {existing_count} embeddings")
            print(f"   Clearing and rebuilding...")
            # Clear existing
            self.collection.delete(where={})
        
        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []
        embeddings = []
        
        print(f"Creating embeddings for {len(chunks)} chunks...")
        
        for chunk in chunks:
            # Create embedding using sentence transformer
            embedding = self.model.encode(chunk["text"]).tolist()
            
            documents.append(chunk["text"])
            metadatas.append({
                "chunk_id": chunk["id"],
                "page": chunk["page"],
                "scene_number": chunk.get("scene_number", 0)
            })
            ids.append(f"chunk_{chunk['id']}")
            embeddings.append(embedding)
        
        # Add to ChromaDB in batch
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )
        
        print(f"✅ Stored {len(chunks)} chunk embeddings in ChromaDB.")
        print(f"   Location: {config.CHROMA_PERSIST_DIRECTORY}/{self.collection_name}")
    
    def search(self, query, top_k=5):
        """Search for similar chunks using ChromaDB."""
        if self.collection.count() == 0:
            print("⚠️  Vector store is empty!")
            return []
        
        # Create query embedding
        query_embedding = self.model.encode(query).tolist()
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results to match old interface
        formatted_results = []
        
        if results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                metadata = results['metadatas'][0][i]
                formatted_results.append({
                    "chunk_id": metadata["chunk_id"],
                    "page": metadata["page"],
                    "text": results['documents'][0][i],
                    "score": 1 - results['distances'][0][i]  # Convert distance to similarity
                })
        
        return formatted_results
    
    def collection_exists(self):
        """Check if collection has embeddings."""
        return self.collection.count() > 0
    
    def get_stats(self):
        """Get collection statistics."""
        return {
            "collection_name": self.collection_name,
            "count": self.collection.count(),
            "persist_directory": config.CHROMA_PERSIST_DIRECTORY
        }
    
    def clear(self):
        """Clear all embeddings from collection."""
        self.collection.delete(where={})
        print(f"🗑️  Cleared collection '{self.collection_name}'")
    
    def delete_collection(self):
        """Delete entire collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            print(f"🗑️  Deleted collection '{self.collection_name}'")
        except Exception as e:
            print(f"⚠️  Could not delete collection: {e}")