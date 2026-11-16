"""
Extraction layer: PDF → Chunks → Entities & Relationships
"""

import re
import json
import time
from pypdf import PdfReader
import config

# ============================================================
# EXTRACTION SETTINGS (edit here)
# ============================================================
MAX_TOKENS_ENTITIES = 4000        # raise for reasoning models (they spend tokens thinking)
MAX_TOKENS_EVENTS = 4500
PARSE_RETRIES = 2                 # extra attempts when the JSON cannot be parsed
USE_JSON_MODE = True              # ask the API for JSON output (ignored if llm.chat does not accept it)
PREVIOUS_EVENTS_CONTEXT = 12      # events from earlier scenes shown to the LLM (cross-scene causality)
ADD_SEQUENCE_EDGES = True         # add HAPPENS_BEFORE between consecutive events (no LLM needed)
ALLOWED_ENTITY_EVENT = {"PERFORMS", "EXPERIENCES", "WITNESSES"}
ALLOWED_EVENT_CHAIN = {"CAUSES", "TRIGGERS", "LEADS_TO", "ENABLES", "PREVENTS", "HAPPENS_BEFORE",
                       "STARTLES", "ANGERS", "SURPRISES"}
META_EVENT_WORDS = ("FADE_IN", "FADE_OUT", "SCENE_", "CUT_TO", "THE_END")   # not story events: dropped


# ============================================================
# PDF READING
# ============================================================

def extract_pdf_text(pdf_path):
    """Extract text from PDF with metadata."""
    print("\n==============================")
    print("STEP 1 - READING PDF")
    print("==============================")

    reader = PdfReader(pdf_path)

    # Extract PDF metadata - handle IndirectObject
    pdf_metadata = {}
    
    if reader.metadata:
        # Get author
        author = reader.metadata.get('/Author', 'Unknown')
        if hasattr(author, 'get_object'):  # It's an IndirectObject
            author = str(author.get_object()) if author else 'Unknown'
        pdf_metadata['author'] = str(author) if author else 'Unknown'
        
        # Get title
        title = reader.metadata.get('/Title', 'Unknown')
        if hasattr(title, 'get_object'):  # It's an IndirectObject
            title = str(title.get_object()) if title else 'Unknown'
        pdf_metadata['title'] = str(title) if title else 'Unknown'
        
        # Get creator
        creator = reader.metadata.get('/Creator', 'Unknown')
        if hasattr(creator, 'get_object'):  # It's an IndirectObject
            creator = str(creator.get_object()) if creator else 'Unknown'
        pdf_metadata['creator'] = str(creator) if creator else 'Unknown'
    else:
        pdf_metadata = {
            'author': 'Unknown',
            'title': 'Unknown',
            'creator': 'Unknown'
        }
    
    # Clean up metadata - remove "IndirectObject" strings
    for key in pdf_metadata:
        if 'IndirectObject' in pdf_metadata[key]:
            pdf_metadata[key] = 'Unknown'
    
    print(f"PDF Metadata (for document info only):")
    print(f"  Author: {pdf_metadata['author']}")
    print(f"  Title: {pdf_metadata['title']}")

    pages = []
    for page_number, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append({
                "page": page_number + 1,
                "text": text
            })

    print(f"Extracted text from {len(pages)} pages.")
    return pages, pdf_metadata


# ============================================================
# CHUNKING
# ============================================================

def is_front_matter_line(line):
    """Detect if a line is front matter (title page, author info, contact)."""
    line_lower = line.lower().strip()
    
    front_matter_patterns = [
        r'written by', r'by\s+[A-Z]', r'draft', r'revision', r'copyright', r'©',
        r'@gmail\.com', r'@[a-z]+\.com', r'\+\d{2,3}\s*\d',
        r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}', r'contact:', r'email:', r'phone:',
        r'version', r'^\s*\d+\s*$', r'fade in', r'title:',
    ]
    
    for pattern in front_matter_patterns:
        if re.search(pattern, line_lower):
            return True
    return False


def detect_scenes(pages):
    """Detect scene boundaries in screenplay format."""
    scenes = []
    current_scene = None
    scene_number = 0
    before_first_scene = True
    front_matter_lines = []
    
    scene_pattern = re.compile(
        r'^(INT\.|EXT\.|INT\./EXT\.)\s+(.+?)\s*[-–—]\s*(.+?)$',
        re.IGNORECASE | re.MULTILINE
    )
    
    for page in pages:
        text = page["text"]
        lines = text.split('\n')
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            match = scene_pattern.match(line_stripped)
            
            if match:
                before_first_scene = False
                if current_scene:
                    scenes.append(current_scene)
                
                scene_number += 1
                int_ext = match.group(1).upper()
                location = match.group(2).strip()
                time = match.group(3).strip()
                
                current_scene = {
                    "scene_number": scene_number,
                    "scene_header": line_stripped,
                    "int_ext": int_ext,
                    "location": location,
                    "time": time,
                    "page": page["page"],
                    "content": [line_stripped]
                }
            else:
                if before_first_scene:
                    if is_front_matter_line(line_stripped):
                        front_matter_lines.append(line_stripped)
                        continue
                    elif len(line_stripped) < 50 and line_stripped.isupper():
                        front_matter_lines.append(line_stripped)
                        continue
                
                if current_scene:
                    current_scene["content"].append(line)
    
    if current_scene:
        scenes.append(current_scene)
    
    if front_matter_lines:
        print(f"\nIgnored {len(front_matter_lines)} front matter lines:")
        for line in front_matter_lines[:5]:
            print(f"  IGNORED: {line[:60]}...")
    
    return scenes


def extract_characters_from_scene(scene_text):
    """Extract character names from scene (ALL CAPS before dialogue)."""
    character_pattern = re.compile(r'^\s*([A-Z][A-Z\s\.]+?)\s*$', re.MULTILINE)
    matches = character_pattern.findall(scene_text)
    
    excluded = {
        'INT', 'EXT', 'FADE IN', 'FADE OUT', 'CUT TO', 'DISSOLVE TO',
        'CONTINUED', 'THE END', 'END', 'TITLE', 'CREDITS'
    }
    
    characters = set()
    for match in matches:
        name = match.strip()
        if (len(name) > 1 and name not in excluded and
            not name.startswith('INT') and not name.startswith('EXT')):
            characters.add(name)
    
    return list(characters)


def chunk_text(pages, max_chunk_size=None, min_chunk_size=None):
    """Scene-aware chunking for screenplay content."""
    if max_chunk_size is None:
        max_chunk_size = config.MAX_CHUNK_SIZE
    if min_chunk_size is None:
        min_chunk_size = config.MIN_CHUNK_SIZE
    
    print("\n==============================")
    print("STEP 2 - SCENE-BASED CHUNKING")
    print("==============================")

    scenes = detect_scenes(pages)
    print(f"Detected {len(scenes)} scenes")

    chunks = []
    for scene in scenes:
        scene_text = '\n'.join(
            line.strip() for line in scene["content"] if line.strip()
        )
        
        characters = extract_characters_from_scene(scene_text)
        
        metadata = {
            "scene_number": scene["scene_number"],
            "scene_header": scene["scene_header"],
            "int_ext": scene["int_ext"],
            "location": scene["location"],
            "time": scene["time"],
            "page": scene["page"],
            "characters": characters
        }
        
        if len(scene_text) <= max_chunk_size:
            chunks.append({"id": len(chunks), "text": scene_text, **metadata})
        else:
            print(f"  Scene {scene['scene_number']} too large ({len(scene_text)} chars), splitting...")
            
            # Split by paragraphs (double newline or single newline for dialogue)
            lines = scene["content"]
            current_chunk_text = ""
            part_number = 1
            
            for line in lines:
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                
                # Check if adding this line would exceed max size
                if len(current_chunk_text) + len(line_stripped) + 1 > max_chunk_size:
                    # Only create chunk if we have meaningful content
                    if len(current_chunk_text) >= min_chunk_size:
                        chunk_metadata = metadata.copy()
                        chunk_metadata["scene_part"] = part_number
                        chunks.append({
                            "id": len(chunks),
                            "text": current_chunk_text.strip(),
                            **chunk_metadata
                        })
                        current_chunk_text = line_stripped
                        part_number += 1
                    else:
                        # Chunk too small, keep adding
                        current_chunk_text += "\n" + line_stripped
                else:
                    # Add line to current chunk
                    if current_chunk_text:
                        current_chunk_text += "\n" + line_stripped
                    else:
                        current_chunk_text = line_stripped
            
            # Don't forget the last chunk!
            if current_chunk_text.strip():
                chunk_metadata = metadata.copy()
                chunk_metadata["scene_part"] = part_number
                chunks.append({
                    "id": len(chunks),
                    "text": current_chunk_text.strip(),
                    **chunk_metadata
                })
                print(f"    Created {part_number} parts for scene {scene['scene_number']}")

    print(f"Created {len(chunks)} scene-based chunks")
    
    scenes_with_parts = sum(1 for c in chunks if "scene_part" in c)
    if scenes_with_parts > 0:
        print(f"  {scenes_with_parts} chunks from split scenes")
    
    # Debug: show chunk distribution
    print(f"  Chunk sizes: min={min(len(c['text']) for c in chunks)}, max={max(len(c['text']) for c in chunks)}, avg={sum(len(c['text']) for c in chunks)//len(chunks)}")

    return chunks


# ============================================================
# ENTITY & RELATIONSHIP EXTRACTION
# ============================================================

# Events from earlier chunks, so a scene can be linked to what caused it
_EVENT_MEMORY = []
# Entity names from earlier chunks, so a scene can refer to characters introduced before
_ENTITY_MEMORY = set()


def _key(name):
    """Comparison key: ignores case, spaces, underscores (PEN_ELOPE == PENELOPE)."""
    return re.sub(r"[^A-Z0-9]", "", str(name or "").upper())


def _repair_split_names(event_name, entity_names):
    """Fix names the LLM split with underscores: PEN_ELOPE_TESTS_ODYSSEUS -> PENELOPE_TESTS_ODYSSEUS."""
    for ent in entity_names:
        if ent.isalpha() and len(ent) >= 4:
            pattern = r"(?<![A-Z])" + "_?".join(ent) + r"(?![A-Z])"
            event_name = re.sub(pattern, ent, event_name)
    return event_name


def _norm_event(name):
    """EVENT names: upper case, words joined with underscores."""
    name = re.sub(r"[^A-Za-z0-9]+", "_", str(name or "")).strip("_").upper()
    return name


def _clean_json(text):
    """Turn an LLM reply into a JSON string (handles None, <think>, fences, extra text)."""
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.replace("```json", "").replace("```", "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end >= start:
        text = text[start:end + 1]
    return text


def _call_json(llm_client, prompt, max_tokens, label):
    """Call the LLM and return (data_dict, seconds, raw_text). Retries on bad JSON."""
    total = 0.0
    raw = ""
    for attempt in range(PARSE_RETRIES + 1):
        t0 = time.time()
        messages = [{"role": "user", "content": prompt}]
        try:
            if USE_JSON_MODE:
                response = llm_client.chat(messages=messages, temperature=0, max_tokens=max_tokens,
                                           response_format={"type": "json_object"})
            else:
                response = llm_client.chat(messages=messages, temperature=0, max_tokens=max_tokens)
        except TypeError:
            # llm.chat does not accept response_format: fall back to a plain call
            response = llm_client.chat(messages=messages, temperature=0, max_tokens=max_tokens)
        total += time.time() - t0

        raw = _clean_json(response.choices[0].message.content)
        try:
            return json.loads(raw), total, raw
        except json.JSONDecodeError as e:
            print(f"\n⚠️  Could not parse {label} JSON (attempt {attempt + 1}/{PARSE_RETRIES + 1}): {e}")
            print(f"Raw output preview: {raw[:200]}")
    return {}, total, raw


def extract_entities_relationships(chunk, llm_client):
    """
    Extract entities and relationships from a chunk using TWO LLM calls.

    Call 1: Entities + Entity-to-Entity relationships
    Call 2: Events + Event chains + Entity-Event links
            (sees the entity list from call 1 and the events of earlier scenes)
    """
    global _EVENT_MEMORY, _ENTITY_MEMORY
    start_time = time.time()

    if chunk.get("id", 0) == 0:
        _EVENT_MEMORY = []          # new document: forget old events and entities
        _ENTITY_MEMORY = set()

    metadata_context = f"""
Scene Metadata:
- Scene Number: {chunk.get('scene_number', 'N/A')}
- Scene Header: {chunk.get('scene_header', 'N/A')}
- Location: {chunk.get('location', 'N/A')}
- Time: {chunk.get('time', 'N/A')}
- Page: {chunk.get('page', 'N/A')}
- Characters Present: {', '.join(chunk.get('characters', [])) if chunk.get('characters') else 'N/A'}
"""

    # ========================================
    # CALL 1: ENTITIES + ENTITY RELATIONSHIPS
    # ========================================

    prompt_entities = f"""
Extract ONLY THE MOST IMPORTANT entities and relationships from this scene.

{metadata_context}

IMPORTANT: BE HIGHLY SELECTIVE! Only extract named entities that are:
1. Main or significant supporting characters
2. Important locations (not every room/hallway)
3. Plot-significant objects

ENTITY EXTRACTION RULES:
1. ONLY extract NAMED entities (characters, locations, objects with proper names)
2. DO NOT extract: pronouns (he, she, it), generic descriptions (the woman, the man)
3. DO NOT extract: Minor background characters with no speaking role or plot impact
4. DO NOT extract: Generic locations (a room, the hallway, outside) - only named places
5. For aliases, only include actual name variations
6. canonical_name must be ONE upper-case name (e.g. ODYSSEUS)

CHARACTER FILTERING:
- Extract: Main characters, recurring characters, characters who drive plot
- Skip: Extras, crowd members, unnamed servants, generic "soldier #3"

LOCATION FILTERING:
- Extract: Named places (THE_PALACE, ITHACA, CYCLOPS_CAVE)
- Skip: Generic places (a room, the street, outside)

OBJECT FILTERING:
- Extract: Plot-important items (SWORD, MAGIC_BOW, IMPORTANT_LETTER)
- Skip: Props, everyday items (cup, chair, table)

ENTITY-TO-ENTITY RELATIONSHIPS:
- Extract ONLY relationships that are story-important
- Use SPECIFIC VERBS or roles from the text (KILLS, EATS, WIFE_OF, SON_OF, CURSES, HELPS...)
- Source and target must be entities from your "entities" list

Return ONLY valid JSON:

{{
    "entities": [
        {{
            "canonical_name": "CHARACTER_NAME_IN_CAPS",
            "type": "Character" | "Location" | "Object",
            "aliases": ["ActualName", "Variation"]
        }}
    ],
    "entity_relationships": [
        {{
            "source": "ENTITY1",
            "relation": "SPECIFIC_VERB",
            "target": "ENTITY2"
        }}
    ]
}}

Scene content:

{chunk['text']}
"""

    data_1, llm_time_1, result_1 = _call_json(llm_client, prompt_entities, MAX_TOKENS_ENTITIES, "entities (Call 1)")
    entities = data_1.get("entities", []) or []
    entity_relationships = data_1.get("entity_relationships", []) or []

    # ========================================
    # POST-PROCESS ENTITIES
    # ========================================

    filtered_entities = []
    PRONOUNS = {"HE", "SHE", "IT", "THEY", "THEM", "HER", "HIS", "THEIR", "I", "YOU", "WE", "US"}
    GENERIC_DESCRIPTORS = {
        "WOMAN", "MAN", "BOY", "GIRL", "CHILD", "PERSON", "PEOPLE",
        "SON", "DAUGHTER", "MOTHER", "FATHER", "FAMILY"
    }
    STOP_WORDS = {"NO", "YES", "OK", "OH", "AH", "WHY", "WHAT", "WELL", "PLEASE", "CONTINUED", "VO", "OS", "BEAT", "NONE"}

    for entity in entities:
        canonical = str(entity.get("canonical_name", "")).upper().strip()
        if not canonical or len(canonical) <= 1 or canonical in PRONOUNS or canonical in GENERIC_DESCRIPTORS or canonical in STOP_WORDS or canonical.isdigit():
            continue
        entity["canonical_name"] = canonical

        if "aliases" in entity:
            cleaned_aliases = []
            for alias in entity["aliases"]:
                if not any(desc in alias.upper() for desc in ["THE ", "HER ", "HIS ", "THEIR "]):
                    cleaned_aliases.append(alias)
            entity["aliases"] = cleaned_aliases

        filtered_entities.append(entity)

    # The scene location is always an entity, so events can be linked to it (OCCURS_AT)
    scene_location = _norm_event(chunk.get("location", ""))
    if scene_location and not any(_key(e["canonical_name"]) == _key(scene_location) for e in filtered_entities):
        filtered_entities.append({
            "canonical_name": scene_location,
            "type": "Location",
            "aliases": [str(chunk.get("location", "")).title()],
        })

    entity_names = {e["canonical_name"] for e in filtered_entities}
    canon = {_key(n): n for n in _ENTITY_MEMORY}
    canon.update({_key(n): n for n in entity_names})          # this scene wins on conflicts

    def resolve(name):
        """Map any spelling to the canonical entity name (or None if unknown)."""
        return canon.get(_key(name))

    cleaned_rels, seen_rels = [], set()
    for r in entity_relationships:
        src, tgt = resolve(r.get("source")), resolve(r.get("target"))
        rel = str(r.get("relation", "")).upper().strip()
        if src and tgt and src != tgt and rel and (src, rel, tgt) not in seen_rels:
            seen_rels.add((src, rel, tgt))
            cleaned_rels.append({"source": src, "relation": rel, "target": tgt})
    entity_relationships = cleaned_rels

    # ========================================
    # CALL 2: EVENTS + EVENT CHAINS
    # ========================================

    previous_events = _EVENT_MEMORY[-PREVIOUS_EVENTS_CONTEXT:]
    previous_block = "\n".join(f"- {e}" for e in previous_events) if previous_events else "(none, this is the start)"

    prompt_events = f"""
Extract ONLY THE MOST SIGNIFICANT events and chains from this scene.

{metadata_context}

ENTITIES YOU MUST USE (exact names): {', '.join(sorted(set(canon.values()))) if canon else 'N/A'}

PREVIOUS EVENTS (from earlier scenes, you may link to them):
{previous_block}

EVENT EXTRACTION - BE SELECTIVE!
- Extract ONLY 2-5 MAJOR events per scene (not every tiny action)
- Focus on plot-advancing moments: conflicts, decisions, discoveries, arrivals/departures
- Skip: minor actions (walking, looking, sitting), dialogue beats, reactions
- Format: SUBJECT_VERB_OBJECT in upper case (e.g., POLYPHEMUS_EATS_CREW, ODYSSEUS_BLINDS_CYCLOPS)

MAJOR EVENTS TO EXTRACT:
✓ Violence/conflict (fights, kills, attacks)
✓ Discoveries (finds item, learns secret)
✓ Decisions (chooses path, makes plan)
✓ Arrivals/Departures (enters location, leaves place)
✓ State changes (becomes king, dies, transforms)
✓ Key dialogue (reveals truth, makes promise)

SKIP THESE MINOR EVENTS:
✗ Movement (walks to, looks at, turns around)
✗ Reactions (smiles, frowns, thinks)
✗ Filler dialogue (greets, says hello)
✗ Props (picks up cup, opens door)

EVENT CHAINS (Causality) - ONLY MEANINGFUL CONNECTIONS:
- CAUSES: Event A directly causes Event B (clear causation)
- TRIGGERS: Event A initiates Event B
- LEADS_TO: Event A results in Event B
- ENABLES: Event A makes Event B possible
- HAPPENS_BEFORE: Important temporal sequence
RULES FOR CHAINS:
- source and target must be EXACT event_name values
- A current event may be caused by a PREVIOUS event (cross-scene cause): add that link
- Only add chains with clear story importance

ENTITY-EVENT LINKS:
- PERFORMS: Entity performs the event (the doer)
- EXPERIENCES: Entity is affected by the event (victim, target)
- WITNESSES: Only for plot-important witnesses
- Use only entities from the list above

Return ONLY valid JSON:

{{
    "events": [
        {{
            "event_name": "SUBJECT_VERB_OBJECT",
            "description": "Brief description",
            "participants": ["ENTITY1", "ENTITY2"],
            "event_type": "action" | "state_change" | "dialogue" | "arrival" | "departure" | "death"
        }}
    ],
    "event_chains": [
        {{
            "source": "EVENT1_NAME",
            "relation": "CAUSES" | "TRIGGERS" | "LEADS_TO" | "ENABLES" | "HAPPENS_BEFORE",
            "target": "EVENT2_NAME or ENTITY_NAME",
            "evidence": "Brief quote from text"
        }}
    ],
    "entity_event_links": [
        {{
            "entity": "ENTITY_NAME",
            "relation": "PERFORMS" | "EXPERIENCES" | "WITNESSES",
            "event": "EVENT_NAME"
        }}
    ]
}}

Scene content:

{chunk['text']}
"""

    data_2, llm_time_2, result_2 = _call_json(llm_client, prompt_events, MAX_TOKENS_EVENTS, "events (Call 2)")
    events = data_2.get("events", []) or []
    event_chains = data_2.get("event_chains", []) or []
    entity_event_links = data_2.get("entity_event_links", []) or []

    # ========================================
    # POST-PROCESS EVENTS (names, relations, dangling links)
    # ========================================

    clean_events, seen_events = [], set()
    for ev in events:
        name = _repair_split_names(_norm_event(ev.get("event_name")), set(canon.values()))
        if not name or name in seen_events or any(w in name for w in META_EVENT_WORDS):
            continue
        seen_events.add(name)
        ev["event_name"] = name
        parts = []
        for p in ev.get("participants", []) or []:
            rp = resolve(p)
            if rp and rp not in parts:
                parts.append(rp)
        ev["participants"] = parts
        ev["location"] = scene_location
        clean_events.append(ev)

    known_events = seen_events | set(_EVENT_MEMORY)

    clean_chains = []
    for ch in event_chains:
        rel = str(ch.get("relation", "")).upper().strip()
        src = _repair_split_names(_norm_event(ch.get("source")), set(canon.values()))
        tgt_raw = str(ch.get("target", "")).strip()
        tgt = _repair_split_names(_norm_event(tgt_raw), set(canon.values()))
        if rel not in ALLOWED_EVENT_CHAIN or src not in known_events:
            continue
        if tgt in known_events:
            target = tgt
        elif resolve(tgt_raw):
            target = resolve(tgt_raw)
        else:
            continue
        clean_chains.append({"source": src, "relation": rel, "target": target,
                             "evidence": ch.get("evidence", "")})

    clean_links, seen_links = [], set()
    for lk in entity_event_links:
        ent = resolve(lk.get("entity"))
        rel = str(lk.get("relation", "")).upper().strip()
        evn = _repair_split_names(_norm_event(lk.get("event")), set(canon.values()))
        if ent and rel in ALLOWED_ENTITY_EVENT and evn in seen_events and (ent, rel, evn) not in seen_links:
            seen_links.add((ent, rel, evn))
            clean_links.append({"entity": ent, "relation": rel, "event": evn})
    # one entity should not both PERFORM and EXPERIENCE the same event: PERFORMS wins
    clean_links = [l for l in clean_links
                   if not (l["relation"] == "EXPERIENCES" and (l["entity"], "PERFORMS", l["event"]) in seen_links)]

    # Deterministic temporal backbone: previous event -> first event -> ... -> last event
    if ADD_SEQUENCE_EDGES and clean_events:
        ordered = [e["event_name"] for e in clean_events]
        if _EVENT_MEMORY:
            ordered = [_EVENT_MEMORY[-1]] + ordered
        existing = {(c["source"], c["target"]) for c in clean_chains}
        for a_, b_ in zip(ordered, ordered[1:]):
            if a_ != b_ and (a_, b_) not in existing:
                clean_chains.append({"source": a_, "relation": "HAPPENS_BEFORE", "target": b_,
                                     "evidence": "scene order"})

    _ENTITY_MEMORY.update(entity_names)
    for ev in clean_events:
        if ev["event_name"] not in _EVENT_MEMORY:
            _EVENT_MEMORY.append(ev["event_name"])

    # Calculate total metrics
    total_time = time.time() - start_time
    total_llm_time = llm_time_1 + llm_time_2
    prompt_tokens = (len(prompt_entities) + len(prompt_events)) // 4
    completion_tokens = (len(result_1) + len(result_2)) // 4

    return {
        "entities": filtered_entities,
        "events": clean_events,
        "entity_relationships": entity_relationships,
        "event_chains": clean_chains,
        "entity_event_links": clean_links,
        "metrics": {
            "extraction_time": total_time,
            "llm_time": total_llm_time,
            "llm_calls": 2,
            "call_1_time": llm_time_1,
            "call_2_time": llm_time_2,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }