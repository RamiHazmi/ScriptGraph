"""
Migration script for multi-document support.
Run this if you have existing data that needs document_id properties.
"""

import sys
sys.path.append('..')

from storage import GraphStore
import config

def check_legacy_data():
    """Check if there's any data without document_id."""
    graph_store = GraphStore()
    
    with graph_store.driver.session() as session:
        # Count nodes without document_id
        result = session.run("""
            MATCH (n)
            WHERE n.document_id IS NULL
            RETURN count(n) as count
        """).single()
        
        legacy_count = result["count"] if result else 0
        
        if legacy_count > 0:
            print(f"\n⚠️  Found {legacy_count} nodes without document_id")
            print("   These are from before multi-document support was added.")
            return True
        else:
            print("\n✅ All nodes have document_id property")
            return False

def migrate_to_document(document_id):
    """Migrate all legacy nodes to a specific document_id."""
    graph_store = GraphStore()
    
    print(f"\n🔄 Migrating legacy nodes to document: {document_id}")
    
    with graph_store.driver.session() as session:
        result = session.run("""
            MATCH (n)
            WHERE n.document_id IS NULL
            SET n.document_id = $document_id
            RETURN count(n) as migrated
        """, document_id=document_id)
        
        migrated = result.single()["migrated"]
        print(f"✅ Migrated {migrated} nodes to document '{document_id}'")

def clear_all_data():
    """Clear all data from Neo4j (use with caution!)."""
    graph_store = GraphStore()
    
    print("\n⚠️  WARNING: This will delete ALL data from Neo4j!")
    confirm = input("Type 'DELETE ALL' to confirm: ")
    
    if confirm == "DELETE ALL":
        graph_store.clear_graph()
        print("✅ All data cleared")
    else:
        print("❌ Cancelled")

def list_documents():
    """List all unique document_ids in the graph."""
    graph_store = GraphStore()
    
    with graph_store.driver.session() as session:
        result = session.run("""
            MATCH (n)
            WHERE n.document_id IS NOT NULL
            RETURN DISTINCT n.document_id as doc_id, count(*) as node_count
            ORDER BY doc_id
        """)
        
        print("\n📚 Documents in Neo4j:")
        for record in result:
            print(f"   - {record['doc_id']}: {record['node_count']} nodes")

def backfill_all_fact_ids():
    """Backfill fact_ids on all relationships and event_ids on all Event nodes in Neo4j."""
    graph_store = GraphStore()
    print("\n🔄 Backfilling fact_ids and event_ids across Neo4j...")
    graph_store.ensure_fact_ids()
    print("✅ All facts and events now have deterministic IDs")

def main():
    print("=" * 60)
    print("Multi-Document Migration Tool")
    print("=" * 60)
    
    # Check for legacy data
    has_legacy = check_legacy_data()
    
    # List existing documents
    list_documents()
    
    print("\n" + "=" * 60)
    print("Options:")
    print("=" * 60)
    print("1. Migrate legacy data to a document" if has_legacy else "1. (No legacy data to migrate)")
    print("2. Backfill fact_ids and event_ids on all graph relationships")
    print("3. Clear all data and start fresh")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ")
    
    if choice == "1" and has_legacy:
        doc_id = input("Enter document_id for legacy data (e.g., 'mini'): ")
        migrate_to_document(doc_id)
        list_documents()
    elif choice == "2":
        backfill_all_fact_ids()
    elif choice == "3":
        clear_all_data()
    else:
        print("Exiting")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()

