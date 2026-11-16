"""
Utility to manage ChromaDB collections
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from chromadb.config import Settings
import config


def list_collections():
    """List all ChromaDB collections."""
    client = chromadb.PersistentClient(
        path=config.CHROMA_PERSIST_DIRECTORY,
        settings=Settings(anonymized_telemetry=False)
    )
    
    collections = client.list_collections()
    
    print("\n" + "=" * 60)
    print("CHROMADB COLLECTIONS")
    print("=" * 60)
    
    if not collections:
        print("No collections found.")
    else:
        for col in collections:
            count = col.count()
            print(f"\n📦 Collection: {col.name}")
            print(f"   Embeddings: {count}")
            print(f"   Metadata: {col.metadata}")
    
    print("\n" + "=" * 60)


def delete_collection(collection_name):
    """Delete a specific collection."""
    client = chromadb.PersistentClient(
        path=config.CHROMA_PERSIST_DIRECTORY,
        settings=Settings(anonymized_telemetry=False)
    )
    
    try:
        client.delete_collection(name=collection_name)
        print(f"✅ Deleted collection: {collection_name}")
    except Exception as e:
        print(f"❌ Error deleting collection: {e}")


def clear_all():
    """Clear all ChromaDB collections."""
    client = chromadb.PersistentClient(
        path=config.CHROMA_PERSIST_DIRECTORY,
        settings=Settings(anonymized_telemetry=False)
    )
    
    collections = client.list_collections()
    
    if not collections:
        print("No collections to clear.")
        return
    
    print(f"\n⚠️  About to delete {len(collections)} collections:")
    for col in collections:
        print(f"   - {col.name} ({col.count()} embeddings)")
    
    confirm = input("\nType 'yes' to confirm: ")
    
    if confirm.lower() == 'yes':
        for col in collections:
            client.delete_collection(name=col.name)
            print(f"✅ Deleted: {col.name}")
        print("\n🗑️  All collections cleared!")
    else:
        print("Cancelled.")


def get_stats():
    """Get ChromaDB statistics."""
    client = chromadb.PersistentClient(
        path=config.CHROMA_PERSIST_DIRECTORY,
        settings=Settings(anonymized_telemetry=False)
    )
    
    collections = client.list_collections()
    total_embeddings = sum(col.count() for col in collections)
    
    print("\n" + "=" * 60)
    print("CHROMADB STATISTICS")
    print("=" * 60)
    print(f"Storage location: {config.CHROMA_PERSIST_DIRECTORY}")
    print(f"Total collections: {len(collections)}")
    print(f"Total embeddings: {total_embeddings}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage ChromaDB collections")
    parser.add_argument("action", choices=["list", "stats", "clear", "delete"],
                        help="Action to perform")
    parser.add_argument("--collection", help="Collection name (for delete action)")
    
    args = parser.parse_args()
    
    if args.action == "list":
        list_collections()
    elif args.action == "stats":
        get_stats()
    elif args.action == "clear":
        clear_all()
    elif args.action == "delete":
        if not args.collection:
            print("❌ --collection required for delete action")
        else:
            delete_collection(args.collection)
