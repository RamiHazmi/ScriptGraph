"""
Clear Neo4j database before running updated GraphRAG pipeline.
"""

from neo4j import GraphDatabase

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"

def clear_database():
    """Delete all nodes and relationships."""
    
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )
    
    try:
        with driver.session() as session:
            
            print("Clearing Neo4j database...")
            
            # Delete all nodes and relationships
            result = session.run(
                """
                MATCH (n)
                DETACH DELETE n
                """
            )
            
            print("✓ All nodes and relationships deleted")
            
            # Verify
            count = session.run(
                """
                MATCH (n)
                RETURN count(n) as count
                """
            ).single()["count"]
            
            print(f"✓ Database cleared. Remaining nodes: {count}")
            
    finally:
        driver.close()


if __name__ == "__main__":
    
    print("=" * 50)
    print("Neo4j Database Cleanup")
    print("=" * 50)
    print()
    print("WARNING: This will delete ALL data in Neo4j!")
    print()
    
    confirm = input("Type 'yes' to continue: ").strip().lower()
    
    if confirm == 'yes':
        clear_database()
        print()
        print("=" * 50)
        print("Database cleared successfully!")
        print("You can now run: python graphrag_pdf.py")
        print("=" * 50)
    else:
        print("Cancelled.")
