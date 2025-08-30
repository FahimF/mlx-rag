#!/usr/bin/env python3
"""
Debug script to test RAG functionality directly.
"""

import os
import sys
import logging

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mlx_rag.database import get_database_manager
from mlx_rag.models import RAGCollection
from mlx_rag.rag_manager import get_rag_manager

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():
    """Test RAG functionality."""
    print("🔍 Testing RAG functionality...")
    
    # Initialize database
    db_manager = get_database_manager()
    
    # Check RAG collections in database
    print("\n📊 RAG Collections in Database:")
    with db_manager.get_session() as session:
        collections = session.query(RAGCollection).all()
        for collection in collections:
            print(f"  • {collection.name}: {collection.status} (Path: {collection.path})")
    
    if not collections:
        print("  No RAG collections found in database.")
        return
    
    # Get RAG manager
    rag_manager = get_rag_manager()
    
    # Test querying each collection
    for collection in collections:
        print(f"\n🔍 Testing RAG collection: {collection.name}")
        
        try:
            # Try to get the ChromaDB collection
            chroma_collection = rag_manager.chroma_client.get_collection(name=collection.name)
            print(f"  ✅ ChromaDB collection found")
            
            # Check how many documents are in the collection
            count_result = chroma_collection.count()
            print(f"  📄 Documents in collection: {count_result}")
            
            if count_result == 0:
                print("  ⚠️  No documents found in ChromaDB collection!")
                continue
            
            # Test query
            test_query = "What does the rag-manager.py do?"
            print(f"  🔍 Testing query: '{test_query}'")
            
            results = chroma_collection.query(
                query_texts=[test_query],
                n_results=3
            )
            
            print(f"  📊 Query results:")
            print(f"    • Found {len(results['documents'][0])} relevant documents")
            
            for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                print(f"    • Document {i+1}:")
                print(f"      Source: {metadata.get('source', 'Unknown')}")
                print(f"      Type: {metadata.get('type', 'Unknown')}")
                print(f"      Name: {metadata.get('name', 'Unknown')}")
                print(f"      Content preview: {doc[:200]}...")
                print()
            
            # Test the full RAG query (with language model if available)
            print(f"  🤖 Testing full RAG query...")
            try:
                response = rag_manager.query(test_query, collection.name)
                print(f"  📝 RAG response: {response[:300]}...")
            except Exception as e:
                print(f"  ❌ RAG query failed: {e}")
                
        except ValueError as e:
            print(f"  ❌ ChromaDB collection not found: {e}")
        except Exception as e:
            print(f"  ❌ Error testing collection: {e}")

if __name__ == "__main__":
    main()
