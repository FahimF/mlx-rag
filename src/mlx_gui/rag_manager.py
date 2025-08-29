"""
Manager for Retrieval-Augmented Generation (RAG) collections.
"""

import logging
import os
from typing import List, Dict, Any, Optional

import chromadb
from tree_sitter import Language, Parser
import tree_sitter_python as tspython

from mlx_gui.database import get_database_manager
from mlx_gui.models import RAGCollection, RAGCollectionStatus

logger = logging.getLogger(__name__)

PYTHON = Language(tspython.language())

class RAGManager:
    """Manages RAG collections."""

    def __init__(self):
        self.db_manager = get_database_manager()
        self.chroma_client = chromadb.Client(
            settings=chromadb.config.Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=os.path.join(self.db_manager.db_dir, "chroma_db"),
            )
        )
        self.parser = Parser()
        self.parser.set_language(PYTHON)

    def create_collection(self, name: str, path: str):
        """Create a new RAG collection."""
        with self.db_manager.get_session() as session:
            collection = RAGCollection(name=name, path=path, status=RAGCollectionStatus.PROCESSING.value)
            session.add(collection)
            session.commit()

        try:
            self._process_folder(collection)
            with self.db_manager.get_session() as session:
                collection.status = RAGCollectionStatus.READY.value
                session.commit()
        except Exception as e:
            logger.error(f"Error creating RAG collection {name}: {e}")
            with self.db_manager.get_session() as session:
                collection.status = RAGCollectionStatus.ERROR.value
                session.commit()

    def _process_folder(self, collection: RAGCollection):
        """Process a folder and add its content to the ChromaDB collection."""
        chroma_collection = self.chroma_client.create_collection(name=collection.name)

        for root, _, files in os.walk(collection.path):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    tree = self.parser.parse(bytes(content, "utf8"))
                    self._extract_chunks(tree.root_node, content, file_path, chroma_collection)

    def _extract_chunks(self, node, text, file_path, chroma_collection):
        """Recursively extract chunks from the syntax tree."""
        if node.type in ["function_definition", "class_definition"]:
            start_byte = node.start_byte
            end_byte = node.end_byte
            chunk_text = text[start_byte:end_byte]
            
            # Get the name of the function/class
            name_node = node.child_by_field_name("name")
            name = text[name_node.start_byte:name_node.end_byte] if name_node else "unknown"

            chroma_collection.add(
                documents=[chunk_text],
                metadatas=[{"source": file_path, "type": node.type, "name": name}],
                ids=[f"{file_path}::{name}"]
            )

        for child in node.children:
            self._extract_chunks(child, text, file_path, chroma_collection)

    def query(self, query: str, collection_name: str) -> str:
        """Query a RAG collection."""
        from mlx_gui.model_manager import get_model_manager
        model_manager = get_model_manager()

        # 1. Get the ChromaDB collection
        try:
            chroma_collection = self.chroma_client.get_collection(name=collection_name)
        except ValueError:
            return f"RAG collection '{collection_name}' not found."

        # 2. Get an embedding model
        # For now, we'll just use the first loaded embedding model
        embedding_model = None
        for model_name, loaded_model in model_manager.get_loaded_models().items():
            if loaded_model.get("config", {}).get("model_type") == "embedding":
                embedding_model = model_manager.get_model_for_inference(model_name)
                break
        
        if not embedding_model:
            return "No embedding model loaded. Please load an embedding model first."

        # 3. Generate an embedding for the query
        query_embedding = embedding_model.mlx_wrapper.generate_embeddings([query])[0]

        # 4. Query ChromaDB
        results = chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )

        # 5. Get a language model
        language_model = None
        for model_name, loaded_model in model_manager.get_loaded_models().items():
            if loaded_model.get("config", {}).get("model_type") == "text":
                language_model = model_manager.get_model_for_inference(model_name)
                break
        
        if not language_model:
            return "No language model loaded. Please load a language model first."

        # 6. Generate a response
        context = "\n".join(results["documents"][0])
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        
        from mlx_gui.mlx_integration import GenerationConfig
        config = GenerationConfig()
        response = language_model.mlx_wrapper.generate(prompt, config)

        return response.text

    def delete_collection(self, name: str):
        """Delete a RAG collection."""
        self.chroma_client.delete_collection(name=name)

    def reprocess_collection(self, name: str):
        """Reprocess a RAG collection."""
        with self.db_manager.get_session() as session:
            collection = session.query(RAGCollection).filter(RAGCollection.name == name).first()
            if not collection:
                return

            try:
                self.chroma_client.delete_collection(name=name)
            except ValueError:
                pass # Collection doesn't exist, that's fine

            collection.status = RAGCollectionStatus.PROCESSING.value
            session.commit()

        try:
            self._process_folder(collection)
            with self.db_manager.get_session() as session:
                collection.status = RAGCollectionStatus.READY.value
                session.commit()
        except Exception as e:
            logger.error(f"Error reprocessing RAG collection {name}: {e}")
            with self.db_manager.get_session() as session:
                collection.status = RAGCollectionStatus.ERROR.value
                session.commit()

# Global RAG manager instance
_rag_manager: Optional[RAGManager] = None

def get_rag_manager() -> RAGManager:
    """Get the global RAG manager instance."""
    global _rag_manager
    if _rag_manager is None:
        _rag_manager = RAGManager()
    return _rag_manager
