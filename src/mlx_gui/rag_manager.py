"""
Manager for Retrieval-Augmented Generation (RAG) collections.
"""

import logging
import os
import threading
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
        # Configure ChromaDB with a local embedding function to avoid HTTP conflicts
        from chromadb.utils import embedding_functions
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        
        self.chroma_client = chromadb.PersistentClient(
            path=os.path.join(self.db_manager.db_dir, "chroma_db")
        )
        self.parser = Parser()
        self.parser.language = PYTHON

    def create_collection(self, name: str, path: str):
        """Create a new RAG collection."""
        # Create the database entry first
        with self.db_manager.get_session() as session:
            collection = RAGCollection(name=name, path=path, status=RAGCollectionStatus.PROCESSING.value)
            session.add(collection)
            session.commit()
            # Get the collection ID for async processing
            collection_id = collection.id

        # Process the collection in a separate thread to avoid blocking the server
        def process_collection_async():
            logger.info(f"Starting async processing of RAG collection '{name}' (ID: {collection_id})")
            try:
                self._process_folder_by_id(collection_id)
                logger.info(f"Successfully processed RAG collection '{name}'")
                with self.db_manager.get_session() as session:
                    collection = session.query(RAGCollection).get(collection_id)
                    if collection:
                        collection.status = RAGCollectionStatus.READY.value
                        session.commit()
                        logger.info(f"RAG collection '{name}' marked as READY")
            except Exception as e:
                logger.error(f"Error creating RAG collection {name}: {e}", exc_info=True)
                with self.db_manager.get_session() as session:
                    collection = session.query(RAGCollection).get(collection_id)
                    if collection:
                        collection.status = RAGCollectionStatus.ERROR.value
                        session.commit()
                        logger.error(f"RAG collection '{name}' marked as ERROR")

        thread = threading.Thread(target=process_collection_async, daemon=True)
        thread.start()

    def _process_folder_by_id(self, collection_id: int):
        """Process a folder by collection ID."""
        logger.info(f"_process_folder_by_id called with collection_id: {collection_id}")
        with self.db_manager.get_session() as session:
            collection = session.query(RAGCollection).get(collection_id)
            if not collection:
                logger.error(f"Collection with ID {collection_id} not found in database")
                return
            logger.info(f"Found collection: {collection.name}, path: {collection.path}")
            self._process_folder(collection)

    def _process_folder(self, collection: RAGCollection):
        """Process a folder and add its content to the ChromaDB collection."""
        logger.info(f"Creating ChromaDB collection '{collection.name}'")
        try:
            chroma_collection = self.chroma_client.create_collection(
                name=collection.name,
                embedding_function=self.embedding_function
            )
            logger.info(f"ChromaDB collection '{collection.name}' created successfully")
        except Exception as e:
            logger.error(f"Failed to create ChromaDB collection '{collection.name}': {e}")
            raise

        logger.info(f"Starting to scan folder: {collection.path}")
        file_count = 0
        processed_files = 0
        
        # Directories to skip
        skip_dirs = {
            '.venv', 'venv', '.env', 'env',  # Virtual environments
            '__pycache__', '.git', '.hg', '.svn',  # Cache and version control
            'node_modules', '.idea', '.vscode',  # IDE and package manager
            'build', 'dist', '.tox', '.pytest_cache',  # Build artifacts
            '.mypy_cache', '.coverage', 'htmlcov'  # Tool caches
        }
        
        for root, dirs, files in os.walk(collection.path):
            # Filter out directories we want to skip
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            
            for file in files:
                if file.endswith(".py"):
                    file_count += 1
                    file_path = os.path.join(root, file)
                    logger.debug(f"Processing Python file: {file_path}")
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        
                        tree = self.parser.parse(bytes(content, "utf8"))
                        self._extract_chunks(tree.root_node, content, file_path, chroma_collection)
                        processed_files += 1
                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {e}")
        
        logger.info(f"Completed processing folder. Files found: {file_count}, Files processed: {processed_files}")

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

        # 2. Query ChromaDB using its built-in embedding function
        # This uses the same embedding model that was used to store the documents
        results = chroma_collection.query(
            query_texts=[query],
            n_results=5
        )

        # 3. Get a language model
        language_model = None
        for model_name, loaded_model in model_manager.get_loaded_models().items():
            if loaded_model.get("config", {}).get("model_type") == "text":
                language_model = model_manager.get_model_for_inference(model_name)
                break
        
        if not language_model:
            return "No language model loaded. Please load a language model first."

        # 4. Generate a response
        context = "\n".join(results["documents"][0])
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        
        from mlx_gui.mlx_integration import GenerationConfig
        config = GenerationConfig()
        response = language_model.mlx_wrapper.generate(prompt, config)

        return response.text

    def delete_collection(self, name: str):
        """Delete a RAG collection."""
        try:
            self.chroma_client.delete_collection(name=name)
        except chromadb.errors.NotFoundError:
            logger.warning(f"Collection '{name}' not found in ChromaDB for deletion. This can happen if collection creation failed.")

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
