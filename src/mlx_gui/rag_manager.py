"""
Manager for Retrieval-Augmented Generation (RAG) collections.
"""

import logging
import os
import threading
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import chromadb
from tree_sitter import Language, Parser

# Import available tree-sitter languages
try:
    import tree_sitter_python as tspython
    PYTHON = Language(tspython.language())
except ImportError:
    PYTHON = None
    logging.warning("tree-sitter-python not available")

try:
    import tree_sitter_javascript as tsjavascript
    JAVASCRIPT = Language(tsjavascript.language())
except ImportError:
    JAVASCRIPT = None
    logging.debug("tree-sitter-javascript not available")

try:
    import tree_sitter_typescript as tstypescript
    TYPESCRIPT = Language(tstypescript.language_typescript())
    TSX = Language(tstypescript.language_tsx())
except ImportError:
    TYPESCRIPT = None
    TSX = None
    logging.debug("tree-sitter-typescript not available")

try:
    import tree_sitter_java as tsjava
    JAVA = Language(tsjava.language())
except ImportError:
    JAVA = None
    logging.debug("tree-sitter-java not available")

try:
    import tree_sitter_cpp as tscpp
    CPP = Language(tscpp.language())
except ImportError:
    CPP = None
    logging.debug("tree-sitter-cpp not available")

try:
    import tree_sitter_c as tsc
    C = Language(tsc.language())
except ImportError:
    C = None
    logging.debug("tree-sitter-c not available")

try:
    import tree_sitter_go as tsgo
    GO = Language(tsgo.language())
except ImportError:
    GO = None
    logging.debug("tree-sitter-go not available")

try:
    import tree_sitter_rust as tsrust
    RUST = Language(tsrust.language())
except ImportError:
    RUST = None
    logging.debug("tree-sitter-rust not available")

try:
    import tree_sitter_bash as tsbash
    BASH = Language(tsbash.language())
except ImportError:
    BASH = None
    logging.debug("tree-sitter-bash not available")

from mlx_gui.database import get_database_manager
from mlx_gui.models import RAGCollection, RAGCollectionStatus

logger = logging.getLogger(__name__)

class RAGManager:
    """Manages RAG collections with multi-language support."""

    def __init__(self):
        self.db_manager = get_database_manager()
        # Configure ChromaDB with a local embedding function to avoid HTTP conflicts
        from chromadb.utils import embedding_functions
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        
        self.chroma_client = chromadb.PersistentClient(
            path=os.path.join(self.db_manager.db_dir, "chroma_db")
        )
        
        # Language configuration mapping file extensions to parsers and patterns
        self.language_config = {
            '.py': {
                'language': PYTHON,
                'parser': Parser() if PYTHON else None,
                'name': 'Python',
                'node_types': ['function_definition', 'class_definition', 'async_function_definition'],
                'name_field': 'name'
            },
            '.js': {
                'language': JAVASCRIPT,
                'parser': Parser() if JAVASCRIPT else None,
                'name': 'JavaScript',
                'node_types': ['function_declaration', 'function_expression', 'arrow_function', 'class_declaration', 'method_definition'],
                'name_field': 'name'
            },
            '.jsx': {
                'language': JAVASCRIPT,
                'parser': Parser() if JAVASCRIPT else None,
                'name': 'JavaScript React',
                'node_types': ['function_declaration', 'function_expression', 'arrow_function', 'class_declaration', 'method_definition'],
                'name_field': 'name'
            },
            '.ts': {
                'language': TYPESCRIPT,
                'parser': Parser() if TYPESCRIPT else None,
                'name': 'TypeScript',
                'node_types': ['function_declaration', 'function_signature', 'method_signature', 'class_declaration', 'interface_declaration', 'type_alias_declaration'],
                'name_field': 'name'
            },
            '.tsx': {
                'language': TSX,
                'parser': Parser() if TSX else None,
                'name': 'TypeScript React',
                'node_types': ['function_declaration', 'function_signature', 'method_signature', 'class_declaration', 'interface_declaration', 'type_alias_declaration'],
                'name_field': 'name'
            },
            '.java': {
                'language': JAVA,
                'parser': Parser() if JAVA else None,
                'name': 'Java',
                'node_types': ['class_declaration', 'interface_declaration', 'method_declaration', 'constructor_declaration'],
                'name_field': 'name'
            },
            '.cpp': {
                'language': CPP,
                'parser': Parser() if CPP else None,
                'name': 'C++',
                'node_types': ['function_definition', 'class_specifier', 'struct_specifier', 'namespace_definition'],
                'name_field': 'name'
            },
            '.cc': {
                'language': CPP,
                'parser': Parser() if CPP else None,
                'name': 'C++',
                'node_types': ['function_definition', 'class_specifier', 'struct_specifier', 'namespace_definition'],
                'name_field': 'name'
            },
            '.cxx': {
                'language': CPP,
                'parser': Parser() if CPP else None,
                'name': 'C++',
                'node_types': ['function_definition', 'class_specifier', 'struct_specifier', 'namespace_definition'],
                'name_field': 'name'
            },
            '.c': {
                'language': C,
                'parser': Parser() if C else None,
                'name': 'C',
                'node_types': ['function_definition', 'struct_specifier'],
                'name_field': 'name'
            },
            '.h': {
                'language': C,
                'parser': Parser() if C else None,
                'name': 'C Header',
                'node_types': ['function_definition', 'function_declarator', 'struct_specifier'],
                'name_field': 'name'
            },
            '.hpp': {
                'language': CPP,
                'parser': Parser() if CPP else None,
                'name': 'C++ Header',
                'node_types': ['function_definition', 'class_specifier', 'struct_specifier', 'namespace_definition'],
                'name_field': 'name'
            },
            '.go': {
                'language': GO,
                'parser': Parser() if GO else None,
                'name': 'Go',
                'node_types': ['function_declaration', 'method_declaration', 'type_declaration'],
                'name_field': 'name'
            },
            '.rs': {
                'language': RUST,
                'parser': Parser() if RUST else None,
                'name': 'Rust',
                'node_types': ['function_item', 'impl_item', 'struct_item', 'enum_item', 'trait_item'],
                'name_field': 'name'
            },
            '.sh': {
                'language': BASH,
                'parser': Parser() if BASH else None,
                'name': 'Shell Script',
                'node_types': ['function_definition'],
                'name_field': 'name'
            },
            '.bash': {
                'language': BASH,
                'parser': Parser() if BASH else None,
                'name': 'Bash Script',
                'node_types': ['function_definition'],
                'name_field': 'name'
            },
            '.dart': {
                'language': None,  # Will use text parsing until tree-sitter-dart is available
                'parser': None,
                'name': 'Dart/Flutter',
                'node_types': ['class', 'function', 'method'],  # For text-based parsing
                'name_field': 'name',
                'text_patterns': [  # Custom patterns for Dart code
                    r'class\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:extends|implements|with)?',
                    r'(?:static\s+)?(?:final\s+)?(?:const\s+)?[A-Za-z_<>\[\]]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(',
                    r'(?:Widget|StatelessWidget|StatefulWidget)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{',
                    r'void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(',
                    r'Future<[^>]*>\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(',
                    r'Stream<[^>]*>\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(',
                ]
            }
        }
        
        # Initialize parsers for available languages
        for config in self.language_config.values():
            if config['parser'] and config['language']:
                config['parser'].language = config['language']

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
            '.mypy_cache', '.coverage', 'htmlcov',  # Tool caches
            'target', 'cmake-build-*', 'Debug', 'Release',  # Build directories
            'bin', 'obj', 'out'  # More build directories
        }
        
        # Supported file extensions
        supported_extensions = set(self.language_config.keys())
        
        for root, dirs, files in os.walk(collection.path):
            # Filter out directories we want to skip
            dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith('.')]
            
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = Path(file).suffix.lower()
                
                if file_ext in supported_extensions:
                    file_count += 1
                    language_config = self.language_config[file_ext]
                    
                    # Process all supported files, even those without tree-sitter parsers
                    logger.debug(f"Processing {language_config['name']} file: {file_path}")
                    try:
                        success = self._process_code_file(file_path, language_config, chroma_collection)
                        if success:
                            processed_files += 1
                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {e}")
        
        logger.info(f"Completed processing folder. Files found: {file_count}, Files processed: {processed_files}")

    def _process_code_file(self, file_path: str, language_config: Dict, chroma_collection) -> bool:
        """Process a single code file."""
        try:
            with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
                content = f.read()
            
            # Skip empty files
            if not content.strip():
                return False
            
            # Check if this is a language with a tree-sitter parser available
            if language_config['parser'] and language_config['language']:
                # Parse the file using tree-sitter
                tree = language_config['parser'].parse(bytes(content, "utf8"))
                
                # Extract chunks
                chunks_found = self._extract_chunks(
                    tree.root_node, 
                    content, 
                    file_path, 
                    chroma_collection, 
                    language_config
                )
                
                # If no chunks found with tree-sitter, fall back to text chunking
                if chunks_found == 0:
                    logger.debug(f"No structured chunks found in {file_path}, using text chunking")
                    self._extract_text_chunks(content, file_path, chroma_collection, language_config)
            else:
                # Use pattern-based parsing for languages without tree-sitter support (e.g., Dart)
                logger.debug(f"Using pattern-based parsing for {language_config['name']}: {file_path}")
                chunks_found = self._extract_pattern_chunks(content, file_path, chroma_collection, language_config)
                
                # If no patterns matched, fall back to text chunking
                if chunks_found == 0:
                    logger.debug(f"No pattern matches found in {file_path}, using text chunking")
                    self._extract_text_chunks(content, file_path, chroma_collection, language_config)
                
            return True
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return False
    
    def _extract_chunks(self, node, text: str, file_path: str, chroma_collection, language_config: Dict) -> int:
        """Recursively extract chunks from the syntax tree."""
        chunks_found = 0
        
        # Check if this node is a type we want to extract
        if node.type in language_config['node_types']:
            start_byte = node.start_byte
            end_byte = node.end_byte
            chunk_text = text[start_byte:end_byte]
            
            # Get the name of the function/class/interface etc.
            name_node = node.child_by_field_name(language_config['name_field'])
            if name_node:
                name = text[name_node.start_byte:name_node.end_byte]
            else:
                # Fallback: try to extract name from first few words
                name = self._extract_name_fallback(chunk_text, node.type)
            
            # Create unique ID
            chunk_id = f"{file_path}::{node.type}::{name}::{start_byte}"
            
            try:
                chroma_collection.add(
                    documents=[chunk_text],
                    metadatas=[{
                        "source": file_path,
                        "type": node.type,
                        "name": name,
                        "language": language_config['name'],
                        "start_byte": start_byte,
                        "end_byte": end_byte
                    }],
                    ids=[chunk_id]
                )
                chunks_found += 1
                logger.debug(f"Extracted {node.type} '{name}' from {file_path}")
            except Exception as e:
                logger.warning(f"Failed to add chunk {chunk_id}: {e}")

        # Recursively process child nodes
        for child in node.children:
            chunks_found += self._extract_chunks(child, text, file_path, chroma_collection, language_config)
        
        return chunks_found
    
    def _extract_pattern_chunks(self, content: str, file_path: str, chroma_collection, language_config: Dict) -> int:
        """Extract chunks using regex patterns for languages without tree-sitter support."""
        import re
        chunks_found = 0
        
        # Get language-specific patterns
        patterns = language_config.get('text_patterns', [])
        if not patterns:
            return 0
        
        # Find all matches for each pattern
        for pattern in patterns:
            for match in re.finditer(pattern, content, re.MULTILINE | re.DOTALL):
                # Get the function/class name from the first capture group
                name = match.group(1) if match.groups() else "unknown"
                
                # Find the complete block by looking for opening and closing braces/indentation
                start_pos = match.start()
                chunk_text, end_pos = self._extract_code_block(content, start_pos, language_config)
                
                if chunk_text and len(chunk_text.strip()) > 20:  # Only include substantial chunks
                    # Determine chunk type based on pattern content
                    chunk_type = self._determine_chunk_type(match.group(0), language_config)
                    
                    chunk_id = f"{file_path}::{chunk_type}::{name}::{start_pos}"
                    
                    try:
                        chroma_collection.add(
                            documents=[chunk_text],
                            metadatas=[{
                                "source": file_path,
                                "type": chunk_type,
                                "name": name,
                                "language": language_config['name'],
                                "start_char": start_pos,
                                "end_char": end_pos,
                                "extraction_method": "pattern_based"
                            }],
                            ids=[chunk_id]
                        )
                        chunks_found += 1
                        logger.debug(f"Extracted {chunk_type} '{name}' from {file_path} using patterns")
                    except Exception as e:
                        logger.warning(f"Failed to add pattern chunk {chunk_id}: {e}")
        
        return chunks_found
    
    def _extract_code_block(self, content: str, start_pos: int, language_config: Dict) -> Tuple[str, int]:
        """Extract a complete code block starting from the given position."""
        lines = content.split('\n')
        start_line = content[:start_pos].count('\n')
        
        # For Dart/Flutter, look for balanced braces
        if language_config['name'] == 'Dart/Flutter':
            return self._extract_brace_block(content, start_pos)
        
        # Default: extract until next function/class or end of file
        end_line = len(lines)
        for i in range(start_line + 1, len(lines)):
            line = lines[i].strip()
            # Look for next top-level declaration
            if line and not line.startswith(' ') and not line.startswith('\t') and not line.startswith('//') and not line.startswith('*'):
                if any(keyword in line for keyword in ['class ', 'function ', 'def ', 'void ', 'Future<', 'Stream<']):
                    end_line = i
                    break
        
        # Get the text from start_line to end_line
        end_pos = len('\n'.join(lines[:end_line]))
        if end_line < len(lines):
            end_pos += 1  # Include the newline
            
        block_text = content[start_pos:end_pos]
        return block_text.strip(), end_pos
    
    def _extract_brace_block(self, content: str, start_pos: int) -> Tuple[str, int]:
        """Extract a block of code by matching opening and closing braces."""
        # Find the first opening brace after start_pos
        brace_pos = content.find('{', start_pos)
        if brace_pos == -1:
            # No braces found, extract until next line that starts with a keyword
            lines = content[start_pos:].split('\n')
            end_lines = []
            for line in lines:
                if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                    if any(keyword in line for keyword in ['class ', 'void ', 'Future<', 'Stream<']) and len(end_lines) > 0:
                        break
                end_lines.append(line)
            
            block_text = '\n'.join(end_lines)
            return block_text.strip(), start_pos + len(block_text)
        
        # Count braces to find the matching closing brace
        brace_count = 0
        pos = brace_pos
        
        while pos < len(content):
            char = content[pos]
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Found the matching closing brace
                    end_pos = pos + 1
                    block_text = content[start_pos:end_pos]
                    return block_text.strip(), end_pos
            pos += 1
        
        # If we reach here, braces weren't balanced - return what we have
        block_text = content[start_pos:]
        return block_text.strip(), len(content)
    
    def _determine_chunk_type(self, match_text: str, language_config: Dict) -> str:
        """Determine the type of code chunk based on the matched text."""
        match_lower = match_text.lower()
        
        if language_config['name'] == 'Dart/Flutter':
            if 'class ' in match_lower:
                if any(widget in match_lower for widget in ['widget', 'statelesswidget', 'statefulwidget']):
                    return 'flutter_widget'
                return 'class'
            elif 'future<' in match_lower:
                return 'async_function'
            elif 'stream<' in match_lower:
                return 'stream_function'
            elif 'void ' in match_lower:
                return 'void_function'
            else:
                return 'function'
        
        # Default fallback
        if 'class' in match_lower:
            return 'class'
        elif 'function' in match_lower:
            return 'function'
        else:
            return 'code_block'
    
    def _extract_name_fallback(self, chunk_text: str, node_type: str) -> str:
        """Extract name from chunk text when tree-sitter field lookup fails."""
        lines = chunk_text.split('\n')
        first_line = lines[0].strip() if lines else ''
        
        # Simple regex patterns for common cases
        import re
        
        # Try to extract name from various patterns
        patterns = [
            r'(?:function|def|class|interface|struct|enum|type)\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'([a-zA-Z_][a-zA-Z0-9_]*)\s*[\(\{:]',
            r'([a-zA-Z_][a-zA-Z0-9_]*)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, first_line, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return f"unnamed_{node_type}"
    
    def _extract_text_chunks(self, content: str, file_path: str, chroma_collection, language_config: Dict):
        """Fallback text chunking when tree-sitter parsing doesn't find structured elements."""
        # Split content into chunks of reasonable size (e.g., ~500 characters)
        chunk_size = 500
        overlap = 50
        
        if len(content) <= chunk_size:
            # File is small enough to be one chunk
            chunk_id = f"{file_path}::text::full"
            try:
                chroma_collection.add(
                    documents=[content],
                    metadatas=[{
                        "source": file_path,
                        "type": "text_chunk",
                        "name": f"full_file",
                        "language": language_config['name'],
                        "chunk_index": 0
                    }],
                    ids=[chunk_id]
                )
                logger.debug(f"Added full file as text chunk: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to add text chunk {chunk_id}: {e}")
            return
        
        # Split into overlapping chunks
        for i, start in enumerate(range(0, len(content), chunk_size - overlap)):
            end = min(start + chunk_size, len(content))
            chunk_text = content[start:end]
            
            # Skip very small chunks at the end
            if len(chunk_text.strip()) < 50:
                continue
                
            chunk_id = f"{file_path}::text::chunk_{i}"
            try:
                chroma_collection.add(
                    documents=[chunk_text],
                    metadatas=[{
                        "source": file_path,
                        "type": "text_chunk",
                        "name": f"chunk_{i}",
                        "language": language_config['name'],
                        "chunk_index": i,
                        "start_char": start,
                        "end_char": end
                    }],
                    ids=[chunk_id]
                )
                logger.debug(f"Added text chunk {i} from {file_path}")
            except Exception as e:
                logger.warning(f"Failed to add text chunk {chunk_id}: {e}")

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
