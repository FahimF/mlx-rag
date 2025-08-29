# Plan for RAG Feature in MLX-GUI

This document outlines the plan for adding a Retrieval-Augmented Generation (RAG) feature to the `mlx-gui` application.

### 1. Core Objective

The main goal is to introduce a "RAG" tab in the admin panel. This interface will empower users to:
*   Create distinct RAG collections from local folders containing source code.
*   Manage these collections (activate, delete, reprocess).
*   Select an "active" collection that can be used for RAG-based chat and other features.

### 2. Ideal Backing Storage: Vector Database

For this functionality, a **vector database** is the ideal backing storage. Here's why:

*   **Efficient Similarity Search:** RAG's effectiveness hinges on finding the most relevant information (in this case, code snippets) for a given query. Vector databases are purpose-built for ultra-fast similarity searches on the high-dimensional vectors that represent the code embeddings.
*   **Scalability:** As users add more projects and the volume of code grows, a vector database can effortlessly scale to manage millions of code chunks.
*   **Rich Metadata Filtering:** Vector databases excel at storing metadata alongside the vectors. This is a critical requirement for this project. I can store metadata for each code chunk, such as:
    *   File path
    *   Function or class name
    *   Programming language
    *   Start and end line numbers
    This will enable sophisticated queries, like asking a question about a specific file or function.
*   **Persistence:** The vector database will save the generated embeddings, eliminating the need to re-process all the source code every time the application is launched.

**Recommendation for `mlx-gui`:**

Given that `mlx-gui` is a local-first desktop application, an **embedded vector database** is the perfect fit. This avoids the complexity of running a separate database server. My top recommendation is:

*   **ChromaDB:** It's an open-source, embedded vector database that is incredibly easy to integrate with Python. It's perfect for rapid prototyping and is powerful enough for this use case. It can be run in-memory or persisted to disk, which is ideal for managing different RAG collections.

### 3. Development Plan

Here is a step-by-step plan to implement this feature:

#### Step 1: Backend Development (`src/mlx_gui/`)

1.  **Create a new `rag_manager.py` file:** This will be the heart of the RAG functionality. It will contain a `RAGManager` class responsible for:
    *   **Code Parsing:** I'll use a library like `tree-sitter` to intelligently parse source code from various languages into meaningful chunks (e.g., functions, classes, methods). This is superior to naive text splitting as it preserves the code's structure and context.
    *   **Embedding Generation:** It will use one of the user's installed embedding models to convert the code chunks into vector embeddings.
    *   **Vector DB Interaction:** It will handle all communication with the ChromaDB instance, including creating collections, storing vectors and metadata, and querying for relevant documents.
    *   **RAG Query Logic:** It will orchestrate the full RAG pipeline: receive a query, retrieve context from the active ChromaDB collection, and then use a language model to generate an answer.

2.  **Update `database.py` and `models.py`:**
    *   I'll add a new `RAGCollection` table to the database to store information about the RAG collections, such as:
        *   `name` (a user-friendly name for the collection)
        *   `path` (the absolute path to the source code folder)
        *   `status` (e.g., "Ready", "Processing", "Error")
        *   `is_active` (a boolean to indicate the currently active collection)

3.  **Create New API Endpoints in `server.py`:**
    *   `POST /v1/rag/collections`: To create a new RAG collection. It will take a folder path and a collection name and will start the indexing process in the background.
    *   `GET /v1/rag/collections`: To list all created RAG collections and their current status.
    *   `POST /v1/rag/collections/{collection_name}/activate`: To set a collection as the active one for RAG queries.
    *   `DELETE /v1/rag/collections/{collection_name}`: To delete a collection and its associated data from ChromaDB.
    *   `POST /v1/rag/collections/{collection_name}/reprocess`: To trigger a full re-indexing of the source code folder.
    *   `POST /v1/rag/query`: The main endpoint for performing a RAG query. It will use the active collection to find context and generate a response.

#### Step 2: Frontend Development (`src/mlx_gui/templates/admin.html`)

1.  **Add a "RAG" Tab:** I'll add a new "RAG" button to the main tab navigation in the admin panel.

2.  **Design the RAG Tab UI:** The new tab will have two main sections:
    *   **Create New Collection:**
        *   An input field for the collection name.
        *   An input field for the absolute path to the folder.
        *   A "Create" button.
    *   **Manage Existing Collections:**
        *   A dynamic list or table that displays all RAG collections from the database.
        *   Each entry will show the collection's name, path, and status.
        *   Each entry will have "Activate", "Reprocess", and "Delete" buttons that call the corresponding API endpoints.
