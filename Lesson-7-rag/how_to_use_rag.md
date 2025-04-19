# How to Use and How it Works: General RAG System

Here is a mind map outlining how to use the General RAG System based on the provided `general_rag.py` file:

*   **1. Get a RAG Instance**
    *   Use the `RAGProvider.get_instance()` class method.
    *   **Required Parameters:**
        *   `source_type`: Specify the type of data source ("dataset", "pdf", or "text").
        *   Provide source details based on `source_type`:
            *   `source_path`: Path to the file for "pdf" or "text" sources.
            *   `dataset_name`: HuggingFace dataset name for the "dataset" source.
    *   **Optional Parameters:**
        *   `embedding_model_name`: Name of the sentence-transformers model (defaults to 'all-MiniLM-L6-v2' or environment variable `RAG_EMBEDDING_MODEL`).
        *   `chunk_size`: Target size of text chunks for "pdf" and "text" (defaults to 500).
        *   `chunk_overlap`: Overlap between consecutive chunks for "pdf" and "text" (defaults to 50).
        *   `num_results`: Number of results to retrieve (defaults to 3 or environment variable `RAG_NUM_RESULTS`).
    *   The `RAGProvider` ensures a single instance is used for a given source configuration.

*   **2. Initialization (Handled Automatically by `get_instance`)**
    *   When `get_instance` is called, if an instance for the specified source doesn't exist or isn't initialized, the `initialize()` method is automatically called.
    *   This process includes:
        *   Loading data from the specified source (`dataset`, `pdf`, or `text` file).
        *   Chunking the loaded text data (for "pdf" and "text" sources).
        *   Loading the specified embedding model.
        *   Creating a FAISS vector index and adding embeddings of the data chunks.

*   **3. Retrieve Information**
    *   Call the `retrieve(query)` method on the obtained RAG instance.
    *   Pass the user's search `query` string as an argument.
    *   This method performs a similarity search against the vector index and returns a list of the most relevant data chunks (up to `num_results`).

*   **4. Format Results**
    *   Call the `format_results(results)` method on the RAG instance.
    *   Pass the list of results obtained from the `retrieve()` method.
    *   This method formats the retrieved chunks and their metadata into a readable string.

*   **Helper Functions for Convenience:**
    *   `query_pdf(pdf_path, query, num_results)`: A simplified function to get an instance for a PDF source, retrieve, and format results in one step.
    *   `query_dataset(dataset_name, query, num_results)`: A simplified function to get an instance for a dataset source, retrieve, and format results in one step.

---

## How the General RAG System Works

The `GeneralRAGSystem` implements a standard Retrieval-Augmented Generation (RAG) workflow to find relevant information from various data sources. Here's a breakdown of the key steps involved:

### 1. Data Loading and Chunking

*   The system first loads data based on the specified `source_type`:
    *   **Dataset:** Loads examples from a HuggingFace dataset using the `datasets` library.
    *   **PDF:** Extracts text content from a PDF file using `PyPDF2`.
    *   **Text:** Reads text content from a plain text file.
*   For PDF and text sources, the raw text is then split into smaller, manageable `chunks`. This is done to ensure that the pieces of text are small enough to be effectively processed by the embedding model and to allow for more granular retrieval. The chunking process includes handling paragraph breaks and adding overlap between chunks to maintain context.

### 2. Embedding Creation

*   A pre-trained `SentenceTransformer` model (defaulting to 'all-MiniLM-L6-v2') is loaded.
*   This model is used to convert each text chunk into a numerical vector representation, called an `embedding`. Embeddings capture the semantic meaning of the text, where chunks with similar meanings will have similar vector representations.

### 3. Vector Indexing (FAISS)

*   The system uses the `faiss` library to create an efficient index of the generated embeddings.
*   A `IndexFlatL2` index is used, which allows for fast similarity search based on L2 (Euclidean) distance.
*   The embeddings of all the chunks are added to this FAISS index.

### 4. Retrieval Process

*   When a user provides a `query`, the same embedding model is used to convert the query string into a query embedding vector.
*   This query embedding is then used to search the FAISS index.
*   The index quickly finds the `num_results` embeddings that are most similar (have the smallest L2 distance) to the query embedding.
*   The system retrieves the original text chunks and their associated metadata corresponding to these most similar embeddings.

### 5. Result Formatting

*   The retrieved chunks, along with their relevance scores (the distance from the query embedding), are formatted into a readable string using the `format_results` method.
*   The formatting adapts slightly based on the type of retrieved item (e.g., dataset prompt examples vs. general text chunks).

This process allows the system to efficiently find and present relevant information from large documents or datasets based on a user's query.