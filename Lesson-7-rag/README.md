# RAG-Enabled Chat Application

This project demonstrates a Retrieval-Augmented Generation (RAG) system integrated with a Mistral AI chat interface. The application uses the `fka/awesome-chatgpt-prompts` dataset to enhance the LLM's responses with relevant examples.

## Architecture

The application consists of two main components:

1. **API Server**: A FastAPI backend that handles chat messages and communicates with the Mistral AI agent
2. **Web UI**: A frontend interface for interacting with the chat system

### Key Features

- **ReAct Agent**: Uses Mistral's large language model with reasoning abilities
- **Tool Integration**: Incorporates external tools like web search
- **RAG Capability**: Enhances responses by retrieving relevant prompt examples
- **Conversation Memory**: Maintains context from previous exchanges

## How RAG is Implemented

The RAG system works in the following way:

1. The `fka/awesome-chatgpt-prompts` dataset is loaded and preprocessed
2. Prompt examples are embedded using the SentenceTransformer model
3. Embeddings are stored in a FAISS vector index for efficient similarity search
4. When a user query requires creative or role-playing content, the agent uses the `retrieve_prompts` tool
5. The tool finds semantically similar examples in the vector store
6. The retrieved examples help the agent generate more appropriate responses

## Setup and Installation

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Installation

1. Clone the repository
2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/macOS
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Configuration

The RAG system can be configured using environment variables:

- `RAG_EMBEDDING_MODEL`: Name of the sentence-transformers model to use (default: 'all-MiniLM-L6-v2')
- `RAG_DATASET_NAME`: HuggingFace dataset identifier (default: "fka/awesome-chatgpt-prompts")
- `RAG_NUM_RESULTS`: Number of similar examples to retrieve (default: 3)

## Usage

1. Start the API server:
   ```
   cd Lesson-7-rag/api
   python main.py
   ```

2. Access the API directly or through the web UI:
   - API endpoint: `http://localhost:8000/chat`
   - Web UI: Open the HTML file in a browser (if implemented)

### API Example

```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={"message": "Act like a pirate and tell me a story"}
)
print(response.json()["response"])
```

## Developer Notes

- The application uses a singleton pattern for the RAG system to ensure efficient resource usage
- Thread-safety is implemented for the vector store operations
- Lazy initialization allows the system to start up quickly

## License

[Specify your license information here] 