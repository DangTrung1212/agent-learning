import os
import re
import faiss
import numpy as np
from typing import List, Dict, Any, Optional, Union, Literal
from sentence_transformers import SentenceTransformer
import threading
import PyPDF2
from datasets import load_dataset
import textwrap


class GeneralRAGSystem:
    """Generalized Retrieval-Augmented Generation system supporting multiple data sources"""
    
    def __init__(self, 
                 source_type: Literal["dataset", "pdf", "text"] = "dataset",
                 source_path: Optional[str] = None,
                 dataset_name: Optional[str] = None,
                 embedding_model_name: Optional[str] = None,
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 num_results: Optional[int] = None):
        """
        Initialize the generalized RAG system.
        
        Args:
            source_type: Type of data source ("dataset", "pdf", or "text")
            source_path: Path to file (for "pdf" or "text" source types)
            dataset_name: HuggingFace dataset name (for "dataset" source type)
            embedding_model_name: Name of the sentence-transformers model
            chunk_size: Target size of text chunks (approximate chars)
            chunk_overlap: Overlap between consecutive chunks
            num_results: Number of results to retrieve
        """
        # Source configuration
        self.source_type = source_type
        self.source_path = source_path
        
        # Load configuration from environment variables or use defaults
        self.embedding_model_name = embedding_model_name or os.environ.get(
            'RAG_EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        self.dataset_name = dataset_name or os.environ.get(
            'RAG_DATASET_NAME', "fka/awesome-chatgpt-prompts")
        self.num_results = num_results or int(os.environ.get('RAG_NUM_RESULTS', '3'))
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize other attributes
        self.chunks = []
        self.chunk_metadata = []
        self.embedding_model = None
        self.index = None
        self.dimension = 384  # Will be properly set during initialization
        self.index_lock = threading.Lock()
        self.is_initialized = False
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate the configuration"""
        if self.source_type not in ["dataset", "pdf", "text"]:
            raise ValueError(f"Invalid source_type: {self.source_type}. Must be 'dataset', 'pdf', or 'text'")
        
        if self.source_type in ["pdf", "text"] and not self.source_path:
            raise ValueError(f"source_path must be provided for source_type '{self.source_type}'")
        
        if self.source_type == "pdf" and not self.source_path.lower().endswith('.pdf'):
            raise ValueError(f"source_path must be a PDF file for source_type 'pdf'")
            
        if self.source_type == "dataset" and not self.dataset_name:
            raise ValueError("dataset_name must be provided for source_type 'dataset'")
    
    def extract_text_from_pdf(self) -> str:
        """Extract all text from a PDF file"""
        print(f"Extracting text from {self.source_path}...")
        
        text = ""
        with open(self.source_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n"
        
        return text
    
    def load_text_from_file(self) -> str:
        """Load text from a plain text file"""
        print(f"Loading text from {self.source_path}...")
        
        with open(self.source_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def load_dataset_examples(self) -> List[Dict[str, Any]]:
        """Load examples from a dataset"""
        print(f"Loading dataset: {self.dataset_name}...")
        
        ds = load_dataset(self.dataset_name)
        examples = []
        
        # Handle specific datasets differently
        if self.dataset_name == "fka/awesome-chatgpt-prompts":
            # Extract act/prompt examples
            for item in ds["train"]:
                prompt_text = item.get("prompt", "")
                act = item.get("act", "")
                if prompt_text and act:
                    # Combine the 'act' and 'prompt' for better context
                    examples.append({
                        "text": f"Act as {act}\n\n{prompt_text}",
                        "metadata": {
                            "act": act,
                            "prompt": prompt_text,
                            "type": "prompt_example"
                        }
                    })
        else:
            # Generic approach for other datasets - assume a "text" field
            if "train" in ds:
                data_split = "train"
            else:
                data_split = list(ds.keys())[0]  # Use first available split
            
            for i, item in enumerate(ds[data_split]):
                if "text" in item:
                    examples.append({
                        "text": item["text"],
                        "metadata": {
                            "id": i,
                            "original_data": item,
                            "type": "dataset_item"
                        }
                    })
        
        return examples
    
    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata"""
        print("Chunking text...")
        
        # Clean text - remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split text into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Create chunks from paragraphs
        chunks = []
        current_chunk = ""
        current_chunk_size = 0
        chunk_start_idx = 0
        
        for i, paragraph in enumerate(paragraphs):
            # If adding this paragraph would exceed chunk size, save current chunk
            if current_chunk_size + len(paragraph) > self.chunk_size and current_chunk:
                chunks.append({
                    "text": current_chunk,
                    "metadata": {
                        "chunk_id": len(chunks),
                        "start_paragraph": chunk_start_idx,
                        "end_paragraph": i - 1,
                        "type": "text_chunk"
                    }
                })
                # Start new chunk with overlap
                words = current_chunk.split()
                overlap_words = words[-self.chunk_overlap:] if len(words) > self.chunk_overlap else words
                current_chunk = " ".join(overlap_words) + " " + paragraph
                current_chunk_size = len(current_chunk)
                chunk_start_idx = i
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += " " + paragraph
                else:
                    current_chunk = paragraph
                current_chunk_size += len(paragraph)
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append({
                "text": current_chunk,
                "metadata": {
                    "chunk_id": len(chunks),
                    "start_paragraph": chunk_start_idx,
                    "end_paragraph": len(paragraphs) - 1,
                    "type": "text_chunk"
                }
            })
        
        print(f"Created {len(chunks)} chunks.")
        return chunks
    
    def initialize(self) -> None:
        """Initialize RAG components based on source type"""
        if self.is_initialized:
            return
        
        # Get documents based on source type
        chunks = []
        if self.source_type == "dataset":
            chunks = self.load_dataset_examples()
        elif self.source_type == "pdf":
            text = self.extract_text_from_pdf()
            chunks = self.chunk_text(text)
        elif self.source_type == "text":
            text = self.load_text_from_file()
            chunks = self.chunk_text(text)
        
        # Store chunks and their metadata
        self.chunks = [chunk["text"] for chunk in chunks]
        self.chunk_metadata = [chunk["metadata"] for chunk in chunks]
        
        # Load embedding model
        print(f"Loading embedding model: {self.embedding_model_name}...")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Create FAISS index
        print("Creating vector store...")
        with self.index_lock:
            # Create embeddings
            embeddings = self.embedding_model.encode(self.chunks)
            
            # Create FAISS index
            self.dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(np.array(embeddings).astype('float32'))
        
        self.is_initialized = True
        print(f"RAG system initialized successfully with {len(self.chunks)} items!")
    
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks based on query"""
        if not self.is_initialized:
            self.initialize()
        
        # Encode the query
        query_embedding = self.embedding_model.encode([query])
        
        # Search the index
        with self.index_lock:
            distances, indices = self.index.search(
                np.array(query_embedding).astype('float32'), 
                min(self.num_results, len(self.chunks))  # Ensure we don't request more than available
            )
        
        # Get results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                results.append({
                    "text": self.chunks[idx],
                    "metadata": self.chunk_metadata[idx],
                    "score": float(distances[0][i])  # Convert to Python float
                })
        
        # Sort by relevance (lower distance is better)
        results.sort(key=lambda x: x["score"])
        
        return results
    
    def format_results(self, results: List[Dict[str, Any]]) -> str:
        """Format retrieval results as a readable string"""
        if not results:
            return "No relevant information found."
            
        formatted_results = []
        
        for i, result in enumerate(results):
            metadata = result["metadata"]
            result_type = metadata.get("type", "")
            
            if result_type == "prompt_example":
                # Format for prompt examples
                formatted_results.append(
                    f"Example {i+1}: Act as {metadata['act']}\n{metadata['prompt']}\n"
                )
            else:
                # Format for text chunks and others
                chunk_id = metadata.get("chunk_id", i)
                formatted_results.append(
                    f"[Chunk {chunk_id}] Relevance Score: {result['score']:.4f}\n"
                    f"{textwrap.fill(result['text'][:500], width=100)}\n"
                    f"{'...' if len(result['text']) > 500 else ''}\n"
                )
        
        return "\n".join(formatted_results)


# Singleton pattern implementation
class RAGProvider:
    """Singleton provider for the RAG system"""
    _instances = {}
    
    @classmethod
    def get_instance(cls, source_type: str = "dataset", **kwargs) -> GeneralRAGSystem:
        """Get or create a RAG system instance for the given source type"""
        # Create a key based on source type and path/dataset
        key = source_type
        if source_type in ["pdf", "text"]:
            key += f":{kwargs.get('source_path', '')}"
        elif source_type == "dataset":
            key += f":{kwargs.get('dataset_name', '')}"
        
        # Create new instance if needed
        if key not in cls._instances:
            cls._instances[key] = GeneralRAGSystem(source_type=source_type, **kwargs)
            try:
                cls._instances[key].initialize()
            except Exception as e:
                print(f"Warning: Could not initialize RAG system: {e}")
                print("The system will try to initialize when first used.")
        
        return cls._instances[key]


# Example usage for PDF
def query_pdf(pdf_path: str, query: str, num_results: int = 5) -> str:
    """Query a PDF file and return formatted results"""
    rag = RAGProvider.get_instance(
        source_type="pdf", 
        source_path=pdf_path,
        num_results=num_results
    )
    results = rag.retrieve(query)
    return rag.format_results(results)


# Example usage for Dataset
def query_dataset(dataset_name: str, query: str, num_results: int = 3) -> str:
    """Query a dataset and return formatted results"""
    rag = RAGProvider.get_instance(
        source_type="dataset", 
        dataset_name=dataset_name,
        num_results=num_results
    )
    results = rag.retrieve(query)
    return rag.format_results(results)


# Simple demo function
def main():
    """Demonstration of the General RAG System"""
    import argparse
    parser = argparse.ArgumentParser(description="General RAG System Demo")
    parser.add_argument("--type", type=str, choices=["pdf", "dataset", "text"], default="dataset",
                        help="Type of source to query")
    parser.add_argument("--source", type=str, help="Path to file or dataset name")
    parser.add_argument("--query", type=str, help="Query to search for")
    parser.add_argument("--results", type=int, default=3, help="Number of results to return")
    
    args = parser.parse_args()
    
    if not args.query:
        # Interactive mode
        if args.type == "pdf" and args.source:
            pdf_path = args.source
            print(f"\n=== PDF RAG System for {pdf_path} ===")
            rag = RAGProvider.get_instance(source_type="pdf", source_path=pdf_path)
            
        elif args.type == "dataset" and args.source:
            dataset_name = args.source
            print(f"\n=== Dataset RAG System for {dataset_name} ===")
            rag = RAGProvider.get_instance(source_type="dataset", dataset_name=dataset_name)
            
        elif args.type == "text" and args.source:
            text_path = args.source
            print(f"\n=== Text RAG System for {text_path} ===")
            rag = RAGProvider.get_instance(source_type="text", source_path=text_path)
            
        else:
            print("Error: Must provide --source for the selected type")
            return
        
        # Initialize the system
        rag.initialize()
        
        print("\nEnter your questions (type 'exit' to quit):")
        while True:
            query = input("\nQuestion: ")
            if query.lower() in ['exit', 'quit', 'q']:
                break
                
            results = rag.retrieve(query)
            print("\n=== Retrieved Passages ===")
            print(rag.format_results(results))
    
    else:
        # One-off query mode
        if args.type == "pdf" and args.source:
            result = query_pdf(args.source, args.query, args.results)
        elif args.type == "dataset" and args.source:
            result = query_dataset(args.source, args.query, args.results)
        elif args.type == "text" and args.source:
            rag = RAGProvider.get_instance(source_type="text", source_path=args.source)
            results = rag.retrieve(args.query)
            result = rag.format_results(results)
        else:
            result = "Error: Must provide --source for the selected type"
        
        print("\n=== Retrieved Passages ===")
        print(result)


if __name__ == "__main__":
    main() 