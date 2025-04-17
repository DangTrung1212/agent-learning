from langchain.tools import tool
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults
from langchain.agents import tool
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import threading
import os
from typing import List, Dict, Any, Optional
from general_rag import RAGProvider




@tool
def search(query: str) -> str:
    """Search for news. then return """
    try:
        search = DuckDuckGoSearchResults()
        results = search.invoke(query)
        return results
    except Exception as e:
        return f"An error occurred: {str(e)}"

@tool
def retrieve_from_pdf(pdf_path: str, query: str) -> str:
    """
    Retrieve relevant information from a PDF document.
    This tool helps find specific information within a PDF file based on your query.
    Provide the full path to the PDF file and your specific question or query.
    
    Args:
        pdf_path: Full path to the PDF file (e.g., "D:/path/to/document.pdf")
        query: Your specific question about the PDF content
    
    Example Input: "D:/AgentLearning/software-testing-with-generative-ai-1nbsped-1633437361-9781633437364_compress.pdf|What is generative AI testing?"
    """
    try:
        # Parse the input - we support either pipe-separated or direct arguments
        if "|" in pdf_path:
            parts = pdf_path.split("|", 1)
            pdf_path = parts[0].strip()
            query = parts[1].strip()
        
        # Validate input
        if not pdf_path.lower().endswith('.pdf'):
            return "Error: The provided path does not appear to be a PDF file."
        
        if not os.path.exists(pdf_path):
            return f"Error: Could not find PDF file at path: {pdf_path}"
        
        if not query.strip():
            return "Error: Please provide a specific question about the PDF content."
        
        # Get RAG system instance from the generalized provider
        rag = RAGProvider.get_instance(
            source_type="pdf", 
            source_path=pdf_path,
            num_results=5  # More results for PDF queries
        )
        
        # Retrieve relevant content
        results = rag.retrieve(query)
        
        # Format results
        return rag.format_results(results)
    
    except Exception as e:
        return f"An error occurred while retrieving from PDF: {str(e)}"


@tool
def software_testing_pdf(query: str) -> str:
    """
    Retrieve information about software testing with generative AI from the book.
    This tool helps answer questions about software testing concepts, generative AI in testing,
    and best practices from the 'Software Testing with Generative AI' book.
    
    Args:
        query: Your specific question about software testing or generative AI testing
    """
    try:
        # Hardcoded path to the software testing PDF
        pdf_path = "D:/AgentLearning/software-testing-with-generative-ai-1nbsped-1633437361-9781633437364_compress.pdf"
        
        # Validate the file exists
        if not os.path.exists(pdf_path):
            return f"Error: Could not find the software testing PDF at path: {pdf_path}"
        
        # Get RAG system instance from the generalized provider
        rag = RAGProvider.get_instance(
            source_type="pdf", 
            source_path=pdf_path,
            num_results=5  # More results for book queries
        )
        
        # Retrieve relevant content
        results = rag.retrieve(query)
        
        # Format results
        return rag.format_results(results)
    
    except Exception as e:
        return f"An error occurred while retrieving from the software testing book: {str(e)}"


tools = [search, software_testing_pdf]
