"""
FastAPI server for Mistral AI Chat Interface

This module implements a REST API server that handles chat communication between
the web frontend and Mistral AI agent. It provides a single endpoint for
processing chat messages and returning AI responses.
"""
from datasets import load_dataset 
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from mistral_agent_raw_prompt import agent_executor
import os

app = FastAPI(
    title="Mistral AI Chat API with RAG",
    description="API server for processing chat messages with RAG-enabled Mistral AI agent",
    version="1.0.0"
)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check if the software testing PDF exists
software_testing_pdf_path = "D:/AgentLearning/software-testing-with-generative-ai-1nbsped-1633437361-9781633437364_compress.pdf"
pdf_exists = os.path.exists(software_testing_pdf_path)

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    print("Starting Mistral AI Chat API with RAG...")
    if pdf_exists:
        print(f"Software Testing PDF found at: {software_testing_pdf_path}")
    else:
        print(f"Warning: Software Testing PDF not found at: {software_testing_pdf_path}")
    print("Server ready!")

@app.get("/", tags=["Root"])
async def root():
    """Return API information"""
    return {
        "message": "Mistral AI Chat API with RAG",
        "instructions": "Send POST requests to /chat endpoint with a message field",
        "features": [
            "General question answering",
            "Web search capability",
            "Prompt retrieval from awesome-chatgpt-prompts",
            "Software Testing with Generative AI book integration" if pdf_exists else 
                "Software Testing PDF not found"
        ],
        "example": "Try asking: 'What are the best practices for generative AI testing?'"
    }

class ChatMessage(BaseModel):
    """
    Pydantic model for chat message validation.
    
    Attributes:
        message (str): The user's chat message
    """
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "What are the best practices for generative AI testing?"
            }
        }

@app.post("/chat", 
    response_model=dict,
    summary="Process chat message",
    description="Sends user message to Mistral AI agent and returns the response")
async def chat(message: ChatMessage):
    """
    Process a chat message and return the AI agent's response.
    
    Args:
        message (ChatMessage): The user's message object
        
    Returns:
        dict: Contains the AI agent's response
        
    Raises:
        HTTPException: If message processing fails
    """
    try:
        # Process message with Mistral agent
        response = agent_executor.invoke({"input": message.message})
        return {"response": response["output"]}
    except Exception as e:
        return {"response": f"Error: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)