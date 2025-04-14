"""
FastAPI server for Mistral AI Chat Interface

This module implements a REST API server that handles chat communication between
the web frontend and Mistral AI agent. It provides a single endpoint for
processing chat messages and returning AI responses.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from mistral_agent_raw_prompt import agent_executor

app = FastAPI(
    title="Mistral AI Chat API",
    description="API server for processing chat messages with Mistral AI agent",
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
                "message": "What's the weather like today?"
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