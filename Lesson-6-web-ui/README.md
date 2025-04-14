# Chat Interface with Mistral AI Agent

This project implements a web-based chat interface that connects to a Mistral AI agent through a FastAPI backend.

## Project Structure
```markdown
# Chat Interface with Mistral AI Agent
```

Lesson-6-web-ui/
├── api/
│   └── main.py                    # FastAPI backend server with CORS and error handling
├── web_ui/
│   ├── index.html                 # Chat interface HTML
│   ├── styles.css                 # CSS with animations and responsive design
│   └── index.js                   # Frontend logic with async communication
├── mistral_agent_raw_prompt.py    # Mistral AI agent configuration
├── agent_tools.py                 # Search tool implementation
└── README.md                      # Project documentation

```plaintext

## Implementation Details

### Frontend Components

#### HTML Structure
- Chat container with message display area
- Text input field for user messages
- Send button with click handler
- Responsive layout design

#### CSS Features
- Modern chat interface styling
- Message bubbles for user and agent
- Loading animation with bouncing dots
- Error message styling
- Responsive design for different screen sizes

#### JavaScript Functionality
- Asynchronous message handling
- Loading state management
- Error handling with user feedback
- Auto-scroll to latest messages
- Enter key support for sending messages

### Backend Components

#### FastAPI Server
- RESTful API implementation
- CORS middleware configuration
- Request/Response validation
- Error handling with HTTP status codes
- API documentation with Swagger UI

#### Mistral AI Integration
- Agent executor implementation
- Custom tool integration
- Conversation memory management
- Error handling and response formatting

## Setup Instructions

1. Environment Setup:
```bash
python -m venv d:\AgentLearning\.venv
d:\AgentLearning\.venv\Scripts\activate
 ```
```

2. Install Dependencies:
```bash
pip install -r d:\AgentLearning\requirements.txt
 ```
```

3. Start Backend Server:
```bash
python d:\AgentLearning\Lesson-6-web-ui\api\main.py
 ```
```

4. Launch Frontend:
```bash
cd d:\AgentLearning\Lesson-6-web-ui\web_ui
python -m http.server 5500
 ```
```

## Usage
1. Access the web interface at: http://localhost:5500
2. Type your message in the input field
3. Send by clicking the button or pressing Enter
4. Wait for the agent's response (animated loading indicator)
5. View API documentation at: http://localhost:8000/docs
## Features
- Real-time chat interface
- Intelligent responses from Mistral AI
- Loading animations for better UX
- Error handling and display
- Web search capability
- Responsive design
- API documentation
## Technical Highlights
- Asynchronous API communication
- CORS support for local development
- Proper error handling both frontend and backend
- Clean and maintainable code structure
- Modular component design
- Documentation with examples
## Future Enhancements
1. User Interface
   
   - Message history persistence
   - Rich text formatting
   - File attachment support
   - Custom themes
2. Backend Features
   
   - User authentication
   - Rate limiting
   - Message logging
   - Additional AI tools
3. Performance
   
   - Response caching
   - Load balancing
   - Message queuing
## Troubleshooting
Common issues and solutions:

- Backend connection errors: Check if the API server is running
- CORS errors: Verify CORS settings in main.py
- Loading never ends: Check network connectivity
- Agent not responding: Verify API key and model settings
## Resources
- FastAPI Documentation
- Mistral AI Documentation
- Python HTTP Server Documentation
- Modern CSS Techniques