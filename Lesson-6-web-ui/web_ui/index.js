async function sendMessage() {
    const userInput = document.getElementById('user-input').value;
    if (userInput.trim() === '') return;

    const chatBox = document.getElementById('chat-box');
    
    // Display user message
    const userMessage = document.createElement('div');
    userMessage.className = 'user-message';
    userMessage.textContent = userInput;
    chatBox.appendChild(userMessage);

    // Clear input
    document.getElementById('user-input').value = '';

    // Add typing indicator
    const typingIndicator = document.createElement('div');
    typingIndicator.className = 'typing-indicator';
    typingIndicator.innerHTML = `
        <div class="typing-dots">
            <div class="dot"></div>
            <div class="dot"></div>
            <div class="dot"></div>
        </div>
    `;
    chatBox.appendChild(typingIndicator);
    chatBox.scrollTop = chatBox.scrollHeight;

    try {
        // Send message to agent and get response
        const response = await fetch('http://localhost:8000/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: userInput })
        });

        const data = await response.json();
        
        // Remove typing indicator
        typingIndicator.remove();
        
        // Display agent message
        const agentMessage = document.createElement('div');
        agentMessage.className = 'agent-message';
        agentMessage.textContent = data.response;
        chatBox.appendChild(agentMessage);
    } catch (error) {
        // Remove typing indicator
        typingIndicator.remove();
        
        // Handle error
        const errorMessage = document.createElement('div');
        errorMessage.className = 'agent-message error';
        errorMessage.textContent = 'Sorry, I encountered an error. Please try again.';
        chatBox.appendChild(errorMessage);
        console.error('Error:', error);
    }

    // Scroll to bottom
    chatBox.scrollTop = chatBox.scrollHeight;
}

// Add enter key support for sending messages
document.getElementById('user-input').addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
});