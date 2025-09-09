let chatbotInitialized = false;

// Check chatbot status periodically
function checkChatbotStatus() {
    fetch('/status')
        .then(response => response.json())
        .then(data => {
            if (data.initialized) {
                chatbotInitialized = true;
                document.getElementById('question-input').disabled = false;
                document.getElementById('send-btn').disabled = false;
                document.getElementById('loading').classList.add('hidden');
                addMessage("Chatbot is ready! Ask me anything about tuberculosis.", 'bot');
            } else {
                setTimeout(checkChatbotStatus, 2000);
            }
        })
        .catch(error => {
            console.error('Error checking status:', error);
            setTimeout(checkChatbotStatus, 2000);
        });
}

function sendQuestion() {
    const input = document.getElementById('question-input');
    const question = input.value.trim();
    
    if (!question || !chatbotInitialized) return;
    
    // Add user message to chat
    addMessage(question, 'user');
    input.value = '';
    
    // Show loading
    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('sources').classList.add('hidden');
    
    // Send question to server
    fetch('/ask', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: question })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            addMessage('Error: ' + data.error, 'bot');
        } else {
            addMessage(data.answer, 'bot');
            if (data.sources && data.sources.length > 0) {
                showSources(data.sources);
            }
        }
    })
    .catch(error => {
        console.error('Error:', error);
        addMessage('Sorry, there was an error processing your question. Please try again.', 'bot');
    })
    .finally(() => {
        document.getElementById('loading').classList.add('hidden');
    });
}

function addMessage(text, type) {
    const messagesContainer = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    messageDiv.textContent = text;
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function showSources(sources) {
    if (!sources || sources.length === 0) return;
    
    const sourcesContainer = document.getElementById('sources-list');
    sourcesContainer.innerHTML = '';
    
    sources.forEach(source => {
        const sourceDiv = document.createElement('div');
        sourceDiv.className = 'source-item';
        sourceDiv.textContent = `${source.document} (Page ~${source.page})`;
        sourcesContainer.appendChild(sourceDiv);
    });
    
    document.getElementById('sources').classList.remove('hidden');
}

// Allow pressing Enter to send message
document.getElementById('question-input').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendQuestion();
    }
});

// Start checking chatbot status when page loads
document.addEventListener('DOMContentLoaded', function() {
    checkChatbotStatus();
    addMessage("Welcome! I'm initializing the Tuberculosis Expert Chatbot. Please wait...", 'bot');
});