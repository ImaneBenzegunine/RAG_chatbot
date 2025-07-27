class Chatbox {
    constructor() {
        this.args = {
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button'),
            chatMessages: document.querySelector('.chatbox__messages')
        };

        this.state = false; // Start with chat closed
        this.messages = [];
    }

    display() {
        const { chatBox, sendButton } = this.args;

        // Toggle chatbox visibility
        this.state = !this.state;
        if (this.state) {
            chatBox.classList.add('chatbox--active');
        } else {
            chatBox.classList.remove('chatbox--active');
        }

        sendButton.addEventListener('click', () => this.onSendButton());
        
        const inputField = chatBox.querySelector('input');
        inputField.addEventListener('keyup', ({ key }) => {
            if (key === 'Enter') {
                this.onSendButton();
            }
        });
    }

    onSendButton() {
        const { chatBox, chatMessages } = this.args;
        const inputField = chatBox.querySelector('input');
        const text = inputField.value;

        if (text.trim() === '') return;

        // Add user message
        this.addMessage({ name: 'User', message: text });
        inputField.value = '';

        // Send to backend
        fetch('/predict', {
            method: 'POST',
            body: JSON.stringify({ message: text }),
            headers: {
                'Content-Type': 'application/json',
            },
        })
        .then(response => response.json())
        .then(data => {
            this.addMessage({ name: 'Sam', message: data.answer });
        })
        .catch(error => {
            console.error('Error:', error);
            this.addMessage({ name: 'Sam', message: 'Sorry, something went wrong. Please try again.' });
        });
    }

    addMessage(message) {
        const { chatMessages } = this.args;
        const messageElement = document.createElement('div');
        
        messageElement.classList.add('messages__item');
        messageElement.classList.add(
            message.name === 'Sam' 
            ? 'messages__item--operator' 
            : 'messages__item--visitor'
        );
        
        messageElement.innerHTML = message.message;
        chatMessages.appendChild(messageElement);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}

const chatbox = new Chatbox();
document.addEventListener('DOMContentLoaded', () => {
    chatbox.display();
});