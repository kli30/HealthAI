from flask import Flask, render_template, request, jsonify, Response
from anthropic import Anthropic
from rag_system import initialize_rag_with_transcripts
import json

app = Flask(__name__)

# Initialize the Anthropic client
client = Anthropic()

# Initialize RAG system
print("Loading RAG system...")
rag = initialize_rag_with_transcripts()
print("RAG system ready!")

# Store conversation history per session (in production, use proper session management)
conversations = {}


@app.route('/')
def index():
    """Serve the chatbox interface."""
    return render_template('chatbox.html')


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages with streaming response."""
    data = request.json
    user_message = data.get('message', '')
    session_id = data.get('session_id', 'default')

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    # Get or create conversation history for this session
    if session_id not in conversations:
        conversations[session_id] = []

    conversation = conversations[session_id]

    # Retrieve relevant context from transcripts
    context = rag.get_context_for_query(user_message, n_results=3)

    # Build the message with context if available
    if context:
        enhanced_message = f"{context}\n\nUser question: {user_message}"
        conversation.append({"role": "user", "content": enhanced_message})
        has_context = True
    else:
        conversation.append({"role": "user", "content": user_message})
        has_context = False

    def generate():
        """Generator function for streaming response."""
        # Send context indicator first
        yield f"data: {json.dumps({'type': 'context', 'has_context': has_context})}\n\n"

        # Stream the Claude response
        stream = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=2000,
            messages=conversation,
            stream=True
        )

        assistant_response = ""
        for chunk in stream:
            if chunk.type == "content_block_delta":
                content = chunk.delta.text
                assistant_response += content
                yield f"data: {json.dumps({'type': 'content', 'text': content})}\n\n"

        # Add assistant response to conversation history
        conversation.append({"role": "assistant", "content": assistant_response})

        # Send completion signal
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return Response(generate(), mimetype='text/event-stream')


@app.route('/clear', methods=['POST'])
def clear_conversation():
    """Clear conversation history for a session."""
    data = request.json
    session_id = data.get('session_id', 'default')

    if session_id in conversations:
        conversations[session_id] = []

    return jsonify({'status': 'success'})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
