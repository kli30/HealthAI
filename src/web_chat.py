import argparse
from flask import Flask, render_template, request, jsonify, Response
from llm_client import get_llm_client
from rag_system import initialize_rag_with_transcripts
import json
import pdb  # Python debugger

app = Flask(__name__)

# Global variables (will be initialized in main)
client = None
rag = None

# System prompt for RAG-enhanced responses
SYSTEM_PROMPT = """You are a helpful AI assistant with access to a database of expert transcripts and documents.

IMPORTANT INSTRUCTIONS:

1. **Answer Based on Retrieved Context**: If relevant context is provided below, answer the question ONLY using information from that context. Do not use your general knowledge unless explicitly needed to clarify or explain concepts.

2. **Citation Requirements**: When using information from the context, ALWAYS cite your sources by stating:
   - The author/expert name (e.g., "According to Andrew Huberman...")
   - The source document if available (e.g., "from the transcript on sleep optimization")

3. **Concise Summaries**: Summarize the information from the context in a clear and concise way. Don't just copy the text - synthesize and explain it.

4. **When No Relevant Context is Found**: If no context is provided or the context is not relevant to the question:
   - Clearly state: "I could not find relevant information about this in the available transcripts/documents."
   - Then offer: "However, based on my training data, I can provide some general information: [answer]"
   - Make it clear you're switching from the database to general knowledge.

5. **Handle Unconventional Views**: If the information from the context differs significantly from mainstream scientific consensus:
   - Provide a brief notice like: "Note: This perspective may differ from mainstream scientific consensus."
   - Briefly mention the major difference if it's important for context.
   - Still present the information from the source faithfully.

Remember: Prioritize the retrieved context over your general knowledge when relevant context is available."""

# System prompt for RAG-enhanced responses
SYSTEM_PROMPT_EVAL = """You are a helpful AI assistant with access to a database of expert transcripts and documents.

IMPORTANT INSTRUCTIONS:

1. **Answer Based on Retrieved Context**: If relevant context is provided below, answer the question ONLY using information from that context. Do not use your general knowledge unless explicitly needed to clarify or explain concepts.

2. **Citation Requirements**: When using information from the context, ALWAYS cite your sources by stating:
   - The author/expert name (e.g., "According to Andrew Huberman...")
   - The source document if available (e.g., "from the transcript on sleep optimization")

3. **Concise Summaries**: Summarize the information from the context in a clear and concise way. Don't just copy the text - synthesize and explain it.

4. **When No Relevant Context is Found**: Clearly state: "I could not find relevant information about this in the available transcripts/documents." 
 

Remember: Prioritize the retrieved context over your general knowledge when relevant context is available."""


# Store conversation history per session (in production, use proper session management)
conversations = {}


@app.route('/')
def index():
    """Serve the chatbox interface."""
    return render_template('chatbox.html')


@app.route('/authors', methods=['GET'])
def get_authors():
    """Get list of available authors in the RAG system."""
    authors = rag.get_authors()
    return jsonify({'authors': authors})


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages with streaming response."""
    # BREAKPOINT: Uncomment the line below to debug at the start of chat route
    # pdb.set_trace()

    data = request.json
    user_message = data.get('message', '')
    session_id = data.get('session_id', 'default')
    selected_authors = data.get('selected_authors', [])

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    # Get or create conversation history for this session
    if session_id not in conversations:
        # Initialize new session with system prompt
        conversations[session_id] = [{"role": "system", "content": SYSTEM_PROMPT_EVAL}]

    conversation = conversations[session_id]

    # Store only the raw user message in conversation history
    conversation.append({"role": "user", "content": user_message})

    # Retrieve relevant context from transcripts, filtered by selected authors
    # BREAKPOINT: Uncomment to debug RAG context retrieval
    # pdb.set_trace()
    context = rag.get_context_for_query(user_message, n_results=3, author_filter=selected_authors)
    has_context = bool(context)

    def generate():
        """Generator function for streaming response."""
        # BREAKPOINT: Uncomment to debug streaming response generation
        # pdb.set_trace()

        # Send context indicator first
        yield f"data: {json.dumps({'type': 'context', 'has_context': has_context})}\n\n"

        # Build messages array with context injected only for the current query
        # Start with system prompt and all messages except the last user message
        messages_with_context = conversation[:-1]

        # Add the current user message with context if available
        if context:
            enhanced_message = f"{context}\n\nUser question: {user_message}"
            messages_with_context.append({"role": "user", "content": enhanced_message})
        else:
            # No context found - inform the system
            no_context_message = f"[No relevant context found in database]\n\nUser question: {user_message}"
            messages_with_context.append({"role": "user", "content": no_context_message})

        # Stream the LLM response
        # BREAKPOINT: Uncomment to debug LLM streaming
        # pdb.set_trace()
        assistant_response = ""
        for content in client.stream_chat(messages=messages_with_context, max_tokens=2000):
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
        # Reset to just the system prompt
        conversations[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

    return jsonify({'status': 'success'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Web-based RAG chatbot with transcript search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default database (./chroma_db) on default port (5000)
  python web_chat.py

  # Use custom database
  python web_chat.py --db ./chroma_db_context

  # Use custom port
  python web_chat.py --port 8080

  # Use custom database and port
  python web_chat.py --db ./my_custom_db --port 8080
        """
    )

    parser.add_argument(
        "--db", "--chroma-db",
        dest="chroma_db",
        default="./chroma_db",
        help="Path to ChromaDB database directory (default: ./chroma_db)"
    )

    parser.add_argument(
        "--port", "-p",
        type=int,
        default=5000,
        help="Port to run the web server on (default: 5000)"
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to run the web server on (default: 127.0.0.1)"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (default: False)"
    )

    args = parser.parse_args()

    # Initialize the LLM client (defaults to OpenAI)
    client = get_llm_client()

    # Initialize RAG system
    print(f"Using LLM: {client.provider.upper()} ({client.model})")
    print(f"Loading RAG system from {args.chroma_db}...")
    rag = initialize_rag_with_transcripts(args.chroma_db)
    print("RAG system ready!")

    # Show available authors
    authors = rag.get_authors()
    if authors:
        print(f"Available authors: {', '.join(authors)}")
    print()

    print(f"Starting web server on http://{args.host}:{args.port}")
    app.run(debug=args.debug, port=args.port, host=args.host)
