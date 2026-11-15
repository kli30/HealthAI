import argparse
from llm_client import get_llm_client
from rag_system import initialize_rag_with_transcripts

# ANSI color codes
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

# System prompt for RAG-enhanced responses
SYSTEM_PROMPT = """You are a helpful AI assistant with access to a database of expert transcripts and documents.

IMPORTANT INSTRUCTIONS:

1. **Answer Based on Retrieved Context**: If relevant context is provided below, answer the question ONLY using information from that context. Do not use your general knowledge unless explicitly needed to clarify or explain concepts.

2. **Citation Requirements**: When using information from the context, ALWAYS cite your sources by stating:
   - The author/expert name (e.g., "According to Andrew Huberman...")
   - The source document if available (e.g., "from the transcript on sleep optimization")

3. **Concise Summaries**: Summarize the information from the context in a clear and concise way. Don't just copy the text - synthesize and explain it.

4. **When No Relevant Context is Found**: If no context is provided or the context is not relevant to the question:
   - Clearly state: "I could not find relevant information about this in the available database."
   - Then offer: "However, based on my training data, I can provide some general information: [answer]"
   - Make it clear you're switching from the database to general knowledge.

5. **Handle Unconventional Views**: If the information from the context differs significantly from mainstream scientific consensus:
   - Provide a brief notice like: "Note: This perspective may differ from mainstream scientific consensus."
   - Briefly mention the major difference if it's important for context.
   - Still present the information from the source faithfully.

Remember: Prioritize the retrieved context over your general knowledge when relevant context is available."""

def chat_with_rag(chroma_db: str = "./chroma_db", use_reranking: bool = True):
    """Interactive chatbot with RAG-enhanced responses.

    Args:
        chroma_db: Path to ChromaDB database directory (default: ./chroma_db)
        use_reranking: If True, use cross-encoder reranking for improved relevance (default: True)
    """
    # Initialize the LLM client (defaults to OpenAI)
    client = get_llm_client()

    provider_name = client.provider.upper()
    model_name = client.model
    print(f"Welcome to the AI Chatbot with RAG! (Using {provider_name}: {model_name})")
    print(f"Database: {chroma_db}")
    print("This chatbot has access to transcript data.")
    print("Type 'quit' to exit the chat.")
    print()

    # Initialize RAG system
    print(f"{YELLOW}Loading RAG system from {chroma_db}...{RESET}")
    rag = initialize_rag_with_transcripts(chroma_db, use_reranking=use_reranking)
    print(f"{YELLOW}RAG system ready!{RESET}")

    # Show available authors
    authors = rag.get_authors()
    if authors:
        print(f"{YELLOW}Available authors: {', '.join(authors)}{RESET}")
    print()

    # Initialize conversation with system prompt
    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        user_input = input(f"{BLUE}You: {RESET}")

        if user_input.lower() == 'quit':
            print("Goodbye!")
            break

        # Retrieve relevant context from transcripts
        context = rag.get_context_for_query(user_input, n_results=3)

        # Build the message with context if available
        if context:
            enhanced_message = f"{context}\n\nUser question: {user_input}"
            conversation.append({"role": "user", "content": enhanced_message})
            print(f"{YELLOW}[Retrieved context from transcripts]{RESET}")
        else:
            # No context found - inform the system
            conversation.append({"role": "user", "content": f"[No relevant context found in database]\n\nUser question: {user_input}"})

        print(f"{GREEN}AI: {RESET}", end="", flush=True)

        assistant_response = ""
        for content in client.stream_chat(messages=conversation, max_tokens=2000):
            print(f"{GREEN}{content}{RESET}", end="", flush=True)
            assistant_response += content

        print()  # New line after the complete response

        conversation.append({"role": "assistant", "content": assistant_response})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAG-enhanced chatbot with transcript search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default database (./chroma_db_context)
  python chat_rag.py

  # Use custom database
  python chat_rag.py --db ./chroma_db_context

  # Use alternative database
  python chat_rag.py --db ./my_custom_db
        """
    )

    parser.add_argument(
        "--db", "--chroma-db",
        dest="chroma_db",
        default="./chroma_db_context",
        help="Path to ChromaDB database directory (default: ./chroma_db_context)"
    )

    parser.add_argument(
        "--no-reranking",
        dest="use_reranking",
        action="store_false",
        help="Disable cross-encoder reranking (enabled by default)"
    )

    args = parser.parse_args()

    chat_with_rag(chroma_db=args.chroma_db, use_reranking=args.use_reranking)
