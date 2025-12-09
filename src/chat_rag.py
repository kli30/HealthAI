from llm_client import get_llm_client
from rag_system import initialize_rag_with_transcripts

# Initialize the LLM client (defaults to OpenAI)
client = get_llm_client()

# ANSI color codes
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def chat_with_rag():
    """Interactive chatbot with RAG-enhanced responses."""
    provider_name = client.provider.upper()
    model_name = client.model
    print(f"Welcome to the AI Chatbot with RAG! (Using {provider_name}: {model_name})")
    print("This chatbot has access to Huberman Lab podcast transcripts.")
    print("Type 'quit' to exit the chat.")
    print()

    # Initialize RAG system
    print(f"{YELLOW}Loading RAG system...{RESET}")
    rag = initialize_rag_with_transcripts()
    print(f"{YELLOW}RAG system ready!{RESET}\n")

    conversation = []

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
            conversation.append({"role": "user", "content": user_input})

        print(f"{GREEN}AI: {RESET}", end="", flush=True)

        assistant_response = ""
        for content in client.stream_chat(messages=conversation, max_tokens=2000):
            print(f"{GREEN}{content}{RESET}", end="", flush=True)
            assistant_response += content

        print()  # New line after the complete response

        conversation.append({"role": "assistant", "content": assistant_response})


if __name__ == "__main__":
    chat_with_rag()
