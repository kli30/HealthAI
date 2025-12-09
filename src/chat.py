from llm_client import get_llm_client
import os

# Initialize the LLM client (defaults to OpenAI)
# Set environment variables:
#   export OPENAI_API_KEY=your_openai_key (for OpenAI - default)
#   export ANTHROPIC_API_KEY=your_anthropic_key (for Anthropic)
#   export LLM_PROVIDER=openai or anthropic (optional, defaults to openai)
client = get_llm_client()

# ANSI color codes
BLUE = "\033[94m"
GREEN = "\033[92m"
RESET = "\033[0m"

def chat_with_claude():
    provider_name = client.provider.upper()
    model_name = client.model
    print(f"Welcome to the AI Chatbot! (Using {provider_name}: {model_name})")
    print("Type 'quit' to exit the chat.")
    
    conversation = []
    
    while True:
        user_input = input(f"{BLUE}You: {RESET}")
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        conversation.append({"role": "user", "content": user_input})

        print(f"{GREEN}AI: {RESET}", end="", flush=True)

        assistant_response = ""
        for content in client.stream_chat(messages=conversation, max_tokens=1000):
            print(f"{GREEN}{content}{RESET}", end="", flush=True)
            assistant_response += content
        
        print()  # New line after the complete response
        
        conversation.append({"role": "assistant", "content": assistant_response})

if __name__ == "__main__":
    chat_with_claude()
