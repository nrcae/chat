import llama_cpp
import os
import argparse
import sys

INITIAL_SYSTEM_PROMPT = {"role": "system", "content": "You are a helpful AI assistant."}

def load_llm(model_path, n_ctx, n_gpu_layers, n_threads, chat_format):
    """Initialize the Llama model with specified parameters."""
    return llama_cpp.Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        chat_format=chat_format,
        verbose=False
    )

def run_minimal_chatbot(
    model_path="app/model/model.gguf", n_gpu_layers=-1, n_threads=None,
    n_ctx=2048, max_tokens_response=350, chat_format='mistral-instruct'
):
    """Runs the main chatbot interactive loop."""
    if n_threads is None:
        n_threads = os.cpu_count()

    try:
        llm = load_llm(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            chat_format=chat_format
        )
        print(f"Model '{model_path}' loaded successfully.")
        # Flag kubernetes readiness
        with open("/tmp/chat_ready", "w") as f:
            f.write("ready")

    except Exception as e:
        print(f"CRITICAL: Error upon loading the model '{model_path}': {e}")
        print("CRITICAL: Assure that the model exists and the path is correct. Chatbot will not start.")
        return

    print(f"AI Chatbot: Hey! What can I do for you? (Type 'exit', 'quit' to cancel, '/clear' or 'help')")

    conversation = [INITIAL_SYSTEM_PROMPT.copy()]
    max_context_tokens = n_ctx - max_tokens_response - 100
    last_user_message_content = None

    while True:
        try:
            user_input = input("You: ").strip()
            user_input_for_processing = None
            if not user_input:
                print("AI Chatbot: Please type a message or command.")
                continue

            if user_input.lower() in ['exit', 'quit']:
                print("AI Chatbot: Goodbye!")
                break
            elif user_input.lower() == '/clear':
                conversation = [INITIAL_SYSTEM_PROMPT.copy()]
                last_user_message_content = None
                print("AI Chatbot: Conversation cleared. How can I help you now?")
                continue
            elif user_input.lower() == '/help':
                print("AI Chatbot: Available commands:")
                print("  /clear          - Clears the conversation history.")
                print("  /status         - Shows the current chatbot parameters.")
                print("  /help           - Shows this help message.")
                print("  exit / quit     - Exits the chatbot.")
                print("  /retry          - Resends the last message for a new response.")
                continue
            elif user_input.lower() == '/status':
                print("AI Chatbot: Current Parameters:")
                print(f"  Model Path:        {model_path}")
                print(f"  Context Size (n_ctx): {n_ctx}")
                print(f"  GPU Layers (n_gpu_layers): {n_gpu_layers}")
                print(f"  CPU Threads (n_threads): {n_threads}")
                print(f"  Max Response Tokens: {max_tokens_response}")
                print(f"  Chat Format:       {chat_format}")
                continue
            elif user_input.lower() == '/retry':
                if last_user_message_content:
                    print(f"AI Chatbot: Retrying your last message: \"{last_user_message_content}\"")
                    user_input_for_processing = last_user_message_content
                    # Logic to ensure the message to retry is correctly in the conversation history
                    if not conversation or conversation[-1]["role"] != "user" or conversation[-1]["content"] != last_user_message_content:
                        if conversation and conversation[-1]["role"] == "assistant":
                            conversation.pop()
                        if not (conversation and conversation[-1]["role"] == "user" and conversation[-1]["content"] == last_user_message_content):
                            conversation.append({"role": "user", "content": last_user_message_content})
                else:
                    user_input_for_processing = user_input
                    last_user_message_content = user_input_for_processing
                    print("AI Chatbot: No previous message to retry.")
                    continue

            conversation.append({"role": "user", "content": user_input_for_processing})

            # Estimate token usage and trim old messages if context is too long
            total_chars = sum(len(msg["content"]) for msg in conversation)
            estimated_tokens = total_chars // 4
            while estimated_tokens > max_context_tokens and len(conversation) > 2:
                # Remove user message
                removed_chars = len(conversation[1]["content"])
                conversation.pop(1)

                # Remove assistant message if present
                if len(conversation) > 1 and conversation[1]["role"] == "assistant":
                    removed_chars += len(conversation[1]["content"])
                    conversation.pop(1)
                
                # Update incrementally
                estimated_tokens -= removed_chars // 4

            print("AI Chatbot: ", end="", flush=True)
            # Generate a response from the model using streaming
            response_stream = llm.create_chat_completion(
                messages=conversation,
                max_tokens=max_tokens_response,
                stream=True
            )

            ai_reply_parts = []
            for chunk in response_stream:
                content = None
                if chunk and 'choices' in chunk and chunk['choices']:
                    delta = chunk['choices'][0].get('delta', {})
                    content = delta.get('content')
                if content:
                    print(content, end="", flush=True)
                    ai_reply_parts.append(content)

            print()
            ai_reply = "".join(ai_reply_parts)

            if not ai_reply.strip():
                ai_reply = "..."

            conversation.append({"role": "assistant", "content": ai_reply})

        except KeyboardInterrupt:
            print("\nAI Chatbot: Goodbye! (Interrupted by user)")
            break
        except EOFError: # Happens if stdin is closed unexpectedly
            print("\nAI Chatbot: Input stream closed. Exiting.")
            break
        except Exception as e:
            print(f"\nERROR: An error occurred during interaction: {type(e).__name__}: {e}")
            print("AI Chatbot: Sorry, I faced an issue. Let's try again or type 'exit', '/clear' or '/help'.")
            # Avoid getting stuck with problematic user message
            if conversation and conversation[-1]["role"] == "user":
                 conversation.pop() # Remove last user message to prevent re-sending a potentially problematic one


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal Llama Chatbot")
    parser.add_argument("--model-path", type=str, default="app/model/model.gguf", help="Path to GGUF model")
    parser.add_argument("--n-gpu-layers", type=int, default=0, help="Number of layers to offload to GPU (0 for CPU only by default as per Docker CMD)")
    parser.add_argument("--n-threads", type=int, default=None, help="Number of CPU threads to use (None for all cores)")
    parser.add_argument("--n-ctx", type=int, default=2048, help="Context window size")
    parser.add_argument("--max-tokens-response", type=int, default=350, help="Max tokens in AI response")
    parser.add_argument("--chat-format", type=str, default='mistral-instruct', help="Chat format string for LlamaCPP")

    args = parser.parse_args()
    if args.n_ctx <= args.max_tokens_response + 100:
        print(f"ERROR: Context size ({args.n_ctx}) must be larger than max_tokens_response + 100 ({args.max_tokens_response + 100})")
        sys.exit(1)

    run_minimal_chatbot(
        model_path=args.model_path,
        n_gpu_layers=args.n_gpu_layers,
        n_threads=args.n_threads,
        n_ctx=args.n_ctx,
        max_tokens_response=args.max_tokens_response,
        chat_format=args.chat_format
    )
