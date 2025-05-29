import llama_cpp
import os
import argparse

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

    except Exception as e:
        print(f"CRITICAL: Error upon loading the model '{model_path}': {e}")
        print("CRITICAL: Assure that the model exists and the path is correct. Chatbot will not start.")
        return # Exit if model loading fails

    print(f"AI Chatbot: Hey! What can I do for you? (Type 'exit', 'quit' to cancel, or '/clear' to reset conversation)")

    conversation = [INITIAL_SYSTEM_PROMPT.copy()]

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                print("AI Chatbot: Please type a message or command.")
                continue

            if user_input.lower() in ['exit', 'quit']:
                print("AI Chatbot: Goodbye!")
                break
            elif user_input.lower() == '/clear':
                conversation = [INITIAL_SYSTEM_PROMPT.copy()]
                print("AI Chatbot: Conversation cleared. How can I help you now?")
                continue

            conversation.append({"role": "user", "content": user_input})

            # Estimate token usage and trim old messages if context is too long
            total_chars = sum(len(msg["content"]) for msg in conversation)
            estimated_tokens = total_chars // 4
            while estimated_tokens > (n_ctx - max_tokens_response - 100) and len(conversation) > 2:
                conversation.pop(1)
                if len(conversation) > 1 and conversation[1]["role"] == "assistant":
                    conversation.pop(1)
                total_chars = sum(len(msg["content"]) for msg in conversation)
                estimated_tokens = total_chars // 4

            print("AI Chatbot: ", end="", flush=True)

            # Generate a response from the model using streaming
            response_stream = llm.create_chat_completion(
                messages=conversation,
                max_tokens=max_tokens_response,
                stream=True
            )

            ai_reply_parts = []
            for chunk in response_stream:
                content_piece = None
                if chunk and 'choices' in chunk and chunk['choices']:
                    delta = chunk['choices'][0].get('delta', {})
                    content_piece = delta.get('content')
                if content_piece:
                    print(content_piece, end="", flush=True)
                    ai_reply_parts.append(content_piece)

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
            print("AI Chatbot: Sorry, I faced an issue. Let's try again or type 'exit' or '/clear'.")
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

    run_minimal_chatbot(
        model_path=args.model_path,
        n_gpu_layers=args.n_gpu_layers,
        n_threads=args.n_threads,
        n_ctx=args.n_ctx,
        max_tokens_response=args.max_tokens_response,
        chat_format=args.chat_format
    )
