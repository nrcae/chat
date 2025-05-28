import llama_cpp
import os
import argparse

# Store the initial system prompt as a constant
INITIAL_SYSTEM_PROMPT = {"role": "system", "content": "You are a helpful AI assistant."}

def load_llm(model_path, n_ctx, n_gpu_layers, n_threads, chat_format):
    """Initialize the Llama model with specified parameters."""
    return llama_cpp.Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        chat_format=chat_format,
        verbose=False # Reduce llama_cpp's own console output
    )

def run_minimal_chatbot(
    model_path="app/model/model.gguf",
    n_gpu_layers=-1,
    n_threads=None,
    n_ctx=2048,
    max_tokens_response=350,
    chat_format='mistral-instruct'
):
    """Runs the main chatbot interactive loop."""
    if n_threads is None:
        n_threads = os.cpu_count()

    try:
        # Load the language model
        llm = load_llm(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            chat_format=chat_format
        )
    except Exception as e:
        print(f"Error upon loading the model '{model_path}': {e}")
        print("Assure that the model exists and the path is correct.")
        return

    print("AI Chatbot: Hey! What can I do for you? (Type 'exit', 'quit' to cancel, or '/clear' to reset conversation)")

    # Initialize conversation with a copy of the system prompt
    conversation = [INITIAL_SYSTEM_PROMPT.copy()]

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ['exit', 'quit']:
                print("AI Chatbot: Goodbye!")
                break
            elif user_input.lower() == '/clear':
                conversation = [INITIAL_SYSTEM_PROMPT.copy()]
                print("AI Chatbot: Conversation cleared. How can I help you now?")
                continue
            elif not user_input:
                continue

            conversation.append({"role": "user", "content": user_input})

            # Estimate token usage and trim old messages if context is too long
            total_chars = sum(len(msg["content"]) for msg in conversation)
            estimated_tokens = total_chars // 4

            # Keep system prompt (index 0) and at least one user/assistant pair if trimming.
            while estimated_tokens > (n_ctx - max_tokens_response - 100) and len(conversation) > 2:
                conversation.pop(1) # Remove oldest user message (after system prompt)
                if len(conversation) > 1 and conversation[1]["role"] == "assistant":
                    conversation.pop(1) # Remove corresponding assistant message

                # Recalculate estimated tokens after trimming
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
                # Standard way to access content in OpenAI-compatible streams
                content_piece = None
                if chunk and 'choices' in chunk and chunk['choices']:
                    delta = chunk['choices'][0].get('delta', {})
                    content_piece = delta.get('content')

                if content_piece:
                    print(content_piece, end="", flush=True)
                    ai_reply_parts.append(content_piece)

            print()

            ai_reply = "".join(ai_reply_parts)

            if not ai_reply.strip(): # If only whitespace or empty
                ai_reply = "..." # Fallback for empty or non-substantive streamed response

            # Add assistant's reply to conversation history
            conversation.append({"role": "assistant", "content": ai_reply})

        except KeyboardInterrupt: # Handle Ctrl+C
            print("\nAI Chatbot: Goodbye! (Interrupted by user)")
            break
        except Exception as e:
            print(f"\nError during interaction: {type(e).__name__}: {e}") # Ensure newline if error happens mid-stream
            print("AI Chatbot: Sorry, I faced an issue. Let's try again or type 'exit' or '/clear'.")
            # Optionally remove the last user message if processing failed
            if conversation and conversation[-1]["role"] == "user":
                 conversation.pop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal Llama Chatbot")
    parser.add_argument("--model-path", type=str, default="app/model/model.gguf", help="Path to GGUF model")
    parser.add_argument("--n-gpu-layers", type=int, default=-1, help="Number of layers to offload to GPU (-1 for model default, 0 for CPU only)")
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
