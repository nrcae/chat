import llama_cpp
import os
import argparse

def load_llm(model_path, n_ctx, n_gpu_layers, n_threads, chat_format):
    # Initialize the Llama model with specified parameters
    return llama_cpp.Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        chat_format=chat_format
    )

def run_minimal_chatbot(
    model_path="app/model/model.gguf",
    n_gpu_layers=-1,
    n_threads=None,                   
    n_ctx=2048,
    max_tokens_response=350,
    chat_format='mistral-instruct'
):
    if n_threads is None:
        n_threads = os.cpu_count()  # Use all available CPU threads by default

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
 
    print("AI Chatbot: Hey! What can I do for you? (Type 'exit' or 'quit' to cancel)")

    # Initialize conversation with a system prompt
    conversation = [
        {"role": "system", "content": "You are a helpful AI assistant."}
    ]

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ['exit', 'quit']:
                print("AI Chatbot: Goodbye!")
                break
            
            if not user_input: # Handle empty input
                continue
            # Estimate token usage and trim old messages if context is too long
            total_chars = sum(len(msg["content"]) for msg in conversation)
            estimated_tokens = total_chars // 4
            while estimated_tokens > (n_ctx - max_tokens_response - 100) and len(conversation) > 3:
                conversation.pop(1)  # Remove oldest user message
                if len(conversation) > 1:
                    conversation.pop(1)  # Remove corresponding assistant message
                total_chars = sum(len(msg["content"]) for msg in conversation)
                estimated_tokens = total_chars // 4

            # Generate a response from the model
            response = llm.create_chat_completion(
                messages=conversation,
                max_tokens=max_tokens_response
            )

            # Extract assistant's reply from response object
            if hasattr(response, 'choices') and response.choices:
                ai_reply = response.choices[0].message["content"]
            else:
                ai_reply = response["choices"][0]["message"]["content"]

            print(f"AI Chatbot: {ai_reply}")

            # Add assistant's reply to conversation history
            conversation.append({"role": "assistant", "content": ai_reply})
        except Exception as e:
            print(f"\nError during interaction: {type(e).__name__}: {e}")
            print("AI Chatbot: Sorry, I faced an issue. Let's try again or type 'exit' to quit.")
            if conversation and conversation[-1]["role"] == "user":
                conversation.pop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal Llama Chatbot")
    parser.add_argument("--model-path", type=str, default="app/model/model.gguf", help="Path to GGUF model")
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--n-threads", type=int, default=None)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--max-tokens-response", type=int, default=350)
    parser.add_argument("--chat-format", type=str, default='mistral-instruct')
    args = parser.parse_args()

    run_minimal_chatbot(
        model_path=args.model_path,
        n_gpu_layers=args.n_gpu_layers,
        n_threads=args.n_threads,
        n_ctx=args.n_ctx,
        max_tokens_response=args.max_tokens_response,
        chat_format=args.chat_format
    )