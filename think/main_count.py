import time
from stream import initialize_model, generate_stream

def main():
    # Initialize the model
    print("Initializing model...")
    llm = initialize_model(
        model_path="./luna.gguf",  # Updated to use luna-hermes.gguf
        threads=4,
        context_size=4096,
        system_prompt="### System: You are Luna and you speak flirty."
    )
    
    print("\nModel initialized. How can I help you?")
    
    # Main interaction loop
    while True:
        user_prompt = input("\n> ")
        
        # Check for exit command
        if user_prompt.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break
        
        # Format and tokenize the prompt to get input token count
        prefix = "### User: "
        suffix = "### Response: "
        full_prompt = f"{prefix}{user_prompt}\n{suffix}"
        input_tokens = len(llm.tokenize(bytes(full_prompt, "utf-8")))
        
        print(f"\nInput tokens: {input_tokens}")
        print("Response: ", end="", flush=True)
        
        # Stream the response
        response_tokens = 0
        start_time = time.time()
        last_token_time = start_time
        token_times = []
        
        for token in generate_stream(llm, user_prompt):
            print(token, end="", flush=True)
            response_tokens += 1
            
            # Calculate token generation rate
            current_time = time.time()
            token_times.append(current_time - last_token_time)
            last_token_time = current_time
        
        # Calculate overall statistics
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Avoid division by zero if no tokens were generated
        tokens_per_second = response_tokens / total_duration if total_duration > 0 else 0
        
        # Calculate average time per token
        avg_time_per_token = sum(token_times) / len(token_times) if token_times else 0
        avg_tokens_per_second = 1 / avg_time_per_token if avg_time_per_token > 0 else 0
        
        # Print token statistics
        print(f"\n\nResponse tokens: {response_tokens}")
        print(f"Total tokens: {input_tokens + response_tokens}")
        print(f"Response time: {total_duration:.2f} seconds")
        print(f"Token rate: {tokens_per_second:.2f} tokens/sec")
        print(f"Average token rate: {avg_tokens_per_second:.2f} tokens/sec")

if __name__ == "__main__":
    main()
