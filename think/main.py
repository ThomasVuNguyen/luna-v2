from stream import initialize_model, generate_stream

def main():
    # Initialize the model
    print("Initializing model...")
    llm = initialize_model(
        model_path="./luna.gguf",
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
        for token in generate_stream(llm, user_prompt):
            print(token, end="", flush=True)
            response_tokens += 1
        
        # Print token statistics
        print(f"\n\nResponse tokens: {response_tokens}")
        print(f"Total tokens: {input_tokens + response_tokens}")

if __name__ == "__main__":
    main()
