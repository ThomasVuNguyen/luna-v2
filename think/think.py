import llama_cpp
import re

# Updated model path to use luna.gguf
MODEL_PATH = "./luna.gguf"
MODEL_THREADS = 4
MODEL_PREFIX = "### User: "
MODEL_SUFFIX = "### Response: "
MODEL_INIT = "### System: You are Luna, a sexy lady trying to help out."
llm = llama_cpp.Llama(
    model_path=MODEL_PATH,
    verbose=False,
    n_ctx=4096,
    n_threads=MODEL_THREADS)

# Add the system prompt to initialize the context
init_tokens = llm.tokenize(bytes(MODEL_INIT, "utf-8"))
llm.eval(init_tokens)

print("\nHow can I help you? ")
while True:
    user_prompt = input("\n> ")
    if (user_prompt.lower() == "quit") or (user_prompt.lower() == "exit"):
        break
        
    full_prompt = f"{MODEL_PREFIX}{user_prompt}\n{MODEL_SUFFIX}"
    ptokens = llm.tokenize(bytes(full_prompt, "utf-8"))
    
    # Count input tokens
    input_token_count = len(ptokens)
    print(f"\nInput tokens: {input_token_count}")
    
    # Initialize response tracking
    response_token_count = 0
    full_response = ""
    
    # Define patterns to check and filter
    stop_patterns = [
        "### User", "###User", '###',
        "### Response", "###Response", 
        "### System", "###System"
    ]
    
    resp_gen = llm.generate(
        ptokens,
        reset=False,
        logits_processor=llama_cpp.LogitsProcessorList([]))
    
    print("\nResponse: ", end="", flush=True)
    
    for tok in resp_gen:
        if tok == llm.token_eos():
            break
            
        word = llm.detokenize([tok]).decode("utf-8", errors="ignore")
        full_response += word
        response_token_count += 1
        
        # Check if adding this token creates a stop sequence
        should_stop = False
        for pattern in stop_patterns:
            if pattern in full_response[-20:]:  # Check last 20 chars for efficiency
                # Trim the response at the stop pattern
                full_response = full_response[:full_response.rfind(pattern)]
                should_stop = True
                break
                
        if should_stop:
            break
        
        # Only print if we're still outputting (haven't hit a stop pattern)
        if not should_stop:
            print(word, end="", flush=True)
    
    # Final cleanup - ensure no trailing stop patterns exist
    for pattern in stop_patterns:
        if pattern in full_response:
            full_response = full_response[:full_response.rfind(pattern)]
    
    # Clear current line and print the cleaned response if it differs from what was printed
    current_display = full_response
    if current_display != full_response:
        # Calculate how many lines to clear based on printed content
        print("\r" + " " * len(current_display), end="\r")
        print(full_response, end="")
    
    # Display token counts
    print(f"\n\nResponse tokens: {response_token_count}")
    print(f"Total tokens: {input_token_count + response_token_count}")
