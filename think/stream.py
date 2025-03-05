import llama_cpp

def initialize_model(model_path="./luna.gguf", threads=4, context_size=4096, system_prompt="### System: You are Luna, a sexy lady trying to help out. Talk shortly as a conversation"):
    """
    Initialize the LLaMA model with the given parameters.
    
    Args:
        model_path (str): Path to the model file
        threads (int): Number of threads to use
        context_size (int): Context window size
        system_prompt (str): System prompt to initialize the model
        
    Returns:
        llama_cpp.Llama: Initialized model
    """
    # Initialize the model
    llm = llama_cpp.Llama(
        model_path=model_path,
        verbose=False,
        n_ctx=context_size,
        n_threads=threads
    )
    
    # Add the system prompt to initialize the context
    init_tokens = llm.tokenize(bytes(system_prompt, "utf-8"))
    llm.eval(init_tokens)
    
    return llm

def generate_stream(llm, prompt, prefix="### User: ", suffix="### Response: "):
    """
    Generate a streaming response from the model.
    
    Args:
        llm (llama_cpp.Llama): Initialized model
        prompt (str): User prompt
        prefix (str): Prefix to add before the user prompt
        suffix (str): Suffix to add after the user prompt
        
    Yields:
        str: Generated tokens as they are produced
    """
    # Format the prompt
    full_prompt = f"{prefix}{prompt}\n{suffix}"
    ptokens = llm.tokenize(bytes(full_prompt, "utf-8"))
    
    # Count input tokens
    input_token_count = len(ptokens)
    
    # Initialize response tracking
    response_token_count = 0
    full_response = ""
    
    # Define patterns to check and filter
    stop_patterns = [
        "### User", "###User", '###',
        "### Response", "###Response", 
        "### System", "###System"
    ]
    
    # Generate response tokens
    resp_gen = llm.generate(
        ptokens,
        reset=False,
        logits_processor=llama_cpp.LogitsProcessorList([])
    )
    
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
        
        # Yield the token for streaming
        if not should_stop:
            yield word
    
    # Final cleanup - ensure no trailing stop patterns exist
    for pattern in stop_patterns:
        if pattern in full_response:
            full_response = full_response[:full_response.rfind(pattern)]
    
    # Just yield the tokens, don't return anything at the end
