import llama_cpp
import re
from collections import deque

# Updated model path to use luna.gguf
MODEL_PATH = "./luna.gguf"
MODEL_THREADS = 4
MODEL_PREFIX = "### User: "
MODEL_SUFFIX = "### Response: "
MODEL_INIT = "### System: You are Luna, a sexy lady trying to help out."

# Conversation history tracking
conversation_history = deque(maxlen=10)  # Keep last 10 exchanges
conversation_tokens = 0

def initialize_model():
    global llm, conversation_tokens
    llm = llama_cpp.Llama(
        model_path=MODEL_PATH,
        verbose=False,
        n_ctx=100,
        n_threads=MODEL_THREADS)
    
    # Add the system prompt to initialize the context
    init_tokens = llm.tokenize(bytes(MODEL_INIT, "utf-8"))
    llm.eval(init_tokens)
    
    # Reset conversation token count and track system prompt tokens
    conversation_tokens = len(init_tokens)
    conversation_history.clear()
    return "Model initialized successfully."

def trim_conversation():
    """Trim conversation history while keeping recent exchanges"""
    global conversation_tokens, llm, conversation_history
    
    # Start fresh but keep the system prompt
    llm = llama_cpp.Llama(
        model_path=MODEL_PATH,
        verbose=False,
        n_ctx=4096,
        n_threads=MODEL_THREADS)
    
    # Add the system prompt to initialize the context
    init_tokens = llm.tokenize(bytes(MODEL_INIT, "utf-8"))
    llm.eval(init_tokens)
    conversation_tokens = len(init_tokens)
    
    # Keep only the 3 most recent exchanges from history
    recent_exchanges = list(conversation_history)[-3:] if conversation_history else []
    
    # Clear the history and re-add recent exchanges
    conversation_history.clear()
    
    # Re-add the most recent exchanges to the context
    for exchange in recent_exchanges:
        user_msg, assistant_msg = exchange
        
        # Re-add user message
        user_tokens = llm.tokenize(bytes(f"{MODEL_PREFIX}{user_msg}\n", "utf-8"))
        llm.eval(user_tokens)
        
        # Re-add assistant message
        assistant_tokens = llm.tokenize(bytes(f"{MODEL_SUFFIX}{assistant_msg}\n", "utf-8"))
        llm.eval(assistant_tokens)
        
        # Add to history
        conversation_history.append((user_msg, assistant_msg))
        
        # Update token count
        conversation_tokens += len(user_tokens) + len(assistant_tokens)
    
    return f"Conversation pruned to the {len(recent_exchanges)} most recent exchanges."

# Initialize model at startup
initialize_model()

print("\nHow can I help you? ")
while True:
    user_prompt = input("\n> ")
    if (user_prompt.lower() == "quit") or (user_prompt.lower() == "exit"):
        break
    
    # Add manual reset command
    if user_prompt.lower() == "reset":
        message = initialize_model()
        print(f"\n{message}")
        print("Conversation has been reset.")
        continue
    
    try:
        full_prompt = f"{MODEL_PREFIX}{user_prompt}\n{MODEL_SUFFIX}"
        ptokens = llm.tokenize(bytes(full_prompt, "utf-8"))
        
        # Count input tokens
        input_token_count = len(ptokens)
        print(f"\nInput tokens: {input_token_count}")
        
        # Check if we're approaching context limit (75% of context size)
        if conversation_tokens + input_token_count > 3000:  # 75% of 4096 is ~3000
            print("\nContext window limit approaching. Trimming older conversation...")
            trim_message = trim_conversation()
            print(trim_message)
            
            # Re-tokenize after trim
            ptokens = llm.tokenize(bytes(full_prompt, "utf-8"))
            input_token_count = len(ptokens)
            print(f"Input tokens after trim: {input_token_count}")
        
        # Update conversation token count
        conversation_tokens += input_token_count
        
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
        
        # Add this exchange to the conversation history
        conversation_history.append((user_prompt, full_response))
        
        # Update conversation tokens count with response tokens
        conversation_tokens += response_token_count
        
        # Display token counts
        print(f"\n\nResponse tokens: {response_token_count}")
        print(f"Total tokens: {input_token_count + response_token_count}")
        print(f"Total conversation tokens: {conversation_tokens}")
        
    except RuntimeError as e:
        if "llama_decode returned 1" in str(e):
            print("\n\n[Context window exceeded. Trimming conversation.]")
            trim_message = trim_conversation()
            print(trim_message)
            print("Could you please repeat your last question?")
        else:
            print(f"\n\nError: {str(e)}")
    except Exception as e:
        print(f"\n\nUnexpected error: {str(e)}")
