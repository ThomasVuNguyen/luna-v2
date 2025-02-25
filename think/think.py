import llama_cpp

MODEL_PATH = "./luna.gguf"
MODEL_THREADS = 4
MODEL_PREFIX = "### User: "
MODEL_SUFFIX = "### Response: "
MODEL_INIT = "### System: You are a coding assistant, skilled in programming."

llm = llama_cpp.Llama(
    model_path=MODEL_PATH,
    verbose=False,
    n_ctx=4096,
    n_threads=MODEL_THREADS)

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
    
    # Initialize counter for response tokens
    response_token_count = 0
    
    resp_gen = llm.generate(
        ptokens,
        reset=False,
        logits_processor=llama_cpp.LogitsProcessorList([]))
    
    for tok in resp_gen:
        if tok == llm.token_eos():
            break
        response_token_count += 1
        word = llm.detokenize([tok]).decode("utf-8", errors="ignore")
        print(word, end="", flush=True)
    
    # Print token counts
    print(f"\n\nResponse tokens: {response_token_count}")
    print(f"Total tokens: {input_token_count + response_token_count}")
