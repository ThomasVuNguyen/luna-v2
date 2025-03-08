from flask import Flask, request, jsonify, Response, stream_with_context
import llama_cpp
import os
import threading
import time
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global model instance
llm = None
model_lock = threading.Lock()

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

def get_model():
    """
    Get the global model instance, initializing it if necessary.
    Uses a lock to ensure thread safety when initializing.
    """
    global llm
    
    if llm is None:
        with model_lock:
            if llm is None:  # Double-check lock pattern
                model_path = os.environ.get('MODEL_PATH', './luna.gguf')
                threads = int(os.environ.get('MODEL_THREADS', '4'))
                context_size = int(os.environ.get('CONTEXT_SIZE', '4096'))
                system_prompt = os.environ.get('SYSTEM_PROMPT', 
                                              "### System: You are Luna, a sexy lady trying to help out. Talk shortly as a conversation")
                
                llm = initialize_model(
                    model_path=model_path,
                    threads=threads,
                    context_size=context_size,
                    system_prompt=system_prompt
                )
                
    return llm

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    API endpoint for generating a complete response (non-streaming).
    """
    try:
        # Get model
        model = get_model()
        
        # Get request data
        data = request.json
        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing prompt"}), 400
            
        prompt = data['prompt']
        
        # Optional parameters
        prefix = data.get('prefix', "### User: ")
        suffix = data.get('suffix', "### Response: ")
        
        # Generate full response
        with model_lock:  # Lock while generating to prevent concurrent access
            response_text = ""
            for token in generate_stream(model, prompt, prefix, suffix):
                response_text += token
        
        return jsonify({
            "response": response_text.strip(),
            "model": os.path.basename(os.environ.get('MODEL_PATH', 'luna.gguf'))
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """
    API endpoint for streaming response generation.
    """
    try:
        # Get model
        model = get_model()
        
        # Get request data
        data = request.json
        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing prompt"}), 400
            
        prompt = data['prompt']
        
        # Optional parameters
        prefix = data.get('prefix', "### User: ")
        suffix = data.get('suffix', "### Response: ")
        
        def generate():
            with model_lock:  # Lock while generating to prevent concurrent access
                for token in generate_stream(model, prompt, prefix, suffix):
                    yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"
        
        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no'
            }
        )
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Simple health check endpoint.
    """
    return jsonify({
        "status": "ok",
        "model": os.path.basename(os.environ.get('MODEL_PATH', 'luna.gguf')),
        "initialized": llm is not None
    })

@app.route('/api/model/reload', methods=['POST'])
def reload_model():
    """
    API endpoint to force model reloading.
    """
    global llm
    
    try:
        with model_lock:
            # Clean up old model if it exists
            if llm is not None:
                del llm
                llm = None
            
            # Initialize a new model instance
            model_path = os.environ.get('MODEL_PATH', './luna.gguf')
            threads = int(os.environ.get('MODEL_THREADS', '4'))
            context_size = int(os.environ.get('CONTEXT_SIZE', '4096'))
            system_prompt = os.environ.get('SYSTEM_PROMPT', 
                                        "### System: You are Luna, a sexy lady trying to help out. Talk shortly as a conversation")
            
            llm = initialize_model(
                model_path=model_path,
                threads=threads,
                context_size=context_size,
                system_prompt=system_prompt
            )
            
        return jsonify({
            "status": "ok",
            "message": "Model reloaded successfully",
            "model": os.path.basename(model_path)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Load environment variables
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Initialize model on startup
    get_model()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=port, debug=debug)
