from flask import Flask, request, jsonify, Response, stream_with_context
import json
import time
import llama_cpp
from flask_cors import CORS

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Default model configuration
DEFAULT_CONFIG = {
    "model_path": "./luna.gguf",
    "model_name": "luna",
    "threads": 4,
    "context_size": 4096,
    "system_prompt": "### System: You are Luna, a sexy lady trying to help out. Talk shortly as a conversation",
    "use_mlock": True  # Default to locking the model in memory
}

# Global model instance
model = None

def initialize_model(model_path=DEFAULT_CONFIG["model_path"], 
                     threads=DEFAULT_CONFIG["threads"], 
                     context_size=DEFAULT_CONFIG["context_size"], 
                     system_prompt=DEFAULT_CONFIG["system_prompt"],
                     use_mlock=DEFAULT_CONFIG["use_mlock"]):  # Updated parameter name
    """
    Initialize the LLaMA model with the given parameters.
    
    Args:
        model_path (str): Path to the model file
        threads (int): Number of threads to use
        context_size (int): Context window size
        system_prompt (str): System prompt to initialize the model
        mlock (bool): Whether to lock the model in memory
        
    Returns:
        llama_cpp.Llama: Initialized model
    """
    # Initialize the model with proper memory options
    params = {
        "model_path": model_path,
        "verbose": False,
        "n_ctx": context_size,
        "n_threads": threads,
        "use_mlock": True  # Use the correct parameter name
    }
    
    # Create model instance
    llm = llama_cpp.Llama(**params)
    print("Model initialized with use_mlock=True")
    
    # Add the system prompt to initialize the context
    init_tokens = llm.tokenize(bytes(system_prompt, "utf-8"))
    llm.eval(init_tokens)
    
    return llm

def init_model(config=None):
    """Initialize the model if not already initialized"""
    global model, DEFAULT_CONFIG
    
    if config is None:
        config = DEFAULT_CONFIG
        
    if model is None:
        print(f"Initializing model: {config['model_name']} from {config['model_path']}...")
        model = initialize_model(
            model_path=config['model_path'],
            threads=config['threads'],
            context_size=config['context_size'],
            system_prompt=config['system_prompt'],
            use_mlock=config.get('use_mlock', True)  # Use correct parameter name
        )
        print("Model initialized successfully")
    return model

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

def get_model_config(request_data):
    """Extract model configuration from request or use defaults"""
    config = DEFAULT_CONFIG.copy()
    
    # Update config with request data if provided
    if 'model_path' in request_data:
        config['model_path'] = request_data['model_path']
    if 'model' in request_data:
        config['model_name'] = request_data['model']
    if 'threads' in request_data:
        config['threads'] = request_data['threads']
    if 'context_size' in request_data:
        config['context_size'] = request_data['context_size']
    if 'system_prompt' in request_data:
        config['system_prompt'] = request_data['system_prompt']
    if 'mlock' in request_data:  # For backward compatibility
        config['use_mlock'] = bool(request_data['mlock'])
    if 'use_mlock' in request_data:  # New correct parameter
        config['use_mlock'] = bool(request_data['use_mlock'])
        
    return config

# Rest of your original code...

@app.route('/v1/models', methods=['GET'])
def list_models():
    """List available models - OpenAI-like endpoint"""
    return jsonify({
        "data": [
            {
                "id": DEFAULT_CONFIG["model_name"],
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local"
            }
        ],
        "object": "list"
    })

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """Handle chat completions endpoint similar to OpenAI"""
    data = request.json
    
    # Extract parameters from request
    messages = data.get('messages', [])
    stream = data.get('stream', False)
    max_tokens = data.get('max_tokens', 256)
    
    # Get model configuration
    config = get_model_config(data)
    
    # Initialize model
    llm = init_model(config)
    
    # Format user prompt from messages
    user_prompt = ""
    for message in messages:
        if message['role'] == 'user' and len(user_prompt) == 0:
            user_prompt = message['content']
    
    # Format and tokenize the prompt to get input token count
    prefix = "### User: "
    suffix = "### Response: "
    full_prompt = f"{prefix}{user_prompt}\n{suffix}"
    input_tokens = len(llm.tokenize(bytes(full_prompt, "utf-8")))
    
    # Handle streaming response
    if stream:
        def generate():
            response_tokens = 0
            full_response = ""
            
            # Stream header
            yield f"data: {json.dumps({'id': f'chatcmpl-{int(time.time())}', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': config['model_name'], 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
            
            # Stream content
            for token in generate_stream(llm, user_prompt):
                response_tokens += 1
                full_response += token
                
                yield f"data: {json.dumps({'id': f'chatcmpl-{int(time.time())}', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': config['model_name'], 'choices': [{'index': 0, 'delta': {'content': token}, 'finish_reason': None}]})}\n\n"
                
                if response_tokens >= max_tokens:
                    break
            
            # Stream completion
            yield f"data: {json.dumps({'id': f'chatcmpl-{int(time.time())}', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': config['model_name'], 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
            
            # End stream
            yield "data: [DONE]\n\n"
            
        return Response(stream_with_context(generate()), content_type='text/event-stream')
    else:
        # Non-streaming response
        response_text = ""
        response_tokens = 0
        
        for token in generate_stream(llm, user_prompt):
            response_text += token
            response_tokens += 1
            if response_tokens >= max_tokens:
                break
        
        response = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": config['model_name'],
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": response_tokens,
                "total_tokens": input_tokens + response_tokens
            }
        }
        
        return jsonify(response)

# Continue with remaining routes and functions...

# Route to update the default model configuration
@app.route('/v1/set-model-config', methods=['POST'])
def set_model_config():
    """Update the default model configuration"""
    global DEFAULT_CONFIG, model
    
    data = request.json
    
    # Update configuration with provided values
    if 'model_path' in data:
        DEFAULT_CONFIG['model_path'] = data['model_path']
    if 'model_name' in data:
        DEFAULT_CONFIG['model_name'] = data['model_name']
    if 'threads' in data:
        DEFAULT_CONFIG['threads'] = data['threads']
    if 'context_size' in data:
        DEFAULT_CONFIG['context_size'] = data['context_size']
    if 'system_prompt' in data:
        DEFAULT_CONFIG['system_prompt'] = data['system_prompt']
    if 'mlock' in data:  # For backward compatibility
        DEFAULT_CONFIG['use_mlock'] = bool(data['mlock'])
    if 'use_mlock' in data:  # New correct parameter
        DEFAULT_CONFIG['use_mlock'] = bool(data['use_mlock'])
    
    # If model should be reinitialized
    if data.get('reinitialize', False):
        model = None
        model = init_model()
    
    return jsonify({
        "status": "success",
        "message": "Model configuration updated",
        "config": DEFAULT_CONFIG
    })

if __name__ == "__main__":
    # Run the Flask app
    app.run(host='0.0.0.0', port=8000, debug=False)
