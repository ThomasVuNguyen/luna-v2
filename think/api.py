from flask import Flask, request, jsonify, Response, stream_with_context
import json
import time
import llama_cpp

def initialize_model(model_path="./luna-hermes.gguf", threads=4, context_size=4096, system_prompt="### System: You are Luna, a sexy lady trying to help out. Talk shortly as a conversation"):
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
from flask_cors import CORS

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


# Global model instance
model = None

def init_model(model_path="./luna-hermes.gguf", threads=4, context_size=4096, system_prompt="### System: You are Luna, a sexy lady trying to help out. Talk shortly as a conversation"):
    """Initialize the model if not already initialized"""
    global model
    if model is None:
        print("Initializing model...")
        model = initialize_model(
            model_path=model_path,
            threads=threads,
            context_size=context_size,
            system_prompt=system_prompt
        )
        print("Model initialized successfully")
    return model

@app.route('/v1/models', methods=['GET'])
def list_models():
    """List available models - OpenAI-like endpoint"""
    return jsonify({
        "data": [
            {
                "id": "luna",
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
    model_name = data.get('model', 'luna')
    
    # Extract model config if provided
    model_path = data.get('model_path', './luna-hermes.gguf')
    threads = data.get('threads', 4)
    context_size = data.get('context_size', 4096)
    system_prompt = data.get('system_prompt', "### System: You are Luna, a sexy lady trying to help out. Talk shortly as a conversation")
    
    # Initialize model
    llm = init_model(model_path, threads, context_size, system_prompt)
    
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
            yield f"data: {json.dumps({'id': f'chatcmpl-{int(time.time())}', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model_name, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
            
            # Stream content
            for token in generate_stream(llm, user_prompt):
                response_tokens += 1
                full_response += token
                
                yield f"data: {json.dumps({'id': f'chatcmpl-{int(time.time())}', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model_name, 'choices': [{'index': 0, 'delta': {'content': token}, 'finish_reason': None}]})}\n\n"
                
                if response_tokens >= max_tokens:
                    break
            
            # Stream completion
            yield f"data: {json.dumps({'id': f'chatcmpl-{int(time.time())}', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model_name, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
            
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
            "model": model_name,
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

@app.route('/v1/chat/completions/stream', methods=['POST'])
def chat_completions_stream():
    """Dedicated streaming endpoint for chat completions"""
    data = request.json
    
    # Extract parameters from request
    messages = data.get('messages', [])
    max_tokens = data.get('max_tokens', 256)
    model_name = data.get('model', 'luna')
    
    # Extract model config if provided
    model_path = data.get('model_path', './luna-hermes.gguf')
    threads = data.get('threads', 4)
    context_size = data.get('context_size', 4096)
    system_prompt = data.get('system_prompt', "### System: You are Luna, a sexy lady trying to help out. Talk shortly as a conversation")
    
    # Initialize model
    llm = init_model(model_path, threads, context_size, system_prompt)
    
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
    
    def generate():
        response_tokens = 0
        full_response = ""
        
        # Stream header
        yield f"data: {json.dumps({'id': f'chatcmpl-{int(time.time())}', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model_name, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
        
        # Stream content
        for token in generate_stream(llm, user_prompt):
            response_tokens += 1
            full_response += token
            
            yield f"data: {json.dumps({'id': f'chatcmpl-{int(time.time())}', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model_name, 'choices': [{'index': 0, 'delta': {'content': token}, 'finish_reason': None}]})}\n\n"
            
            if response_tokens >= max_tokens:
                break
        
        # Stream completion
        yield f"data: {json.dumps({'id': f'chatcmpl-{int(time.time())}', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model_name, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        
        # End stream
        yield "data: [DONE]\n\n"
        
    return Response(stream_with_context(generate()), content_type='text/event-stream')

@app.route('/v1/completions', methods=['POST'])
def completions():
    """Handle text completions endpoint similar to OpenAI"""
    data = request.json
    
    # Extract parameters from request
    prompt = data.get('prompt', '')
    stream = data.get('stream', False)
    max_tokens = data.get('max_tokens', 256)
    model_name = data.get('model', 'luna')
    
    # Extract model config if provided
    model_path = data.get('model_path', './luna-hermes.gguf')
    threads = data.get('threads', 4)
    context_size = data.get('context_size', 4096)
    system_prompt = data.get('system_prompt', "### System: You are Luna, a sexy lady trying to help out. Talk shortly as a conversation")
    
    # Initialize model
    llm = init_model(model_path, threads, context_size, system_prompt)
    
    # Format and tokenize the prompt to get input token count
    prefix = "### User: "
    suffix = "### Response: "
    full_prompt = f"{prefix}{prompt}\n{suffix}"
    input_tokens = len(llm.tokenize(bytes(full_prompt, "utf-8")))
    
    # Handle streaming response
    if stream:
        def generate():
            response_tokens = 0
            full_response = ""
            
            # Stream header
            yield f"data: {json.dumps({'id': f'cmpl-{int(time.time())}', 'object': 'text_completion.chunk', 'created': int(time.time()), 'model': model_name, 'choices': [{'index': 0, 'text': '', 'finish_reason': None}]})}\n\n"
            
            # Stream content
            for token in generate_stream(llm, prompt):
                response_tokens += 1
                full_response += token
                
                yield f"data: {json.dumps({'id': f'cmpl-{int(time.time())}', 'object': 'text_completion.chunk', 'created': int(time.time()), 'model': model_name, 'choices': [{'index': 0, 'text': token, 'finish_reason': None}]})}\n\n"
                
                if response_tokens >= max_tokens:
                    break
            
            # Stream completion
            yield f"data: {json.dumps({'id': f'cmpl-{int(time.time())}', 'object': 'text_completion.chunk', 'created': int(time.time()), 'model': model_name, 'choices': [{'index': 0, 'text': '', 'finish_reason': 'stop'}]})}\n\n"
            
            # End stream
            yield "data: [DONE]\n\n"
            
        return Response(stream_with_context(generate()), content_type='text/event-stream')
    else:
        # Non-streaming response
        response_text = ""
        response_tokens = 0
        
        for token in generate_stream(llm, prompt):
            response_text += token
            response_tokens += 1
            if response_tokens >= max_tokens:
                break
        
        response = {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "text": response_text,
                    "index": 0,
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
        
@app.route('/v1/completions/stream', methods=['POST'])
def completions_stream():
    """Dedicated streaming endpoint for text completions"""
    data = request.json
    
    # Extract parameters from request
    prompt = data.get('prompt', '')
    max_tokens = data.get('max_tokens', 256)
    model_name = data.get('model', 'luna')
    
    # Extract model config if provided
    model_path = data.get('model_path', './luna-hermes.gguf')
    threads = data.get('threads', 4)
    context_size = data.get('context_size', 4096)
    system_prompt = data.get('system_prompt', "### System: You are Luna, a sexy lady trying to help out. Talk shortly as a conversation")
    
    # Initialize model
    llm = init_model(model_path, threads, context_size, system_prompt)
    
    # Format and tokenize the prompt to get input token count
    prefix = "### User: "
    suffix = "### Response: "
    full_prompt = f"{prefix}{prompt}\n{suffix}"
    input_tokens = len(llm.tokenize(bytes(full_prompt, "utf-8")))
    
    def generate():
        response_tokens = 0
        full_response = ""
        
        # Stream header
        yield f"data: {json.dumps({'id': f'cmpl-{int(time.time())}', 'object': 'text_completion.chunk', 'created': int(time.time()), 'model': model_name, 'choices': [{'index': 0, 'text': '', 'finish_reason': None}]})}\n\n"
        
        # Stream content
        for token in generate_stream(llm, prompt):
            response_tokens += 1
            full_response += token
            
            yield f"data: {json.dumps({'id': f'cmpl-{int(time.time())}', 'object': 'text_completion.chunk', 'created': int(time.time()), 'model': model_name, 'choices': [{'index': 0, 'text': token, 'finish_reason': None}]})}\n\n"
            
            if response_tokens >= max_tokens:
                break
        
        # Stream completion
        yield f"data: {json.dumps({'id': f'cmpl-{int(time.time())}', 'object': 'text_completion.chunk', 'created': int(time.time()), 'model': model_name, 'choices': [{'index': 0, 'text': '', 'finish_reason': 'stop'}]})}\n\n"
        
        # End stream
        yield "data: [DONE]\n\n"
        
    return Response(stream_with_context(generate()), content_type='text/event-stream')

@app.route('/v1/stream', methods=['POST'])
def generic_stream():
    """General purpose streaming endpoint that supports both chat and text formats"""
    data = request.json
    
    # Determine if we're using chat or completion mode
    is_chat = 'messages' in data
    
    if is_chat:
        # Extract chat parameters
        messages = data.get('messages', [])
        max_tokens = data.get('max_tokens', 256)
        model_name = data.get('model', 'luna')
        
        # Extract user message
        user_prompt = ""
        for message in messages:
            if message['role'] == 'user' and len(user_prompt) == 0:
                user_prompt = message['content']
    else:
        # Extract completion parameters
        user_prompt = data.get('prompt', '')
        max_tokens = data.get('max_tokens', 256)
        model_name = data.get('model', 'luna')
    
    # Extract model config if provided
    model_path = data.get('model_path', './luna-hermes.gguf')
    threads = data.get('threads', 4)
    context_size = data.get('context_size', 4096)
    system_prompt = data.get('system_prompt', "### System: You are Luna, a sexy lady trying to help out. Talk shortly as a conversation")
    
    # Initialize model
    llm = init_model(model_path, threads, context_size, system_prompt)
    
    # Format and tokenize the prompt to get input token count
    prefix = "### User: "
    suffix = "### Response: "
    full_prompt = f"{prefix}{user_prompt}\n{suffix}"
    input_tokens = len(llm.tokenize(bytes(full_prompt, "utf-8")))
    
    def generate():
        response_tokens = 0
        full_response = ""
        
        # Generate unique ID
        response_id = int(time.time())
        
        # Stream header based on type
        if is_chat:
            yield f"data: {json.dumps({'id': f'chatcmpl-{response_id}', 'object': 'chat.completion.chunk', 'created': response_id, 'model': model_name, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
        else:
            yield f"data: {json.dumps({'id': f'cmpl-{response_id}', 'object': 'text_completion.chunk', 'created': response_id, 'model': model_name, 'choices': [{'index': 0, 'text': '', 'finish_reason': None}]})}\n\n"
        
        # Stream content
        for token in generate_stream(llm, user_prompt):
            response_tokens += 1
            full_response += token
            
            if is_chat:
                yield f"data: {json.dumps({'id': f'chatcmpl-{response_id}', 'object': 'chat.completion.chunk', 'created': response_id, 'model': model_name, 'choices': [{'index': 0, 'delta': {'content': token}, 'finish_reason': None}]})}\n\n"
            else:
                yield f"data: {json.dumps({'id': f'cmpl-{response_id}', 'object': 'text_completion.chunk', 'created': response_id, 'model': model_name, 'choices': [{'index': 0, 'text': token, 'finish_reason': None}]})}\n\n"
            
            if response_tokens >= max_tokens:
                break
        
        # Stream completion
        if is_chat:
            yield f"data: {json.dumps({'id': f'chatcmpl-{response_id}', 'object': 'chat.completion.chunk', 'created': response_id, 'model': model_name, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        else:
            yield f"data: {json.dumps({'id': f'cmpl-{response_id}', 'object': 'text_completion.chunk', 'created': response_id, 'model': model_name, 'choices': [{'index': 0, 'text': '', 'finish_reason': 'stop'}]})}\n\n"
        
        # End stream
        yield "data: [DONE]\n\n"
        
    return Response(stream_with_context(generate()), content_type='text/event-stream')

if __name__ == "__main__":
    # Run the Flask app
    app.run(host='0.0.0.0', port=8000, debug=False)