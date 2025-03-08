from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import llama_cpp
import os
import threading
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("luna-api")

# Initialize Flask app
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
    import time
    
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🔄 Starting prompt evaluation")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 📝 Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    
    # Format the prompt
    full_prompt = f"{prefix}{prompt}\n{suffix}"
    
    tokenization_start = time.time()
    ptokens = llm.tokenize(bytes(full_prompt, "utf-8"))
    tokenization_time = time.time() - tokenization_start
    
    # Count input tokens
    input_token_count = len(ptokens)
    
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🔢 Tokenized {input_token_count} tokens in {tokenization_time:.2f}s")
    
    # Initialize response tracking
    response_token_count = 0
    full_response = ""
    generation_start_time = time.time()
    last_token_time = generation_start_time
    
    # Define patterns to check and filter
    stop_patterns = [
        "### User", "###User", '###',
        "### Response", "###Response", 
        "### System", "###System"
    ]
    
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🚀 Starting token generation")
    
    # Generate response tokens
    resp_gen = llm.generate(
        ptokens,
        reset=False,
        logits_processor=llama_cpp.LogitsProcessorList([])
    )
    
    for tok in resp_gen:
        current_time = time.time()
        time_since_last = current_time - last_token_time
        total_time = current_time - generation_start_time
        
        if tok == llm.token_eos():
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🛑 Reached end-of-sequence token after {response_token_count} tokens")
            break
            
        word = llm.detokenize([tok]).decode("utf-8", errors="ignore")
        full_response += word
        response_token_count += 1
        
        # Log token generation (limit frequency to avoid overwhelming logs)
        if response_token_count == 1 or response_token_count % 10 == 0 or time_since_last > 1.0:
            tokens_per_second = response_token_count / total_time if total_time > 0 else 0
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🔤 Token #{response_token_count}: '{word}' ({tokens_per_second:.1f} tokens/sec)")
        
        last_token_time = current_time
        
        # Check if adding this token creates a stop sequence
        should_stop = False
        for pattern in stop_patterns:
            if pattern in full_response[-20:]:  # Check last 20 chars for efficiency
                # Trim the response at the stop pattern
                full_response = full_response[:full_response.rfind(pattern)]
                should_stop = True
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🚫 Stop pattern '{pattern}' detected after {response_token_count} tokens")
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
    
    total_generation_time = time.time() - generation_start_time
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ✅ Generation complete: {response_token_count} tokens in {total_generation_time:.2f}s ({response_token_count/total_generation_time:.1f} tokens/sec)")


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
    import time
    
    try:
        request_start = time.time()
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 📡 Received non-streaming request from {request.remote_addr}")
        
        # Get model
        model = get_model()
        
        # Get request data
        data = request.json
        if not data or 'prompt' not in data:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ❌ Error: Missing prompt in request")
            return jsonify({"error": "Missing prompt"}), 400
            
        prompt = data['prompt']
        prompt_length = len(prompt)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 📨 Prompt received: {prompt_length} characters")
        
        # Optional parameters
        prefix = data.get('prefix', "### User: ")
        suffix = data.get('suffix', "### Response: ")
        
        # Generate full response
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🔒 Acquiring model lock")
        generation_start = time.time()
        
        with model_lock:  # Lock while generating to prevent concurrent access
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ✅ Model lock acquired, beginning generation")
            response_text = ""
            token_count = 0
            
            for token in generate_stream(model, prompt, prefix, suffix):
                response_text += token
                token_count += 1
        
        generation_time = time.time() - generation_start
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🏁 Generation complete: {token_count} tokens, {len(response_text)} chars in {generation_time:.2f}s")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 📊 Performance: {token_count/generation_time:.1f} tokens/sec")
        
        response = {
            "response": response_text.strip(),
            "model": os.path.basename(os.environ.get('MODEL_PATH', 'luna.gguf')),
            "tokens": token_count,
            "generation_time": round(generation_time, 2)
        }
        
        total_time = time.time() - request_start
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🚀 Response sent in {total_time:.2f}s (includes {total_time - generation_time:.2f}s overhead)")
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """
    API endpoint for streaming response generation.
    """
    import time
    
    try:
        request_start = time.time()
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 📡 Received streaming request from {request.remote_addr}")
        
        # Get model
        model = get_model()
        
        # Get request data
        data = request.json
        if not data or 'prompt' not in data:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ❌ Error: Missing prompt in request")
            return jsonify({"error": "Missing prompt"}), 400
            
        prompt = data['prompt']
        prompt_length = len(prompt)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 📨 Prompt received: {prompt_length} characters")
        
        # Optional parameters
        prefix = data.get('prefix', "### User: ")
        suffix = data.get('suffix', "### Response: ")
        
        def generate():
            token_count = 0
            stream_start = time.time()
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🔒 Acquiring model lock for streaming")
            
            with model_lock:  # Lock while generating to prevent concurrent access
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ✅ Model lock acquired, beginning token generation")
                
                # Send an initial empty data event to establish the connection quickly
                yield f"data: \n\n"
                
                for token in generate_stream(model, prompt, prefix, suffix):
                    token_count += 1
                    # Log every 50 tokens streamed to avoid excessive logging
                    if token_count == 1 or token_count % 50 == 0:
                        elapsed = time.time() - stream_start
                        tokens_per_sec = token_count / elapsed if elapsed > 0 else 0
                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 📤 Streamed {token_count} tokens ({tokens_per_sec:.1f} tokens/sec)")
                    yield f"data: {token}\n\n"
            
            # Send completion token
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🏁 Streaming complete, sent {token_count} tokens in {time.time() - stream_start:.2f}s")
            yield "data: [DONE]\n\n"
        
        response = Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no'
            }
        )
        
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🚀 Response stream initiated in {time.time() - request_start:.4f}s")
        return response
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Simple health check endpoint.
    """
    import time
    import psutil
    
    try:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🔍 Health check from {request.remote_addr}")
        
        # Get system metrics
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent(interval=0.1)
        
        # Get model info
        model_path = os.environ.get('MODEL_PATH', 'luna.gguf')
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024) if os.path.exists(model_path) else 0
        
        response = {
            "status": "ok",
            "model": os.path.basename(model_path),
            "model_size_mb": round(model_size_mb, 2),
            "initialized": llm is not None,
            "server_time": time.strftime('%Y-%m-%d %H:%M:%S'),
            "uptime_seconds": time.time() - process.create_time(),
            "memory_mb": round(memory_info.rss / (1024 * 1024), 2),
            "cpu_percent": round(cpu_percent, 2),
            "threads": process.num_threads()
        }
        
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ✅ Health check successful")
        return jsonify(response)
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ❌ Health check error: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e),
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
    
    # Add required dependencies
    try:
        import psutil
    except ImportError:
        print("Installing psutil for system monitoring...")
        import subprocess
        subprocess.check_call(["pip", "install", "psutil"])
        import psutil
    
    # Print startup banner
    import time
    print("\n" + "="*80)
    print(f"🌙 LUNA API SERVER - Starting up at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    model_path = os.environ.get('MODEL_PATH', './luna.gguf')
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024) if os.path.exists(model_path) else 0
    
    print(f"📁 Model path: {model_path}")
    print(f"📊 Model size: {model_size_mb:.2f} MB")
    print(f"🧵 Threads: {os.environ.get('MODEL_THREADS', '4')}")
    print(f"📝 Context size: {os.environ.get('CONTEXT_SIZE', '4096')}")
    print(f"🔌 Server port: {port}")
    print(f"🐞 Debug mode: {debug}")
    
    print("🚀 Initializing model...")
    start_time = time.time()
    # Initialize model on startup
    model = get_model()
    loading_time = time.time() - start_time
    print(f"✅ Model initialized in {loading_time:.2f} seconds")
    
    print("📡 Starting server...")
    print("="*80)
    print(f"API endpoints:")
    print(f"  - POST /api/chat         - Generate complete response")
    print(f"  - POST /api/chat/stream  - Stream response as it's generated")
    print(f"  - GET  /api/health       - Check server health")
    print(f"  - POST /api/model/reload - Reload model")
    print("="*80 + "\n")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=port, debug=debug)
