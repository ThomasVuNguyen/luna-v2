import requests
import json
import time

# Server configuration
SERVER_URL = "http://localhost:8000"  # Change this if your server is on a different port/host

def test_model_config():
    """Test updating the model configuration with use_mlock option"""
    url = f"{SERVER_URL}/v1/set-model-config"
    
    payload = {
        "use_mlock": True,
        "reinitialize": True  # Force model reinitialization
    }
    
    print("Setting model config with use_mlock=True...")
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        print("Config updated successfully!")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

def test_chat_completion(message="Tell me about memory management in large language models"):
    """Test chat completion endpoint with a simple message"""
    url = f"{SERVER_URL}/v1/chat/completions"
    
    payload = {
        "model": "luna",  # Use the model name from your server config
        "messages": [
            {
                "role": "user",
                "content": message
            }
        ],
        "max_tokens": 256,
        "use_mlock": True  # Explicitly set use_mlock
    }
    
    print(f"Sending chat completion request with message: '{message}'")
    start_time = time.time()
    
    response = requests.post(url, json=payload)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    if response.status_code == 200:
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        tokens = result["usage"]["completion_tokens"]
        
        print(f"Response received in {elapsed:.2f} seconds ({tokens} tokens):")
        print("-" * 40)
        print(content)
        print("-" * 40)
        print(f"Usage stats: {json.dumps(result['usage'], indent=2)}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

def test_streaming_completion(message="Explain why memory-locking is useful for LLMs"):
    """Test streaming chat completion endpoint"""
    url = f"{SERVER_URL}/v1/chat/completions"
    
    payload = {
        "model": "luna",
        "messages": [
            {
                "role": "user",
                "content": message
            }
        ],
        "max_tokens": 256,
        "stream": True,
        "use_mlock": True
    }
    
    print(f"Sending streaming chat completion request with message: '{message}'")
    start_time = time.time()
    
    response = requests.post(url, json=payload, stream=True)
    
    if response.status_code == 200:
        # Process the streaming response
        print("\nStreaming response:")
        print("-" * 40)
        
        full_response = ""
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                # Skip the "data: " prefix and the [DONE] message
                if line.startswith("data: ") and "delta" in line and not line.endswith("[DONE]"):
                    try:
                        data = json.loads(line[6:])  # Skip "data: " prefix
                        if "delta" in data["choices"][0] and "content" in data["choices"][0]["delta"]:
                            content = data["choices"][0]["delta"]["content"]
                            full_response += content
                            # Print without newline to simulate streaming
                            print(content, end="", flush=True)
                    except json.JSONDecodeError:
                        pass
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        print("\n" + "-" * 40)
        print(f"Response completed in {elapsed:.2f} seconds")
        print(f"Total response length: {len(full_response)} characters")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    # First, update the model config to ensure use_mlock is enabled
    test_model_config()
    
    # Wait a moment for the model to initialize
    print("\nWaiting for model to initialize...")
    time.sleep(2)
    
    # Test regular chat completion
    test_chat_completion()
    
    # Test streaming chat completion
    print("\n")
    test_streaming_completion()
