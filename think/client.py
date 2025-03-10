import requests
import json
import time
import sys

def chat_with_luna(prompt, base_url="http://localhost:8000"):
    """Send a chat message to Luna API and get response"""
    try:
        url = f"{base_url}/v1/chat/completions"
        data = {
            "model": "luna",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500
        }
        
        # Send request
        response = requests.post(url, json=data)
        response.raise_for_status()
        
        # Parse JSON response
        result = response.json()
        
        # Extract and return text
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            return "[Error: Unexpected response format]"
    except Exception as e:
        return f"[Error: {str(e)}]"

def stream_chat_with_luna(prompt, base_url="http://localhost:8000"):
    """Send a chat message to Luna API and stream the response"""
    try:
        url = f"{base_url}/v1/chat/completions/stream"
        data = {
            "model": "luna",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500
        }
        
        # Send request
        response = requests.post(url, json=data, stream=True)
        response.raise_for_status()
        
        # Process streaming response
        full_response = ""
        for line in response.iter_lines():
            if not line:
                continue
                
            line = line.decode('utf-8')
            
            if line == 'data: [DONE]':
                break
                
            if line.startswith('data: '):
                json_str = line[6:]  # Remove 'data: ' prefix
                try:
                    chunk = json.loads(json_str)
                    
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        choice = chunk['choices'][0]
                        
                        # Handle chat delta format
                        if 'delta' in choice and 'content' in choice['delta']:
                            content = choice['delta']['content']
                            full_response += content
                            print(content, end="", flush=True)
                except json.JSONDecodeError:
                    pass
        
        return full_response
    except Exception as e:
        error_msg = f"[Error: {str(e)}]"
        print(error_msg)
        return error_msg

def main():
    print("=" * 50)
    print("Luna AI Chat Client")
    print("Type 'exit' or 'quit' to end the conversation")
    print("=" * 50)
    
    # Try to check if server is running
    try:
        requests.get("http://localhost:8000/v1/models", timeout=2)
    except:
        print("\nWarning: Could not connect to Luna API server.")
        print("Make sure the server is running at http://localhost:8000")
        print("Continue anyway? (y/n)")
        if input().lower() != 'y':
            print("Goodbye!")
            return
    
    # Set streaming mode preference
    print("\nUse streaming mode? (y/n)")
    streaming = input().lower() == 'y'
    
    # Start chat loop
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        # Check for exit command
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        print("\nLuna: ", end="", flush=True)
        
        # Get response
        if streaming:
            stream_chat_with_luna(user_input)
            print()  # Add newline after streaming
        else:
            response = chat_with_luna(user_input)
            print(response)

if __name__ == "__main__":
    main()