import requests

def synthesize_text(text, output_file="output.opus", api_url="http://0.0.0.0:8848/api/v1/synthesise"):
    """
    Send text to the speech synthesis API and save the response to a file.
    
    Args:
        text (str): The text to be synthesized into speech
        output_file (str): The name of the file to save the audio to (default: output.opus)
        api_url (str): The URL of the synthesis API (default: http://0.0.0.0:8848/api/v1/synthesise)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Prepare the JSON payload
        payload = {"text": text}
        
        # Set the appropriate headers
        headers = {'Content-Type': 'application/json'}
        
        # Make the POST request to the API
        response = requests.post(api_url, json=payload, headers=headers)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Save the binary content to the output file
            with open(output_file, 'wb') as f:
                f.write(response.content)
            print(f"Successfully saved audio to {output_file}")
            return True
        else:
            print(f"Error: API returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return False

# Example usage
if __name__ == "__main__":
    sample_text = "To be or not to be, that is the question"
    synthesize_text(sample_text, "test.opus")
