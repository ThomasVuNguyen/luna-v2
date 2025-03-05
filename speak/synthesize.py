import requests
import re
import os
import threading
import tempfile
import time
import queue

def synthesize_text(text, output_file="output.opus", api_url="http://0.0.0.0:8848/api/v1/synthesise"):
    """
    Send text to the speech synthesis API and save the response to a file.
    Filters out markdown symbols and other non-text elements.
    
    Args:
        text (str): The text to be synthesized into speech
        output_file (str): The name of the file to save the audio to (default: output.opus)
        api_url (str): The URL of the synthesis API (default: http://0.0.0.0:8848/api/v1/synthesise)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Filter out markdown and other non-text symbols
        filtered_text = filter_non_text(text)
        
        # Prepare the JSON payload
        payload = {"text": filtered_text}
        
        # Set the appropriate headers
        headers = {'Content-Type': 'application/json'}
        
        # Make the POST request to the API
        response = requests.post(api_url, json=payload, headers=headers)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Save the binary content to the output file
            with open(output_file, 'wb') as f:
                f.write(response.content)
            return True
        else:
            print(f"Error: API returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return False

def filter_non_text(text):
    """
    Filter out markdown symbols and other non-text elements.
    
    Args:
        text (str): The text to be filtered
        
    Returns:
        str: Filtered text
    """
    import re
    
    # Remove markdown headings (# symbols)
    text = re.sub(r'#+\s+', '', text)
    
    # Remove markdown emphasis (* and _)
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'_+', '', text)
    
    # Remove markdown code blocks (```), inline code (`), and block quotes (>)
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`[^`]*`', '', text)
    text = re.sub(r'^\s*>\s*', '', text, flags=re.MULTILINE)
    
    # Remove markdown links
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]*>', '', text)
    
    # Remove multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def play_audio_file(file_path):
    """Play an audio file using the appropriate player for the operating system."""
    try:
        if os.name == 'nt':  # Windows
            os.system(f'start /wait {file_path}')
            return True
        elif os.name == 'posix':  # macOS or Linux
            # Try different players depending on what's available
            if os.system('which ffplay > /dev/null 2>&1') == 0:
                return os.system(f'ffplay -autoexit -nodisp -loglevel quiet "{file_path}"') == 0
            elif os.system('which mplayer > /dev/null 2>&1') == 0:
                return os.system(f'mplayer "{file_path}" > /dev/null 2>&1') == 0
            elif os.system('which afplay > /dev/null 2>&1') == 0:  # macOS
                return os.system(f'afplay "{file_path}"') == 0
            elif os.system('which mpv > /dev/null 2>&1') == 0:
                return os.system(f'mpv --no-video "{file_path}" > /dev/null 2>&1') == 0
            else:
                print("No suitable audio player found.")
                return False
        return False
    except Exception as e:
        print(f"Error playing audio: {str(e)}")
        return False

def synthesize_and_play_stream(text_stream, api_url="http://0.0.0.0:8848/api/v1/synthesise"):
    """
    Process a stream of text, synthesize each sentence, and play them in sequence.
    While one sentence is playing, the next one is being synthesized.
    
    Args:
        text_stream: An iterable of strings (can be a generator)
        api_url (str): The URL of the synthesis API
        
    Returns:
        None
    """
    # Create a temporary directory for audio files
    temp_dir = tempfile.mkdtemp()
    print(f"Using temporary directory: {temp_dir}")
    
    # Create queues for communication between threads
    audio_queue = queue.Queue()
    finished_event = threading.Event()
    
    # Keep track of sentence count
    sentence_count = 0
    
    # Worker thread to play audio files as they become available
    def player_worker():
        while not finished_event.is_set() or not audio_queue.empty():
            try:
                # Get the next audio file to play with a timeout
                audio_file = audio_queue.get(timeout=1)
                print(f"Playing: {os.path.basename(audio_file)}")
                
                # Play the audio file
                play_audio_file(audio_file)
                
                # Remove the file after playing
                os.remove(audio_file)
                audio_queue.task_done()
            except queue.Empty:
                # No audio files ready yet, keep waiting
                continue
            except Exception as e:
                print(f"Error in player thread: {str(e)}")
        print("Player thread finished")
    
    # Start the player thread
    player_thread = threading.Thread(target=player_worker)
    player_thread.start()
    
    try:
        # Process buffer into sentences
        buffer = ""
        sentence_pattern = re.compile(r'(?<=[.!?])\s+')
        
        for text_chunk in text_stream:
            buffer += text_chunk
            
            # Split buffer into sentences
            sentences = sentence_pattern.split(buffer)
            
            # Keep the last incomplete sentence in the buffer
            if sentences and not buffer.rstrip().endswith(('.', '!', '?')):
                buffer = sentences.pop()
            else:
                buffer = ""
                
            # Process complete sentences
            for sentence in sentences:
                if not sentence.strip():
                    continue
                    
                sentence_count += 1
                audio_file = os.path.join(temp_dir, f"sentence_{sentence_count}.wav")
                
                print(f"Synthesizing: {sentence}")
                success = synthesize_text(sentence, audio_file, api_url)
                
                if success:
                    # Add to the queue for playback
                    audio_queue.put(audio_file)
                else:
                    print(f"Failed to synthesize: {sentence}")
        
        # Process any remaining text in the buffer
        if buffer.strip():
            sentence_count += 1
            audio_file = os.path.join(temp_dir, f"sentence_{sentence_count}.wav")
            
            print(f"Synthesizing final part: {buffer}")
            if synthesize_text(buffer, audio_file, api_url):
                audio_queue.put(audio_file)
        
        # Wait for all queued audio files to be played
        audio_queue.join()
    finally:
        # Signal player thread to exit and wait for it
        finished_event.set()
        player_thread.join()
        
        # Clean up any remaining files in the temp directory
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error removing {file_path}: {str(e)}")
        
        # Remove the temporary directory
        try:
            os.rmdir(temp_dir)
            print(f"Removed temporary directory: {temp_dir}")
        except Exception as e:
            print(f"Error removing temporary directory: {str(e)}")

# Example usage
if __name__ == "__main__":
    sample_text = "To be or not to be, that is the question"
    synthesize_text(sample_text, "test.wav")
    
    # Example of using the stream function with a list of chunks
    def example_stream():
        text_chunks = [
            "Hello, this is a test of the streaming synthesis. ",
            "It breaks text into sentences. Then it creates audio files. ",
            "While one sentence is playing, the next one is being prepared. ",
            "This approach reduces the perceived latency for the user."
        ]
        for chunk in text_chunks:
            yield chunk
            time.sleep(0.5)  # Simulate delay between incoming chunks
    
    # Uncomment to test the streaming function
    synthesize_and_play_stream(example_stream())
