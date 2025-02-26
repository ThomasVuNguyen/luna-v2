import os
import tempfile
from client import synthesize_text  # Import the synthesize_text function

def text_to_speech_and_play(text, keep_file=False, output_file=None):
    """
    Convert text to speech, save it to a file, and play it.
    
    Args:
        text (str): The text to convert to speech
        keep_file (bool): Whether to keep the audio file after playing (default: False)
        output_file (str): Path to save the audio file (default: temporary file)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create a temporary file if no output file is specified
        if output_file is None:
            temp_dir = tempfile.gettempdir()
            output_file = os.path.join(temp_dir, "temp_speech.opus")
            
        # Synthesize the text to an audio file
        success = synthesize_text(text, output_file)
        
        if not success:
            print("Failed to synthesize speech")
            return False
            
        # Play the audio file
        if os.name == 'nt':  # Windows
            os.system(f'start {output_file}')
        elif os.name == 'posix':  # macOS or Linux
            # Try different players depending on what's available
            if os.system('which ffplay > /dev/null 2>&1') == 0:
                os.system(f'ffplay -autoexit -nodisp -loglevel quiet "{output_file}"')
            elif os.system('which mplayer > /dev/null 2>&1') == 0:
                os.system(f'mplayer "{output_file}" > /dev/null 2>&1')
            elif os.system('which afplay > /dev/null 2>&1') == 0:  # macOS
                os.system(f'afplay "{output_file}"')
            elif os.system('which mpv > /dev/null 2>&1') == 0:
                os.system(f'mpv --no-video "{output_file}" > /dev/null 2>&1')
            else:
                print("No suitable audio player found. Please install ffplay, mplayer, mpv, or afplay.")
                return False
        else:
            print(f"Unsupported operating system: {os.name}")
            return False
            
        print("Audio played successfully")
        
        # Clean up the temporary file if not keeping it
        if not keep_file and output_file.startswith(tempfile.gettempdir()):
            os.remove(output_file)
            print(f"Temporary file {output_file} removed")
            
        return True
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return False

# Example usage
if __name__ == "__main__":
    user_text = input("Enter text to convert to speech: ")
    text_to_speech_and_play(user_text)
