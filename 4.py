#!/usr/bin/env python3
import sys
import os
import subprocess
import time

# Import from our custom modules
from think.stream import initialize_model, generate_stream
from speak.synthesize import synthesize_text, synthesize_and_play_stream
from hear.transcribe import initialize_whisper_model, transcribe_audio, get_full_transcript

# Global variables
MODEL_PATH = "think/luna.gguf"
WHISPER_MODEL_SIZE = "base.en"
WHISPER_MODEL = None
LLM_MODEL = None

def start_recording(output_file="/tmp/recording.wav"):
    """Start recording using the record.py functionality"""
    try:
        # Make sure the file doesn't exist
        if os.path.exists(output_file):
            os.remove(output_file)
            
        # Start the recording process
        cmd = ["python3", "-m", "hear.record", "main", output_file, "noplay"]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        print("Recording started... (Press Enter to stop)")
        return process
        
    except Exception as e:
        print(f"Error starting recording: {str(e)}")
        return None

def stop_recording(recording_process):
    """Stop the recording process"""
    if recording_process and recording_process.poll() is None:
        try:
            # Try to terminate the process
            recording_process.terminate()
            recording_process.wait(timeout=2)
            print("Recording stopped")
            return True
        except Exception as e:
            print(f"Error stopping recording: {str(e)}")
            # If terminate fails, try to kill it
            try:
                recording_process.kill()
                print("Recording process killed")
                return True
            except:
                pass
    return False

def transcribe_audio_file(audio_file):
    """Transcribe an audio file using Whisper"""
    global WHISPER_MODEL
    
    if not os.path.exists(audio_file):
        print(f"Audio file not found: {audio_file}")
        return None
    
    # Check if the file is large enough to contain audio
    if os.path.getsize(audio_file) < 1000:
        print(f"Warning: Recording file is too small, might be empty.")
        return None
    
    print(f"Transcribing audio...")
    
    try:
        # Transcribe the audio file
        segments, info, elapsed_time = transcribe_audio(WHISPER_MODEL, audio_file)
        segment_list = list(segments)
        transcript = get_full_transcript(segment_list)
        
        print(f"Transcription completed in {elapsed_time:.2f} seconds")
        print(f"You said: \"{transcript}\"")
        
        return transcript
        
    except Exception as e:
        print(f"Error transcribing audio: {str(e)}")
        return None

def generate_response(prompt):
    """Generate a response from the LLM and return the stream directly"""
    global LLM_MODEL
    
    print("Thinking...")
    try:
        # Get the response generator and return it directly
        return generate_stream(LLM_MODEL, prompt)
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return None

def main():
    global WHISPER_MODEL, LLM_MODEL
    
    # Initialize models
    print(f"Initializing Whisper model ({WHISPER_MODEL_SIZE})...")
    WHISPER_MODEL = initialize_whisper_model(WHISPER_MODEL_SIZE)
    
    print(f"Initializing LLM from {MODEL_PATH}...")
    LLM_MODEL = initialize_model(model_path=MODEL_PATH)
    
    print("\n=== Voice Conversation System ===")
    print("Press Enter to start recording")
    print("Press Enter again to stop recording")
    print("Type 'exit' or 'quit' to end the conversation")
    print("===================================\n")
    
    recording_active = False
    recording_file = "/tmp/recording.wav"
    
    while True:
        try:
            # Wait for user input to start/stop recording
            user_input = input("")
            
            # Check if user wants to exit
            if user_input.lower() in ["exit", "quit"]:
                print("Ending conversation. Goodbye!")
                break
                
            if not recording_active:
                # Start recording
                recording_process = start_recording(recording_file)
                if recording_process:
                    recording_active = True
            else:
                # Stop recording
                if 'recording_process' in locals():
                    stop_recording(recording_process)
                    recording_active = False
                    
                    # Wait for file to be properly saved
                    time.sleep(1.0)
                    
                    # Transcribe the audio if file exists
                    if os.path.exists(recording_file):
                        transcript = transcribe_audio_file(recording_file)
                        
                        if transcript and transcript.strip():
                            # Generate response stream and pass directly to speech synthesis
                            response_stream = generate_response(transcript)
                            
                            if response_stream:
                                # Synthesize and play the response as it's being generated
                                print("Speaking and generating response simultaneously...")
                                synthesize_and_play_stream(response_stream)
                                print("Response complete")
                        else:
                            print("No transcript obtained. Try again.")
                    else:
                        print(f"Recording file not found. Try running:")
                        print(f"python3 -m hear.record test /tmp/test.wav")
                        
        except KeyboardInterrupt:
            print("\nInterrupted. Ending conversation.")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
    
    print("Voice conversation system stopped.")

if __name__ == "__main__":
    main()
