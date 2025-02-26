#!/usr/bin/env python3
import sys
import os
import subprocess
import threading
import time
import queue

# Import from our custom modules
from think.stream import initialize_model, generate_stream
from speak.synthesize import synthesize_text, synthesize_and_play_stream
# Import our new transcribe functions
from hear.transcribe import initialize_whisper_model, transcribe_audio, get_full_transcript

# Global variables
MODEL_PATH = "think/luna.gguf"
WHISPER_MODEL_SIZE = "base.en"
RECORD_OUTPUT_DIR = "/tmp"
WHISPER_MODEL = None
LLM_MODEL = None
RECORDING_ACTIVE = False
CURRENT_RECORDING_FILE = None  # Will store the current recording filename

def generate_random_filename():
    """Generate a random filename for recording"""
    import uuid
    import datetime
    
    # Create a unique filename using timestamp and UUID
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]  # Use first 8 chars of UUID
    
    return f"{RECORD_OUTPUT_DIR}/recording_{timestamp}_{unique_id}.wav"

def start_recording(record_type="main", max_retries=3, retry_delay=1.0):
    """Start recording using the record.py functionality with retries for busy device"""
    global CURRENT_RECORDING_FILE
    
    for attempt in range(max_retries):
        try:
            # Generate a new random filename for this recording
            CURRENT_RECORDING_FILE = generate_random_filename()
            print(f"Using new recording file: {CURRENT_RECORDING_FILE}")
            
            # Start the recording process
            cmd = ["python", "-m", "hear.record", record_type, CURRENT_RECORDING_FILE, "play"]
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Check if the process started successfully by polling it once
            if process.poll() is None:
                print("Recording started... (Press Enter to stop)")
                return process
            else:
                # Process exited immediately, let's check the error
                stdout, stderr = process.communicate()
                stdout_str = stdout.decode('utf-8', errors='ignore').strip()
                stderr_str = stderr.decode('utf-8', errors='ignore').strip()
                
                if stderr_str:
                    print(f"Recording process error: {stderr_str}")
                if stdout_str:
                    print(f"Recording process output: {stdout_str}")
                
                if "busy" in stderr_str.lower() or "busy" in stdout_str.lower() or "device" in stderr_str.lower():
                    print(f"Audio device appears to be busy. Attempt {attempt+1}/{max_retries}. Waiting...")
                    time.sleep(retry_delay)
                    continue
                else:
                    print("Recording failed for a different reason.")
                    break
                    
        except Exception as e:
            print(f"Error starting recording: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying... Attempt {attempt+2}/{max_retries}")
                time.sleep(retry_delay)
            else:
                print("Maximum retry attempts reached. Could not start recording.")
    
    CURRENT_RECORDING_FILE = None
    return None

def release_microphone():
    """Explicitly release the microphone after recording"""
    try:
        # For Linux (ALSA/PulseAudio)
        # Try using pactl to unload module-suspend-on-idle to force a release
        # This forces PulseAudio to release unused resources
        subprocess.run(["pactl", "unload-module", "module-suspend-on-idle"], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL,
                      timeout=1)
        # Then reload it
        subprocess.run(["pactl", "load-module", "module-suspend-on-idle"], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL,
                      timeout=1)
        
        # On some Linux systems, we can try to restart PulseAudio
        # in a way that doesn't disrupt the user experience
        try:
            subprocess.run(["pacmd", "suspend", "true"], 
                          stdout=subprocess.DEVNULL, 
                          stderr=subprocess.DEVNULL,
                          timeout=1)
            time.sleep(0.2)
            subprocess.run(["pacmd", "suspend", "false"], 
                          stdout=subprocess.DEVNULL, 
                          stderr=subprocess.DEVNULL,
                          timeout=1)
        except:
            pass  # Ignore errors, as this is just an additional attempt
        
        print("Microphone released")
        return True
    except Exception as e:
        # Silently fail - this is an enhancement, not critical functionality
        # print(f"Note: Could not explicitly release microphone: {str(e)}")
        return False

def stop_recording(recording_process):
    """Stop the recording process and explicitly release the microphone"""
    success = False
    
    if recording_process and recording_process.poll() is None:
        try:
            # Before terminating, check if the process is actually recording
            # by checking if it's still running
            if recording_process.poll() is None:
                # Try to terminate the process gracefully
                print("Stopping recording process...")
                recording_process.terminate()
                # Wait longer for it to finish
                try:
                    # Capture any output during termination
                    stdout, stderr = recording_process.communicate(timeout=3)
                    if stderr:
                        stderr_str = stderr.decode('utf-8', errors='ignore').strip()
                        if stderr_str:
                            print(f"Recording process stderr: {stderr_str}")
                    
                    print("Recording stopped")
                    success = True
                except subprocess.TimeoutExpired:
                    print("Recording process didn't terminate in time, forcing kill")
                    recording_process.kill()
                    try:
                        recording_process.wait(timeout=2)
                        print("Recording process killed")
                        success = True
                    except subprocess.TimeoutExpired:
                        print("Warning: Process still not terminated after kill")
            else:
                print("Recording process already ended")
                success = True
        except Exception as e:
            print(f"Error stopping recording: {str(e)}")
            # If terminate fails, try to kill it
            try:
                recording_process.kill()
                recording_process.wait(timeout=1)
                print("Recording process killed")
                success = True
            except Exception as e2:
                print(f"Failed to kill process: {str(e2)}")
    else:
        print("No active recording process to stop")
    
    # Explicitly try to release the microphone regardless of process termination success
    release_microphone()
    
    return success

def transcribe_audio_file(audio_file):
    """Transcribe an audio file using Whisper"""
    global WHISPER_MODEL
    
    # Enhanced file checks with better error messages
    if not audio_file:
        print("Error: No audio file specified for transcription")
        return None
        
    if not os.path.exists(audio_file):
        print(f"Audio file not found: {audio_file}")
        # Check if directory exists
        dir_path = os.path.dirname(audio_file)
        if not os.path.exists(dir_path):
            print(f"Directory does not exist: {dir_path}")
        else:
            print(f"Directory exists but file is missing. Available files:")
            try:
                files = os.listdir(dir_path)
                for file in files[:5]:  # Show first 5 files
                    print(f" - {file}")
                if len(files) > 5:
                    print(f" ... and {len(files)-5} more files")
            except Exception as e:
                print(f"Could not list directory contents: {str(e)}")
        return None
    
    # Check if the file is large enough to contain audio
    try:
        file_size = os.path.getsize(audio_file)
        print(f"Audio file size: {file_size} bytes")
        
        if file_size < 1000:  # arbitrary small size check
            print(f"Warning: Recording file {audio_file} is too small, might be empty.")
            return None
    except Exception as e:
        print(f"Error checking file size: {str(e)}")
        return None
    
    print(f"Transcribing audio from: {audio_file}")
    
    try:
        # Transcribe the audio file
        segments, info, elapsed_time = transcribe_audio(WHISPER_MODEL, audio_file)
        
        # Convert segments to a list to ensure we can iterate through it
        segment_list = list(segments)
        
        # Get the full transcript
        transcript = get_full_transcript(segment_list)
        
        print(f"Transcription completed in {elapsed_time:.2f} seconds")
        print(f"You said: \"{transcript}\"")
        
        # Clean up the file after successful transcription
        try:
            os.remove(audio_file)
            print(f"Removed used recording file: {audio_file}")
        except Exception as e:
            print(f"Note: Could not remove used recording file: {str(e)}")
        
        return transcript
        
    except Exception as e:
        print(f"Error transcribing audio: {str(e)}")
        return None

def generate_response(prompt):
    """Generate a response from the LLM"""
    global LLM_MODEL
    
    print("Thinking...")
    try:
        # Get the response generator
        response_generator = generate_stream(LLM_MODEL, prompt)
        
        # Collect the full response for display
        full_response = ""
        response_generator_for_display = []
        
        for token in response_generator:
            full_response += token
            response_generator_for_display.append(token)
        
        print(f"AI response: \"{full_response}\"")
        
        # Return a generator that yields the collected tokens
        return (token for token in response_generator_for_display)
        
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return None

def ensure_audio_device_available():
    """Check if the audio device is available by trying to list audio devices"""
    try:
        # First try to explicitly release the microphone
        # This might free up a busy device without the user having to request it
        release_microphone()
        
        # Then check if devices are available
        # Use arecord -l to list recording devices (Linux)
        result = subprocess.run(["arecord", "-l"], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               timeout=2)
        
        if result.returncode == 0:
            # Found devices
            return True
        else:
            stderr = result.stderr.decode('utf-8', errors='ignore')
            if "busy" in stderr.lower() or "device" in stderr.lower():
                print("Warning: Audio devices may be busy.")
                # Try one more time to explicitly release
                release_microphone()
                return False
            else:
                print("Could not list audio devices.")
                return False
    except FileNotFoundError:
        # arecord not available, try something else or assume it's OK
        print("arecord not found, cannot check audio devices")
        return True
    except Exception as e:
        print(f"Error checking audio devices: {str(e)}")
        return True  # Assume it's OK if we can't check

def reset_audio_devices():
    """Attempt to reset audio devices that might be stuck"""
    try:
        print("\nAttempting to reset audio devices...")
        
        # Try to release the microphone using our dedicated function
        release_result = release_microphone()
        
        # On Linux, try a more aggressive approach if the user explicitly requested it
        try:
            # Try to restart PulseAudio - this is more aggressive and might interrupt audio
            # but the user explicitly asked for reset, so it's okay
            subprocess.run(["pulseaudio", "-k"], 
                          stdout=subprocess.DEVNULL, 
                          stderr=subprocess.DEVNULL,
                          timeout=2)
            print("PulseAudio service restarted")
            time.sleep(1)  # Give it a moment to restart
            return True
        except:
            # If that didn't work, just print manual instructions
            print("\nAutomatic reset attempt completed. If still having issues, you could try:")
            print("1. Check for other applications using the microphone and close them")
            print("2. If on Linux, run: 'pulseaudio -k' to restart audio")
            print("3. If on Windows, check the Sound control panel")
            print("4. If on macOS, restart the CoreAudio service\n")
            return release_result
            
    except Exception as e:
        print(f"Error attempting to reset audio: {str(e)}")
        print("Please try manually closing any applications that might be using the microphone.")
        return False

def main():
    global WHISPER_MODEL, LLM_MODEL, RECORDING_ACTIVE, CURRENT_RECORDING_FILE
    
    # Initialize models
    print(f"Initializing Whisper model ({WHISPER_MODEL_SIZE})...")
    WHISPER_MODEL = initialize_whisper_model(WHISPER_MODEL_SIZE)
    
    print(f"Initializing LLM from {MODEL_PATH}...")
    LLM_MODEL = initialize_model(model_path=MODEL_PATH)
    
    # Release any potentially held microphone resources at startup
    release_microphone()
    
    # Initial check for audio devices
    audio_available = ensure_audio_device_available()
    if not audio_available:
        print("Warning: Audio devices may not be available. Will try anyway.")
        # Try one more aggressive reset at startup if devices seem unavailable
        reset_audio_devices()
    
    print("\n=== Voice Conversation System ===")
    print("Press Enter to start recording")
    print("Press Enter again to stop recording")
    print("Type 'exit' or 'quit' to end the conversation")
    print("Type 'reset' to attempt to reset audio devices if they're busy")
    print("Type 'release' to explicitly release the microphone")
    print("=====================================\n")
    
    while True:
        try:
            # Wait for user input to start/stop recording
            user_input = input("")
            
            # Check if user wants to exit
            if user_input.lower() in ["exit", "quit"]:
                print("Ending conversation. Goodbye!")
                break
                
            # Check if user wants to reset audio devices
            if user_input.lower() == "reset":
                print("Attempting to reset audio devices...")
                reset_audio_devices()
                # Force any recording process to stop
                if RECORDING_ACTIVE and 'recording_process' in locals():
                    stop_recording(recording_process)
                    RECORDING_ACTIVE = False
                continue
                
            # Check if user wants to explicitly release the microphone
            if user_input.lower() == "release":
                print("Explicitly releasing microphone...")
                release_microphone()
                # Also stop any active recording for safety
                if RECORDING_ACTIVE and 'recording_process' in locals():
                    stop_recording(recording_process)
                    RECORDING_ACTIVE = False
                continue
                
            if not RECORDING_ACTIVE:
                # Check if audio device is available before starting
                if not ensure_audio_device_available():
                    print("Audio device appears to be busy. Try the 'reset' command or wait a moment.")
                    continue
                
                # Start recording with a new random filename
                recording_process = start_recording()
                if recording_process:
                    RECORDING_ACTIVE = True
                else:
                    print("Could not start recording. Audio device may be busy.")
                    print("Try the 'reset' command or wait a moment before trying again.")
            else:
                # Stop recording
                if 'recording_process' in locals() and CURRENT_RECORDING_FILE:
                    stop_recording(recording_process)
                    RECORDING_ACTIVE = False
                    
                    # Store the current recording file path before it might change
                    current_file = CURRENT_RECORDING_FILE
                    
                    # Give more time for the file to be properly closed and flushed to disk
                    print("Waiting for recording file to be finalized...")
                    time.sleep(2.0)
                    
                    # Verify the file exists and has content
                    if os.path.exists(current_file):
                        print(f"Recording file found: {current_file}")
                        file_size = os.path.getsize(current_file)
                        print(f"File size: {file_size} bytes")
                        
                        if file_size > 1000:  # Arbitrary minimum size
                            # Transcribe the audio
                            transcript = transcribe_audio_file(current_file)
                            
                            if transcript and transcript.strip():
                                # Generate response
                                response_generator = generate_response(transcript)
                                
                                if response_generator:
                                    # Synthesize and play the response
                                    print("Speaking...")
                                    synthesize_and_play_stream(response_generator)
                                    print("\nPress Enter to start recording again")
                            else:
                                print("No valid transcript obtained. Please try again.")
                                print("Press Enter to start recording again")
                        else:
                            print("Recording file is too small, no valid audio captured.")
                            print("Press Enter to start recording again")
                    else:
                        print(f"Error: Recording file not found at {current_file}")
                        print("Press Enter to start recording again")
        
        except KeyboardInterrupt:
            print("\nInterrupted. Ending conversation.")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Press Enter to continue or type 'exit' to quit")
    
    print("Voice conversation system stopped.")

if __name__ == "__main__":
    main()