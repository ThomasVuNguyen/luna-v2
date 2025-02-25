#!/usr/bin/env python3
import subprocess
import os
import time
import signal
import sys
import re
from faster_whisper import WhisperModel

def get_card_number(pattern):
    """Get the card number for a specific audio device pattern"""
    try:
        aplay_output = subprocess.check_output(["aplay", "-l"], text=True)
        for line in aplay_output.splitlines():
            if pattern in line:
                # Extract card number
                match = re.search(r'card (\d+):', line)
                if match:
                    return match.group(1)
    except subprocess.SubprocessError:
        pass
    return None

def setup_mixer(card, record_type, board):
    """Configure the mixer settings for recording with maximum volume"""
    # Set all mixer settings to moderate volume level 6
    subprocess.run(["amixer", "-c", card, "cset", "name='ALC Capture Max PGA'", "6"], 
                  stdout=subprocess.DEVNULL)
    
    subprocess.run(["amixer", "-c", card, "cset", "name='ALC Capture Min PGA'", "6"], 
                  stdout=subprocess.DEVNULL)
    
    subprocess.run(["amixer", "-c", card, "cset", "name='Capture Digital Volume'", "192"], 
                  stdout=subprocess.DEVNULL)  # Using a proportional value for digital volume
    
    subprocess.run(["amixer", "-c", card, "cset", "name='Left Channel Capture Volume'", "6"], 
                  stdout=subprocess.DEVNULL)
    
    subprocess.run(["amixer", "-c", card, "cset", "name='Right Channel Capture Volume'", "6"], 
                  stdout=subprocess.DEVNULL)
    
    # Board-specific settings
    if board == "orangepi900":
        if record_type == "main":
            subprocess.run(["amixer", "-c", card, "cset", "name='Left Line Mux'", "1"], 
                          stdout=subprocess.DEVNULL)
            subprocess.run(["amixer", "-c", card, "cset", "name='Right Line Mux'", "1"], 
                          stdout=subprocess.DEVNULL)
            subprocess.run(["amixer", "-c", card, "cset", "name='Left Mixer Left Playback Switch'", "1"], 
                          stdout=subprocess.DEVNULL)
        else:
            subprocess.run(["amixer", "-c", card, "cset", "name='Left Line Mux'", "0"], 
                          stdout=subprocess.DEVNULL)
            subprocess.run(["amixer", "-c", card, "cset", "name='Right Line Mux'", "0"], 
                          stdout=subprocess.DEVNULL)
            subprocess.run(["amixer", "-c", card, "cset", "name='Left Mixer Left Playback Switch'", "0"], 
                          stdout=subprocess.DEVNULL)
    else:
        if record_type == "main":
            subprocess.run(["amixer", "-c", card, "cset", "name='Left PGA Mux'", "1"], 
                          stdout=subprocess.DEVNULL)
            subprocess.run(["amixer", "-c", card, "cset", "name='Right PGA Mux'", "1"], 
                          stdout=subprocess.DEVNULL)
            subprocess.run(["amixer", "-c", card, "cset", "name='Differential Mux'", "1"], 
                          stdout=subprocess.DEVNULL)
        else:
            subprocess.run(["amixer", "-c", card, "cset", "name='Left PGA Mux'", "0"], 
                          stdout=subprocess.DEVNULL)
            subprocess.run(["amixer", "-c", card, "cset", "name='Right PGA Mux'", "0"], 
                          stdout=subprocess.DEVNULL)
            subprocess.run(["amixer", "-c", card, "cset", "name='Differential Mux'", "0"], 
                          stdout=subprocess.DEVNULL)

def main():
    # Initialize the whisper model
    model_size = "distil-medium.en"
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    
    # Get board information
    board = None
    try:
        with open("/etc/orangepi-release", "r") as f:
            for line in f:
                if line.startswith("BOARD="):
                    board = line.strip().split("=")[1]
                    break
    except FileNotFoundError:
        pass
    
    # Get card numbers
    card = get_card_number("es8388")
    hdmi0_card = get_card_number("hdmi0")
    hdmi1_card = None
    
    if board == "orangepi5ultra":
        hdmi1_card = get_card_number("hdmi1")
    
    if not card:
        print("Could not find the es8388 audio card.")
        sys.exit(1)
    
    # Set default recording type to "main"
    record_type = "main"
    
    # Counter for unique filenames
    counter = 0
    
    print("Press Enter to start recording, Ctrl+C to exit the program")
    
    try:
        while True:
            # Generate a unique filename for this recording
            audio_file = f"recording_{counter}.wav"
            counter += 1
            
            # Wait for Enter to start recording
            input("Press Enter to START recording...")
            
            # Setup mixer with maximum volume
            setup_mixer(card, record_type, board)
            
            # Start recording
            print(f"Recording to {audio_file}... (Press Enter to stop)")
            
            # Start the recording process
            recording_process = subprocess.Popen(
                ["arecord", "-D", f"hw:{card},0", "-f", "cd", "-t", "wav", audio_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Wait for Enter to stop recording
            input()
            
            # Stop the recording process
            recording_process.terminate()
            recording_process.wait()
            print("Recording stopped.")
            
            # Transcribe the recorded audio
            print("Transcribing...")
            start_time = time.time()
            
            segments, info = model.transcribe(audio_file, beam_size=5)
            
            # Print transcription info
            elapsed_time = time.time() - start_time
            print(f"Detected language '{info.language}' with probability {info.language_probability}")
            print(f"Transcription completed in {elapsed_time:.2f} seconds")
            start = time.time()
            
            # Print each segment of the transcription
            print("\n--- TRANSCRIPTION ---")
            for segment in segments:
                print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
            
            print("\n" + "-"*50 + "\n")
            print(time.time() - start)
    except KeyboardInterrupt:
        print("\nExiting program.")
        
    # Clean up temporary recordings if desired
    # Uncomment the following lines if you want to delete the recording files
    """
    for i in range(counter):
        try:
            os.remove(f"recording_{i}.wav")
        except:
            pass
    """

if __name__ == "__main__":
    main()
