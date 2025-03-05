#!/usr/bin/env python3
import os
import subprocess
import signal
import time
import wiringpi
from datetime import datetime
from hear import record, transcribe
from think.stream import initialize_model, generate_stream
from speak.synthesize import synthesize_and_play_stream

# GPIO Configuration
PIN = 3  # wPi number for GPIO1_B7 (physical pin 3)

def generate_filename():
    """Generate a unique filename based on current timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"/tmp/audio_{timestamp}.wav"

def setup_gpio():
    """Setup GPIO for button detection."""
    wiringpi.wiringPiSetup()
    wiringpi.pinMode(PIN, wiringpi.INPUT)
    wiringpi.pullUpDnControl(PIN, wiringpi.PUD_UP)

def wait_for_button_press():
    """Wait for button to be pressed (connected to ground)."""
    print("Press the button to start recording...")
    while wiringpi.digitalRead(PIN) == 1:  # Wait for LOW (button press)
        time.sleep(0.1)
    time.sleep(0.3)  # Simple debounce

def wait_for_button_release():
    """Wait for button to be released (disconnected from ground)."""
    print("Recording... Press the button again to stop.")
    time.sleep(0.3)  # Add debounce delay to avoid immediate trigger
    while wiringpi.digitalRead(PIN) == 0:  # Wait for HIGH (button release)
        time.sleep(0.1)
    time.sleep(0.3)  # Simple debounce

def main():
    # Setup GPIO
    setup_gpio()
    
    print("Initializing Whisper Model")
    model = transcribe.initialize_whisper_model()
    print("Successfully initialized whisper")

    print("Initializing luna language model")
    llm = initialize_model(model_path='think/luna.gguf')
    print("Luna initialized successfully")
    
    try:
        while True:
            wait_for_button_press()  # Wait for button press to start recording
            
            # Generate a unique filename
            output_file = generate_filename()
            print(f"Recording to: {output_file}")
            
            # Start recording in a separate process
            recording_process = subprocess.Popen(
                ["python3", "-c", f"from hear import record; record.record(output_file='{output_file}')"],
                preexec_fn=os.setsid
            )
            
            wait_for_button_release()  # Wait for button press to stop recording
            
            # Stop the recording by sending a signal to the process group
            os.killpg(os.getpgid(recording_process.pid), signal.SIGINT)
            recording_process.wait()

            segments, info, elapsed_time = transcribe.transcribe_audio(model=model, audio_file=output_file)
            transcription = transcribe.get_full_transcript(segments=segments)
            print(transcription)

            llm_stream = generate_stream(llm=llm, prompt=transcription)
            synthesize_and_play_stream(text_stream=llm_stream)
            
            print(f"Recording saved to: {output_file}")
            print("Ready for next recording...")

    except KeyboardInterrupt:
        print("\nExiting recorder.")

if __name__ == "__main__":
    main()