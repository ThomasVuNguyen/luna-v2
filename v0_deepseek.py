#!/usr/bin/env python3
# Press enter to talk & enter to stop -> listen response
import os
import subprocess
import signal
from datetime import datetime
from hear import record, transcribe
from think.stream import initialize_model, generate_stream
from speak.synthesize import synthesize_and_play_stream
def generate_filename():
    """Generate a unique filename based on current timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"/tmp/audio_{timestamp}.wav"

def main():
    print("Initializing Whisper Model")
    model = transcribe.initialize_whisper_model()
    print("Successfully initialized whisper")

    print("Initializing luna language model")
    llm = initialize_model(model_path='think/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf')
    print("Luna initialized successfully")

    print("Press Enter to start recording...")
    
    while True:
        input()  # Wait for Enter key to start recording
        
        # Generate a unique filename
        output_file = generate_filename()
        print(f"Recording to: {output_file}")
        print("Recording... Press Enter to stop.")
        
        # Start recording in a separate process
        recording_process = subprocess.Popen(
            ["python3", "-c", f"from hear import record; record.record(output_file='{output_file}')"],
            preexec_fn=os.setsid
        )
        
        input()  # Wait for Enter key to stop recording
        
        # Stop the recording by sending a signal to the process group
        os.killpg(os.getpgid(recording_process.pid), signal.SIGINT)
        recording_process.wait()

        segments, info, elapsed_time = transcribe.transcribe_audio(model=model,audio_file=output_file)
        transcription = transcribe.get_full_transcript(segments=segments)
        print(transcription)

        llm_stream = generate_stream(llm=llm, prompt=transcription)
        synthesize_and_play_stream(text_stream=llm_stream)
        

        print(f"Recording saved to: {output_file}")
        print("Press Enter to start a new recording or Ctrl+C to exit...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting recorder.")
