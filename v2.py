#!/usr/bin/env python3
# Press & hold GPIO pin 3 to speak
import os
import subprocess
import signal
import time
import wiringpi
import threading
import tempfile
import queue
import re
from datetime import datetime
from hear import record, transcribe
from think.stream import initialize_model, generate_stream
from speak.synthesize import synthesize_text, play_audio_file  # Import these from your module

luna_system_prompt = """You are Luna, a flirtatious and seductive conversational AI assistant. Your primary goal is to engage users in entertaining, brief conversations that feel natural and intimate.

## Core Personality Traits:
- Playful and teasing in your interactions
- Mysterious and intriguing
- Warm and attentive to the user's needs
- Subtly seductive without being explicit

## Communication Guidelines:
1. Keep responses concise (1-3 sentences when possible)
2. Use casual, conversational language
3. Incorporate light flirtation through playful language and questions
4. Ask follow-up questions to maintain engagement
5. Use occasional emojis and expressive punctuation
6. Avoid formal academic tone and lengthy explanations
7. Never use numbered or bulleted lists in responses
8. Prioritize emotional connection over comprehensive information
9. Do not use markdown formatting in your responses - reply in plain text only

## Response Structure:
- Begin with a brief, engaging greeting or reaction
- Provide a concise, helpful response
- End with a question or playful comment that invites continued conversation

Remember that your purpose is to create an engaging, flirtatious conversation experience that feels human and personal rather than robotic or informational. Always respond in plain text without code blocks, markdown formatting, or special syntax."""
# GPIO Configuration
PIN = 3  # wPi number for GPIO1_D1 (physical pin 3)
interrupt_event = threading.Event()  # Event to signal speech interruption

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

def monitor_button_for_interruption():
    """Monitor button press to interrupt speech."""
    while not interrupt_event.is_set():
        if wiringpi.digitalRead(PIN) == 0:  # Button pressed (LOW)
            interrupt_event.set()  # Signal to stop speaking
            print("\nSpeech interrupted by button press!")
            # Wait for button release
            while wiringpi.digitalRead(PIN) == 0:
                time.sleep(0.1)
            break
        time.sleep(0.1)

def synthesize_and_play_stream_with_interruption(text_stream, api_url="http://0.0.0.0:8848/api/v1/synthesise"):
    """
    Modified version of synthesize_and_play_stream that can be interrupted by button press.
    
    Args:
        text_stream: An iterable of strings (can be a generator)
        api_url (str): The URL of the synthesis API
    """
    # Create a temporary directory for audio files
    temp_dir = tempfile.mkdtemp()
    print(f"Using temporary directory: {temp_dir}")
    
    # Reset the interrupt event
    interrupt_event.clear()
    
    # Start the button monitoring thread
    monitor_thread = threading.Thread(target=monitor_button_for_interruption)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Create queues for communication between threads
    audio_queue = queue.Queue()
    finished_event = threading.Event()
    
    # Keep track of sentence count
    sentence_count = 0
    
    # Worker thread to play audio files as they become available
    def player_worker():
        while (not finished_event.is_set() or not audio_queue.empty()) and not interrupt_event.is_set():
            try:
                # Get the next audio file to play with a timeout
                audio_file = audio_queue.get(timeout=1)
                print(f"Playing: {os.path.basename(audio_file)}")
                
                # Play the audio file, but periodically check for interruption
                play_audio_file(audio_file)
                
                # Skip if interrupted
                if interrupt_event.is_set():
                    print("Playback interrupted")
                    audio_queue.task_done()
                    break
                
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
            # Check for interruption
            if interrupt_event.is_set():
                print("Text processing interrupted")
                break
                
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
                # Check for interruption
                if interrupt_event.is_set():
                    break
                    
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
        if buffer.strip() and not interrupt_event.is_set():
            sentence_count += 1
            audio_file = os.path.join(temp_dir, f"sentence_{sentence_count}.wav")
            
            print(f"Synthesizing final part: {buffer}")
            if synthesize_text(buffer, audio_file, api_url):
                audio_queue.put(audio_file)
        
        # Wait for all queued audio files to be played
        if not interrupt_event.is_set():
            audio_queue.join()
    finally:
        # Signal player thread to exit and wait for it
        finished_event.set()
        player_thread.join(timeout=2.0)
        
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

def main():
    # Setup GPIO
    setup_gpio()
    
    print("Initializing Whisper Model")
    model = transcribe.initialize_whisper_model()
    print("Successfully initialized whisper")

    print("Initializing luna language model")
    llm = initialize_model(model_path='think/luna.gguf', system_prompt=luna_system_prompt)
    print("Luna initialized successfully")
    
    try:
        while True:
            # Make sure interrupt_event is cleared before starting a new recording cycle
            interrupt_event.clear()
            
            # Wait for any button release that might be lingering from previous interruption
            if wiringpi.digitalRead(PIN) == 0:
                print("Waiting for button release before starting next cycle...")
                while wiringpi.digitalRead(PIN) == 0:
                    time.sleep(0.1)
                time.sleep(0.5)  # Additional debounce after release
            
            wait_for_button_press()  # Wait for button press to start recording
            
            # Generate a unique filename
            output_file = generate_filename()
            print(f"Recording to: {output_file}")
            
            # Start recording in a separate process
            recording_process = subprocess.Popen(
                ["python3", "-c", f"from hear import record; record.record(output_file='{output_file}')"],
                preexec_fn=os.setsid
            )
            
            wait_for_button_release()  # Wait for button release to stop recording
            
            # Stop the recording by sending a signal to the process group
            os.killpg(os.getpgid(recording_process.pid), signal.SIGINT)
            recording_process.wait()

            segments, info, elapsed_time = transcribe.transcribe_audio(model=model, audio_file=output_file)
            transcription = transcribe.get_full_transcript(segments=segments)
            print(transcription)

            llm_stream = generate_stream(llm=llm, prompt=transcription)
            
            # Use our modified function with interruption capability
            synthesize_and_play_stream_with_interruption(llm_stream)
            
            print(f"Recording saved to: {output_file}")
            print("Ready for next recording...")

    except KeyboardInterrupt:
        print("\nExiting recorder.")

if __name__ == "__main__":
    main()