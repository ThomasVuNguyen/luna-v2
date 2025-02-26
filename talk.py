#!/usr/bin/env python3
import sys
import os
import subprocess
import threading
import time
import queue
import keyboard
# Import from our custom modules
from think.stream import initialize_model, generate_stream
from speak.synthesize import synthesize_text, synthesize_and_play_stream
# Import our new transcribe functions
from hear.transcribe import initialize_whisper_model, transcribe_audio, get_full_transcript
from hear.record import record
import threading

def input_thread(L):
    char = input()
    L.append(char)

L = []
thread = threading.Thread(target=input_thread, args=(L,))
thread.start()



# Global variables
MODEL_PATH = "think/luna.gguf"
WHISPER_MODEL_SIZE = "base.en"
RECORD_OUTPUT_DIR = "/tmp"
WHISPER_MODEL = None
LLM_MODEL = None
RECORDING_ACTIVE = False
CURRENT_RECORDING_FILE = None  # Will store the current recording filename
RECORDING_COMMAND = "python -m hear.record main test.wav noplay"
TMUX_SESSION_RECORD_NAME = "record"
def main():
    global WHISPER_MODEL, LLM_MODEL, RECORDING_ACTIVE, CURRENT_RECORDING_FILE
    
    # Initialize models
    print(f"Initializing Whisper model ({WHISPER_MODEL_SIZE})...")
    WHISPER_MODEL = initialize_whisper_model(WHISPER_MODEL_SIZE)
    
    print(f"Initializing LLM from {MODEL_PATH}...")
    LLM_MODEL = initialize_model(model_path=MODEL_PATH)

    record('ok.wav')


if __name__ == "__main__":
    main()