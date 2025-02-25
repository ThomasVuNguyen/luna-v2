#!/usr/bin/env python3
import sys
import os
import signal
import subprocess
import re

def cleanup(output_file, play_option, card, hdmi0_card, hdmi1_card=None, board=None, signum=None, frame=None):
    print("\nRecording stopped.")
    # Only try to play if the file exists and playback is enabled
    if os.path.exists(output_file) and play_option != "noplay":
        print("Start playing")
        subprocess.run(["aplay", output_file, "-D", f"hw:{card},0"])
        
        if board == "orangepi5ultra" and hdmi1_card is not None:
            subprocess.run(["aplay", output_file, "-D", f"hw:{hdmi1_card},0"])
        elif hdmi0_card is not None:
            subprocess.run(["aplay", output_file, "-D", f"hw:{hdmi0_card},0"])
    elif not os.path.exists(output_file):
        print("No recording to play back.")
    else:
        print(f"Playback skipped. Recording saved to: {output_file}")
    
    sys.exit(0)

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
    """Configure the mixer settings for recording"""
    # Set common mixer settings
# In the setup_mixer function, try reducing these values:

    # Reduce the ALC Capture PGA (Programmable Gain Amplifier) values
    subprocess.run(["amixer", "-c", card, "cset", "name='ALC Capture Max PGA'", 
                "2" if record_type == "main" else "1"], stdout=subprocess.DEVNULL)

    subprocess.run(["amixer", "-c", card, "cset", "name='ALC Capture Min PGA'", 
                "1" if record_type == "main" else "0"], stdout=subprocess.DEVNULL)

    # Lower the Capture Digital Volume (from 192 to a lower value like 160)
    subprocess.run(["amixer", "-c", card, "cset", "name='Capture Digital Volume'", "192"], 
                stdout=subprocess.DEVNULL)

    # Reduce the channel capture volumes (from 4 to 3 or 2)
    subprocess.run(["amixer", "-c", card, "cset", "name='Left Channel Capture Volume'", "4"], 
                stdout=subprocess.DEVNULL)

    subprocess.run(["amixer", "-c", card, "cset", "name='Right Channel Capture Volume'", "4"], 
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
    # Check command line arguments
    if len(sys.argv) < 2 or sys.argv[1] not in ["main", "headset"]:
        print("usage: test_record.py main/headset [output_file] [play|noplay]")
        print("Recording will continue until interrupted with Ctrl+C")
        sys.exit(1)
    
    record_type = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "/tmp/test.wav"
    play_option = sys.argv[3] if len(sys.argv) > 3 else "play"
    
    # Get board information from /etc/orangepi-release
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
    
    # Setup signal handler for clean interrupt
    signal.signal(signal.SIGINT, lambda signum, frame: 
                 cleanup(output_file, play_option, card, hdmi0_card, hdmi1_card, board))
    
    # Setup mixer
    setup_mixer(card, record_type, board)
    
    # Start recording
    print(f"Start recording: {output_file} (Press Ctrl+C to stop)")
    try:
        subprocess.run(["arecord", "-D", f"hw:{card},0", "-f", "cd", "-t", "wav", output_file])
    except KeyboardInterrupt:
        # Will be caught by signal handler
        pass

if __name__ == "__main__":
    main()
