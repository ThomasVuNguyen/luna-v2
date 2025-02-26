from faster_whisper import WhisperModel
import time

def initialize_whisper_model(model_size="base.en", device="cpu", compute_type="int8"):
    """
    Initialize a Whisper model with specified parameters.
    
    Args:
        model_size (str): Size/version of the Whisper model to use
        device (str): Device to run the model on ('cpu' or 'cuda')
        compute_type (str): Compute type ('int8', 'float16', 'float32')
    
    Returns:
        WhisperModel: Initialized Whisper model
    """
    return WhisperModel(model_size, device=device, compute_type=compute_type)

def transcribe_audio(model, audio_file, beam_size=5, language=None):
    """
    Transcribe an audio file using a Whisper model.
    
    Args:
        model (WhisperModel): Initialized Whisper model
        audio_file (str): Path to the audio file to transcribe
        beam_size (int): Beam size for transcription
        language (str, optional): Language code to force for transcription
    
    Returns:
        tuple: (segments, info, elapsed_time)
            - segments: Iterator of transcription segments
            - info: Information about the transcription
            - elapsed_time: Time taken for transcription
    """
    start_time = time.time()
    segments, info = model.transcribe(audio_file, beam_size=beam_size, language=language)
    elapsed_time = time.time() - start_time
    
    return segments, info, elapsed_time

def get_full_transcript(segments):
    """
    Convert transcription segments into a single string.
    
    Args:
        segments: Transcription segments from the model
    
    Returns:
        str: Complete transcript as a string
    """
    return " ".join(segment.text for segment in segments)

def print_transcription_info(info, elapsed_time):
    """
    Print information about the transcription.
    
    Args:
        info: Information from the transcription
        elapsed_time (float): Time taken for transcription
    """
    print(f"Detected language '{info.language}' with probability {info.language_probability:.4f}")
    print(f"Transcription completed in {elapsed_time:.2f} seconds")

def print_segments(segments):
    """
    Print each segment of the transcription with timing information.
    
    Args:
        segments: Transcription segments from the model
    """
    for segment in segments:
        start = time.time()
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        processing_time = time.time() - start
        print(f"Segment processing time: {processing_time:.6f}s")

# Example of how to use these functions
def transcribe_file(file_path, model_size="base.en"):
    """
    Transcribe an audio file and print the results.
    
    Args:
        file_path (str): Path to the audio file
        model_size (str): Size of the Whisper model to use
    
    Returns:
        str: The full transcript
    """
    # Initialize the model
    model = initialize_whisper_model(model_size)
    
    # Start timing
    total_start_time = time.time()
    
    # Transcribe the audio
    segments, info, transcription_time = transcribe_audio(model, file_path)
    
    # Print information
    print_transcription_info(info, transcription_time)
    
    # Create a list from segments to allow multiple iterations
    segment_list = list(segments)
    
    # Print individual segments
    print_segments(segment_list)
    
    # Get the full transcript
    transcript = get_full_transcript(segment_list)
    
    # Print total time
    total_time = time.time() - total_start_time
    print(f"Total processing time: {total_time:.2f} seconds")
    
    return transcript

if __name__ == "__main__":
    # Example usage
    transcript = transcribe_file("test.wav")
    print("\nFull transcript:")
    print(transcript)
