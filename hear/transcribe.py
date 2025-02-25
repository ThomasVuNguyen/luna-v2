from faster_whisper import WhisperModel
import time

# Use tiny model with English-only
model_size = "base.en"

# Run on CPU with INT8
model = WhisperModel(model_size, device="cpu", compute_type="int8")

# Start timing
start_time = time.time()

# Transcribe the test7.wav file
segments, info = model.transcribe("test.wav", beam_size=5)

# Calculate elapsed time
elapsed_time = time.time() - start_time

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
print("Transcription completed in %.2f seconds" % elapsed_time)

for segment in segments:
    start = time.time()
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    end = time.time() - start
    print(end)
print(time.time() - start_time)
