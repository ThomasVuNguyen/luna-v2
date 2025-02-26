from speak.synthesize import synthesize_and_play_stream
from think.stream import initialize_model, generate_stream

# Initialize the LLM
llm = initialize_model(model_path='think/luna.gguf')

# Generate response and synthesize/play it simultaneously
def text_generator(prompt):
    for token in generate_stream(llm, prompt):
        yield token

synthesize_and_play_stream(text_generator("Tell me a short story"))
