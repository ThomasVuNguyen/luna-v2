import numpy as np
from llama_cpp import Llama

# Path to your local GGUF embedding model
model_path = "luna-embed.gguf"  # Adjust path if needed

# Initialize the model with embedding mode enabled
llm = Llama(
    model_path=model_path,
    n_ctx=512,  # Context size matches your model's context length
    embedding=True,  # Enable embedding mode
    n_threads=4  # Adjust based on your CPU
)

# Get embeddings for text
text = "This is a sample text for embedding."
embedding = llm.embed(text)

# Convert to numpy array if it's a list
if isinstance(embedding, list):
    embedding = np.array(embedding)

print(f"Embedding shape: {embedding.shape}")
print(f"Embedding sample: {embedding[:364]}")  # First 5 values
