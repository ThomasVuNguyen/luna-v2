import chromadb
import numpy as np
import time
import psutil
import os
import uuid
from tqdm import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import random
import string

def generate_random_text(min_length=100, max_length=500):
    """Generate random text of variable length."""
    length = random.randint(min_length, max_length)
    return ''.join(random.choice(string.ascii_letters + ' ' * 10) for _ in range(length))

def memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def benchmark_chroma(num_chunks=2000, batch_size=100, vector_size=768, collection_name=None):
    """
    Benchmark ChromaDB performance with specified number of text chunks.
    
    Args:
        num_chunks: Number of text chunks to process
        batch_size: Batch size for adding documents
        vector_size: Dimension of embedding vectors
        collection_name: Name for the collection (random if None)
    
    Returns:
        dict: Dictionary containing benchmark results
    """
    # Create client and collection
    client = chromadb.Client()
    
    # Use UUID for collection name if not specified
    if collection_name is None:
        collection_name = f"benchmark_{uuid.uuid4().hex[:8]}"
    
    # Try to get collection, delete if exists
    try:
        client.delete_collection(collection_name)
    except:
        pass
    
    collection = client.create_collection(name=collection_name)
    
    # Generate random texts
    print(f"Generating {num_chunks} random text chunks...")
    texts = [generate_random_text() for _ in range(num_chunks)]
    
    # Generate random embeddings
    print(f"Generating random embeddings with dimension {vector_size}...")
    embeddings = [list(np.random.random(vector_size)) for _ in range(num_chunks)]
    
    # Generate IDs
    ids = [f"id_{i}" for i in range(num_chunks)]
    
    # Add data in batches and measure time
    print(f"Adding {num_chunks} documents in batches of {batch_size}...")
    
    metrics = {
        "add_times": [],
        "memory_usage": [],
        "query_times": [],
        "query_k_times": {},
        "batch_sizes": []
    }
    
    initial_memory = memory_usage()
    metrics["memory_usage"].append((0, initial_memory))
    
    total_add_time = 0
    
    for i in tqdm(range(0, num_chunks, batch_size)):
        batch_end = min(i + batch_size, num_chunks)
        batch_size_actual = batch_end - i
        
        start_time = time.time()
        collection.add(
            documents=texts[i:batch_end],
            embeddings=embeddings[i:batch_end],
            ids=ids[i:batch_end]
        )
        end_time = time.time()
        
        add_time = end_time - start_time
        total_add_time += add_time
        
        metrics["add_times"].append(add_time)
        metrics["batch_sizes"].append(batch_size_actual)
        metrics["memory_usage"].append((i + batch_size_actual, memory_usage()))
    
    metrics["total_add_time"] = total_add_time
    metrics["avg_add_time_per_doc"] = total_add_time / num_chunks
    
    # Run sample queries with different numbers of results (k values)
    k_values = [1, 5, 10, 50, 100]
    query_samples = 10
    
    print("Running sample queries with different k values...")
    
    for k in k_values:
        query_times = []
        
        for _ in tqdm(range(query_samples), desc=f"Queries with k={k}"):
            # Use a random embedding for query
            query_embedding = list(np.random.random(vector_size))
            
            start_time = time.time()
            collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )
            end_time = time.time()
            
            query_times.append(end_time - start_time)
        
        metrics["query_k_times"][k] = query_times
        metrics["query_times"].extend(query_times)
    
    # Calculate summary statistics
    metrics["total_memory_mb"] = metrics["memory_usage"][-1][1] - metrics["memory_usage"][0][1]
    metrics["avg_query_time"] = sum(metrics["query_times"]) / len(metrics["query_times"])
    metrics["avg_query_k_times"] = {k: sum(times)/len(times) for k, times in metrics["query_k_times"].items()}
    
    return metrics

def plot_results(metrics, output_file="chroma_benchmark_results.png"):
    """Plot benchmark results."""
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Add times per batch
    axs[0, 0].plot(metrics["add_times"])
    axs[0, 0].set_title("Add Time per Batch")
    axs[0, 0].set_xlabel("Batch Number")
    axs[0, 0].set_ylabel("Time (seconds)")
    
    # Plot 2: Memory usage
    docs, mem = zip(*metrics["memory_usage"])
    axs[0, 1].plot(docs, mem)
    axs[0, 1].set_title("Memory Usage")
    axs[0, 1].set_xlabel("Documents Added")
    axs[0, 1].set_ylabel("Memory (MB)")
    
    # Plot 3: Query times for different k values
    k_values = list(metrics["avg_query_k_times"].keys())
    avg_times = list(metrics["avg_query_k_times"].values())
    axs[1, 0].bar(k_values, avg_times)
    axs[1, 0].set_title("Average Query Time by k")
    axs[1, 0].set_xlabel("k (number of results)")
    axs[1, 0].set_ylabel("Time (seconds)")
    
    # Plot 4: Summary statistics
    summary = [
        f"Total documents: 2000",
        f"Total add time: {metrics['total_add_time']:.2f}s",
        f"Avg add time per doc: {metrics['avg_add_time_per_doc']*1000:.2f}ms",
        f"Total memory: {metrics['total_memory_mb']:.2f}MB",
        f"Avg query time: {metrics['avg_query_time']*1000:.2f}ms"
    ]
    
    axs[1, 1].axis('off')
    axs[1, 1].text(0.1, 0.5, "\n".join(summary), fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Results saved to {output_file}")
    
    return fig

def main():
    """Main function to run the benchmark."""
    print("ChromaDB Performance Benchmark")
    print("=============================")
    
    # Print system info
    print("\nSystem Information:")
    print(f"CPU: {psutil.cpu_count(logical=True)} logical cores")
    print(f"Memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    
    # Run benchmark
    print("\nStarting benchmark...")
    start_time = time.time()
    
    metrics = benchmark_chroma(
        num_chunks=200000,
        batch_size=100,
        vector_size=512
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Print results
    print("\nBenchmark Results:")
    print(f"Total runtime: {total_time:.2f} seconds")
    print(f"Total memory used: {metrics['total_memory_mb']:.2f} MB")
    print(f"Average time to add a document: {metrics['avg_add_time_per_doc']*1000:.2f} ms")
    print(f"Average query time: {metrics['avg_query_time']*1000:.2f} ms")
    print("\nAverage query times by k:")
    for k, time_avg in metrics["avg_query_k_times"].items():
        print(f"  k={k}: {time_avg*1000:.2f} ms")
    
    # Plot results
    fig = plot_results(metrics)
    plt.show()
    
    return metrics

if __name__ == "__main__":
    main()
