import requests
import json
import time
import argparse
import re
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from datetime import datetime

console = Console()

class LlamaPerformanceTester:
    def __init__(self, server_url="http://localhost:8000", model_name="luna"):
        self.server_url = server_url
        self.model_name = model_name
        self.streaming_stats = {
            "total_tokens": 0,
            "elapsed_time": 0,
            "tokens_per_second": 0,
            "words": 0,
            "words_per_second": 0,
            "response_text": "",
        }
        self.benchmark_results = []
    
    def count_words(self, text):
        """Count words in text."""
        # Split by whitespace and count non-empty strings
        return len([w for w in re.split(r'\s+', text) if w])
    
    def run_benchmark(self, prompt, use_mlock=True, max_tokens=256, num_runs=1):
        """Run a benchmark test for a given prompt and configuration."""
        console.print(f"\n[bold blue]Running benchmark with prompt:[/bold blue] '{prompt}'")
        console.print(f"[bold]Configuration:[/bold] use_mlock={use_mlock}, max_tokens={max_tokens}, runs={num_runs}")
        
        results = []
        
        for run in range(1, num_runs + 1):
            console.print(f"\n[bold]Run {run}/{num_runs}[/bold]")
            
            # Reset stats for this run
            self.streaming_stats = {
                "total_tokens": 0,
                "elapsed_time": 0,
                "tokens_per_second": 0,
                "words": 0,
                "words_per_second": 0,
                "response_text": "",
            }
            
            result = self.test_streaming_completion(prompt, use_mlock, max_tokens)
            results.append(result)
            
            # Wait a bit between runs
            if run < num_runs:
                time.sleep(1)
        
        # Calculate averages
        avg_tokens_per_second = sum(r["tokens_per_second"] for r in results) / len(results)
        avg_words_per_second = sum(r["words_per_second"] for r in results) / len(results)
        avg_total_tokens = sum(r["total_tokens"] for r in results) / len(results)
        avg_time = sum(r["elapsed_time"] for r in results) / len(results)
        
        # Store benchmark results
        benchmark_result = {
            "prompt": prompt,
            "use_mlock": use_mlock,
            "max_tokens": max_tokens,
            "runs": num_runs,
            "avg_tokens_per_second": avg_tokens_per_second,
            "avg_words_per_second": avg_words_per_second,
            "avg_total_tokens": avg_total_tokens,
            "avg_time": avg_time,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.benchmark_results.append(benchmark_result)
        
        # Display summary
        console.print("\n[bold green]Benchmark Summary:[/bold green]")
        console.print(f"Average tokens per second: [bold]{avg_tokens_per_second:.2f}[/bold]")
        console.print(f"Average words per second: [bold]{avg_words_per_second:.2f}[/bold]")
        console.print(f"Average total tokens: [bold]{avg_total_tokens:.1f}[/bold]")
        console.print(f"Average time: [bold]{avg_time:.2f}[/bold] seconds")
        
        return benchmark_result
    
    def test_streaming_completion(self, prompt, use_mlock=True, max_tokens=256):
        """Test streaming chat completion with performance metrics."""
        url = f"{self.server_url}/v1/chat/completions"
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "stream": True,
            "use_mlock": use_mlock
        }
        
        start_time = time.time()
        last_update_time = start_time
        streaming_started = False
        first_token_time = None
        token_count = 0
        cumulative_text = ""
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Processing..."),
            BarColumn(),
            TextColumn("[bold]{task.fields[tokens]} tokens | {task.fields[tps]:.2f} tokens/sec | {task.fields[wps]:.2f} words/sec"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            # Add a task
            task = progress.add_task("Generating", tokens=0, tps=0, wps=0)
            
            try:
                with requests.post(url, json=payload, stream=True) as response:
                    if response.status_code != 200:
                        console.print(f"[bold red]Error: {response.status_code}[/bold red]")
                        console.print(response.text)
                        return self.streaming_stats
                    
                    # Process the streaming response
                    for line in response.iter_lines():
                        if line:
                            line = line.decode('utf-8')
                            
                            # Skip the [DONE] message
                            if line == "data: [DONE]":
                                continue
                                
                            # Skip lines that don't start with "data: "
                            if not line.startswith("data: "):
                                continue
                                
                            # Parse the JSON
                            try:
                                data = json.loads(line[6:])  # Skip "data: " prefix
                                
                                if "choices" in data and data["choices"] and "delta" in data["choices"][0]:
                                    delta = data["choices"][0]["delta"]
                                    
                                    # Track when we get the first content token
                                    if "content" in delta and delta["content"]:
                                        if not streaming_started:
                                            streaming_started = True
                                            first_token_time = time.time()
                                        
                                        token = delta["content"]
                                        token_count += 1
                                        cumulative_text += token
                                        
                                        # Update stats
                                        current_time = time.time()
                                        if current_time - last_update_time >= 0.1:  # Update every 100ms
                                            elapsed = current_time - first_token_time if first_token_time else 0
                                            if elapsed > 0:
                                                tokens_per_second = token_count / elapsed
                                                word_count = self.count_words(cumulative_text)
                                                words_per_second = word_count / elapsed
                                                
                                                # Update progress
                                                progress.update(
                                                    task, 
                                                    completed=min(token_count / max_tokens * 100, 100),
                                                    tokens=token_count,
                                                    tps=tokens_per_second,
                                                    wps=words_per_second
                                                )
                                            
                                            last_update_time = current_time
                            except json.JSONDecodeError:
                                pass
            except Exception as e:
                console.print(f"[bold red]Error: {str(e)}[/bold red]")
        
        # Calculate final stats
        end_time = time.time()
        elapsed = end_time - (first_token_time or start_time)
        tokens_per_second = token_count / elapsed if elapsed > 0 else 0
        word_count = self.count_words(cumulative_text)
        words_per_second = word_count / elapsed if elapsed > 0 else 0
        
        result = {
            "total_tokens": token_count,
            "elapsed_time": elapsed,
            "tokens_per_second": tokens_per_second,
            "words": word_count,
            "words_per_second": words_per_second,
            "response_text": cumulative_text
        }
        
        console.print(f"\n[bold]Results:[/bold]")
        console.print(f"Total tokens: [bold]{token_count}[/bold]")
        console.print(f"Total words: [bold]{word_count}[/bold]")
        console.print(f"Time elapsed: [bold]{elapsed:.2f}[/bold] seconds")
        console.print(f"Tokens per second: [bold]{tokens_per_second:.2f}[/bold]")
        console.print(f"Words per second: [bold]{words_per_second:.2f}[/bold]")
        
        # Print the first 100 characters of the response
        preview = cumulative_text[:100] + "..." if len(cumulative_text) > 100 else cumulative_text
        console.print(f"\n[bold]Response preview:[/bold] {preview}")
        
        return result
    
    def compare_mlock_performance(self, prompt, max_tokens=256, runs=3):
        """Compare performance with and without memory locking."""
        console.print("\n[bold yellow]===== PERFORMANCE COMPARISON: WITH vs WITHOUT MEMORY LOCKING =====[/bold yellow]")
        
        # First run with memory locking enabled
        console.print("\n[bold magenta]Testing WITH memory locking (use_mlock=True):[/bold magenta]")
        with_mlock = self.run_benchmark(prompt, use_mlock=True, max_tokens=max_tokens, num_runs=runs)
        
        # Wait a bit between tests
        time.sleep(2)
        
        # Then run without memory locking
        console.print("\n[bold magenta]Testing WITHOUT memory locking (use_mlock=False):[/bold magenta]")
        without_mlock = self.run_benchmark(prompt, use_mlock=False, max_tokens=max_tokens, num_runs=runs)
        
        # Calculate improvement
        tps_improvement = (with_mlock["avg_tokens_per_second"] / without_mlock["avg_tokens_per_second"] - 1) * 100
        wps_improvement = (with_mlock["avg_words_per_second"] / without_mlock["avg_words_per_second"] - 1) * 100
        
        # Display comparison table
        table = Table(title="Memory Locking Performance Comparison")
        table.add_column("Metric", style="cyan")
        table.add_column("With mlock", style="green")
        table.add_column("Without mlock", style="yellow")
        table.add_column("Improvement", style="magenta")
        
        table.add_row(
            "Tokens/sec", 
            f"{with_mlock['avg_tokens_per_second']:.2f}", 
            f"{without_mlock['avg_tokens_per_second']:.2f}",
            f"{tps_improvement:.1f}%"
        )
        table.add_row(
            "Words/sec", 
            f"{with_mlock['avg_words_per_second']:.2f}", 
            f"{without_mlock['avg_words_per_second']:.2f}",
            f"{wps_improvement:.1f}%"
        )
        table.add_row(
            "Avg Time (sec)", 
            f"{with_mlock['avg_time']:.2f}", 
            f"{without_mlock['avg_time']:.2f}",
            f"{-((with_mlock['avg_time'] / without_mlock['avg_time'] - 1) * 100):.1f}%"
        )
        
        console.print(table)
        
        return {
            "with_mlock": with_mlock,
            "without_mlock": without_mlock,
            "tps_improvement": tps_improvement,
            "wps_improvement": wps_improvement
        }

def main():
    parser = argparse.ArgumentParser(description='Benchmark llama.cpp server with performance metrics')
    parser.add_argument('--url', type=str, default='http://localhost:8000', help='Server URL')
    parser.add_argument('--model', type=str, default='luna', help='Model name')
    parser.add_argument('--prompt', type=str, default='Explain the benefits of memory locking in large language models', 
                      help='Prompt to test with')
    parser.add_argument('--max-tokens', type=int, default=256, help='Maximum tokens to generate')
    parser.add_argument('--runs', type=int, default=3, help='Number of runs for each test')
    parser.add_argument('--compare', action='store_true', help='Compare performance with and without memory locking')
    parser.add_argument('--mlock', type=bool, default=True, help='Whether to use memory locking')
    
    args = parser.parse_args()
    
    tester = LlamaPerformanceTester(server_url=args.url, model_name=args.model)
    
    if args.compare:
        tester.compare_mlock_performance(args.prompt, args.max_tokens, args.runs)
    else:
        tester.run_benchmark(args.prompt, args.mlock, args.max_tokens, args.runs)

if __name__ == "__main__":
    main()
