import llama_cpp
import re
from typing import Generator, List, Optional

class LunaChat:
    def __init__(
        self, 
        model_path: str = "./luna.gguf",
        model_threads: int = 4,
        context_size: int = 4096,
        system_prompt: str = "You are a coding assistant, skilled in programming."
    ):
        self.model_path = model_path
        self.model_threads = model_threads
        self.context_size = context_size
        self.system_prompt = system_prompt
        
        # Define message formats
        self.prefix = "### User: "
        self.suffix = "### Response: "
        self.sys_prefix = "### System: "
        
        # Initialize model
        self.llm = llama_cpp.Llama(
            model_path=self.model_path,
            verbose=False,
            n_ctx=self.context_size,
            n_threads=self.model_threads
        )
        
        # Initialize with system prompt
        self._initialize_context()
        
        # Define patterns to filter out
        self.stop_patterns = [
            "### User", "###User", 
            "### Response", "###Response", 
            "### System", "###System"
        ]
    
    def _initialize_context(self):
        """Initialize the model with the system prompt"""
        system_message = f"{self.sys_prefix}{self.system_prompt}"
        init_tokens = self.llm.tokenize(bytes(system_message, "utf-8"))
        self.llm.eval(init_tokens)
    
    def get_response_stream(self, prompt: str) -> Generator[str, None, tuple]:
        """
        Get a streaming response for the given prompt
        
        Args:
            prompt: The user prompt to send to the model
            
        Yields:
            Tokens from the model response as they're generated
            
        Returns:
            A tuple of (final_response, input_tokens, output_tokens)
        """
        # Format the prompt
        full_prompt = f"{self.prefix}{prompt}\n{self.suffix}"
        ptokens = self.llm.tokenize(bytes(full_prompt, "utf-8"))
        
        # Count input tokens
        input_token_count = len(ptokens)
        
        # Initialize response tracking
        response_token_count = 0
        full_response = ""
        
        # Generate the response
        resp_gen = self.llm.generate(
            ptokens,
            reset=False,
            logits_processor=llama_cpp.LogitsProcessorList([])
        )
        
        for tok in resp_gen:
            if tok == self.llm.token_eos():
                break
                
            word = self.llm.detokenize([tok]).decode("utf-8", errors="ignore")
            full_response += word
            response_token_count += 1
            
            # Check if adding this token creates a stop sequence
            should_stop = False
            for pattern in self.stop_patterns:
                if pattern in full_response[-20:]:
                    # Trim the response at the stop pattern
                    full_response = full_response[:full_response.rfind(pattern)]
                    should_stop = True
                    break
                    
            if should_stop:
                break
            
            # Only yield if we're still outputting
            if not should_stop:
                yield word
        
        # Final cleanup of response
        for pattern in self.stop_patterns:
            if pattern in full_response:
                full_response = full_response[:full_response.rfind(pattern)]
        
        return full_response, input_token_count, response_token_count
    
    def get_response(self, prompt: str) -> tuple:
        """
        Get a complete response (non-streaming) for the given prompt
        
        Args:
            prompt: The user prompt to send to the model
            
        Returns:
            A tuple of (response, input_tokens, output_tokens)
        """
        response = ""
        for token in self.get_response_stream(prompt):
            response += token
        
        # The final values are returned from the generator
        final_response, input_tokens, output_tokens = next(self.get_response_stream(prompt))
        
        return final_response, input_tokens, output_tokens

# Example usage
if __name__ == "__main__":
    chat = LunaChat()
    
    print("\nHow can I help you? ")
    while True:
        user_prompt = input("\n> ")
        if user_prompt.lower() in ["quit", "exit"]:
            break
            
        print("\nResponse: ", end="", flush=True)
        
        # Use the streaming function to print response as it's generated
        try:
            # Initial call to consume the generator and print tokens
            for token in chat.get_response_stream(user_prompt):
                print(token, end="", flush=True)
            
            # Second call to get the stats (this is a bit inefficient but works for demonstration)
            final_response, input_tokens, output_tokens = chat.get_response(user_prompt)
            
            # Print token stats
            print(f"\n\nInput tokens: {input_tokens}")
            print(f"Response tokens: {output_tokens}")
            print(f"Total tokens: {input_tokens + output_tokens}")
            
        except Exception as e:
            print(f"\nError: {e}")
