from typing import List
import torch.nn as nn
import torch

class QwenVLLMQueryEnhancer(nn.Module):
    def __init__(self, model_name="Qwen/Qwen-7B"):
        super().__init__()
        try:
            from vllm import LLM, SamplingParams
            self.use_vllm = True
            
            # Initialize vLLM
            self.llm = LLM(
                model=model_name,
                trust_remote_code=True,
                tensor_parallel_size=1  # Adjust based on available GPUs
            )
            
            # Initialize standard tokenizer for consistency in interface
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True,
                cache_dir="huggingface_cache"
            )
            
            # Configure sampling parameters
            self.sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=128
            )
            
            print("Successfully initialized vLLM for Qwen")
            
        except ImportError:
            print("vLLM not installed. Please install it: pip install vllm")
            print("Falling back to standard implementation")
            self.use_vllm = False
            
            # Fall back to standard implementation
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True,
                cache_dir="huggingface_cache"
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir="huggingface_cache",
                device_map="auto"
            )
    
    def forward(self, queries: List[str]) -> List[str]:
        # System prompt template
        prompt_template = """You are a query enhancement assistant. Your task is to improve the given query to make it more specific and detailed for code generation.
Original query: {query}
Enhanced query:"""
        
        enhanced_queries = []
        
        if self.use_vllm:
            # vLLM approach
            prompts = [prompt_template.format(query=query) for query in queries]
            outputs = self.llm.generate(prompts, self.sampling_params)
            
            for output in outputs:
                text = output.outputs[0].text
                # Extract the enhanced query portion
                enhanced_query = text.split("Enhanced query:")[-1].strip()
                enhanced_queries.append(enhanced_query)
        else:
            # Standard approach (same as original implementation)
            for query in queries:
                prompt = prompt_template.format(query=query)
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                enhanced_query = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                enhanced_query = enhanced_query.split("Enhanced query:")[-1].strip()
                enhanced_queries.append(enhanced_query)
                
        return enhanced_queries
    
    def forward_with_loss(self, queries: List[str]):
        # For vLLM or any LLM engine, we don't need traditional loss computation
        # Just return generated results and a placeholder loss
        enhanced_queries = self.forward(queries)
        loss = torch.tensor(0.0, requires_grad=True)
        return loss, enhanced_queries