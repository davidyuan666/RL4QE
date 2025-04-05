import os
# Set environment variables to disable Flash Attention
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
# This is important to prevent loading the problematic flash_attn module
os.environ["DISABLE_FLASH_ATTN"] = "true"
# Additional environment variables to prevent Flash Attention loading
os.environ["USE_FLASH_ATTENTION"] = "0"

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
from typing import List
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QwenQueryEnhancer(nn.Module):
    def __init__(self, model_name="Qwen/Qwen-7B"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
            cache_dir="huggingface_cache"
        )
        # For Qwen, we need to set pad_token to an existing token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Disable Flash Attention to avoid CUDA symbol errors
        try:
            # First, try to clear any loaded modules that might cause issues
            torch.cuda.empty_cache()
            
            # Explicitly prevent transformers from using flash attention
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir="huggingface_cache",
                device_map="auto",
                # Explicitly disable flash attention
                attn_implementation="eager",
                use_flash_attention_2=False,
                torch_dtype=torch.bfloat16  # Use lower precision for memory efficiency
            )
            logger.info("Successfully loaded Qwen model with standard attention")
        except Exception as e:
            logger.error(f"Standard model loading failed: {e}")
            logger.info("Trying with 8-bit quantization")
            
            # Try 8-bit quantization as fallback
            try:
                import bitsandbytes as bnb
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    cache_dir="huggingface_cache",
                    device_map="auto",
                    load_in_8bit=True,
                    use_flash_attention_2=False,
                    attn_implementation="eager"
                )
                logger.info("Successfully loaded 8-bit quantized model")
            except Exception as e:
                logger.error(f"8-bit quantization failed: {e}")
                logger.error("Try using a smaller model or check CUDA installation")
                
                # Final fallback - try smaller Qwen model
                try:
                    logger.info("Attempting to load smaller Qwen model as fallback")
                    fallback_model = "Qwen/Qwen-1.8B"
                    self.model = AutoModelForCausalLM.from_pretrained(
                        fallback_model,
                        trust_remote_code=True,
                        cache_dir="huggingface_cache",
                        device_map="auto",
                        torch_dtype=torch.float16,
                        attn_implementation="eager",
                        use_flash_attention_2=False
                    )
                    logger.info(f"Successfully loaded smaller model: {fallback_model}")
                except Exception as e:
                    logger.error(f"All loading attempts failed: {e}")
                    raise
        
    def forward(self, queries: List[str]) -> List[str]:
        # 添加系统提示和任务描述
        prompt_template = """You are a query enhancement assistant. Your task is to improve the given query to make it more specific and detailed for code generation.
Original query: {query}
Enhanced query:"""
        
        enhanced_queries = []
        for query in queries:
            prompt = prompt_template.format(query=query)
            # Process one at a time to save memory
            with torch.no_grad():  # Disable gradient calculation
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
            # 移除原始提示，只保留生成的部分
            enhanced_query = enhanced_query.split("Enhanced query:")[-1].strip()
            enhanced_queries.append(enhanced_query)
            
            # Clear memory after each generation
            del outputs, inputs
            torch.cuda.empty_cache()
            
        return enhanced_queries
        
    def forward_with_loss(self, queries: List[str]):
        # 对于Qwen这样的大模型，我们可能不需要传统的loss计算
        # 直接返回生成的结果和一个占位符loss
        enhanced_queries = self.forward(queries)
        # Create a tensor with requires_grad=True
        loss = torch.tensor(0.0, requires_grad=True)
        return loss, enhanced_queries