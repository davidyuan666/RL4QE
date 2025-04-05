from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
from typing import List
import torch

class QwenQueryEnhancer(nn.Module):
    def __init__(self, model_name="Qwen/Qwen-7B"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
            cache_dir="huggingface_cache"
        )
        # For Qwen, we need to set pad_token to an existing token
        # Make sure the pad token is properly set and persists
        if self.tokenizer.pad_token is None:
            # Qwen doesn't support adding custom special tokens
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Make sure the model is updated with the new token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir="huggingface_cache",
            device_map="auto"
        )
        # No need to resize token embeddings as we're using an existing token
        
    def forward(self, queries: List[str]) -> List[str]:
        # 添加系统提示和任务描述
        prompt_template = """You are a query enhancement assistant. Your task is to improve the given query to make it more specific and detailed for code generation.
Original query: {query}
Enhanced query:"""
        
        enhanced_queries = []
        for query in queries:
            prompt = prompt_template.format(query=query)
            # Ensure pad_token_id is explicitly passed and pad_token exists
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            enhanced_query = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 移除原始提示，只保留生成的部分
            enhanced_query = enhanced_query.split("Enhanced query:")[-1].strip()
            enhanced_queries.append(enhanced_query)
            
        return enhanced_queries
        
    def forward_with_loss(self, queries: List[str]):
        # 对于Qwen这样的大模型，我们可能不需要传统的loss计算
        # 直接返回生成的结果和一个占位符loss
        enhanced_queries = self.forward(queries)
        return torch.tensor(0.0), enhanced_queries