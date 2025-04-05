from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict
import torch.nn as nn

class T5SmallQueryEnhancer(nn.Module):
    def __init__(self, model_name="t5-small"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="huggingface_cache")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir="huggingface_cache")
        
    def forward(self, queries: List[str]) -> List[str]:
        # 添加任务前缀
        prefixed_queries = ["enhance: " + query for query in queries]
        inputs = self.tokenizer(prefixed_queries, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model.generate(
            **inputs,
            max_length=128,  # 设置最大长度
            num_beams=4,     # 使用beam search
            temperature=0.7,  # 添加一些随机性
            do_sample=True   # 启用采样
        )
        enhanced_queries = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return enhanced_queries