from transformers import AutoTokenizer, AutoModelForSeq2SeqGeneration
from typing import List
import torch.nn as nn

class T5SmallQueryEnhancer(nn.Module):
    def __init__(self, model_name="t5-small"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="cache")
        self.model = AutoModelForSeq2SeqGeneration.from_pretrained(model_name, cache_dir="cache")
        
    def forward(self, queries: List[str]) -> List[str]:
        inputs = self.tokenizer(queries, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model.generate(**inputs)
        enhanced_queries = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return enhanced_queries