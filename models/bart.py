from transformers import AutoTokenizer, AutoModelForSeq2SeqGeneration
import torch.nn as nn
from typing import List

class BARTQueryEnhancer(nn.Module):
    def __init__(self, model_name="facebook/bart-large-cnn"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="cache")
        self.model = AutoModelForSeq2SeqGeneration.from_pretrained(model_name, cache_dir="cache")

    def forward(self, queries: List[str]) -> List[str]:
        inputs = self.tokenizer(queries, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model.generate(**inputs)
        enhanced_queries = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return enhanced_queries


