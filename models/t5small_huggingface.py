from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict
import torch
import torch.nn as nn

class T5SmallQueryEnhancer(nn.Module):
    def __init__(self, model_name="t5-small"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="huggingface_cache")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir="huggingface_cache")
        
    def forward(self, queries: List[str]) -> List[str]:
        inputs = self.tokenizer(queries, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model.generate(**inputs)
        enhanced_queries = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return enhanced_queries
    
    def forward_with_loss(self, queries: List[str]) -> Dict:
        # Tokenize input
        inputs = self.tokenizer(queries, return_tensors="pt", padding=True, truncation=True)
        
        # Generate enhanced queries
        outputs = self.model.generate(**inputs)
        enhanced_queries = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Calculate loss
        outputs = self.model(**inputs, labels=inputs['input_ids'])
        
        return {
            'generated_text': enhanced_queries,
            'loss': outputs.loss
        }