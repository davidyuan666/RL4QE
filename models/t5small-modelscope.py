from modelscope.pipelines import pipeline
from modelscope.models import Model
from modelscope.preprocessors import Preprocessor
from typing import List
import torch.nn as nn

class T5SmallQueryEnhancer(nn.Module):
    def __init__(self, model_name="damo/nlp_t5_small-text2text-generation"):
        super().__init__()
        self.model = Model.from_pretrained(model_name, cache_dir="modelscope_cache")
        self.preprocessor = Preprocessor.from_pretrained(model_name, cache_dir="modelscope_cache")
        self.pipeline = pipeline(
            task='text2text-generation',
            model=self.model,
            preprocessor=self.preprocessor
        )
        
    def forward(self, queries: List[str]) -> List[str]:
        enhanced_queries = []
        for query in queries:
            result = self.pipeline(query)
            enhanced_queries.append(result['text'])
        return enhanced_queries