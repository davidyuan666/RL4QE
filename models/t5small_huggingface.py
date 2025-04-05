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
        
    def forward_with_loss(self, queries: List[str]):
        # Similar to forward but also computes loss
        prefixed_queries = ["enhance: " + query for query in queries]
        inputs = self.tokenizer(prefixed_queries, return_tensors="pt", padding=True, truncation=True)
        
        # Get model outputs with loss
        outputs = self.model(**inputs, labels=inputs["input_ids"])
        
        # Generate enhanced queries
        generated = self.model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            temperature=0.7,
            do_sample=True
        )
        enhanced_queries = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        
        return outputs.loss, enhanced_queries
    


if __name__ == "__main__":
    # 创建模型实例
    model = T5SmallQueryEnhancer()
    
    # 测试样例查询
    test_queries = [
        "如何学习python",
        "推荐一本机器学习的书",
        "什么是强化学习"
    ]
    
    print("====== 测试forward方法 ======")
    enhanced = model.forward(test_queries)
    
    # 打印输入和输出对比
    for i, (query, result) in enumerate(zip(test_queries, enhanced)):
        print(f"\n样例 {i+1}:")
        print(f"输入: {query}")
        print(f"输出: {result}")
    
    print("\n====== 测试forward_with_loss方法 ======")
    loss, enhanced_with_loss = model.forward_with_loss(test_queries)
    print(f"损失值: {loss.item()}")
    
    for i, (query, result) in enumerate(zip(test_queries, enhanced_with_loss)):
        print(f"\n样例 {i+1}:")
        print(f"输入: {query}")
        print(f"输出: {result}")