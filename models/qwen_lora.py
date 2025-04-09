import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple
import numpy as np
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


# 添加QwenLoRAQueryEnhancer实现
class QwenLoRAQueryEnhancer(nn.Module):
    def __init__(self, lora_r=8, lora_alpha=32, lora_dropout=0.1):
        super().__init__()
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name=os.getenv("MODEL_NAME"), 
            trust_remote_code=True,
            cache_dir="huggingface_cache"
        )
        
        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # 加载基础模型 - 使用低精度和量化以节省内存
        print("正在加载Qwen基础模型...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir="huggingface_cache",
            device_map="auto",
            load_in_8bit=True,  # 使用8位量化
            torch_dtype=torch.float16  # 使用半精度
        )

        # Add this before creating the LoRA config
        print("Available modules in the model:")
        for name, _ in self.base_model.named_modules():
            if any(keyword in name for keyword in ['attn', 'attention', 'proj', 'mlp']):
                print(name)
        
        # 确保模型配置有pad_token_id
        self.base_model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # 配置LoRA
        print("正在应用LoRA适配器...")
        # In models/qwen_lora.py, around line 46-50
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,                     # LoRA秩
            lora_alpha=lora_alpha,        # LoRA缩放因子
            lora_dropout=lora_dropout,    # LoRA dropout
            bias="none",                  # 不要训练偏置项
            # Change this line to use Qwen's actual module names
            target_modules=["c_attn", "c_proj", "w1", "w2"],  # For Qwen
        )

        # lora_config = LoraConfig(
        #     task_type=TaskType.CAUSAL_LM,
        #     r=lora_r,                     # LoRA秩
        #     lora_alpha=lora_alpha,        # LoRA缩放因子
        #     lora_dropout=lora_dropout,    # LoRA dropout
        #     bias="none",                  # 不要训练偏置项
        #     # Update to match Qwen1.5 architecture
        #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # )
        
        # 将LoRA应用到模型
        self.model = get_peft_model(self.base_model, lora_config)
        print(f"LoRA适配器已加载! 训练参数数量: {self.model.print_trainable_parameters()}")
        
    def forward(self, queries: List[str]) -> List[str]:
        # 添加系统提示和任务描述
        prompt_template = """You are a query enhancement assistant. Your task is to improve the given query to make it more specific and detailed for code generation.
Original query: {query}
Enhanced query:"""
        
        enhanced_queries = []
        for query in queries:
            prompt = prompt_template.format(query=query)
            
            # 使用no_grad以降低内存使用
            with torch.no_grad():
                # 处理单个查询
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                # 生成增强查询
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
                
                enhanced_query = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # 提取生成的部分
                enhanced_query = enhanced_query.split("Enhanced query:")[-1].strip()
                enhanced_queries.append(enhanced_query)
                
                # 释放内存
                del outputs, inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
        return enhanced_queries
        
    def forward_with_loss(self, queries: List[str]):
        # 记录梯度计算操作
        prompt_template = """You are a query enhancement assistant. Your task is to improve the given query to make it more specific and detailed for code generation.
Original query: {query}
Enhanced query:"""
        
        batch_loss = None
        enhanced_queries = []
        
        for query in queries:
            prompt = prompt_template.format(query=query)
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # 获取模型输出和损失
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            # 处理损失
            if batch_loss is None:
                batch_loss = outputs.loss
            else:
                batch_loss = batch_loss + outputs.loss
            
            # 生成增强查询
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
                
                enhanced_query = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                enhanced_query = enhanced_query.split("Enhanced query:")[-1].strip()
                enhanced_queries.append(enhanced_query)
            
            # 释放不需要的张量
            del generated
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 平均批次损失
        batch_loss = batch_loss / len(queries)
        return batch_loss, enhanced_queries

    def save_lora_weights(self, path):
        """保存LoRA权重"""
        self.model.save_pretrained(path)
        
    def load_lora_weights(self, path):
        """加载LoRA权重"""
        self.model = PeftModel.from_pretrained(self.base_model, path)

