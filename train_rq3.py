# train_rq3.py
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple, Union, Optional
import numpy as np
import json
import os
import gc
import time
import argparse
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig
)

from llm.deepseek import DeepseekAPI
from utils.reward_util import RewardCalculator

# 加载环境变量
load_dotenv()

# 获取奖励计算方法
REWARD_METHOD = os.getenv("REWARD_METHOD", "overlap")

class ModelType:
    CAUSAL_LM = "causal_lm"  # GPT类自回归模型 (Qwen, Llama等)
    SEQ2SEQ_LM = "seq2seq_lm"  # 编码器-解码器模型 (T5, BART等)

class BaseQueryEnhancer(nn.Module):
    """所有查询增强模型的基类"""
    def __init__(self):
        super().__init__()
        
    def forward(self, queries: List[str]) -> List[str]:
        """处理输入查询并返回增强后的查询"""
        raise NotImplementedError("子类必须实现forward方法")
        
    def forward_with_loss(self, queries: List[str]):
        """处理输入查询并返回损失和增强后的查询"""
        raise NotImplementedError("子类必须实现forward_with_loss方法")
        
    def save_lora_weights(self, path):
        """保存LoRA权重"""
        if hasattr(self, 'model') and hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(path)
        else:
            raise NotImplementedError("模型不支持保存LoRA权重")
            
    def load_lora_weights(self, path):
        """加载LoRA权重"""
        raise NotImplementedError("子类必须实现load_lora_weights方法")

class CausalLMQueryEnhancer(BaseQueryEnhancer):
    """自回归语言模型查询增强器 (Qwen, Llama等)"""
    def __init__(
        self, 
        model_name: str,
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        use_4bit: bool = False,
        use_8bit: bool = False,
        device_map: str = "auto"
    ):
        super().__init__()
        self.model_name = model_name
        self.use_lora = use_lora
        self.model_type = ModelType.CAUSAL_LM
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
            cache_dir="huggingface_cache"
        )
        
        # 设置pad_token (如果不存在)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # 量化配置
        quantization_config = None
        if use_4bit:
            print(f"正在使用4位量化加载模型 {model_name}...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif use_8bit:
            print(f"正在使用8位量化加载模型 {model_name}...")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        
        # 加载基础模型
        print(f"正在加载基础模型 {model_name}...")
        if use_lora:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir="huggingface_cache",
                device_map=device_map,
                quantization_config=quantization_config,
                torch_dtype=torch.float16  # 使用半精度
            )
            
            # 确保模型配置有pad_token_id
            self.base_model.config.pad_token_id = self.tokenizer.pad_token_id
            
            # 配置LoRA
            print("正在应用LoRA适配器...")
            # 为不同模型家族设置目标模块
            target_modules = self._get_target_modules_for_model()
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=target_modules
            )
            
            # 将LoRA应用到模型
            self.model = get_peft_model(self.base_model, lora_config)
            print(f"LoRA适配器已加载! 训练参数数量: {self.model.print_trainable_parameters()}")
        else:
            # 全参数模型 - 使用半精度以减少内存使用
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir="huggingface_cache",
                device_map=device_map,
                quantization_config=quantization_config,
                torch_dtype=torch.float16
            )
            # 确保模型配置有pad_token_id
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
    
    def _get_target_modules_for_model(self) -> List[str]:
        """根据模型名称获取适合的LoRA目标模块"""
        model_name_lower = self.model_name.lower()
        
        # 为每个模型家族指定适当的目标模块
        if "llama" in model_name_lower:
            return ["q_proj", "v_proj", "k_proj", "o_proj"]
        elif "qwen" in model_name_lower:
            return ["c_attn", "c_proj", "w1", "w2"]
        elif "mistral" in model_name_lower:
            return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "bloom" in model_name_lower:
            return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
        else:
            # 默认目标模块 - 通用设置
            return ["query", "key", "value", "dense"]
    
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
    
    def load_lora_weights(self, path):
        """加载LoRA权重"""
        if self.use_lora:
            self.model = PeftModel.from_pretrained(self.base_model, path)
        else:
            raise ValueError("非LoRA模型无法加载LoRA权重")

class Seq2SeqLMQueryEnhancer(BaseQueryEnhancer):
    """序列到序列模型查询增强器 (T5, BART等)"""
    def __init__(
        self, 
        model_name: str,
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        use_4bit: bool = False,
        use_8bit: bool = False,
        device_map: str = "auto"
    ):
        super().__init__()
        self.model_name = model_name
        self.use_lora = use_lora
        self.model_type = ModelType.SEQ2SEQ_LM
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir="huggingface_cache"
        )
        
        # 量化配置
        quantization_config = None
        if use_4bit:
            print(f"正在使用4位量化加载模型 {model_name}...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif use_8bit:
            print(f"正在使用8位量化加载模型 {model_name}...")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        
        # 加载基础模型
        print(f"正在加载基础模型 {model_name}...")
        if use_lora:
            self.base_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                cache_dir="huggingface_cache",
                device_map=device_map,
                quantization_config=quantization_config,
                torch_dtype=torch.float16
            )
            
            # 配置LoRA
            print("正在应用LoRA适配器...")
            target_modules = self._get_target_modules_for_model()
            
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=target_modules
            )
            
            # 将LoRA应用到模型
            self.model = get_peft_model(self.base_model, lora_config)
            print(f"LoRA适配器已加载! 训练参数数量: {self.model.print_trainable_parameters()}")
        else:
            # 全参数模型
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                cache_dir="huggingface_cache",
                device_map=device_map,
                quantization_config=quantization_config,
                torch_dtype=torch.float16
            )
    
    def _get_target_modules_for_model(self) -> List[str]:
        """根据模型名称获取适合的LoRA目标模块"""
        model_name_lower = self.model_name.lower()
        
        if "t5" in model_name_lower:
            return ["q", "v", "k", "o", "wi", "wo"]
        elif "bart" in model_name_lower:
            return ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
        else:
            # 默认目标模块
            return ["query", "value", "key", "dense"]
    
    def forward(self, queries: List[str]) -> List[str]:
        # 添加任务前缀
        prefixed_queries = ["enhance: " + query for query in queries]
        
        enhanced_queries = []
        for query in prefixed_queries:
            # 使用no_grad以降低内存使用
            with torch.no_grad():
                # 处理单个查询
                inputs = self.tokenizer(query, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                # 生成增强查询
                outputs = self.model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=4,
                    temperature=0.7,
                    do_sample=True
                )
                
                enhanced_query = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                enhanced_queries.append(enhanced_query)
                
                # 释放内存
                del outputs, inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
        return enhanced_queries
        
    def forward_with_loss(self, queries: List[str]):
        # 添加任务前缀
        prefixed_queries = ["enhance: " + query for query in queries]
        
        batch_loss = None
        enhanced_queries = []
        
        for query in prefixed_queries:
            inputs = self.tokenizer(query, return_tensors="pt", truncation=True, max_length=512)
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
                    max_length=128,
                    num_beams=4,
                    temperature=0.7,
                    do_sample=True
                )
                
                enhanced_query = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                enhanced_queries.append(enhanced_query)
            
            # 释放不需要的张量
            del generated
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 平均批次损失
        batch_loss = batch_loss / len(queries)
        return batch_loss, enhanced_queries
    
    def load_lora_weights(self, path):
        """加载LoRA权重"""
        if self.use_lora:
            self.model = PeftModel.from_pretrained(self.base_model, path)
        else:
            raise ValueError("非LoRA模型无法加载LoRA权重")

class RLTrainer:
    def __init__(self, 
                 query_enhancer: Union[CausalLMQueryEnhancer, Seq2SeqLMQueryEnhancer],
                 deepseek_api: DeepseekAPI,
                 reward_calculator: RewardCalculator,
                 learning_rate: float = 1e-4,
                 checkpoint_dir: str = "checkpoints",
                 gradient_accumulation_steps: int = 1):
        self.query_enhancer = query_enhancer
        self.deepseek_api = deepseek_api
        self.reward_calculator = reward_calculator
        
        # 设置优化器 - 只训练需要训练的参数
        if hasattr(self.query_enhancer, 'use_lora') and self.query_enhancer.use_lora:
            if hasattr(self.query_enhancer, 'model') and hasattr(self.query_enhancer.model, 'parameters'):
                trainable_params = [p for p in self.query_enhancer.model.parameters() if p.requires_grad]
                self.optimizer = optim.Adam(trainable_params, lr=learning_rate)
                self.is_lora = True
        else:
            self.optimizer = optim.Adam(query_enhancer.parameters(), lr=learning_rate)
            self.is_lora = False
            
        self.checkpoint_dir = checkpoint_dir
        # 确保检查点目录存在
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_reward = -float('inf')
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
    def train_step(self, 
               original_query: str,
               ground_truth: str,
               step: int) -> Tuple[float, str, str]:
        # 设置为评估模式
        self.query_enhancer.eval()
        
        with torch.no_grad():
            # 1. 生成增强查询
            _, enhanced_queries = self.query_enhancer.forward_with_loss([original_query])
            enhanced_query = enhanced_queries[0]

            # 2. 获取DeepSeek响应
            response = self.deepseek_api.get_response(enhanced_query)
            generated_code = None
            try:
                # 尝试提取<answer>标签之间的内容
                if "<answer>" in response and "</answer>" in response:
                    generated_code = response.split("<answer>")[1].split("</answer>")[0].strip()
                    # 移除代码块标记
                    if generated_code.startswith("```"):
                        # 找到第一个换行符以跳过语言说明行
                        first_newline = generated_code.find("\n")
                        if first_newline != -1:
                            # 找到最后的```并排除
                            last_marker = generated_code.rfind("```")
                            if last_marker != -1:
                                generated_code = generated_code[first_newline:last_marker].strip()
                            else:
                                generated_code = generated_code[first_newline:].strip()
                else:
                    generated_code = ""
                    print(f"警告: 响应中未找到<answer>标签: {response}")
            except Exception as e:
                print(f"解析响应时出错: {e}")
                generated_code = ""

            # 3. 计算奖励
            reward = self.reward_calculator.calculate(generated_code, ground_truth)
        
        # 切换到训练模式，计算梯度
        self.query_enhancer.train()
        
        # 4. 计算损失
        model_loss, _ = self.query_enhancer.forward_with_loss([original_query])
        
        # 根据梯度累积步骤缩放损失
        loss = -torch.mean(torch.tensor(reward, device=model_loss.device) * model_loss) / self.gradient_accumulation_steps
        
        # 5. 反向传播
        loss.backward()
        
        # 仅在适当的步骤更新权重
        if (step + 1) % self.gradient_accumulation_steps == 0:
            # 应用梯度裁剪以防止梯度爆炸
            if self.is_lora and hasattr(self.query_enhancer, 'model'):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.query_enhancer.model.parameters() if p.requires_grad], 
                    max_norm=1.0
                )
            else:
                torch.nn.utils.clip_grad_norm_(self.query_enhancer.parameters(), max_norm=1.0)
                
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # 显式垃圾回收以释放内存
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return reward, enhanced_query, generated_code
    
    def save_checkpoint(self, epoch, avg_reward, is_best=False):
        """保存检查点"""
        # 根据模型类型选择保存方式
        if self.is_lora:
            # 创建检查点目录
            checkpoint_dir = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # 保存LoRA权重
            self.query_enhancer.save_lora_weights(checkpoint_dir)
            
            # 保存额外信息
            meta_info = {
                'epoch': epoch,
                'reward': avg_reward,
                'model_name': self.query_enhancer.model_name,
                'model_type': self.query_enhancer.model_type,
                'optimizer_state': self.optimizer.state_dict()
            }
            torch.save(meta_info, os.path.join(checkpoint_dir, "meta_info.pt"))
            
            print(f"已保存LoRA检查点到: {checkpoint_dir}")
            
            # 保存最佳模型
            if is_best:
                best_model_dir = os.path.join(self.checkpoint_dir, 'best_model')
                os.makedirs(best_model_dir, exist_ok=True)
                
                self.query_enhancer.save_lora_weights(best_model_dir)
                # 复制meta信息
                torch.save(meta_info, os.path.join(best_model_dir, "meta_info.pt"))
                print(f"已保存最佳LoRA模型! Reward: {avg_reward:.4f}")
        else:
            # 全参数模型保存
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.query_enhancer.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'reward': avg_reward,
                'model_name': self.query_enhancer.model_name,
                'model_type': self.query_enhancer.model_type
            }
            
            checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save(checkpoint, checkpoint_path)
            print(f"已保存全参数检查点到: {checkpoint_path}")
            
            # 保存最佳模型
            if is_best:
                best_model_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
                torch.save(checkpoint, best_model_path)
                print(f"已保存最佳全参数模型! Reward: {avg_reward:.4f}")
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        if not os.path.exists(checkpoint_path):
            print(f"检查点不存在: {checkpoint_path}")
            return 0, -float('inf')  # 返回初始epoch和reward
            
        # 根据模型类型选择加载方式
        if self.is_lora and os.path.isdir(checkpoint_path):
            # 加载LoRA权重
            self.query_enhancer.load_lora_weights(checkpoint_path)
            
            # 加载meta信息
            meta_path = os.path.join(checkpoint_path, "meta_info.pt")
            if os.path.exists(meta_path):
                meta_info = torch.load(meta_path, map_location='cpu')
                self.optimizer.load_state_dict(meta_info['optimizer_state'])
                epoch = meta_info['epoch']
                reward = meta_info['reward']
                self.best_reward = reward
                print(f"已加载LoRA检查点: {checkpoint_path}, Epoch: {epoch}, Reward: {reward:.4f}")
                return epoch, reward
            else:
                print(f"无法找到meta信息: {meta_path}")
                return 0, -float('inf')
        else:
            # 加载全参数模型
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.query_enhancer.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            reward = checkpoint['reward']
            self.best_reward = reward
            print(f"已加载全参数检查点: {checkpoint_path}, Epoch: {epoch}, Reward: {reward:.4f}")
            return epoch, reward

def get_model(
    model_name: str, 
    model_type: str = None,
    use_lora: bool = True,
    use_4bit: bool = False,
    use_8bit: bool = True
) -> Union[CausalLMQueryEnhancer, Seq2SeqLMQueryEnhancer]:
    """根据模型名称和类型创建合适的模型"""
    
    # 如果没有指定模型类型，尝试根据模型名称推断
    if model_type is None:
        model_type = infer_model_type(model_name)
        
    if model_type == ModelType.CAUSAL_LM:
        return CausalLMQueryEnhancer(
            model_name=model_name,
            use_lora=use_lora,
            use_4bit=use_4bit,
            use_8bit=use_8bit
        )
    elif model_type == ModelType.SEQ2SEQ_LM:
        return Seq2SeqLMQueryEnhancer(
            model_name=model_name,
            use_lora=use_lora,
            use_4bit=use_4bit,
            use_8bit=use_8bit
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

def infer_model_type(model_name: str) -> str:
    """根据模型名称推断模型类型"""
    model_name_lower = model_name.lower()
    
    seq2seq_models = ["t5", "bart", "pegasus", "mt5", "mbart"]
    causal_models = ["llama", "qwen", "gpt", "mistral", "bloom", "chat", "falcon"]
    
    for model in seq2seq_models:
        if model in model_name_lower:
            return ModelType.SEQ2SEQ_LM
            
    for model in causal_models:
        if model in model_name_lower:
            return ModelType.CAUSAL_LM
    
    # 默认为因果语言模型
    print(f"警告: 无法确定模型类型 '{model_name}'，默认使用因果语言模型类型")
    return ModelType.CAUSAL_LM

def optimize_memory():
    """优化内存使用以在有限显存下运行"""
    # 设置PyTorch内存管理
    if torch.cuda.is_available():
        # 设置内存分配器以减少内存碎片
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # 设置CUDA分配器以减少内存碎片
        torch.cuda.set_per_process_memory_fraction(0.9)  # 保留10%显存给系统
        torch.cuda.empty_cache()
        
        # 启用CUDNN确定性模式以减少内存波动
        torch.backends.cudnn.deterministic = True
        
    # 设置较低的预缓存区
    os.environ['TRANSFORMERS_CACHE'] = 'huggingface_cache'
    os.environ['HF_HOME'] = 'huggingface_cache'
    
    # 启用梯度检查点以减少内存使用
    os.environ['PYTORCH_CHECKPOINT_WARNING'] = '1'

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练RQ3的各种基础模型")
    
    # 模型相关参数
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen-7B", 
                       help="要训练的模型名称，例如: 'Qwen/Qwen-7B', 'meta-llama/Llama-2-7b-hf', 't5-small'等")
    parser.add_argument("--model_type", type=str, choices=[ModelType.CAUSAL_LM, ModelType.SEQ2SEQ_LM], default=None,
                       help="模型类型: 'causal_lm'或'seq2seq_lm'")
    
    # 训练配置
    parser.add_argument("--use_lora", action="store_true", default=True,
                      help="使用LoRA进行高效微调")
    parser.add_argument("--use_4bit", action="store_true", default=False,
                      help="使用4位量化以减少内存使用")
    parser.add_argument("--use_8bit", action="store_true", default=True,
                      help="使用8位量化以减少内存使用")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                      help="学习率")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                      help="梯度累积步数")
    parser.add_argument("--num_epochs", type=int, default=5,
                      help="训练轮数")
    
    # 数据和检查点
    parser.add_argument("--data_path", type=str, default="dataset/train.jsonl",
                      help="训练数据路径")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                      help="检查点保存目录，默认为'checkpoints_model名称'")
    parser.add_argument("--resume", action="store_true", default=False,
                      help="从最新检查点恢复训练")
    
    args = parser.parse_args()
    
    # 处理一些默认值
    if args.checkpoint_dir is None:
        model_name_short = args.model_name.split("/")[-1]
        args.checkpoint_dir = f"checkpoints_{model_name_short}"
    
    return args

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 优化内存使用
    optimize_memory()
    
    print(f"=== RQ3实验: 多种基础模型的转移性评估 ===")
    print(f"当前模型: {args.model_name}")
    print(f"使用LoRA: {args.use_lora}")
    print(f"量化设置: 4bit={args.use_4bit}, 8bit={args.use_8bit}")
    
    # 创建检查点目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 加载模型
    model_start_time = time.time()
    query_enhancer = get_model(
        model_name=args.model_name,
        model_type=args.model_type,
        use_lora=args.use_lora,
        use_4bit=args.use_4bit,
        use_8bit=args.use_8bit
    )
    model_load_time = time.time() - model_start_time
    print(f"模型加载完成，耗时: {model_load_time:.2f}秒")
    
    # 初始化其他组件
    deepseek_api = DeepseekAPI()
    reward_calculator = RewardCalculator(method=REWARD_METHOD)
    
    # 初始化训练器
    trainer = RLTrainer(
        query_enhancer=query_enhancer,
        deepseek_api=deepseek_api,
        reward_calculator=reward_calculator,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    # 加载检查点（如果需要恢复）
    start_epoch = 0
    if args.resume:
        if args.use_lora:
            resume_checkpoint = os.path.join(args.checkpoint_dir, "latest_checkpoint")
        else:
            resume_checkpoint = os.path.join(args.checkpoint_dir, "latest_checkpoint.pt")
        
        if os.path.exists(resume_checkpoint):
            start_epoch, _ = trainer.load_checkpoint(resume_checkpoint)
            start_epoch += 1  # 从下一个epoch开始
            print(f"已从epoch {start_epoch}恢复训练")
    
    # 加载训练数据
    print("正在加载数据集...")
    training_data = []
    with open(args.data_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            try:
                item = json.loads(line)
                if 'query' in item and 'ground_truth' in item:
                    training_data.append(item)
            except json.JSONDecodeError:
                print(f"警告: 无法解析JSON行: {line}")
                continue

    print(f"加载了 {len(training_data)} 条有效训练样本")
    
    # 截取少量样本以便快速评估不同模型的性能
    # 可通过环境变量控制训练样本数量
    max_samples = int(os.getenv("MAX_SAMPLES", "0"))
    if max_samples > 0 and max_samples < len(training_data):
        print(f"截取前 {max_samples} 条样本用于快速评估")
        training_data = training_data[:max_samples]
    
    # 训练循环
    print(f"\n开始训练...")
    for epoch in range(start_epoch, args.num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.num_epochs} ---")
        
        epoch_start_time = time.time()
        total_reward = 0
        
        # 确保优化器梯度清零开始新的epoch
        trainer.optimizer.zero_grad()
        
        for idx, data in enumerate(training_data):
            try:
                step_start_time = time.time()
                reward, enhanced_query, generated_code = trainer.train_step(
                    data["query"],
                    data["ground_truth"],
                    idx
                )
                step_time = time.time() - step_start_time
                
                print(f"样本 {idx+1}/{len(training_data)}, 奖励: {reward:.4f}, 步骤时间: {step_time:.2f}秒")
                print(f"原始查询: {data['query']}")
                print(f"增强查询: {enhanced_query}")
                if len(generated_code) > 100:
                    print(f"生成代码: {generated_code[:100]}...")
                else:
                    print(f"生成代码: {generated_code}")
                print("-" * 50)
                
                total_reward += reward
                
                # 每处理一定数量样本后手动执行垃圾回收
                if (idx + 1) % 5 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"警告: CUDA内存不足。释放缓存并跳过当前样本: {str(e)}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    print(f"训练过程中发生错误: {str(e)}")
                    raise e
            except Exception as e:
                print(f"处理样本时出错: {str(e)}")
                continue
            
        avg_reward = total_reward / len(training_data)
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1}, 平均奖励: {avg_reward:.4f}, 耗时: {epoch_time:.2f}秒")
        
        # 保存检查点
        trainer.save_checkpoint(epoch + 1, avg_reward)
        
        # 保存最新检查点的副本作为恢复点
        if args.use_lora:
            latest_checkpoint = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}")
            if os.path.exists(latest_checkpoint):
                os.system(f"cp -r {latest_checkpoint} {os.path.join(args.checkpoint_dir, 'latest_checkpoint')}")
        else:
            latest_checkpoint = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pt')
            if os.path.exists(latest_checkpoint):
                checkpoint_data = torch.load(latest_checkpoint, map_location='cpu')
                torch.save(checkpoint_data, os.path.join(args.checkpoint_dir, "latest_checkpoint.pt"))
        
        # 保存最佳模型
        if avg_reward > trainer.best_reward:
            trainer.best_reward = avg_reward
            trainer.save_checkpoint(epoch + 1, avg_reward, is_best=True)
            print(f"新的最佳模型! 奖励: {avg_reward:.4f}")
        
        # 每个epoch结束后执行全面内存清理
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    # 打印最终结果
    print(f"\n训练完成! 模型: {args.model_name}")
    print(f"最佳奖励: {trainer.best_reward:.4f}")
    print(f"检查点保存在: {args.checkpoint_dir}")


'''
# 使用Llama-2-7B模型与LoRA
python train_rq3.py --model_name "meta-llama/Llama-2-7b-hf" --use_lora --use_8bit

# 使用T5模型
python train_rq3.py --model_name "t5-base" --model_type seq2seq_lm --use_lora

# 恢复训练
python train_rq3.py --model_name "Qwen/Qwen-7B" --resume

# 使用4位量化以进一步减少内存使用
python train_rq3.py --model_name "meta-llama/Llama-2-7b-hf" --use_4bit --gradient_accumulation_steps 16

# 限制训练样本数量以快速评估多个模型
MAX_SAMPLES=100 python train_rq3.py --model_name "mistralai/Mistral-7B-v0.1" --num_epochs 2

'''
if __name__ == "__main__":
    main()