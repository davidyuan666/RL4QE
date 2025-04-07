# train_rq2.py
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple, Union
import numpy as np
import json
import os
import gc
from dotenv import load_dotenv

from models.qwen import QwenQueryEnhancer
from models.qwen_lora import QwenLoRAQueryEnhancer
from llm.deepseek import DeepseekAPI
from utils.reward_util import RewardCalculator

# 加载环境变量
load_dotenv()

# 获取训练模式环境变量，默认为"lora"
TRAINING_MODE = os.getenv("TRAINING_MODE", "lora").lower()

class RLTrainer:
    def __init__(self, 
                 query_enhancer: Union[QwenQueryEnhancer, QwenLoRAQueryEnhancer],
                 deepseek_api: DeepseekAPI,
                 reward_calculator: RewardCalculator,
                 learning_rate: float = 1e-4,
                 checkpoint_dir: str = "checkpoints",
                 gradient_accumulation_steps: int = 1,
                 is_lora: bool = True):
        self.query_enhancer = query_enhancer
        self.deepseek_api = deepseek_api
        self.reward_calculator = reward_calculator
        self.is_lora = is_lora
        
        # 根据模型类型设置优化器
        if self.is_lora:
            # 只训练LoRA参数
            if hasattr(self.query_enhancer, 'model') and hasattr(self.query_enhancer.model, 'parameters'):
                trainable_params = [p for p in self.query_enhancer.model.parameters() if p.requires_grad]
                self.optimizer = optim.Adam(trainable_params, lr=learning_rate)
        else:
            # 全参数训练
            self.optimizer = optim.Adam(query_enhancer.parameters(), lr=learning_rate)
            
        self.checkpoint_dir = checkpoint_dir
        # 确保检查点目录存在
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_reward = -float('inf')
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
    def train_step(self, 
               original_query: str,
               ground_truth: str,
               step: int) -> Tuple[float, str, str]:
        # 设置为评估模式（不跟踪梯度），以减少内存使用
        self.query_enhancer.eval()
        
        with torch.no_grad():
            # 1. Generate enhanced query without computing gradients
            _, enhanced_queries = self.query_enhancer.forward_with_loss([original_query])
            enhanced_query = enhanced_queries[0]  # Get the first query since we only input one

            # 2. Get Deepseek response
            response = self.deepseek_api.get_response(enhanced_query)
            generated_code = None
            try:
                # First try to extract content between <answer> tags
                if "<answer>" in response and "</answer>" in response:
                    generated_code = response.split("<answer>")[1].split("</answer>")[0].strip()
                    # Remove code block markers if present
                    if generated_code.startswith("```"):
                        # Find the first newline to skip the language specifier line
                        first_newline = generated_code.find("\n")
                        if first_newline != -1:
                            # Find the last ``` and exclude it
                            last_marker = generated_code.rfind("```")
                            if last_marker != -1:
                                generated_code = generated_code[first_newline:last_marker].strip()
                            else:
                                generated_code = generated_code[first_newline:].strip()
                else:
                    # If no answer tags found, return empty string or handle as needed
                    generated_code = ""
                    print(f"Warning: Could not find <answer> tags in response: {response}")
            except Exception as e:
                print(f"Error parsing response: {e}")
                generated_code = ""

            # 3. Calculate reward
            reward = self.reward_calculator.calculate(generated_code, ground_truth)
        
        # 切换到训练模式，计算梯度
        self.query_enhancer.train()
        
        # 4. Compute loss with minimal memory usage
        # Only compute gradients for the forward pass
        model_loss, _ = self.query_enhancer.forward_with_loss([original_query])
        
        # Scale loss by gradient accumulation steps
        loss = -torch.mean(torch.tensor(reward, device=model_loss.device) * model_loss) / self.gradient_accumulation_steps
        
        # 5. Backward pass and optimizer step only at appropriate steps
        loss.backward()
        
        # Only update weights after accumulating gradients
        if (step + 1) % self.gradient_accumulation_steps == 0:
            # Apply gradient clipping to prevent exploding gradients
            if self.is_lora and hasattr(self.query_enhancer, 'model'):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.query_enhancer.model.parameters() if p.requires_grad], 
                    max_norm=1.0
                )
            else:
                torch.nn.utils.clip_grad_norm_(self.query_enhancer.parameters(), max_norm=1.0)
                
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Explicit garbage collection to free memory
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
                'model_type': 'QwenLoRAQueryEnhancer',
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
            model_type = 'QwenQueryEnhancer'
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.query_enhancer.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'reward': avg_reward,
                'model_type': model_type
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

def main():
    # 设置PyTorch内存管理
    if torch.cuda.is_available():
        # 设置内存分配器以减少内存碎片
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # 获取训练模式
    training_mode = TRAINING_MODE
    print(f"当前训练模式: {training_mode}")
    
    # 创建检查点目录
    checkpoint_dir = f"checkpoints_{training_mode}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 根据训练模式初始化不同的模型
    is_lora = training_mode == "lora"
    
    if is_lora:
        # 使用LoRA模式
        print("初始化Qwen LoRA模型...")
        query_enhancer = QwenLoRAQueryEnhancer()
    else:
        # 使用全参数训练模式
        print("初始化Qwen全参数模型...")
        query_enhancer = QwenQueryEnhancer()
        
        # 对于全参数训练，可以使用半精度以节省内存
        if hasattr(query_enhancer.model, "half") and torch.cuda.is_available():
            query_enhancer.model = query_enhancer.model.half()
    
    # 初始化其他组件
    deepseek_api = DeepseekAPI()
    reward_calculator = RewardCalculator()
    
    # 配置梯度累积步数 - 全参数训练可能需要更多累积步数
    gradient_accumulation_steps = 8 if not is_lora else 4
    
    # 初始化训练器
    trainer = RLTrainer(
        query_enhancer=query_enhancer,
        deepseek_api=deepseek_api,
        reward_calculator=reward_calculator,
        checkpoint_dir=checkpoint_dir,
        gradient_accumulation_steps=gradient_accumulation_steps,
        is_lora=is_lora
    )
    
    # 加载检查点（如果存在）
    start_epoch = 0
    if is_lora:
        resume_checkpoint = os.path.join(checkpoint_dir, "latest_checkpoint")
    else:
        resume_checkpoint = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
    
    if os.path.exists(resume_checkpoint):
        start_epoch, _ = trainer.load_checkpoint(resume_checkpoint)
        start_epoch += 1  # 从下一个epoch开始
    
    # 加载训练数据
    print("正在加载数据集...")
    with open("dataset/train.jsonl", "r") as f:
        lines = f.readlines()
        raw_training_data = [json.loads(line) for line in lines]

    print(f"加载了 {len(raw_training_data)} 条训练样本")
    
    # 处理数据集
    training_data = []
    for idx, item in enumerate(raw_training_data):
        if 'query' in item and 'ground_truth' in item:
            training_data.append(item)
    
    print(f"有效训练样本: {len(training_data)} 条")
    
    # 训练循环
    num_epochs = 10
    batch_size = 1  # 单次处理一个样本，使用梯度累积
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        print(f"训练模式: {'LoRA' if is_lora else '全参数'}")
        
        total_reward = 0
        
        # 确保优化器梯度清零开始新的epoch
        trainer.optimizer.zero_grad()
        
        for idx, data in enumerate(training_data):
            try:
                reward, enhanced_query, generated_code = trainer.train_step(
                    data["query"],
                    data["ground_truth"],
                    idx
                )
                print(f"样本 {idx+1}/{len(training_data)}, 奖励: {reward:.4f}")
                print(f"原始查询: {data['query']}")
                print(f"增强查询: {enhanced_query}")
                print(f"生成代码: {generated_code[:100]}..." if len(generated_code) > 100 else f"生成代码: {generated_code}")
                print("-" * 50)
                
                total_reward += reward
                
                # 每处理一定数量样本后手动执行垃圾回收
                if (idx + 1) % 5 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("警告: CUDA内存不足。释放缓存并跳过当前样本。")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            
        avg_reward = total_reward / len(training_data)
        print(f"Epoch {epoch + 1}, 平均奖励: {avg_reward:.4f}")
        
        # 保存检查点
        trainer.save_checkpoint(epoch + 1, avg_reward)
        
        # 保存最新检查点的副本作为恢复点
        if is_lora:
            latest_checkpoint = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}")
            if os.path.exists(latest_checkpoint):
                os.system(f"cp -r {latest_checkpoint} {os.path.join(checkpoint_dir, 'latest_checkpoint')}")
        else:
            latest_checkpoint = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pt')
            if os.path.exists(latest_checkpoint):
                checkpoint_data = torch.load(latest_checkpoint, map_location='cpu')
                torch.save(checkpoint_data, os.path.join(checkpoint_dir, "latest_checkpoint.pt"))
        
        # 保存最佳模型
        if avg_reward > trainer.best_reward:
            trainer.best_reward = avg_reward
            trainer.save_checkpoint(epoch + 1, avg_reward, is_best=True)
            print(f"新的最佳模型! 奖励: {avg_reward:.4f}")
            
    # 打印最终结果
    print(f"\n训练完成! 训练模式: {'LoRA' if is_lora else '全参数'}")
    print(f"最佳奖励: {trainer.best_reward:.4f}")

if __name__ == "__main__":
    main()