import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple
import numpy as np
import json
import os
import gc


from models.t5small_huggingface import T5SmallQueryEnhancer
from models.qwen import QwenQueryEnhancer
from llm.deepseek import DeepseekAPI
from utils.reward_util import RewardCalculator

class RLTrainer:
    def __init__(self, 
                 query_enhancer: T5SmallQueryEnhancer,
                 deepseek_api: DeepseekAPI,
                 reward_calculator: RewardCalculator,
                 learning_rate: float = 1e-4,
                 checkpoint_dir: str = "checkpoints",
                 gradient_accumulation_steps: int = 1):
        self.query_enhancer = query_enhancer
        self.deepseek_api = deepseek_api
        self.reward_calculator = reward_calculator
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
        model_type = type(self.query_enhancer).__name__
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.query_enhancer.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'reward': avg_reward,
            'model_type': model_type
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"已保存检查点到: {checkpoint_path}")
        
        # 保存最佳模型
        if is_best:
            best_model_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_model_path)
            print(f"已保存最佳模型到: {best_model_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        if not os.path.exists(checkpoint_path):
            print(f"检查点不存在: {checkpoint_path}")
            return 0, -float('inf')  # 返回初始epoch和reward
        
        # 尝试使用CPU加载，以节省GPU内存
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.query_enhancer.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        reward = checkpoint['reward']
        self.best_reward = reward
        print(f"已加载检查点: {checkpoint_path}, Epoch: {epoch}, Reward: {reward:.4f}")
        return epoch, reward

def main():
    # 设置PyTorch内存管理
    if torch.cuda.is_available():
        # 设置内存分配器以减少内存碎片
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
    # 创建检查点目录
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 初始化组件
    query_enhancer = QwenQueryEnhancer()
    
    # 使用低精度减少内存使用
    if hasattr(query_enhancer.model, "half") and torch.cuda.is_available():
        query_enhancer.model = query_enhancer.model.half()  # 使用FP16精度
    
    deepseek_api = DeepseekAPI()
    reward_calculator = RewardCalculator()
    
    # 配置梯度累积步数
    gradient_accumulation_steps = 4  # 增大以减少内存使用
    
    trainer = RLTrainer(
        query_enhancer=query_enhancer,
        deepseek_api=deepseek_api,
        reward_calculator=reward_calculator,
        checkpoint_dir=checkpoint_dir,
        gradient_accumulation_steps=gradient_accumulation_steps
    )
    
    # 加载检查点（如果存在）
    start_epoch = 0
    resume_checkpoint = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
    
    if os.path.exists(resume_checkpoint):
        start_epoch, _ = trainer.load_checkpoint(resume_checkpoint)
        start_epoch += 1  # 从下一个epoch开始
    
    with open("dataset/train.json", "r") as f:
        training_data = json.load(f)
    
    # 训练循环
    num_epochs = 10
    save_frequency = 2  # 每隔多少个epoch保存一次检查点
    
    for epoch in range(start_epoch, num_epochs):
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
                print(f"Epoch {epoch + 1}, Sample {idx+1}/{len(training_data)}, Reward: {reward:.4f}")
                print(f"Original Query: {data['query']}")
                print(f"Enhanced Query: {enhanced_query}")
                print(f"Generated Code: {generated_code[:100]}..." if len(generated_code) > 100 else f"Generated Code: {generated_code}")
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
        print(f"Epoch {epoch + 1}, Average Reward: {avg_reward:.4f}")
        
        # 保存最新检查点
        trainer.save_checkpoint(epoch + 1, avg_reward)
        
        # 保存最新检查点的副本作为resume点
        latest_checkpoint = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pt')
        if os.path.exists(latest_checkpoint):
            # 使用CPU加载以减少GPU内存使用
            checkpoint_data = torch.load(latest_checkpoint, map_location='cpu')
            torch.save(checkpoint_data, resume_checkpoint)
        
        # 如果是最佳模型，则标记保存
        if avg_reward > trainer.best_reward:
            trainer.best_reward = avg_reward
            trainer.save_checkpoint(epoch + 1, avg_reward, is_best=True)
            print(f"新的最佳模型! Reward: {avg_reward:.4f}")

if __name__ == "__main__":
    main()