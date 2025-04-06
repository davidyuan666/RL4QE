
from models.qwen_lora import QwenLoRAQueryEnhancer
from llm.deepseek import DeepseekAPI
from utils.reward_util import RewardCalculator
from torch import optim
import torch
import os
import gc
import json
from typing import Tuple


class RLTrainer:
    def __init__(self, 
                 query_enhancer,
                 deepseek_api: DeepseekAPI,
                 reward_calculator: RewardCalculator,
                 learning_rate: float = 1e-4,
                 checkpoint_dir: str = "checkpoints",
                 gradient_accumulation_steps: int = 1):
        self.query_enhancer = query_enhancer
        self.deepseek_api = deepseek_api
        self.reward_calculator = reward_calculator
        
        # 只训练LoRA参数
        if hasattr(self.query_enhancer, 'model') and hasattr(self.query_enhancer.model, 'parameters'):
            trainable_params = [p for p in self.query_enhancer.model.parameters() if p.requires_grad]
            self.optimizer = optim.Adam(trainable_params, lr=learning_rate)
        else:
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
            if hasattr(self.query_enhancer, 'model'):
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
        # 创建检查点目录
        checkpoint_dir = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 对于使用LoRA的模型，使用特殊的保存方法
        if hasattr(self.query_enhancer, 'save_lora_weights'):
            self.query_enhancer.save_lora_weights(checkpoint_dir)
            
            # 保存额外信息
            meta_info = {
                'epoch': epoch,
                'reward': avg_reward,
                'model_type': type(self.query_enhancer).__name__,
                'optimizer_state': self.optimizer.state_dict()
            }
            torch.save(meta_info, os.path.join(checkpoint_dir, "meta_info.pt"))
            
            print(f"已保存LoRA检查点到: {checkpoint_dir}")
        else:
            # 原始保存方法
            model_type = type(self.query_enhancer).__name__
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.query_enhancer.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'reward': avg_reward,
                'model_type': model_type
            }
            
            checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save(checkpoint, checkpoint_path)
            print(f"已保存检查点到: {checkpoint_path}")
        
        # 保存最佳模型
        if is_best:
            best_model_dir = os.path.join(self.checkpoint_dir, 'best_model')
            os.makedirs(best_model_dir, exist_ok=True)
            
            if hasattr(self.query_enhancer, 'save_lora_weights'):
                self.query_enhancer.save_lora_weights(best_model_dir)
                
                # 复制meta信息
                meta_info = {
                    'epoch': epoch,
                    'reward': avg_reward,
                    'model_type': type(self.query_enhancer).__name__,
                    'optimizer_state': self.optimizer.state_dict()
                }
                torch.save(meta_info, os.path.join(best_model_dir, "meta_info.pt"))
            else:
                best_model_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
                torch.save(checkpoint, best_model_path)
                
            print(f"已保存最佳模型! Reward: {avg_reward:.4f}")
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        if not os.path.exists(checkpoint_path):
            print(f"检查点不存在: {checkpoint_path}")
            return 0, -float('inf')
            
        # 对于LoRA模型，需要特别处理
        if hasattr(self.query_enhancer, 'load_lora_weights') and os.path.isdir(checkpoint_path):
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
    
    # 初始化组件 - 使用LoRA版本的Qwen
    print("初始化LoRA增强的Qwen模型...")
    query_enhancer = QwenLoRAQueryEnhancer()
    
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
    resume_checkpoint = os.path.join(checkpoint_dir, "latest_checkpoint")
    best_model_path = os.path.join(checkpoint_dir, "best_model")
    
    if os.path.exists(resume_checkpoint):
        start_epoch, _ = trainer.load_checkpoint(resume_checkpoint)
        start_epoch += 1  # 从下一个epoch开始
    
    # 1. 加载和预处理数据集
    print("正在加载DS1000数据集...")
    with open("dataset/train.jsonl", "r") as f:
        lines = f.readlines()
        raw_training_data = [json.loads(line) for line in lines]

    # 添加数据预处理和分析步骤
    training_data = []
    for idx, item in enumerate(raw_training_data):
        # 确保数据包含所有必要字段
        if "prompt" not in item or "reference_code" not in item:
            print(f"警告: 跳过样本 {idx+1}，缺少必要字段")
            continue

        training_data.append(item)
        
        
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
                    data["prompt"],
                    data["reference_code"],
                    idx
                )
                print(f"Epoch {epoch + 1}, Sample {idx+1}/{len(training_data)}, Reward: {reward:.4f}")
                print(f"Original Query: {data['prompt']}")
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
        epoch_checkpoint_dir = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}")
        trainer.save_checkpoint(epoch + 1, avg_reward)
        
        # 将最新检查点链接为latest_checkpoint
        latest_checkpoint = os.path.join(checkpoint_dir, "latest_checkpoint")
        if os.path.exists(latest_checkpoint) and os.path.isdir(latest_checkpoint):
            import shutil
            shutil.rmtree(latest_checkpoint)
        elif os.path.exists(latest_checkpoint):
            os.remove(latest_checkpoint)
            
        # 复制最新检查点作为恢复点
        if os.path.exists(epoch_checkpoint_dir) and os.path.isdir(epoch_checkpoint_dir):
            import shutil
            shutil.copytree(epoch_checkpoint_dir, latest_checkpoint)
        
        # 如果是最佳模型，则标记保存
        if avg_reward > trainer.best_reward:
            trainer.best_reward = avg_reward
            trainer.save_checkpoint(epoch + 1, avg_reward, is_best=True)
            print(f"新的最佳模型! Reward: {avg_reward:.4f}")

if __name__ == "__main__":
    main()