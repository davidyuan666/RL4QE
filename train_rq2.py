# train_rq2.py
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple, Union
import numpy as np
import json
import os
import gc
import time
from datetime import datetime
from dotenv import load_dotenv
import psutil  # Add this import
from models.qwen_lora import QwenLoRAQueryEnhancer
from models.qwen_full import QwenFullQueryEnhancer
from llm.deepseek import DeepseekAPI
from utils.reward_util import RewardCalculator
from torch.cuda.amp import GradScaler, autocast  # Add this import
# 加载环境变量
load_dotenv()

# 获取训练模式环境变量，默认为"lora"
TRAINING_MODE = os.getenv("TRAINING_MODE", "lora").lower()

class RLTrainer:
    def __init__(self, 
                 query_enhancer: Union[QwenFullQueryEnhancer, QwenLoRAQueryEnhancer],
                 deepseek_api: DeepseekAPI,
                 reward_calculator: RewardCalculator,
                 learning_rate: float = 1e-4,
                 checkpoint_dir: str = "rq2/checkpoints",
                 log_dir: str = "rq2/logs",
                 gradient_accumulation_steps: int = 1,
                 is_lora: bool = True,
                 use_amp: bool = True):
        self.query_enhancer = query_enhancer
        self.deepseek_api = deepseek_api
        self.reward_calculator = reward_calculator
        self.is_lora = is_lora
        self.use_amp = use_amp and torch.cuda.is_available() and not is_lora
        
        # 初始化梯度缩放器
        self.scaler = GradScaler() if self.use_amp else None
        
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

        # 添加日志相关的初始化
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 创建训练记录文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.train_log_file = os.path.join(self.log_dir, f"train_log_{timestamp}.jsonl")
        self.val_log_file = os.path.join(self.log_dir, f"val_log_{timestamp}.jsonl")
        self.metrics_file = os.path.join(self.log_dir, f"metrics_{timestamp}.json")
        
        # 初始化性能跟踪记录
        self.metrics = {
            "train_rewards": [],
            "val_rewards": [],
            "best_reward": -float('inf'),
            "best_epoch": 0,
            "training_time": 0,
            "start_time": time.time(),
            "memory_usage": [],
            "gpu_memory_usage": [],
            "epoch_times": [],
            "steps_completed": 0,
            "total_samples_processed": 0,
            "oom_events": 0
        }

        self.metrics.update({
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
            "val_css": [],
            "test_precision": 0.0,
            "test_recall": 0.0,
            "test_f1": 0.0,
            "test_css": 0.0
        })

    def get_memory_stats(self):
        """获取内存使用统计"""
        stats = {
            "ram_used": psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024,  # MB
            "ram_percent": psutil.virtual_memory().percent
        }
        
        if torch.cuda.is_available():
            stats.update({
                "gpu_used": torch.cuda.memory_allocated() / 1024 / 1024,  # MB
                "gpu_cached": torch.cuda.memory_reserved() / 1024 / 1024  # MB
            })
        
        return stats

    def train_step(self, 
                  original_query: str,
                  ground_truth: str,
                  step: int) -> Tuple[float, str, str]:
        try:
            # 记录内存使用
            memory_stats = self.get_memory_stats()
            self.metrics["memory_usage"].append(memory_stats)
            
            # 设置为评估模式
            self.query_enhancer.eval()
            
            with torch.no_grad():
                # 1. Generate enhanced query
                _, enhanced_queries = self.query_enhancer.forward_with_loss([original_query])
                enhanced_query = enhanced_queries[0]

                # 2. Get Deepseek response
                response = self.deepseek_api.get_response(enhanced_query)
                generated_code = self._parse_response(response)

                # 3. Calculate reward
                reward = self.reward_calculator.calculate(generated_code, ground_truth)
            
            # 切换到训练模式
            self.query_enhancer.train()
            
            # 4. Compute loss with mixed precision
            with autocast(enabled=self.use_amp):
                model_loss, _ = self.query_enhancer.forward_with_loss([original_query])
                loss = -torch.mean(torch.tensor(reward, device=model_loss.device) * model_loss) / self.gradient_accumulation_steps

            # 5. Backward pass with gradient scaling
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 6. Optimizer step if needed
            if (step + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                # Gradient clipping
                max_norm = 1.0
                if self.is_lora:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.query_enhancer.model.parameters() if p.requires_grad],
                        max_norm
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(self.query_enhancer.parameters(), max_norm)
                
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Memory cleanup
                self._cleanup_memory()

            # 更新训练指标
            self.metrics["train_rewards"].append(float(reward))
            self.metrics["steps_completed"] += 1
            self.metrics["total_samples_processed"] += 1
            
            # 记录训练数据
            self._log_training_step(original_query, enhanced_query, ground_truth, 
                                  generated_code, reward, step, loss)
            
            return reward, enhanced_query, generated_code
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                self.metrics["oom_events"] += 1
                self._cleanup_memory()
                print(f"内存不足错误 (OOM #{self.metrics['oom_events']})")
                raise e
            raise e

    def _parse_response(self, response: str) -> str:
        """解析API响应"""
        try:
            if "<answer>" in response and "</answer>" in response:
                code = response.split("<answer>")[1].split("</answer>")[0].strip()
                if code.startswith("```"):
                    first_newline = code.find("\n")
                    if first_newline != -1:
                        last_marker = code.rfind("```")
                        if last_marker != -1:
                            code = code[first_newline:last_marker].strip()
                        else:
                            code = code[first_newline:].strip()
                return code
            print(f"Warning: Could not find <answer> tags in response: {response}")
            return ""
        except Exception as e:
            print(f"Error parsing response: {e}")
            return ""

    def _cleanup_memory(self):
        """清理内存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _log_training_step(self, original_query, enhanced_query, ground_truth, 
                          generated_code, reward, step, loss):
        """记录训练步骤数据"""
        log_entry = {
            "original_query": original_query,
            "enhanced_query": enhanced_query,
            "ground_truth": ground_truth,
            "generated_code": generated_code,
            "reward": float(reward),
            "step": step,
            "loss": float(loss.item()),
            "memory_stats": self.get_memory_stats(),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(self.train_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def save_metrics(self):
        """保存训练指标数据"""
        self.metrics["training_time"] = time.time() - self.metrics["start_time"]
        
        # 计算平均值和统计信息
        if self.metrics["train_rewards"]:
            self.metrics.update({
                "avg_train_reward": sum(self.metrics["train_rewards"]) / len(self.metrics["train_rewards"]),
                "max_train_reward": max(self.metrics["train_rewards"]),
                "min_train_reward": min(self.metrics["train_rewards"]),
                "final_memory_stats": self.get_memory_stats()
            })
        
        with open(self.metrics_file, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=2)
        print(f"已保存训练指标到: {self.metrics_file}")
    
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
                'optimizer_state': self.optimizer.state_dict(),
                'metrics': self.metrics  # 添加训练指标
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
                'model_type': model_type,
                'metrics': self.metrics  # 添加训练指标
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
                # 恢复训练指标
                if 'metrics' in meta_info:
                    self.metrics.update(meta_info['metrics'])
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
            # 恢复训练指标
            if 'metrics' in checkpoint:
                self.metrics.update(checkpoint['metrics'])
            print(f"已加载全参数检查点: {checkpoint_path}, Epoch: {epoch}, Reward: {reward:.4f}")
            return epoch, reward

def main():
    # 设置PyTorch内存管理
    if torch.cuda.is_available():
        # 更激进的内存优化设置
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # 获取训练模式
    training_mode = TRAINING_MODE
    print(f"当前训练模式: {training_mode}")
    
    # 创建检查点和日志目录
    checkpoint_dir = f"rq2/checkpoints_{training_mode}"
    log_dir = f"rq2/logs_{training_mode}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 根据训练模式初始化不同的模型
    is_lora = training_mode == "lora"
    
    if is_lora:
        print("初始化Qwen LoRA模型...")
        query_enhancer = QwenLoRAQueryEnhancer()
    else:
        print("初始化Qwen全参数模型...")
        query_enhancer = QwenFullQueryEnhancer()
        
        if torch.cuda.is_available():
            print("应用模型量化...")
            try:
                import bitsandbytes as bnb
                # 使用更安全的量化方式
                if hasattr(query_enhancer, 'model'):
                    # 直接使用 bnb 的 convert_module_to_8bit 函数
                    query_enhancer.model = bnb.nn.convert_module_to_8bit(
                        query_enhancer.model,
                        has_fp16_weights=False,
                        threshold=6.0
                    )
                    print("已应用8位量化")
            except ImportError:
                print("未找到bitsandbytes库，使用FP16量化")
                if hasattr(query_enhancer, 'model'):
                    query_enhancer.model = query_enhancer.model.half()
            except Exception as e:
                print(f"8位量化失败，回退到FP16量化: {str(e)}")
                if hasattr(query_enhancer, 'model'):
                    query_enhancer.model = query_enhancer.model.half()
    
    # 初始化其他组件
    deepseek_api = DeepseekAPI()
    reward_calculator = RewardCalculator(method="bleu")
    
    # 增加梯度累积步数以减少内存压力
    gradient_accumulation_steps = 16 if not is_lora else 8
    
    # 初始化训练器
    trainer = RLTrainer(
        query_enhancer=query_enhancer,
        deepseek_api=deepseek_api,
        reward_calculator=reward_calculator,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        gradient_accumulation_steps=gradient_accumulation_steps,
        is_lora=is_lora,
        use_amp=not is_lora  # 只在全参数模式下使用AMP
    )
    
    # 加载检查点（如果存在）
    start_epoch = 0
    if is_lora:
        resume_checkpoint = os.path.join(checkpoint_dir, "latest_checkpoint")
    else:
        resume_checkpoint = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
    
    if os.path.exists(resume_checkpoint):
        start_epoch, _ = trainer.load_checkpoint(resume_checkpoint)
        start_epoch += 1
    
    # 加载训练数据
    print("正在加载数据集...")
    training_data = load_and_process_data("dataset/train.jsonl", sample_ratio=0.1)  # 加载50%的样本
    print(f"有效训练样本: {len(training_data)} 条")
    
    # 训练循环
    num_epochs = 1
    try:
        for epoch in range(start_epoch, num_epochs):
            run_training_epoch(trainer, training_data, epoch, num_epochs)
                
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练发生错误: {str(e)}")
        raise e
    finally:
        # 保存最终的训练指标
        trainer.save_metrics()
        
    # 打印最终结果
    print_training_summary(trainer)



def load_and_process_data(data_path: str, sample_ratio: float = 0.5) -> List[Dict]:
    """加载和处理训练数据
    
    Args:
        data_path: 数据文件路径
        sample_ratio: 采样比例,范围0-1,默认0.5表示加载一半样本
    """
    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        raw_training_data = [json.loads(line) for line in lines]

    print(f"原始训练样本数: {len(raw_training_data)}")
    
    # 随机打乱数据
    import random
    random.shuffle(raw_training_data)
    
    # 根据采样比例选择样本
    sample_size = int(len(raw_training_data) * sample_ratio)
    raw_training_data = raw_training_data[:sample_size]
    
    training_data = []
    for item in raw_training_data:
        if 'prompt' in item and 'reference_code' in item:
            training_data.append(item)
    
    print(f"采样后的有效训练样本: {len(training_data)} 条 (采样比例: {sample_ratio:.1%})")
    
    return training_data



def run_training_epoch(trainer: RLTrainer, 
                      training_data: List[Dict], 
                      epoch: int, 
                      num_epochs: int):
    """运行一个训练epoch"""
    epoch_start_time = time.time()
    print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
    print(f"训练模式: {'LoRA' if trainer.is_lora else '全参数'}")
    
    total_reward = 0
    trainer.optimizer.zero_grad()
    
    # 对数据集进行分批处理以减少内存使用
    batch_size = 1  # 减小批处理大小
    for idx in range(0, len(training_data), batch_size):
        batch_data = training_data[idx:idx + batch_size]
        
        try:
            for batch_idx, data in enumerate(batch_data):
                reward, enhanced_query, generated_code = trainer.train_step(
                    data["prompt"],
                    data["reference_code"],
                    idx + batch_idx
                )
                print(f"样本 {idx+batch_idx+1}/{len(training_data)}, 奖励: {reward:.4f}")
                total_reward += reward
                
            # 定期进行内存清理
            if (idx + 1) % (batch_size * 2) == 0:
                trainer._cleanup_memory()
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"警告: CUDA内存不足。尝试减小批处理大小并释放缓存。")
                trainer._cleanup_memory()
                
                # 如果发生OOM，尝试逐个处理样本
                for single_data in batch_data:
                    try:
                        reward, _, _ = trainer.train_step(
                            single_data["prompt"],
                            single_data["reference_code"],
                            idx
                        )
                        total_reward += reward
                    except RuntimeError:
                        print(f"警告: 跳过当前样本")
                        continue
            else:
                raise e
    
    # 计算和记录epoch统计信息
    avg_reward = total_reward / len(training_data)
    epoch_time = time.time() - epoch_start_time
    
    # 更新训练指标
    trainer.metrics["epoch_times"].append(epoch_time)
    trainer.metrics["epochs_completed"] = epoch + 1
    
    print(f"Epoch {epoch + 1} 完成")
    print(f"平均奖励: {avg_reward:.4f}")
    print(f"耗时: {epoch_time:.2f}秒")
    
    # 保存检查点和指标
    trainer.save_checkpoint(epoch + 1, avg_reward)
    if avg_reward > trainer.best_reward:
        trainer.best_reward = avg_reward
        trainer.metrics["best_reward"] = avg_reward
        trainer.metrics["best_epoch"] = epoch + 1
        trainer.save_checkpoint(epoch + 1, avg_reward, is_best=True)
        print(f"新的最佳模型! 奖励: {avg_reward:.4f}")
    
    trainer.save_metrics()

def print_training_summary(trainer: RLTrainer):
    """打印训练总结"""
    print("\n=== 训练总结 ===")
    print(f"训练模式: {'LoRA' if trainer.is_lora else '全参数'}")
    print(f"最佳奖励: {trainer.best_reward:.4f}")
    print(f"最佳epoch: {trainer.metrics['best_epoch']}")
    print(f"总训练时间: {trainer.metrics['training_time']:.2f}秒")
    print(f"处理的样本总数: {trainer.metrics['total_samples_processed']}")
    print(f"OOM事件数: {trainer.metrics['oom_events']}")
    
    if trainer.metrics.get("memory_usage"):
        last_memory = trainer.metrics["memory_usage"][-1]
        print("\n内存使用情况:")
        print(f"RAM使用: {last_memory['ram_used']:.2f}MB ({last_memory['ram_percent']}%)")
        if 'gpu_used' in last_memory:
            print(f"GPU内存使用: {last_memory['gpu_used']:.2f}MB")
            print(f"GPU缓存: {last_memory['gpu_cached']:.2f}MB")



if __name__ == "__main__":
    main()