from models.qwen_lora import QwenLoRAQueryEnhancer
from llm.deepseek import DeepseekAPI
from utils.reward_util import RewardCalculator
from torch import optim
import torch
import os
import gc
import json
import time
from typing import Tuple, List, Dict
from datetime import datetime
from dotenv import load_dotenv
import argparse
from utils.code_metric import CodeMetricsCalculator

load_dotenv()

class RLTrainer:
    def __init__(self, 
                 query_enhancer,
                 deepseek_api: DeepseekAPI,
                 reward_calculator: RewardCalculator,
                 learning_rate: float = 1e-4,
                 checkpoint_dir: str = "checkpoints",
                 log_dir: str = "logs",
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
        self.log_dir = log_dir
        # 确保检查点和日志目录存在
        os.makedirs(self.checkpoint_dir, exist_ok=True)
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
            "start_time": time.time()
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
        
        # 记录训练数据
        log_entry = {
            "original_query": original_query,
            "enhanced_query": enhanced_query,
            "ground_truth": ground_truth,
            "generated_code": generated_code,
            "reward": reward,
            "step": step
        }
        
        with open(self.train_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            
        return reward, enhanced_query, generated_code
    
    def validate(self, validation_data) -> float:
        """对验证集进行评估"""
        print("开始验证...")
        self.query_enhancer.eval()  # 设置为评估模式
        total_reward = 0.0
        val_results = []

        # 添加指标累计器
        total_metrics = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "css": 0.0
        }
    
        
        with torch.no_grad():  # 不计算梯度
            for idx, data in enumerate(validation_data):
                try:
                    # 提取查询和参考代码
                    original_query = data["prompt"]
                    ground_truth = data["reference_code"]
                    
                    # 生成增强查询
                    _, enhanced_queries = self.query_enhancer.forward_with_loss([original_query])
                    enhanced_query = enhanced_queries[0]
                    
                    # 获取Deepseek响应
                    response = self.deepseek_api.get_response(enhanced_query)
                    
                    # 解析生成的代码
                    generated_code = ""
                    try:
                        if "<answer>" in response and "</answer>" in response:
                            generated_code = response.split("<answer>")[1].split("</answer>")[0].strip()
                            if generated_code.startswith("```"):
                                first_newline = generated_code.find("\n")
                                if first_newline != -1:
                                    last_marker = generated_code.rfind("```")
                                    if last_marker != -1:
                                        generated_code = generated_code[first_newline:last_marker].strip()
                                    else:
                                        generated_code = generated_code[first_newline:].strip()
                    except Exception as e:
                        print(f"验证时解析响应出错: {e}")
                    
                    # 计算奖励
                    reward = self.reward_calculator.calculate(generated_code, ground_truth)
                    total_reward += reward
                    
                    # 计算其他代码指标
                    metrics = CodeMetricsCalculator.calculate_metrics(generated_code, ground_truth)
                    for key, value in metrics.items():
                        total_metrics[key] += value


                    # 记录验证结果
                    val_entry = {
                        "original_query": original_query,
                        "enhanced_query": enhanced_query,
                        "ground_truth": ground_truth,
                        "generated_code": generated_code,
                        "reward": reward,
                        "idx": idx
                    }
                    # 添加新指标到验证结果
                    val_entry.update(metrics)
                    val_results.append(val_entry)
                    
                    # 打印指标信息
                    print(f"验证样本 {idx+1}/{len(validation_data)}, Reward: {reward:.4f}, F1: {metrics['f1']:.4f}, CSS: {metrics['css']:.4f}")
                
                    # 每处理一定数量样本后手动执行垃圾回收
                    if (idx + 1) % 5 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"警告: CUDA内存不足。释放缓存并跳过验证样本 {idx+1}。")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
        
        # 保存验证结果
        with open(self.val_log_file, "a", encoding="utf-8") as f:
            for entry in val_results:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        # 计算平均指标
        dataset_size = len(validation_data) if validation_data else 1
        avg_reward = total_reward / dataset_size
        avg_metrics = {key: value / dataset_size for key, value in total_metrics.items()}
        
        # 更新累计指标结果
        if "val_metrics" not in self.metrics:
            self.metrics["val_metrics"] = []
        
        self.metrics["val_metrics"].append(avg_metrics)
        self.metrics["val_precision"].append(avg_metrics["precision"])
        self.metrics["val_recall"].append(avg_metrics["recall"])
        self.metrics["val_f1"].append(avg_metrics["f1"])
        self.metrics["val_css"].append(avg_metrics["css"])
        
        # 打印结果
        print(f"验证集平均奖励: {avg_reward:.4f}")
        print(f"验证集平均指标: Precision: {avg_metrics['precision']:.4f}, " + 
            f"Recall: {avg_metrics['recall']:.4f}, F1: {avg_metrics['f1']:.4f}, " + 
            f"CSS: {avg_metrics['css']:.4f}")
        
        return avg_reward
    
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
        
    def save_metrics(self):
        """保存训练指标数据"""
        self.metrics["training_time"] = time.time() - self.metrics["start_time"]
        with open(self.metrics_file, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=2)
        print(f"已保存训练指标到: {self.metrics_file}")


def load_jsonl_data(file_path):
    """从JSONL文件加载数据"""
    if not os.path.exists(file_path):
        print(f"数据文件不存在: {file_path}")
        return []
    
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                if "prompt" in item and "reference_code" in item:
                    data.append(item)
                else:
                    print(f"警告: 跳过不完整的数据项: {line[:50]}...")
            except json.JSONDecodeError:
                print(f"警告: 跳过无效的JSON行: {line[:50]}...")
    
    print(f"从 {file_path} 加载了 {len(data)} 条数据")
    return data


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="训练RL模型")
    parser.add_argument("--reward-method", type=str, default="overlap", 
                       choices=["overlap", "rouge", "bleu", "f1"],
                       help="奖励计算方法")
    parser.add_argument("--num-epochs", type=int, default=1,
                       help="训练的轮数")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                       help="梯度累积步数")
    args = parser.parse_args()


    # 设置PyTorch内存管理
    if torch.cuda.is_available():
        # 设置内存分配器以减少内存碎片
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # 创建检查点和日志目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = f"checkpoints/run_{timestamp}"
    log_dir = f"logs/run_{timestamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 初始化组件 - 使用LoRA版本的Qwen
    print("初始化LoRA增强的Qwen模型...")
    query_enhancer = QwenLoRAQueryEnhancer(model_name=os.getenv("MODEL_NAME"))
    
    deepseek_api = DeepseekAPI()
    reward_calculator = RewardCalculator(method=args.reward_method)
    reward_method = reward_calculator.get_method()
    print(f"使用奖励计算方法: {reward_method}")
    
    # 创建reward method子目录
    reward_checkpoint_dir = os.path.join(checkpoint_dir, reward_method)
    reward_log_dir = os.path.join(log_dir, reward_method)
    os.makedirs(reward_checkpoint_dir, exist_ok=True)
    os.makedirs(reward_log_dir, exist_ok=True)
    
    # 配置梯度累积步数
    gradient_accumulation_steps = args.gradient_accumulation_steps
    
    trainer = RLTrainer(
        query_enhancer=query_enhancer,
        deepseek_api=deepseek_api,
        reward_calculator=reward_calculator,
        checkpoint_dir=reward_checkpoint_dir,  # 使用包含reward method的目录
        log_dir=reward_log_dir,  # 使用包含reward method的目录
        gradient_accumulation_steps=gradient_accumulation_steps
    )
    
    # 加载检查点（如果存在）
    start_epoch = 0
    latest_checkpoint = os.path.join(reward_checkpoint_dir, "latest_checkpoint")
    
    if os.path.exists(latest_checkpoint):
        start_epoch, _ = trainer.load_checkpoint(latest_checkpoint)
        start_epoch += 1  # 从下一个epoch开始
    
    # 加载训练、验证和测试数据集
    print("正在加载数据集...")
    train_data = load_jsonl_data("dataset/train.jsonl")
    val_data = load_jsonl_data("dataset/val.jsonl")
    test_data = load_jsonl_data("dataset/test.jsonl")
    
    # 记录数据集大小
    dataset_info = {
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
        "timestamp": timestamp
    }
    
    with open(os.path.join(reward_log_dir, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)

    # 训练循环
    num_epochs = args.num_epochs
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\n====== Epoch {epoch + 1}/{num_epochs} ======")
        epoch_start_time = time.time()
        total_reward = 0
        
        # 确保优化器梯度清零开始新的epoch
        trainer.optimizer.zero_grad()
        
        # 训练阶段
        for idx, data in enumerate(train_data):
            try:
                reward, enhanced_query, generated_code = trainer.train_step(
                    data["prompt"],
                    data["reference_code"],
                    idx
                )
                print(f"=> 训练: Epoch {epoch + 1}, Sample {idx+1}/{len(train_data)},Reward method: {reward_method}, Reward: {reward:.4f}")
                
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
            
        avg_train_reward = total_reward / len(train_data)
        print(f"[*] Epoch {epoch + 1}, Average Train Reward: {avg_train_reward:.4f}")
        trainer.metrics["train_rewards"].append(avg_train_reward)
        
        # 验证阶段
        val_reward = trainer.validate(val_data)
        trainer.metrics["val_rewards"].append(val_reward)
        
        # 保存每个epoch的检查点
        trainer.save_checkpoint(epoch + 1, val_reward)
        
        # 将最新检查点作为恢复点
        latest_checkpoint_dir = os.path.join(checkpoint_dir, reward_method, "latest_checkpoint")
        if os.path.exists(latest_checkpoint_dir) and os.path.isdir(latest_checkpoint_dir):
            import shutil
            shutil.rmtree(latest_checkpoint_dir)
        elif os.path.exists(latest_checkpoint_dir):
            os.remove(latest_checkpoint_dir)
            
        # 复制最新检查点作为恢复点
        epoch_checkpoint_dir = os.path.join(checkpoint_dir,reward_method, f"checkpoint_epoch_{epoch+1}")
        if os.path.exists(epoch_checkpoint_dir) and os.path.isdir(epoch_checkpoint_dir):
            import shutil
            shutil.copytree(epoch_checkpoint_dir, latest_checkpoint_dir)
        
        # 如果是最佳模型，则标记保存
        if val_reward > trainer.best_reward:
            trainer.best_reward = val_reward
            trainer.metrics["best_reward"] = val_reward
            trainer.metrics["best_epoch"] = epoch + 1
            trainer.save_checkpoint(epoch + 1, val_reward, is_best=True)
            print(f"新的最佳模型! Validation Reward: {val_reward:.4f}")
            
            # 每次保存最佳模型时额外再次评估验证集结果
            print("\n====== 对最佳模型执行验证集评估 ======")
            best_validation_reward = trainer.validate(val_data)
            print(f"最佳模型验证集评估结果 - Average Reward: {best_validation_reward:.4f}")
            # 将最新评估结果保存在单独的指标中
            trainer.metrics["best_model_validation_reward"] = best_validation_reward
            if trainer.metrics["val_metrics"]:
                latest_metrics = trainer.metrics["val_metrics"][-1]
                trainer.metrics["best_model_precision"] = latest_metrics["precision"]
                trainer.metrics["best_model_recall"] = latest_metrics["recall"]
                trainer.metrics["best_model_f1"] = latest_metrics["f1"]
                trainer.metrics["best_model_css"] = latest_metrics["css"]
                print(f"最佳模型验证集评估结果 - Precision: {latest_metrics['precision']:.4f}, " + 
                    f"Recall: {latest_metrics['recall']:.4f}, F1: {latest_metrics['f1']:.4f}, " + 
                    f"CSS: {latest_metrics['css']:.4f}")
                with open(os.path.join(reward_log_dir, "best_model_validation_results.json"), "w", encoding="utf-8") as f:
                    json.dump(latest_metrics, f, ensure_ascii=False, indent=2)
            # 保存最新的指标
            trainer.save_metrics()
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1} 完成，用时: {epoch_time:.2f}秒")
        
        # 每个epoch后保存指标
        trainer.save_metrics()
    
    # 训练结束后，使用测试集评估最终模型
    print("\n====== 使用测试集评估最终模型 ======")
    
    # 加载最佳模型
    best_model_dir = os.path.join(checkpoint_dir, reward_method, "best_model")
    if os.path.exists(best_model_dir):
        trainer.load_checkpoint(best_model_dir)
        print("已加载最佳模型进行测试评估")
    
    # 在测试集上评估
    test_reward = trainer.validate(test_data)
    trainer.metrics["test_reward"] = test_reward

    # 添加最新验证指标到test指标
    if trainer.metrics["val_metrics"]:
        latest_metrics = trainer.metrics["val_metrics"][-1]
        trainer.metrics["test_precision"] = latest_metrics["precision"]
        trainer.metrics["test_recall"] = latest_metrics["recall"]
        trainer.metrics["test_f1"] = latest_metrics["f1"]
        trainer.metrics["test_css"] = latest_metrics["css"]
        
        # 打印完整测试结果
        print(f"最终测试集评估结果 - Average Reward: {test_reward:.4f}")
        print(f"最终测试集评估结果 - Precision: {latest_metrics['precision']:.4f}, " + 
            f"Recall: {latest_metrics['recall']:.4f}, F1: {latest_metrics['f1']:.4f}, " + 
            f"CSS: {latest_metrics['css']:.4f}")
    # 保存最终指标
    trainer.save_metrics()
    print("\n====== 训练完成 ======")


if __name__ == "__main__":
    main()