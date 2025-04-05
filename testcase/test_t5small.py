import time
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Optional
from ..models.t5small_huggingface import T5SmallQueryEnhancer

def test_model(model: nn.Module, 
               test_cases: List[str], 
               prefix: Optional[str] = None,
               additional_params: Optional[Dict[str, Any]] = None,
               verbose: bool = True) -> Tuple[List[str], float]:
    """
    专门用来测试模型的输入和输出的通用方法
    
    Args:
        model: 需要测试的模型
        test_cases: 测试样例列表
        prefix: 可选的前缀，用于添加到测试样例前
        additional_params: 可选的额外参数，用于模型生成
        verbose: 是否打印详细信息
    
    Returns:
        Tuple[List[str], float]: 返回增强后的查询列表和处理时间
    """
    if verbose:
        print(f"测试模型: {model.__class__.__name__}")
        print(f"测试样例数量: {len(test_cases)}")
    
    # 如果有前缀，添加到测试样例
    if prefix:
        processed_cases = [f"{prefix} {case}" for case in test_cases]
    else:
        processed_cases = test_cases
    
    # 记录开始时间
    start_time = time.time()
    
    # 执行模型前向传播
    try:
        if hasattr(model, 'forward_with_loss'):
            loss, results = model.forward_with_loss(processed_cases)
            if verbose:
                print(f"模型损失: {loss.item()}")
        else:
            results = model.forward(processed_cases)
    except Exception as e:
        print(f"模型执行出错: {str(e)}")
        return [], 0.0
    
    # 计算处理时间
    process_time = time.time() - start_time
    
    if verbose:
        print(f"处理时间: {process_time:.4f}秒")
        print(f"平均每个样例处理时间: {process_time/len(test_cases):.4f}秒")
        
        # 打印输入和输出对比
        for i, (query, result) in enumerate(zip(test_cases, results)):
            print(f"\n样例 {i+1}:")
            print(f"输入: {query}")
            print(f"输出: {result}")
    
    return results, process_time


if __name__ == "__main__":
    # 创建模型实例
    model = T5SmallQueryEnhancer()
    
    # 测试样例查询
    test_queries = [
        "如何学习python",
        "推荐一本机器学习的书",
        "什么是强化学习"
    ]
    
    # 使用新的测试方法测试模型
    print("====== 使用test_model方法测试 ======")
    results, time_taken = test_model(model, test_queries)
    
    # 测试使用英文样例
    english_test_queries = [
        "how to learn python",
        "recommend a book on machine learning",
        "what is reinforcement learning"
    ]
    
    print("\n====== 测试英文样例 ======")
    english_results, english_time = test_model(model, english_test_queries)
    
    print("\n====== 测试不同前缀 ======")
    prefix_results, prefix_time = test_model(
        model, 
        english_test_queries,
        prefix="translate to Chinese:",
        verbose=True
    )
    
    # 测试带有额外参数的生成
    print("\n====== 测试带有额外参数的生成 ======")
    extra_params = {
        "max_length": 150,
        "num_beams": 5,
        "temperature": 0.8,
        "do_sample": True
    }
    
    params_results, params_time = test_model(
        model,
        test_queries,
        additional_params=extra_params,
        verbose=True
    )
    
    # 比较不同测试情况的结果
    print("\n====== 比较不同测试情况的结果 ======")
    print(f"标准中文测试时间: {time_taken:.4f}秒")
    print(f"英文测试时间: {english_time:.4f}秒")
    print(f"带前缀测试时间: {prefix_time:.4f}秒")
    print(f"带额外参数测试时间: {params_time:.4f}秒")