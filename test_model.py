import time
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Optional, Callable
from models.t5small_huggingface import T5SmallQueryEnhancer
from models.qwen import QwenQueryEnhancer

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
            if verbose and hasattr(loss, 'item'):
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

def get_test_queries(language="zh") -> List[str]:
    """获取测试用例，支持中文和英文"""
    if language.lower() in ["zh", "chinese", "cn"]:
        return [
            "如何学习python",
            "推荐一本机器学习的书",
            "什么是强化学习"
        ]
    else:  # 默认为英文
        return [
            "how to learn python",
            "recommend a book on machine learning",
            "what is reinforcement learning"
        ]

def test_model_with_configurations(model_class, model_name=None, test_configurations=None):
    """
    通用模型测试函数，用于测试不同配置下的模型表现
    
    Args:
        model_class: 模型类
        model_name: 可选的模型名称，用于初始化模型
        test_configurations: 测试配置字典列表，每个配置包含测试所需的参数
    
    Returns:
        Dict: 包含各项测试结果的字典
    """
    # 初始化模型
    if model_name:
        model = model_class(model_name=model_name)
    else:
        model = model_class()
    
    # 如果没有提供测试配置，使用默认配置
    if test_configurations is None:
        test_configurations = [
            {"name": "标准中文测试", "queries": get_test_queries("zh")},
            {"name": "英文测试", "queries": get_test_queries("en")},
            {"name": "带前缀测试", "queries": get_test_queries("en"), "prefix": "translate to Chinese:"},
            {"name": "带额外参数测试", "queries": get_test_queries("zh"), "additional_params": {
                "max_length": 150,
                "num_beams": 5,
                "temperature": 0.8,
                "do_sample": True
            }}
        ]
    
    results = {}
    print(f"\n====== 测试模型: {model.__class__.__name__} ======")
    
    # 执行每个测试配置
    for config in test_configurations:
        config_name = config["name"]
        print(f"\n====== {config_name} ======")
        
        # 提取参数
        queries = config["queries"]
        prefix = config.get("prefix")
        additional_params = config.get("additional_params")
        verbose = config.get("verbose", True)
        
        # 执行测试
        outputs, time_taken = test_model(
            model,
            queries,
            prefix=prefix,
            additional_params=additional_params,
            verbose=verbose
        )
        
        # 保存结果
        results[config_name] = {
            "outputs": outputs,
            "time_taken": time_taken
        }
    
    # 打印时间比较
    print("\n====== 各测试配置处理时间比较 ======")
    for config_name, data in results.items():
        print(f"{config_name}时间: {data['time_taken']:.4f}秒")
    
    return results

def test_t5small():
    """测试T5Small模型"""
    return test_model_with_configurations(T5SmallQueryEnhancer)

def test_qwen():
    """测试Qwen模型"""
    return test_model_with_configurations(QwenQueryEnhancer)


def test_all_models():
    """测试所有可用模型"""
    print("====== 开始测试所有模型 ======")
    
    results = {}
    
    # 测试T5Small模型
    print("\n====== 测试T5Small模型 ======")
    results["T5Small"] = test_t5small()
    
    # 测试Qwen模型
    print("\n====== 测试Qwen模型 ======")
    results["Qwen"] = test_qwen()

    # 比较不同模型的性能
    print("\n====== 各模型性能比较 ======")
    for model_name, model_results in results.items():
        print(f"\n{model_name}模型:")
        for config_name, data in model_results.items():
            print(f"  {config_name}时间: {data['time_taken']:.4f}秒")
    
    return results

if __name__ == "__main__":
    # 测试单个模型
    # test_t5small()
    # test_qwen()
    
    # 测试所有模型
    test_all_models()