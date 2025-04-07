# utils/code_metrics.py
import re
import difflib
from typing import Dict, Tuple, List, Any


class CodeMetricsCalculator:
    """
    计算代码相似度的多种度量指标，包括精确度、召回率、F1分数和代码结构相似度(CSS)
    """
    
    @staticmethod
    def tokenize_code(code: str) -> List[str]:
        """将代码分割成标记序列（token）"""
        # 去除空白字符并转小写
        code = code.strip().lower()
        
        # 基本的代码标记化方法
        # 保留标识符、数字、运算符、括号等
        tokens = re.findall(r'[a-zA-Z_]\w*|[-+*/=<>!&|^%(){}[\],.;:]|\d+(?:\.\d+)?', code)
        return tokens
    
    @staticmethod
    def calculate_metrics(generated_code: str, ground_truth: str) -> Dict[str, float]:
        """
        计算所有指标
        
        Args:
            generated_code: 模型生成的代码
            ground_truth: 标准答案代码
            
        Returns:
            Dict: 包含precision, recall, f1, css等指标的字典
        """
        # 如果代码为空，返回零分
        if not generated_code or not ground_truth:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "css": 0.0
            }
        
        # 标记化代码
        generated_tokens = CodeMetricsCalculator.tokenize_code(generated_code)
        truth_tokens = CodeMetricsCalculator.tokenize_code(ground_truth)
        
        # 计算常见标记
        generated_set = set(generated_tokens)
        truth_set = set(truth_tokens)
        common_tokens = generated_set.intersection(truth_set)
        
        # 计算精确度和召回率
        precision = len(common_tokens) / len(generated_set) if generated_set else 0.0
        recall = len(common_tokens) / len(truth_set) if truth_set else 0.0
        
        # 计算F1分数
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # 计算代码结构相似度 (CSS)
        # 使用difflib的序列匹配比较代码结构
        matcher = difflib.SequenceMatcher(None, generated_tokens, truth_tokens)
        css = matcher.ratio()  # 返回0到1之间的相似度分数
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "css": css
        }