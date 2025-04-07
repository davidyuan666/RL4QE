from rouge import Rouge
import nltk
from nltk.translate.bleu_score import sentence_bleu
import os
from dotenv import load_dotenv

load_dotenv()

class RewardCalculator:
    def __init__(self, method):
        """初始化奖励计算器
        
        Args:
            method: 奖励计算方法，可选值有 "overlap", "rouge", "bleu", "exact_match", "f1"
        """
        self.method = method


    def get_method(self):
        return self.method
        
    def calculate(self, response: str, ground_truth: str) -> float:
        """根据选择的方法计算奖励
        
        Args:
            response: 模型生成的回答
            ground_truth: 标准答案
            
        Returns:
            float: 奖励分数，范围通常为0到1
        """
        if self.method == "overlap":
            # 简单集合重叠率
            return self._overlap_score(response, ground_truth)
        elif self.method == "rouge":
            # ROUGE-L 评分
            return self._rouge_score(response, ground_truth)
        elif self.method == "bleu":
            # BLEU 评分
            return self._bleu_score(response, ground_truth)
        elif self.method == "exact_match":
            # 精确匹配
            return 1.0 if response.strip() == ground_truth.strip() else 0.0
        elif self.method == "f1":
            # F1 分数
            return self._f1_score(response, ground_truth)
        else:
            raise ValueError(f"不支持的奖励计算方法: {self.method}")
    
    def _overlap_score(self, response: str, ground_truth: str) -> float:
        """计算词汇重叠率"""
        response_words = set(response.lower().split())
        truth_words = set(ground_truth.lower().split())
        if not truth_words:
            return 0.0
        return len(response_words & truth_words) / len(truth_words)
    
    def _rouge_score(self, response: str, ground_truth: str) -> float:
        """计算ROUGE-L分数"""
        try:
            # Handle empty responses or ground truths
            if not response or not ground_truth:
                return 0.0
                
            # Add a space if response or ground_truth is too short
            if len(response.strip()) < 2:
                response = response.strip() + " dummy"
            if len(ground_truth.strip()) < 2:
                ground_truth = ground_truth.strip() + " dummy"
                
            rouge = Rouge()
            scores = rouge.get_scores(response, ground_truth)
            return scores[0]["rouge-l"]["f"]
        except Exception as e:
            print(f"Rouge计算错误: {e}")
            # Fallback to overlap score when rouge fails
            return self._overlap_score(response, ground_truth)
    
    def _bleu_score(self, response: str, ground_truth: str) -> float:
        """计算BLEU分数"""
        try:
            response_tokens = response.lower().split()
            reference_tokens = [ground_truth.lower().split()]
            
            return sentence_bleu(reference_tokens, response_tokens, 
                                weights=(0.25, 0.25, 0.25, 0.25))
        except ImportError:
            print("请安装nltk库: pip install nltk")
            return self._overlap_score(response, ground_truth)
    
    def _f1_score(self, response: str, ground_truth: str) -> float:
        """计算基于词的F1分数"""
        response_tokens = set(response.lower().split())
        truth_tokens = set(ground_truth.lower().split())
        
        if len(response_tokens) == 0 or len(truth_tokens) == 0:
            return 0.0
            
        common = response_tokens & truth_tokens
        precision = len(common) / len(response_tokens) if response_tokens else 0
        recall = len(common) / len(truth_tokens) if truth_tokens else 0
        
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)