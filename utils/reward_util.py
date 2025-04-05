class RewardCalculator:
    def __init__(self):
        pass
        
    def calculate(self, response: str, ground_truth: str) -> float:
        # 实现基于ground truth的奖励计算
        # 这里可以使用ROUGE分数、BLEU分数或其他相似度度量
        # 示例使用简单的字符串匹配率
        return len(set(response.split()) & set(ground_truth.split())) / len(set(ground_truth.split()))
