import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple
import numpy as np
import json
from models.t5small_huggingface import T5SmallQueryEnhancer
from llm.deepseek import DeepseekAPI
from utils.reward_util import RewardCalculator

class RLTrainer:
    def __init__(self, 
                 query_enhancer: T5SmallQueryEnhancer,
                 deepseek_api: DeepseekAPI,
                 reward_calculator: RewardCalculator,
                 learning_rate: float = 1e-4):
        self.query_enhancer = query_enhancer
        self.deepseek_api = deepseek_api
        self.reward_calculator = reward_calculator
        self.optimizer = optim.Adam(query_enhancer.parameters(), lr=learning_rate)
        
    def train_step(self, 
               original_query: str,
               ground_truth: str) -> Tuple[float, str]:
        # 1. Generate enhanced query and get loss
        output = self.query_enhancer.forward_with_loss([original_query])
        enhanced_query = output['generated_text'][0]
        model_loss = output['loss']

        print(f'enhanced_query: {enhanced_query}')
        
        # 2. Get Deepseek response
        response = self.deepseek_api.get_response(enhanced_query)

        generated_code = response.split("<answer>")[1].split("</answer>")[0].strip()

        print(f'generated_code: {generated_code}')

        # 3. Calculate reward
        reward = self.reward_calculator.calculate(generated_code, ground_truth)
        
        # 4. Calculate policy gradient and update model
        self.optimizer.zero_grad()
        # Use REINFORCE algorithm with the properly captured loss
        loss = -torch.mean(torch.tensor(reward) * model_loss)
        loss.backward()
        self.optimizer.step()
        
        return reward, enhanced_query,generated_code

def main():
    # 初始化组件
    query_enhancer = T5SmallQueryEnhancer()
    deepseek_api = DeepseekAPI()
    reward_calculator = RewardCalculator()
    
    trainer = RLTrainer(
        query_enhancer=query_enhancer,
        deepseek_api=deepseek_api,
        reward_calculator=reward_calculator
    )
    
    with open("dataset/train.json", "r") as f:
        training_data = json.load(f)
    
    # 训练循环
    num_epochs = 10
    for epoch in range(num_epochs):
        total_reward = 0
        for data in training_data:
            reward, enhanced_query, generated_code = trainer.train_step(
                data["query"],
                data["ground_truth"]
            )
            print(f"Epoch {epoch + 1}, Reward: {reward:.4f}, original Query: {data['query']}, Enhanced Query: {enhanced_query}, Response: {generated_code}")
            total_reward += reward
            
        avg_reward = total_reward / len(training_data)
        print(f"Epoch {epoch + 1}, Average Reward: {avg_reward:.4f}")

if __name__ == "__main__":
    main()