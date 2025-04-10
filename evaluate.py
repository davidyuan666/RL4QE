import os
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def read_jsonl_data(file_path):
    """Read data from a JSONL file"""
    rewards = []
    steps = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                # Extract reward and step
                if 'reward' in data:
                    rewards.append(data['reward'])
                    
                    # Extract step if available, otherwise use the current position
                    step = data.get('step', len(rewards) - 1)
                    steps.append(step)
            except json.JSONDecodeError:
                print(f"Error decoding JSON in file {os.path.basename(file_path)}, line: {line[:50]}...")
                continue
    
    return rewards, steps

def create_reward_plot(rewards, steps, file_name, output_path):
    """Create a reward wave plot"""
    plt.figure(figsize=(12, 7))
    
    plt.plot(steps, rewards, marker='o', linestyle='-', linewidth=2, markersize=8, color='blue')
    plt.title(f'Reward over Steps - {file_name}', fontsize=16)
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Reward', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Calculate statistics
    avg_reward = np.mean(rewards)
    median_reward = np.median(rewards)
    
    # Add horizontal lines for statistics
    plt.axhline(y=avg_reward, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {avg_reward:.4f}')
    plt.axhline(y=median_reward, color='green', linestyle='-.', linewidth=2, 
               label=f'Median: {median_reward:.4f}')
    
    # Add trend line (moving average)
    window_size = min(5, len(rewards))
    if window_size > 1:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        valid_steps = steps[window_size-1:]
        plt.plot(valid_steps, moving_avg, color='purple', linestyle='-', 
                label=f'{window_size}-point Moving Avg', linewidth=3, alpha=0.7)
    
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Set up plot style
    plt.style.use('ggplot')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    
    # Find JSONL files in the result directory
    result_dir = 'result/f1'
    json_files = [f for f in os.listdir(result_dir) if f.startswith('train_') and f.endswith('.jsonl')]
    
    # Create output directory for plots
    output_dir = os.path.join(result_dir, 'plots')
    os.makedirs(output_dir, exist_ok=True)
    
    for json_file in json_files:
        file_path = os.path.join(result_dir, json_file)
        rewards, steps = read_jsonl_data(file_path)
        
        if rewards:
            base_name = os.path.splitext(json_file)[0]
            
            # Create reward plot
            reward_path = os.path.join(output_dir, f'{base_name}_reward.png')
            create_reward_plot(rewards, steps, json_file, reward_path)
            print(f"Created reward plot: {reward_path}")
        else:
            print(f"No reward data found in {json_file}")
    
    print("\nReward plot creation completed.")

if __name__ == "__main__":
    main()