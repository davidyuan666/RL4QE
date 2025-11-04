# RL4QE
A Reinforcement Learning framework for Large Language Models (LLMs), focusing on code generation tasks. This repository implements various RL training methods to enhance LLMs' ability to generate high-quality code based on user queries.

## Installation

There are two ways to install and run the project:

### Method 1: Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/RL4QE.git
cd RL4QE

# Set up environment
cp .env_example .env  # Copy example env file and modify as needed

# Install dependencies
pip install .  # Install using pyproject.toml
```

### Method 2: Using PDM (Python Development Master)

```bash
# Clone the repository
git clone https://github.com/yourusername/RL4QE.git
cd RL4QE

# Set up environment
cp .env_example .env  # Copy example env file and modify as needed

# Install PDM if you haven't
pip install pdm

# Install dependencies using PDM
make install  # or run: pdm install

# Start training
make start    # or run: pdm run python train.py
```

Choose the method that best suits your workflow. Method 2 (PDM) provides better dependency management and isolation.

## Features
- Implementation of various RL algorithms for LLM training
- Code generation task optimization
- Customizable reward functions
- Support for multiple LLM architectures
- Easy-to-use training pipeline

## Usage

### Basic Training

```python
from rl4llm import Trainer, LLMAgent, Environment

# Initialize the environment and agent
env = Environment()
agent = LLMAgent(model_name="your-base-model")

# Create trainer
trainer = Trainer(
    agent=agent,
    environment=env,
    learning_rate=1e-5,
    batch_size=16
)

# Start training
trainer.train(num_episodes=1000)
```

### Custom Reward Function

```python
def custom_reward_function(generated_code, reference):
    # Implement your custom reward logic
    return reward_score

# Use custom reward in training
trainer.set_reward_function(custom_reward_function)
```

## Configuration

Key configuration parameters:
- `model_name`: Base LLM model to use
- `learning_rate`: Learning rate for RL training
- `batch_size`: Batch size for training
- `num_episodes`: Number of training episodes
- `reward_type`: Type of reward function to use

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{yuan2025enhancing,
  title={Enhancing queries for code generation with reinforcement learning},
  author={Yuan, Dawei and Liang, Guojun and Li, Tingting and Liu, Suping},
  journal={Scientific Reports},
  volume={15},
  number={1},
  pages={37300},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```
Dataset and Additional Resources: [https://doi.org/10.6084/m9.figshare.28767299.v2](https://doi.org/10.6084/m9.figshare.28767299.v2)

## License
[MIT License](LICENSE)
