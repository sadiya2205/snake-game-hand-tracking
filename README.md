# Snake Game AI using Q-Learning and DQN

An intelligent agent implementation that learns to play Snake using reinforcement learning techniques - specifically Q-Learning and Deep Q-Networks (DQN). The project demonstrates the power of AI in game playing through reward-based learning.

## 🎯 Project Overview

This project implements and compares two different reinforcement learning approaches:

1. **Q-Learning**: A table-based approach that learns optimal actions for different game states
2. **Deep Q-Network (DQN)**: A neural network-based approach using TensorFlow for complex state representations

## 🚀 Features

- **Complete Snake Game Implementation**: Built with Pygame for smooth gameplay
- **Dual AI Agents**: Both Q-Learning and DQN implementations
- **Training Scripts**: Automated training with progress monitoring
- **Visualization Tools**: Comprehensive analysis and plotting of training results
- **Model Persistence**: Save and load trained models
- **Performance Comparison**: Side-by-side analysis of different approaches

## 📁 Project Structure

```
snake-game-ai/
├── src/
│   └── snake_game.py          # Core Snake game implementation
├── agents/
│   ├── q_learning_agent.py    # Q-Learning agent
│   └── dqn_agent.py          # Deep Q-Network agent
├── models/                    # Saved trained models
├── data/                      # Training data and visualizations
├── docs/                      # Additional documentation
├── train_qlearning.py         # Q-Learning training script
├── train_dqn.py              # DQN training script
├── visualize.py              # Visualization and analysis tools
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd snake-game-ai
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import pygame, tensorflow, numpy, matplotlib; print('All dependencies installed successfully!')"
   ```

## 🎮 Usage

### Training Agents

#### Q-Learning Agent
```bash
# Train for 1000 episodes (default)
python train_qlearning.py

# Train for custom episodes
python train_qlearning.py --episodes 2000

# Test trained model
python train_qlearning.py --mode test --model models/q_learning_final.pkl
```

#### DQN Agent
```bash
# Train for 2000 episodes (default)
python train_dqn.py

# Train for custom episodes
python train_dqn.py --episodes 3000

# Test trained model
python train_dqn.py --mode test --model models/dqn_final

# Compare both agents
python train_dqn.py --mode compare
```

### Visualization and Analysis

```bash
# View training comparison
python visualize.py --mode comparison

# Analyze agent performance
python visualize.py --mode analysis --agent both --episodes 50

# View Q-table heatmap
python visualize.py --mode qtable

# Run all visualizations
python visualize.py --mode all
```

## 🧠 How It Works

### Game Environment

The Snake game provides a structured environment with:
- **State Space**: 11-dimensional vector representing:
  - Danger detection (straight, right, left)
  - Current direction (4 boolean values)
  - Food location relative to head (4 boolean values)
- **Action Space**: 3 possible actions [straight, right, left]
- **Reward System**:
  - +10 for eating food
  - -10 for collision/death
  - 0 for regular moves

### Q-Learning Approach

- Uses a Q-table to store state-action values
- Implements ε-greedy exploration strategy
- Updates Q-values using the Bellman equation:
  ```
  Q(s,a) = Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
  ```
- **Advantages**: Simple, interpretable, guaranteed convergence
- **Limitations**: Limited to discrete state spaces

### Deep Q-Network (DQN) Approach

- Uses neural networks to approximate Q-values
- Implements experience replay for stable learning
- Features target network for stable updates
- **Architecture**: 128→128→64→3 fully connected layers
- **Advantages**: Handles complex state spaces, scalable
- **Limitations**: Requires more computational resources

## 📊 Performance Metrics

### Training Metrics
- **Episode Score**: Points achieved in each game
- **Mean Score**: Running average of scores
- **Epsilon**: Exploration rate over time
- **Q-table Size**: Number of unique states discovered (Q-Learning)
- **Memory Buffer**: Experience replay buffer utilization (DQN)

### Evaluation Metrics
- Average score over test episodes
- Maximum score achieved
- Score consistency (standard deviation)
- Average steps per episode

## 📈 Expected Results

After proper training, you can expect:

- **Q-Learning**: Scores typically range 5-15, with occasional higher scores
- **DQN**: Generally achieves higher and more consistent scores (10-25+)
- **Learning Curve**: Both agents show improvement over episodes
- **Convergence**: DQN typically converges faster due to function approximation

## 🔧 Configuration

### Hyperparameters

#### Q-Learning
- Learning Rate: 0.1
- Discount Factor: 0.95
- Initial Epsilon: 1.0
- Epsilon Decay: 0.995
- Epsilon Min: 0.01

#### DQN
- Learning Rate: 0.001
- Discount Factor: 0.95
- Initial Epsilon: 1.0
- Epsilon Decay: 0.995
- Epsilon Min: 0.01
- Memory Size: 10,000
- Batch Size: 32
- Target Update Frequency: 100 episodes

### Customization

You can modify these parameters in the respective agent files to experiment with different learning behaviors.

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**:
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python path configuration

2. **Slow Training**:
   - Increase game speed parameter in training scripts
   - Reduce visualization frequency
   - Consider using GPU for DQN training

3. **Poor Performance**:
   - Increase training episodes
   - Adjust hyperparameters
   - Ensure proper reward structure

4. **Memory Issues**:
   - Reduce DQN memory buffer size
   - Lower batch size for DQN training

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## 📚 Further Reading

- [Q-Learning Algorithm](https://en.wikipedia.org/wiki/Q-learning)
- [Deep Q-Networks Paper](https://arxiv.org/abs/1312.5602)
- [Pygame Documentation](https://www.pygame.org/docs/)
- [TensorFlow Guide](https://www.tensorflow.org/guide)

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for reinforcement learning research
- Pygame community for game development tools
- TensorFlow team for deep learning framework
- Research community for Q-Learning and DQN algorithms

## 📧 Contact

For questions, suggestions, or contributions, please open an issue on the repository.

---

**Happy Learning!** 🐍🤖