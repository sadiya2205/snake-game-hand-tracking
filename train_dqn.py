import sys
import os
sys.path.append('src')
sys.path.append('agents')

import numpy as np
import matplotlib.pyplot as plt
from snake_game import SnakeGame
from dqn_agent import DQNAgent
import argparse

def train_dqn_agent(episodes=2000, save_interval=200, target_update_frequency=100):
    """
    Train DQN agent to play Snake game
    
    Args:
        episodes: Number of training episodes
        save_interval: How often to save the model
        target_update_frequency: How often to update target network
    """
    # Initialize game and agent
    game = SnakeGame(speed=1000)  # Fast speed for training
    agent = DQNAgent()
    
    # Training statistics
    scores = []
    mean_scores = []
    total_score = 0
    record = 0
    
    print(f"Starting DQN training for {episodes} episodes...")
    print(f"Initial epsilon: {agent.epsilon:.4f}")
    print(f"Target network update frequency: {target_update_frequency}")
    
    try:
        for episode in range(episodes):
            # Reset game
            state = game.reset()
            score = 0
            done = False
            steps = 0
            
            while not done:
                # Get action from agent
                action = agent.get_action(state, training=True)
                
                # Execute action
                reward, done, score = game.play_step(action)
                next_state = game.get_state()
                
                # Store experience in replay buffer
                agent.remember(state, action, reward, next_state, done)
                
                # Train the neural network
                agent.replay()
                
                state = next_state
                steps += 1
            
            # Update target network periodically
            if (episode + 1) % target_update_frequency == 0:
                agent.update_target_network()
                print(f"Target network updated at episode {episode + 1}")
            
            # Update statistics
            scores.append(score)
            agent.training_scores.append(score)
            agent.training_steps.append(steps)
            total_score += score
            
            if score > record:
                record = score
            
            mean_score = total_score / (episode + 1)
            mean_scores.append(mean_score)
            
            # Print progress
            if (episode + 1) % 100 == 0:
                avg_loss = np.mean(agent.losses[-100:]) if len(agent.losses) >= 100 else 0
                print(f'Episode {episode + 1:4d}, Score: {score:2d}, Record: {record:2d}, '
                      f'Mean Score: {mean_score:.2f}, Epsilon: {agent.epsilon:.4f}, '
                      f'Memory: {len(agent.memory)}, Avg Loss: {avg_loss:.4f}')
            
            # Save model periodically
            if (episode + 1) % save_interval == 0:
                model_path = f'models/dqn_model_ep{episode + 1}'
                os.makedirs('models', exist_ok=True)
                agent.save_model(model_path)
        
        # Final save
        os.makedirs('models', exist_ok=True)
        agent.save_model('models/dqn_final')
        
        # Plot training results
        plt.figure(figsize=(15, 10))
        
        # Scores plot
        plt.subplot(2, 3, 1)
        plt.plot(scores)
        plt.plot(mean_scores)
        plt.title('DQN Training Scores')
        plt.ylabel('Score')
        plt.xlabel('Episode')
        plt.legend(['Score', 'Mean Score'])
        
        # Epsilon decay plot
        plt.subplot(2, 3, 2)
        epsilon_history = [1.0 * (agent.epsilon_decay ** i) for i in range(episodes)]
        epsilon_history = [max(e, agent.epsilon_min) for e in epsilon_history]
        plt.plot(epsilon_history)
        plt.title('Epsilon Decay')
        plt.ylabel('Epsilon')
        plt.xlabel('Episode')
        
        # Loss plot
        plt.subplot(2, 3, 3)
        if agent.losses:
            # Moving average of losses
            window_size = 100
            if len(agent.losses) >= window_size:
                moving_avg_loss = []
                for i in range(window_size - 1, len(agent.losses)):
                    moving_avg_loss.append(np.mean(agent.losses[i - window_size + 1:i + 1]))
                plt.plot(moving_avg_loss)
                plt.title('Training Loss (Moving Average)')
                plt.ylabel('Loss')
                plt.xlabel('Training Step')
        
        # Memory size plot
        plt.subplot(2, 3, 4)
        memory_sizes = [min(i * 10, agent.memory_size) for i in range(episodes // 10)]
        plt.plot(memory_sizes)
        plt.title('Experience Replay Buffer Size')
        plt.ylabel('Memory Size')
        plt.xlabel('Episode (x10)')
        
        # Steps per episode
        plt.subplot(2, 3, 5)
        if agent.training_steps:
            plt.plot(agent.training_steps)
            plt.title('Steps per Episode')
            plt.ylabel('Steps')
            plt.xlabel('Episode')
        
        # Score distribution
        plt.subplot(2, 3, 6)
        plt.hist(scores, bins=20, alpha=0.7)
        plt.title('Score Distribution')
        plt.ylabel('Frequency')
        plt.xlabel('Score')
        
        plt.tight_layout()
        os.makedirs('data', exist_ok=True)
        plt.savefig('data/dqn_training.png')
        plt.show()
        
        print(f"\nTraining completed!")
        print(f"Final statistics:")
        print(f"- Episodes: {episodes}")
        print(f"- Final Score: {score}")
        print(f"- Best Score: {record}")
        print(f"- Mean Score: {mean_score:.2f}")
        print(f"- Final Epsilon: {agent.epsilon:.4f}")
        print(f"- Memory Buffer Size: {len(agent.memory)}")
        print(f"- Total Training Losses: {len(agent.losses)}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        agent.save_model('models/dqn_interrupted')
        print("Model saved as 'dqn_interrupted'")

def test_dqn_agent(model_path, episodes=10):
    """Test trained DQN agent"""
    game = SnakeGame(speed=100)  # Normal speed for testing
    agent = DQNAgent()
    agent.load_model(model_path)
    
    scores = []
    
    print(f"Testing DQN agent for {episodes} episodes...")
    
    for episode in range(episodes):
        state = game.reset()
        score = 0
        done = False
        
        while not done:
            action = agent.get_action(state, training=False)
            _, done, score = game.play_step(action)
            state = game.get_state()
        
        scores.append(score)
        print(f'Test Episode {episode + 1}: Score = {score}')
    
    print(f"\nTest Results:")
    print(f"Average Score: {np.mean(scores):.2f}")
    print(f"Best Score: {np.max(scores)}")
    print(f"Worst Score: {np.min(scores)}")

def compare_agents():
    """Compare Q-Learning and DQN agents"""
    from q_learning_agent import QLearningAgent
    
    print("Comparing Q-Learning and DQN agents...")
    
    # Test Q-Learning agent
    print("\nTesting Q-Learning agent:")
    game = SnakeGame(speed=100)
    q_agent = QLearningAgent()
    try:
        q_agent.load_model('models/q_learning_final.pkl')
        scores = []
        for episode in range(20):
            state = game.reset()
            score = 0
            done = False
            while not done:
                action = q_agent.get_action(state, training=False)
                _, done, score = game.play_step(action)
                state = game.get_state()
            scores.append(score)
        print(f"Q-Learning Average Score: {np.mean(scores):.2f}")
    except Exception as e:
        print(f"Q-Learning agent not available: {e}")
    
    # Test DQN agent
    print("\nTesting DQN agent:")
    test_dqn_agent('models/dqn_final', episodes=20)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or test DQN agent')
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'test', 'compare'],
                        help='Mode: train, test, or compare agents')
    parser.add_argument('--episodes', type=int, default=2000,
                        help='Number of episodes for training')
    parser.add_argument('--model', type=str, default='models/dqn_final',
                        help='Model path for testing')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_dqn_agent(episodes=args.episodes)
    elif args.mode == 'test':
        test_dqn_agent(args.model, episodes=10)
    else:  # compare
        compare_agents()