import sys
import os
sys.path.append('src')
sys.path.append('agents')

import numpy as np
import matplotlib.pyplot as plt
from snake_game import SnakeGame
from q_learning_agent import QLearningAgent
import argparse

def train_q_learning_agent(episodes=1000, save_interval=100):
    """
    Train Q-Learning agent to play Snake game
    
    Args:
        episodes: Number of training episodes
        save_interval: How often to save the model
    """
    # Initialize game and agent
    game = SnakeGame(speed=1000)  # Fast speed for training
    agent = QLearningAgent()
    
    # Training statistics
    scores = []
    mean_scores = []
    total_score = 0
    record = 0
    
    print(f"Starting Q-Learning training for {episodes} episodes...")
    print(f"Initial epsilon: {agent.epsilon:.4f}")
    
    try:
        for episode in range(episodes):
            # Reset game
            state = game.reset()
            score = 0
            done = False
            
            while not done:
                # Get action from agent
                action = agent.get_action(state, training=True)
                
                # Execute action
                reward, done, score = game.play_step(action)
                next_state = game.get_state()
                
                # Update Q-table
                agent.update_q_table(state, action, reward, next_state, done)
                
                state = next_state
            
            # Update statistics
            scores.append(score)
            agent.training_scores.append(score)
            total_score += score
            
            if score > record:
                record = score
            
            mean_score = total_score / (episode + 1)
            mean_scores.append(mean_score)
            
            # Print progress
            if (episode + 1) % 100 == 0:
                print(f'Episode {episode + 1:4d}, Score: {score:2d}, Record: {record:2d}, '
                      f'Mean Score: {mean_score:.2f}, Epsilon: {agent.epsilon:.4f}, '
                      f'Q-table size: {len(agent.q_table)}')
            
            # Save model periodically
            if (episode + 1) % save_interval == 0:
                model_path = f'models/q_learning_model_ep{episode + 1}'
                os.makedirs('models', exist_ok=True)
                agent.save_model(f'{model_path}.pkl')
        
        # Final save
        os.makedirs('models', exist_ok=True)
        agent.save_model('models/q_learning_final.pkl')
        
        # Plot training results
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(scores)
        plt.plot(mean_scores)
        plt.title('Q-Learning Training Scores')
        plt.ylabel('Score')
        plt.xlabel('Episode')
        plt.legend(['Score', 'Mean Score'])
        
        plt.subplot(1, 2, 2)
        epsilon_history = [1.0 * (agent.epsilon_decay ** i) for i in range(episodes)]
        epsilon_history = [max(e, agent.epsilon_min) for e in epsilon_history]
        plt.plot(epsilon_history)
        plt.title('Epsilon Decay')
        plt.ylabel('Epsilon')
        plt.xlabel('Episode')
        
        plt.tight_layout()
        plt.savefig('data/q_learning_training.png')
        plt.show()
        
        print(f"\nTraining completed!")
        print(f"Final statistics:")
        print(f"- Episodes: {episodes}")
        print(f"- Final Score: {score}")
        print(f"- Best Score: {record}")
        print(f"- Mean Score: {mean_score:.2f}")
        print(f"- Final Epsilon: {agent.epsilon:.4f}")
        print(f"- Q-table size: {len(agent.q_table)} states")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        agent.save_model('models/q_learning_interrupted.pkl')
        print("Model saved as 'q_learning_interrupted.pkl'")

def test_q_learning_agent(model_path, episodes=10):
    """Test trained Q-Learning agent"""
    game = SnakeGame(speed=100)  # Normal speed for testing
    agent = QLearningAgent()
    agent.load_model(model_path)
    
    scores = []
    
    print(f"Testing Q-Learning agent for {episodes} episodes...")
    
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or test Q-Learning agent')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Mode: train or test')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes for training')
    parser.add_argument('--model', type=str, default='models/q_learning_final.pkl',
                        help='Model path for testing')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_q_learning_agent(episodes=args.episodes)
    else:
        test_q_learning_agent(args.model, episodes=10)