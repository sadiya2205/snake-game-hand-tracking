import sys
import os
sys.path.append('src')
sys.path.append('agents')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from snake_game import SnakeGame
from q_learning_agent import QLearningAgent
from dqn_agent import DQNAgent
import argparse
import pickle

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_training_comparison():
    """Compare training progress of Q-Learning and DQN agents"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Load Q-Learning results
    try:
        q_agent = QLearningAgent()
        q_agent.load_model('models/q_learning_final.pkl')
        q_scores = q_agent.training_scores
        
        # Plot Q-Learning scores
        axes[0, 0].plot(q_scores, alpha=0.7, label='Episode Score')
        axes[0, 0].plot(pd.Series(q_scores).rolling(100).mean(), label='Moving Average (100)')
        axes[0, 0].set_title('Q-Learning Training Scores')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        print(f"Q-Learning stats: Mean={np.mean(q_scores):.2f}, Max={np.max(q_scores)}")
        
    except Exception as e:
        print(f"Could not load Q-Learning results: {e}")
        axes[0, 0].text(0.5, 0.5, 'Q-Learning data not available', 
                       ha='center', va='center', transform=axes[0, 0].transAxes)
    
    # Load DQN results
    try:
        dqn_agent = DQNAgent()
        dqn_agent.load_model('models/dqn_final')
        dqn_scores = dqn_agent.training_scores
        dqn_losses = dqn_agent.losses
        
        # Plot DQN scores
        axes[0, 1].plot(dqn_scores, alpha=0.7, label='Episode Score')
        axes[0, 1].plot(pd.Series(dqn_scores).rolling(100).mean(), label='Moving Average (100)')
        axes[0, 1].set_title('DQN Training Scores')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot DQN losses
        if dqn_losses:
            axes[1, 0].plot(pd.Series(dqn_losses).rolling(50).mean())
            axes[1, 0].set_title('DQN Training Loss (Moving Average)')
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True, alpha=0.3)
        
        print(f"DQN stats: Mean={np.mean(dqn_scores):.2f}, Max={np.max(dqn_scores)}")
        
    except Exception as e:
        print(f"Could not load DQN results: {e}")
        axes[0, 1].text(0.5, 0.5, 'DQN data not available', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[1, 0].text(0.5, 0.5, 'DQN loss data not available', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
    
    # Comparison plot
    try:
        if 'q_scores' in locals() and 'dqn_scores' in locals():
            # Ensure same length for comparison
            min_len = min(len(q_scores), len(dqn_scores))
            q_avg = pd.Series(q_scores[:min_len]).rolling(100).mean()
            dqn_avg = pd.Series(dqn_scores[:min_len]).rolling(100).mean()
            
            axes[1, 1].plot(q_avg, label='Q-Learning', linewidth=2)
            axes[1, 1].plot(dqn_avg, label='DQN', linewidth=2)
            axes[1, 1].set_title('Agent Comparison (Moving Average)')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Comparison not available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
    except:
        axes[1, 1].text(0.5, 0.5, 'Comparison not available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    plt.savefig('data/training_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_agent_performance(agent_type='both', episodes=50):
    """Analyze and visualize agent performance"""
    results = {}
    
    if agent_type in ['q_learning', 'both']:
        print("Analyzing Q-Learning agent...")
        try:
            game = SnakeGame(speed=1000)  # Fast for analysis
            agent = QLearningAgent()
            agent.load_model('models/q_learning_final.pkl')
            
            scores = []
            steps_list = []
            
            for episode in range(episodes):
                state = game.reset()
                score = 0
                done = False
                steps = 0
                
                while not done:
                    action = agent.get_action(state, training=False)
                    _, done, score = game.play_step(action)
                    state = game.get_state()
                    steps += 1
                
                scores.append(score)
                steps_list.append(steps)
                
                if (episode + 1) % 10 == 0:
                    print(f"Q-Learning Episode {episode + 1}: Score = {score}")
            
            results['q_learning'] = {
                'scores': scores,
                'steps': steps_list,
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'max_score': np.max(scores),
                'mean_steps': np.mean(steps_list)
            }
            
        except Exception as e:
            print(f"Error analyzing Q-Learning agent: {e}")
    
    if agent_type in ['dqn', 'both']:
        print("Analyzing DQN agent...")
        try:
            game = SnakeGame(speed=1000)  # Fast for analysis
            agent = DQNAgent()
            agent.load_model('models/dqn_final')
            
            scores = []
            steps_list = []
            
            for episode in range(episodes):
                state = game.reset()
                score = 0
                done = False
                steps = 0
                
                while not done:
                    action = agent.get_action(state, training=False)
                    _, done, score = game.play_step(action)
                    state = game.get_state()
                    steps += 1
                
                scores.append(score)
                steps_list.append(steps)
                
                if (episode + 1) % 10 == 0:
                    print(f"DQN Episode {episode + 1}: Score = {score}")
            
            results['dqn'] = {
                'scores': scores,
                'steps': steps_list,
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'max_score': np.max(scores),
                'mean_steps': np.mean(steps_list)
            }
            
        except Exception as e:
            print(f"Error analyzing DQN agent: {e}")
    
    # Visualize results
    if results:
        plot_performance_analysis(results)
        
    return results

def plot_performance_analysis(results):
    """Plot detailed performance analysis"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    agents = list(results.keys())
    colors = ['skyblue', 'lightcoral']
    
    # Score distribution
    for i, agent in enumerate(agents):
        scores = results[agent]['scores']
        axes[0, 0].hist(scores, alpha=0.7, label=f'{agent.replace("_", "-").title()}', 
                       bins=15, color=colors[i])
    axes[0, 0].set_title('Score Distribution')
    axes[0, 0].set_xlabel('Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Score timeline
    for i, agent in enumerate(agents):
        scores = results[agent]['scores']
        axes[0, 1].plot(scores, alpha=0.8, label=f'{agent.replace("_", "-").title()}', 
                       color=colors[i])
    axes[0, 1].set_title('Score Over Episodes')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Steps per episode
    for i, agent in enumerate(agents):
        steps = results[agent]['steps']
        axes[0, 2].plot(steps, alpha=0.8, label=f'{agent.replace("_", "-").title()}', 
                       color=colors[i])
    axes[0, 2].set_title('Steps per Episode')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Steps')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Performance comparison bar chart
    if len(agents) > 1:
        metrics = ['mean_score', 'max_score', 'mean_steps']
        x = np.arange(len(metrics))
        width = 0.35
        
        for i, agent in enumerate(agents):
            values = [results[agent][metric] for metric in metrics]
            axes[1, 0].bar(x + i*width, values, width, 
                          label=f'{agent.replace("_", "-").title()}', color=colors[i])
        
        axes[1, 0].set_title('Performance Comparison')
        axes[1, 0].set_xlabel('Metrics')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].set_xticks(x + width/2)
        axes[1, 0].set_xticklabels(['Mean Score', 'Max Score', 'Mean Steps'])
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Box plots for score comparison
    score_data = [results[agent]['scores'] for agent in agents]
    agent_labels = [agent.replace('_', '-').title() for agent in agents]
    
    box_plot = axes[1, 1].boxplot(score_data, labels=agent_labels, patch_artist=True)
    for patch, color in zip(box_plot['boxes'], colors[:len(agents)]):
        patch.set_facecolor(color)
    axes[1, 1].set_title('Score Distribution Comparison')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Performance statistics table
    axes[1, 2].axis('off')
    table_data = []
    for agent in agents:
        table_data.append([
            agent.replace('_', '-').title(),
            f"{results[agent]['mean_score']:.2f}",
            f"{results[agent]['std_score']:.2f}",
            f"{results[agent]['max_score']}",
            f"{results[agent]['mean_steps']:.1f}"
        ])
    
    table = axes[1, 2].table(cellText=table_data,
                            colLabels=['Agent', 'Mean Score', 'Std Score', 'Max Score', 'Mean Steps'],
                            cellLoc='center',
                            loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    axes[1, 2].set_title('Performance Statistics')
    
    plt.tight_layout()
    plt.savefig('data/performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_q_table_heatmap():
    """Visualize Q-table as heatmap for Q-Learning agent"""
    try:
        agent = QLearningAgent()
        agent.load_model('models/q_learning_final.pkl')
        
        # Extract Q-values for visualization
        states = list(agent.q_table.keys())[:50]  # Show first 50 states
        q_values_matrix = []
        
        for state in states:
            q_values_matrix.append(agent.q_table[state])
        
        if q_values_matrix:
            plt.figure(figsize=(12, 8))
            sns.heatmap(q_values_matrix, 
                       annot=True, 
                       fmt='.2f', 
                       cmap='RdYlBu_r',
                       xticklabels=['Straight', 'Right', 'Left'],
                       yticklabels=[f'State {i+1}' for i in range(len(states))])
            plt.title('Q-Values Heatmap (Sample States)')
            plt.ylabel('Game States')
            plt.xlabel('Actions')
            plt.tight_layout()
            plt.savefig('data/q_table_heatmap.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"Q-table contains {len(agent.q_table)} unique states")
        else:
            print("No Q-table data available")
            
    except Exception as e:
        print(f"Error creating Q-table heatmap: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize training results and agent performance')
    parser.add_argument('--mode', type=str, default='comparison', 
                       choices=['comparison', 'analysis', 'qtable', 'all'],
                       help='Visualization mode')
    parser.add_argument('--agent', type=str, default='both', 
                       choices=['q_learning', 'dqn', 'both'],
                       help='Which agent to analyze')
    parser.add_argument('--episodes', type=int, default=50,
                       help='Number of episodes for performance analysis')
    
    args = parser.parse_args()
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    if args.mode == 'comparison' or args.mode == 'all':
        print("Creating training comparison plots...")
        plot_training_comparison()
    
    if args.mode == 'analysis' or args.mode == 'all':
        print("Analyzing agent performance...")
        analyze_agent_performance(args.agent, args.episodes)
    
    if args.mode == 'qtable' or args.mode == 'all':
        print("Creating Q-table heatmap...")
        plot_q_table_heatmap()
    
    print("Visualization complete! Check the 'data' folder for saved plots.")