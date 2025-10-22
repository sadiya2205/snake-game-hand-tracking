import numpy as np
import random
import pickle
from collections import defaultdict

class QLearningAgent:
    def __init__(self, state_size=11, action_size=3, learning_rate=0.1, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """
        Q-Learning agent for Snake game
        
        Args:
            state_size: Size of the state space
            action_size: Size of the action space (3 for snake: straight, right, left)
            learning_rate: Learning rate for Q-value updates
            discount_factor: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table as defaultdict for dynamic state discovery
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        
        # Training statistics
        self.training_scores = []
        self.training_steps = []
    
    def _state_to_string(self, state):
        """Convert state array to string for Q-table indexing"""
        return str(state.tolist())
    
    def get_action(self, state, training=True):
        """
        Get action using epsilon-greedy policy
        
        Args:
            state: Current game state
            training: Whether in training mode (affects epsilon usage)
            
        Returns:
            Action as one-hot encoded array
        """
        state_key = self._state_to_string(state)
        
        # Epsilon-greedy action selection
        if training and random.random() < self.epsilon:
            # Random action (exploration)
            action_idx = random.randint(0, self.action_size - 1)
        else:
            # Best action from Q-table (exploitation)
            q_values = self.q_table[state_key]
            action_idx = np.argmax(q_values)
        
        # Convert to one-hot encoding
        action = np.zeros(self.action_size)
        action[action_idx] = 1
        return action
    
    def update_q_table(self, state, action, reward, next_state, done):
        """
        Update Q-table using Q-learning update rule
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is finished
        """
        state_key = self._state_to_string(state)
        next_state_key = self._state_to_string(next_state)
        action_idx = np.argmax(action)
        
        # Q-learning update rule
        current_q = self.q_table[state_key][action_idx]
        
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[next_state_key])
            target_q = reward + self.discount_factor * max_next_q
        
        # Update Q-value
        self.q_table[state_key][action_idx] = current_q + self.learning_rate * (target_q - current_q)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, filepath):
        """Save Q-table and agent parameters"""
        model_data = {
            'q_table': dict(self.q_table),
            'epsilon': self.epsilon,
            'training_scores': self.training_scores,
            'training_steps': self.training_steps,
            'hyperparameters': {
                'state_size': self.state_size,
                'action_size': self.action_size,
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'epsilon_decay': self.epsilon_decay,
                'epsilon_min': self.epsilon_min
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load Q-table and agent parameters"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Restore Q-table
            self.q_table = defaultdict(lambda: np.zeros(self.action_size))
            self.q_table.update(model_data['q_table'])
            
            # Restore parameters
            self.epsilon = model_data.get('epsilon', self.epsilon_min)
            self.training_scores = model_data.get('training_scores', [])
            self.training_steps = model_data.get('training_steps', [])
            
            print(f"Model loaded from {filepath}")
            print(f"Q-table size: {len(self.q_table)} states")
            print(f"Current epsilon: {self.epsilon:.4f}")
            
        except FileNotFoundError:
            print(f"No saved model found at {filepath}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def get_stats(self):
        """Get training statistics"""
        if not self.training_scores:
            return "No training data available"
        
        return {
            'games_played': len(self.training_scores),
            'average_score': np.mean(self.training_scores),
            'max_score': np.max(self.training_scores),
            'q_table_size': len(self.q_table),
            'current_epsilon': self.epsilon
        }
    
    def print_q_table_sample(self, num_states=5):
        """Print a sample of Q-table for inspection"""
        print(f"\nQ-Table Sample (showing {min(num_states, len(self.q_table))} states):")
        print("-" * 60)
        
        for i, (state_key, q_values) in enumerate(list(self.q_table.items())[:num_states]):
            print(f"State {i+1}: {state_key}")
            print(f"Q-values: {q_values}")
            print(f"Best action: {np.argmax(q_values)}")
            print("-" * 30)