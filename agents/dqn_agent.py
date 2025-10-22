import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

class DQNAgent:
    def __init__(self, state_size=11, action_size=3, learning_rate=0.001, discount_factor=0.95, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, memory_size=10000, batch_size=32):
        """
        Deep Q-Network agent for Snake game
        
        Args:
            state_size: Size of the state space
            action_size: Size of the action space (3 for snake: straight, right, left)
            learning_rate: Learning rate for neural network
            discount_factor: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
            memory_size: Size of experience replay buffer
            batch_size: Batch size for training
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory_size = memory_size
        self.batch_size = batch_size
        
        # Experience replay buffer
        self.memory = deque(maxlen=memory_size)
        
        # Neural networks
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        
        # Update target network initially
        self.update_target_network()
        
        # Training statistics
        self.training_scores = []
        self.training_steps = []
        self.losses = []
    
    def _build_network(self):
        """Build the neural network for Q-value approximation"""
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(self.state_size,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                     loss='mse')
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        action_idx = np.argmax(action)
        self.memory.append((state, action_idx, reward, next_state, done))
    
    def get_action(self, state, training=True):
        """
        Get action using epsilon-greedy policy
        
        Args:
            state: Current game state
            training: Whether in training mode (affects epsilon usage)
            
        Returns:
            Action as one-hot encoded array
        """
        if training and random.random() < self.epsilon:
            # Random action (exploration)
            action_idx = random.randint(0, self.action_size - 1)
        else:
            # Best action from neural network (exploitation)
            state_tensor = tf.expand_dims(state, 0)
            q_values = self.q_network(state_tensor, training=False)
            action_idx = np.argmax(q_values[0])
        
        # Convert to one-hot encoding
        action = np.zeros(self.action_size)
        action[action_idx] = 1
        return action
    
    def replay(self):
        """Train the neural network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample random batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        # Get current Q-values
        current_q_values = self.q_network(states)
        
        # Get next Q-values from target network
        next_q_values = self.target_network(next_states)
        max_next_q_values = np.max(next_q_values, axis=1)
        
        # Calculate target Q-values
        target_q_values = current_q_values.numpy()
        for i in range(self.batch_size):
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i]
            else:
                target_q_values[i][actions[i]] = rewards[i] + self.discount_factor * max_next_q_values[i]
        
        # Train the network
        history = self.q_network.fit(states, target_q_values, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        self.losses.append(loss)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def save_model(self, filepath):
        """Save the trained model"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the main network
        self.q_network.save(f"{filepath}_q_network.h5")
        self.target_network.save(f"{filepath}_target_network.h5")
        
        # Save additional parameters
        import pickle
        params = {
            'epsilon': self.epsilon,
            'training_scores': self.training_scores,
            'training_steps': self.training_steps,
            'losses': self.losses,
            'hyperparameters': {
                'state_size': self.state_size,
                'action_size': self.action_size,
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'epsilon_decay': self.epsilon_decay,
                'epsilon_min': self.epsilon_min,
                'memory_size': self.memory_size,
                'batch_size': self.batch_size
            }
        }
        
        with open(f"{filepath}_params.pkl", 'wb') as f:
            pickle.dump(params, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        try:
            # Load networks
            self.q_network = keras.models.load_model(f"{filepath}_q_network.h5")
            self.target_network = keras.models.load_model(f"{filepath}_target_network.h5")
            
            # Load parameters
            import pickle
            with open(f"{filepath}_params.pkl", 'rb') as f:
                params = pickle.load(f)
            
            self.epsilon = params.get('epsilon', self.epsilon_min)
            self.training_scores = params.get('training_scores', [])
            self.training_steps = params.get('training_steps', [])
            self.losses = params.get('losses', [])
            
            print(f"Model loaded from {filepath}")
            print(f"Current epsilon: {self.epsilon:.4f}")
            print(f"Training games: {len(self.training_scores)}")
            
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
            'average_loss': np.mean(self.losses) if self.losses else 0,
            'memory_size': len(self.memory),
            'current_epsilon': self.epsilon
        }
    
    def clear_memory(self):
        """Clear the experience replay buffer"""
        self.memory.clear()
        print("Experience replay buffer cleared")
