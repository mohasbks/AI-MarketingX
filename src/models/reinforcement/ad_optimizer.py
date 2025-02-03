import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import gym
from gym import spaces
from collections import deque
import random
from typing import Dict, List, Tuple
import logging

class AdEnvironment(gym.Env):
    """Custom Environment for Ad Campaign Optimization"""
    def __init__(self, initial_budget: float, initial_metrics: Dict):
        super().__init__()
        
        # Define action space (budget adjustment, bid adjustment, targeting adjustment)
        self.action_space = spaces.Box(
            low=np.array([-0.5, -0.5, -0.5]),  # Maximum 50% decrease
            high=np.array([0.5, 0.5, 0.5]),    # Maximum 50% increase
            dtype=np.float32
        )
        
        # Define observation space
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0]),  # CTR, CPC, Conv Rate, ROI, Budget, Time
            high=np.array([1, 1000, 1, 10, 1000000, 24]),
            dtype=np.float32
        )
        
        self.initial_budget = initial_budget
        self.initial_metrics = initial_metrics
        self.reset()
        
    def reset(self):
        """Reset environment to initial state"""
        self.budget = self.initial_budget
        self.current_metrics = self.initial_metrics.copy()
        self.steps = 0
        return self._get_state()
        
    def _get_state(self):
        """Get current state observation"""
        return np.array([
            self.current_metrics['ctr'],
            self.current_metrics['cpc'],
            self.current_metrics['conversion_rate'],
            self.current_metrics['roi'],
            self.budget,
            self.steps % 24  # Time of day
        ])
        
    def step(self, action):
        """Execute action and return new state, reward, done, info"""
        # Apply actions
        budget_change = action[0]
        bid_change = action[1]
        targeting_change = action[2]
        
        # Update budget
        self.budget *= (1 + budget_change)
        
        # Simulate effect of actions on metrics
        self.current_metrics = self._simulate_metric_changes(
            budget_change, bid_change, targeting_change
        )
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Update step count
        self.steps += 1
        
        # Check if episode is done
        done = self.steps >= 24  # One day of optimization
        
        return self._get_state(), reward, done, {}
        
    def _simulate_metric_changes(
        self,
        budget_change: float,
        bid_change: float,
        targeting_change: float
    ) -> Dict:
        """Simulate how actions affect metrics"""
        new_metrics = self.current_metrics.copy()
        
        # Simulate CTR changes
        new_metrics['ctr'] *= (1 + targeting_change * 0.5)
        new_metrics['ctr'] = min(max(new_metrics['ctr'], 0.001), 0.2)
        
        # Simulate CPC changes
        new_metrics['cpc'] *= (1 + bid_change)
        new_metrics['cpc'] = min(max(new_metrics['cpc'], 0.1), 100.0)
        
        # Simulate conversion rate changes
        new_metrics['conversion_rate'] *= (1 + targeting_change * 0.3)
        new_metrics['conversion_rate'] = min(max(new_metrics['conversion_rate'], 0.001), 0.2)
        
        # Calculate ROI
        new_metrics['roi'] = (
            new_metrics['ctr'] *
            new_metrics['conversion_rate'] *
            100 /  # Average conversion value
            new_metrics['cpc']
        )
        
        return new_metrics
        
    def _calculate_reward(self) -> float:
        """Calculate reward based on ROI and budget efficiency"""
        roi_component = self.current_metrics['roi'] - self.initial_metrics['roi']
        efficiency_component = (
            1 - abs(self.budget - self.initial_budget) / self.initial_budget
        )
        return roi_component + efficiency_component

class DQNAgent:
    """Deep Q-Network Agent for Ad Optimization"""
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.001,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        memory_size: int = 10000
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = deque(maxlen=memory_size)
        
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
    def _build_model(self) -> models.Model:
        """Build neural network model"""
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_dim=self.state_size),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        return model
        
    def update_target_model(self):
        """Update target model weights"""
        self.target_model.set_weights(self.model.get_weights())
        
    def remember(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state: np.ndarray) -> np.ndarray:
        """Choose action based on epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            return self.action_space.sample()
            
        act_values = self.model.predict(state.reshape(1, -1))
        return np.argmax(act_values[0])
        
    def replay(self, batch_size: int):
        """Train model on experiences"""
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.discount_factor * \
                    np.amax(self.target_model.predict(next_state.reshape(1, -1))[0])
                    
            target_f = self.model.predict(state.reshape(1, -1))
            target_f[0][action] = target
            
            self.model.fit(
                state.reshape(1, -1),
                target_f,
                epochs=1,
                verbose=0
            )
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def save_model(self, filepath: str):
        """Save model weights"""
        self.model.save_weights(filepath)
        
    def load_model(self, filepath: str):
        """Load model weights"""
        self.model.load_weights(filepath)

class AdCampaignOptimizer:
    """Main class for optimizing ad campaigns using reinforcement learning"""
    def __init__(
        self,
        initial_budget: float,
        initial_metrics: Dict,
        learning_rate: float = 0.001,
        episodes: int = 1000
    ):
        self.env = AdEnvironment(initial_budget, initial_metrics)
        self.agent = DQNAgent(
            state_size=6,  # State space size
            action_size=3,  # Action space size
            learning_rate=learning_rate
        )
        self.episodes = episodes
        
    def train(self, batch_size: int = 32) -> List[float]:
        """Train the optimization agent"""
        scores = []
        
        for episode in range(self.episodes):
            state = self.env.reset()
            total_reward = 0
            
            for time in range(24):  # 24 hours in a day
                action = self.agent.act(state)
                next_state, reward, done, _ = self.env.step(action)
                
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
                if done:
                    break
                    
            self.agent.replay(batch_size)
            
            if episode % 10 == 0:
                self.agent.update_target_model()
                logging.info(f"Episode: {episode}, Score: {total_reward}")
                
            scores.append(total_reward)
            
        return scores
        
    def optimize_campaign(self, current_state: np.ndarray) -> Tuple[np.ndarray, float]:
        """Generate optimization recommendations for current campaign state"""
        action = self.agent.act(current_state)
        next_state, reward, _, _ = self.env.step(action)
        
        return action, reward
        
    def save_agent(self, filepath: str):
        """Save trained agent"""
        self.agent.save_model(filepath)
        
    def load_agent(self, filepath: str):
        """Load trained agent"""
        self.agent.load_model(filepath)
