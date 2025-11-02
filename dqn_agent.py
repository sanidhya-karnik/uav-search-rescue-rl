import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import List, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm

class QNetwork(nn.Module):
    """Neural network for Q-value approximation"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256, 128]):
        super(QNetwork, self).__init__()
        
        layers = []
        prev_dim = state_dim
        
        # Hidden layers with ReLU
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer (linear)
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """Deep Q-Network Agent"""
    
    def __init__(self,
                 env,
                 hidden_dims: List[int] = [256, 256, 128],
                 learning_rate: float = 0.001,
                 gamma: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 buffer_size: int = 10000,
                 batch_size: int = 64,
                 target_update_freq: int = 100):
        """
        Initialize DQN agent
        
        Args:
            env: UAVEnvironment instance
            hidden_dims: List of hidden layer dimensions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon
            buffer_size: Replay buffer capacity
            batch_size: Batch size for training
            target_update_freq: How often to update target network
        """
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # State and action dimensions
        # State: one-hot(node) + battery + time + visited_mask
        self.n_nodes = len(env.nodes)
        self.state_dim = self.n_nodes + 2 + self.n_nodes  # one-hot + battery/time + visited
        self.action_dim = self.n_nodes
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Networks
        self.policy_net = QNetwork(self.state_dim, self.action_dim, hidden_dims).to(self.device)
        self.target_net = QNetwork(self.state_dim, self.action_dim, hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.epsilon_history = []
        self.losses = []
        self.update_counter = 0
    
    def encode_state(self, node_id: int, battery: float, time: float, visited: set) -> np.ndarray:
        """Encode state as one-hot vector + continuous features + visited mask"""
        state = np.zeros(self.state_dim)
        
        # One-hot encode current node
        state[node_id] = 1
        
        # Normalized battery and time
        state[self.n_nodes] = battery / self.env.max_battery
        state[self.n_nodes + 1] = time / self.env.time_limit
        
        # Visited mask (1 if visited, 0 otherwise)
        for v in visited:
            if v < self.n_nodes:
                state[self.n_nodes + 2 + v] = 1
        
        return state
    
    def select_action(self, state: np.ndarray, available_actions: List[int], 
                     training: bool = True) -> int:
        """Select action using epsilon-greedy policy with masking"""
        if training and random.random() < self.epsilon:
            return random.choice(available_actions)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]
            
            # Mask unavailable actions
            masked_q = np.full(self.action_dim, -np.inf)
            masked_q[available_actions] = q_values[available_actions]
            
            return int(np.argmax(masked_q))
    
    def train_step(self):
        """Perform one training step using experience replay"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q-values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Target Q-values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss
        loss = self.loss_fn(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()
        
        return loss.item()
    
    def train_episode(self) -> Tuple[float, int, bool]:
        """Run one training episode"""
        current_node = 0
        battery = self.env.max_battery
        time_remaining = self.env.time_limit
        visited = set()
        
        total_reward = 0
        steps = 0
        done = False
        success = False
        
        while not done and steps < 1000:
            # Get current state
            state = self.encode_state(current_node, battery, time_remaining, visited)
            
            # Get available actions
            available_actions = self.env.get_available_actions(
                current_node, battery, time_remaining, visited
            )
            
            if not available_actions:
                break
            
            # Select action
            action = self.select_action(state, available_actions, training=True)
            
            # Take action
            reward, new_battery, new_time, done = self.env.step(
                current_node, action, battery, time_remaining
            )
            
            # Update visited
            new_visited = visited.copy()
            if self.env.nodes[action].node_type == 'service':
                new_visited.add(action)
            
            # Get next state
            next_state = self.encode_state(action, new_battery, new_time, new_visited)
            
            # Store experience
            self.replay_buffer.push(state, action, reward, next_state, done)
            
            # Train
            loss = self.train_step()
            if loss is not None:
                self.losses.append(loss)
            
            # Update target network
            self.update_counter += 1
            if self.update_counter % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            total_reward += reward
            steps += 1
            
            if done and reward >= 0:
                success = True
            
            # Update state
            current_node = action
            battery = new_battery
            time_remaining = new_time
            visited = new_visited
            
            if reward < 0:
                break
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return total_reward, steps, success
    
    def train(self, n_episodes: int, verbose: bool = True):
        """Train the agent for multiple episodes"""
        iterator = tqdm(range(n_episodes)) if verbose else range(n_episodes)
        
        for episode in iterator:
            reward, steps, success = self.train_episode()
            
            self.episode_rewards.append(reward)
            self.episode_lengths.append(steps)
            self.epsilon_history.append(self.epsilon)
            
            if verbose and episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_loss = np.mean(self.losses[-100:]) if self.losses else 0
                iterator.set_description(
                    f"Ep {episode} | Reward: {avg_reward:.1f} | Loss: {avg_loss:.4f} | ε: {self.epsilon:.3f}"
                )
        
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'epsilon_history': self.epsilon_history,
            'losses': self.losses
        }
    
    def get_route(self, max_steps: int = 100) -> Tuple[List[int], float]:
        """Get greedy route using learned policy"""
        current_node = 0
        battery = self.env.max_battery
        time_remaining = self.env.time_limit
        visited = set()
        
        route = [current_node]
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < max_steps:
            state = self.encode_state(current_node, battery, time_remaining, visited)
            available_actions = self.env.get_available_actions(
                current_node, battery, time_remaining, visited
            )
            
            if not available_actions:
                break
            
            action = self.select_action(state, available_actions, training=False)
            
            reward, battery, time_remaining, done = self.env.step(
                current_node, action, battery, time_remaining
            )
            
            if reward < 0:
                break
            
            route.append(action)
            total_reward += reward
            
            if self.env.nodes[action].node_type == 'service':
                visited.add(action)
            
            current_node = action
            steps += 1
        
        return route, total_reward
    
    def save(self, filename: str):
        """Save model"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'losses': self.losses
        }, filename)
    
    def load(self, filename: str):
        """Load model"""
        checkpoint = torch.load(filename)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.episode_rewards = checkpoint['episode_rewards']
        self.losses = checkpoint['losses']
    
    def plot_training_progress(self, save_path: str = None):
        """Plot training statistics"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Rewards
        window = 100
        if len(self.episode_rewards) >= window:
            smoothed = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
            axes[0].plot(smoothed, linewidth=2)
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel(f'Avg Reward (window={window})')
        axes[0].set_title('Training Progress: Reward')
        axes[0].grid(True, alpha=0.3)
        
        # Loss
        if self.losses:
            if len(self.losses) >= window:
                smoothed = np.convolve(self.losses, np.ones(window)/window, mode='valid')
                axes[1].plot(smoothed, linewidth=2, color='red')
            axes[1].set_xlabel('Training Step')
            axes[1].set_ylabel(f'Avg Loss (window={window})')
            axes[1].set_title('Training Progress: Loss')
            axes[1].grid(True, alpha=0.3)
        
        # Epsilon
        axes[2].plot(self.epsilon_history, linewidth=2, color='orange')
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Epsilon')
        axes[2].set_title('Exploration Rate')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


if __name__ == "__main__":
    from uav_environment import UAVEnvironment
    
    print("Loading environment...")
    env = UAVEnvironment.load_from_file('env_20.json')
    
    print("Training DQN agent...")
    agent = DQNAgent(env)
    
    stats = agent.train(n_episodes=5000, verbose=True)
    
    agent.save('dqn_agent_20.pth')
    print("\nAgent saved")
    
    route, reward = agent.get_route()
    print(f"\nBest route: {route}")
    print(f"Total reward: {reward}")
    
    fig = agent.plot_training_progress('training_progress_dqn.png')
    plt.show()
    
    fig = env.visualize(routes=[route], title=f"DQN Route (Reward: {reward})")
    plt.savefig('route_dqn.png', dpi=150, bbox_inches='tight')
    plt.show()