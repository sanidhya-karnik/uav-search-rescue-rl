import numpy as np
from collections import defaultdict
from typing import List, Tuple, Set, Dict
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

class QLearningNDTS:
    """Q-Learning with Non-Decreasing Tree Search Update"""
    
    def __init__(self,
                 env,
                 battery_bins: int = 10,
                 time_bins: int = 10,
                 alpha: float = 0.1,
                 gamma: float = 0.95,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.9995,
                 epsilon_min: float = 0.01):
        """
        Initialize Q-Learning agent with NDTS
        
        Args:
            env: UAVEnvironment instance
            battery_bins: Number of bins to discretize battery
            time_bins: Number of bins to discretize time
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon value
        """
        self.env = env
        self.battery_bins = battery_bins
        self.time_bins = time_bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: dictionary mapping (state, action) to Q-value
        self.Q = defaultdict(float)
        
        # Statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.epsilon_history = []
        
    def discretize_battery(self, battery: float) -> int:
        """Discretize continuous battery value"""
        normalized = battery / self.env.max_battery
        bin_idx = int(normalized * self.battery_bins)
        return min(bin_idx, self.battery_bins - 1)
    
    def discretize_time(self, time: float) -> int:
        """Discretize continuous time value"""
        normalized = time / self.env.time_limit
        bin_idx = int(normalized * self.time_bins)
        return min(bin_idx, self.time_bins - 1)
    
    def get_state_key(self, node_id: int, battery: float, time: float, 
                     visited: Set[int]) -> Tuple:
        """Convert state to hashable key for Q-table"""
        battery_bin = self.discretize_battery(battery)
        time_bin = self.discretize_time(time)
        visited_tuple = tuple(sorted(visited))
        return (node_id, battery_bin, time_bin, visited_tuple)
    
    def select_action(self, state_key: Tuple, available_actions: List[int], 
                     training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.choice(available_actions)
        else:
            # Exploit: best action based on Q-values
            q_values = [self.Q[(state_key, a)] for a in available_actions]
            max_q = max(q_values)
            # Handle ties randomly
            best_actions = [a for a, q in zip(available_actions, q_values) if q == max_q]
            return np.random.choice(best_actions)
    
    def train_episode(self) -> Tuple[float, int, bool]:
        """
        Run one training episode with NDTS update
        
        Returns:
            total_reward: Total reward collected
            steps: Number of steps taken
            success: Whether episode completed successfully
        """
        # Initialize episode
        current_node = 0  # Start at depot
        battery = self.env.max_battery
        time_remaining = self.env.time_limit
        visited = set()
        
        # Store trajectory for backward update
        trajectory = []
        total_reward = 0
        steps = 0
        done = False
        success = False
        
        # Episode loop
        while not done and steps < 1000:  # Max steps to prevent infinite loops
            # Get current state
            state_key = self.get_state_key(current_node, battery, time_remaining, visited)
            
            # Get available actions (masking)
            available_actions = self.env.get_available_actions(
                current_node, battery, time_remaining, visited
            )
            
            if not available_actions:
                # No feasible actions, force return to depot
                next_node = 1
            else:
                # Select action
                next_node = self.select_action(state_key, available_actions, training=True)
            
            # Take action
            reward, new_battery, new_time, done = self.env.step(
                current_node, next_node, battery, time_remaining
            )
            
            # Update visited set
            new_visited = visited.copy()
            if self.env.nodes[next_node].node_type == 'service':
                new_visited.add(next_node)
            
            # Store experience
            trajectory.append({
                'state': state_key,
                'action': next_node,
                'reward': reward,
                'next_state': self.get_state_key(next_node, new_battery, new_time, new_visited),
                'next_node': next_node,
                'done': done,
                'available_actions': available_actions
            })
            
            total_reward += reward
            steps += 1
            
            # Check for success (returned to depot without penalty)
            if done and reward >= 0:
                success = True
            
            # Move to next state
            current_node = next_node
            battery = new_battery
            time_remaining = new_time
            visited = new_visited
            
            # Early termination on infeasible action
            if reward < 0:
                done = True
                break
        
        # NDTS: Backward replay with non-decreasing update
        self._ndts_update(trajectory)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return total_reward, steps, success
    
    def _ndts_update(self, trajectory: List[Dict]):
        """
        Non-Decreasing Tree Search Q-value update
        Updates Q-values backward through trajectory
        """
        if not trajectory:
            return
        
        # Backward update
        for i in range(len(trajectory) - 1, -1, -1):
            exp = trajectory[i]
            state = exp['state']
            action = exp['action']
            reward = exp['reward']
            next_state = exp['next_state']
            done = exp['done']
            
            # Current Q-value
            current_q = self.Q[(state, action)]
            
            if done:
                # Terminal state
                target_q = reward
            else:
                # Get max Q-value for next state among available actions
                # Need to get available actions for next state
                next_node = exp['next_node']
                
                # Get next available actions (excluding visited service nodes)
                next_visited = set()
                for j in range(i + 1):
                    if trajectory[j]['reward'] > 0:  # Service node visited
                        next_visited.add(trajectory[j]['action'])
                
                # Approximate next available actions
                next_available = [a for a in range(len(self.env.nodes)) 
                                if a not in next_visited or 
                                self.env.nodes[a].node_type in ['charging', 'depot']]
                
                if next_available:
                    max_next_q = max([self.Q[(next_state, a)] for a in next_available])
                else:
                    max_next_q = 0
                
                target_q = reward + self.gamma * max_next_q
            
            # Non-decreasing update: only update if target is higher
            if target_q > current_q:
                self.Q[(state, action)] = current_q + self.alpha * (target_q - current_q)
    
    def train(self, n_episodes: int, verbose: bool = True) -> Dict:
        """
        Train the agent for multiple episodes
        
        Args:
            n_episodes: Number of training episodes
            verbose: Whether to show progress bar
            
        Returns:
            Dictionary with training statistics
        """
        iterator = tqdm(range(n_episodes)) if verbose else range(n_episodes)
        
        for episode in iterator:
            reward, steps, success = self.train_episode()
            
            self.episode_rewards.append(reward)
            self.episode_lengths.append(steps)
            self.epsilon_history.append(self.epsilon)
            
            if verbose and episode % 1000 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                iterator.set_description(
                    f"Ep {episode} | Avg Reward: {avg_reward:.1f} | ε: {self.epsilon:.3f}"
                )
        
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'epsilon_history': self.epsilon_history
        }
    
    def get_route(self, max_steps: int = 100) -> Tuple[List[int], float]:
        """
        Get greedy route using learned policy
        
        Returns:
            route: List of node IDs visited
            total_reward: Total reward collected
        """
        current_node = 0
        battery = self.env.max_battery
        time_remaining = self.env.time_limit
        visited = set()
        
        route = [current_node]
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < max_steps:
            state_key = self.get_state_key(current_node, battery, time_remaining, visited)
            available_actions = self.env.get_available_actions(
                current_node, battery, time_remaining, visited
            )
            
            if not available_actions:
                break
            
            # Greedy action selection
            next_node = self.select_action(state_key, available_actions, training=False)
            
            reward, battery, time_remaining, done = self.env.step(
                current_node, next_node, battery, time_remaining
            )
            
            if reward < 0:  # Infeasible
                break
            
            route.append(next_node)
            total_reward += reward
            
            if self.env.nodes[next_node].node_type == 'service':
                visited.add(next_node)
            
            current_node = next_node
            steps += 1
        
        return route, total_reward
    
    def save(self, filename: str):
        """Save Q-table to file"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'Q': dict(self.Q),
                'battery_bins': self.battery_bins,
                'time_bins': self.time_bins,
                'episode_rewards': self.episode_rewards,
                'episode_lengths': self.episode_lengths
            }, f)
    
    def load(self, filename: str):
        """Load Q-table from file"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.Q = defaultdict(float, data['Q'])
            self.battery_bins = data['battery_bins']
            self.time_bins = data['time_bins']
            self.episode_rewards = data['episode_rewards']
            self.episode_lengths = data['episode_lengths']
    
    def plot_training_progress(self, save_path: str = None):
        """Plot training statistics"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot rewards
        window = 100
        if len(self.episode_rewards) >= window:
            smoothed_rewards = np.convolve(
                self.episode_rewards, 
                np.ones(window)/window, 
                mode='valid'
            )
            axes[0].plot(smoothed_rewards, linewidth=2)
            axes[0].set_xlabel('Episode')
            axes[0].set_ylabel(f'Average Reward (window={window})')
            axes[0].set_title('Training Progress: Reward')
            axes[0].grid(True, alpha=0.3)
        
        # Plot epsilon
        axes[1].plot(self.epsilon_history, linewidth=2, color='orange')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Epsilon')
        axes[1].set_title('Exploration Rate Decay')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


if __name__ == "__main__":
    from uav_environment import UAVEnvironment
    
    # Load environment
    print("Loading environment...")
    env = UAVEnvironment.load_from_file('env_20.json')
    
    # Create and train agent
    print("Training Q-Learning agent with NDTS...")
    agent = QLearningNDTS(env, battery_bins=10, time_bins=10)
    
    stats = agent.train(n_episodes=10000, verbose=True)
    
    # Save trained agent
    agent.save('q_learning_ndts_20.pkl')
    print("\nAgent saved to q_learning_ndts_20.pkl")
    
    # Get best route
    route, reward = agent.get_route()
    print(f"\nBest route found: {route}")
    print(f"Total reward: {reward}")
    
    # Plot training progress
    fig = agent.plot_training_progress(save_path='training_progress_ndts.png')
    plt.show()
    
    # Visualize route
    fig = env.visualize(routes=[route], title=f"Q-Learning NDTS Route (Reward: {reward})")
    plt.savefig('route_ndts.png', dpi=150, bbox_inches='tight')
    plt.show()