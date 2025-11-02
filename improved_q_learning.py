import numpy as np
from collections import defaultdict
from typing import List, Tuple, Set, Dict
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

class ImprovedQLearningNDTS:
    """Improved Q-Learning with Reward-Biased Exploration and Better Reward Shaping"""
    
    def __init__(self,
                 env,
                 battery_bins: int = 10,
                 time_bins: int = 10,
                 alpha: float = 0.1,
                 gamma: float = 0.95,
                 epsilon: float = 0.3,
                 epsilon_decay: float = 0.9995,
                 epsilon_min: float = 0.01,
                 reward_bias: float = 0.5):
        """
        Initialize Improved Q-Learning agent
        
        Args:
            env: UAVEnvironment instance
            battery_bins: Number of bins to discretize battery
            time_bins: Number of bins to discretize time
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon value
            reward_bias: Weight for reward-biased exploration (0-1)
        """
        self.env = env
        self.battery_bins = battery_bins
        self.time_bins = time_bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.reward_bias = reward_bias
        
        # Q-table
        self.Q = defaultdict(float)
        
        # Statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.epsilon_history = []
        
        # Precompute node rewards for faster access
        self.node_rewards = {node.id: node.reward for node in env.nodes}
    
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
    
    def select_action_reward_biased(self, state_key: Tuple, available_actions: List[int],
                                   current_node: int, battery: float, 
                                   time_remaining: float) -> int:
        """
        Select action with reward-biased exploration
        Combines Q-values with node rewards and feasibility
        """
        if np.random.random() < self.epsilon:
            # Reward-biased exploration instead of pure random
            return self._reward_biased_sample(available_actions, current_node, 
                                             battery, time_remaining)
        else:
            # Exploitation with reward consideration
            return self._best_action_with_reward_tie_break(state_key, available_actions)
    
    def _reward_biased_sample(self, available_actions: List[int], 
                             current_node: int, battery: float, 
                             time_remaining: float) -> int:
        """Sample action biased by node rewards and feasibility"""
        if not available_actions:
            return 1  # Return to depot
        
        # Calculate scores for each action
        scores = []
        for action in available_actions:
            node = self.env.nodes[action]
            
            # Base score from reward
            reward_score = node.reward if node.node_type == 'service' else 0
            
            # Distance factor (prefer closer nodes early, farther nodes later)
            distance = self.env.distance_matrix[current_node, action]
            distance_score = 1.0 / (1.0 + distance / 10)
            
            # Feasibility margin (prefer actions with more battery/time margin)
            battery_margin = battery - self.env.battery_cost_matrix[current_node, action]
            time_margin = time_remaining - self.env.time_cost_matrix[current_node, action]
            feasibility_score = min(battery_margin, time_margin) / 10
            
            # Combined score
            total_score = (self.reward_bias * reward_score + 
                          (1 - self.reward_bias) * distance_score +
                          0.1 * feasibility_score)
            
            scores.append(max(total_score, 0.1))  # Ensure positive
        
        # Sample proportional to scores
        scores = np.array(scores)
        probabilities = scores / scores.sum()
        
        return np.random.choice(available_actions, p=probabilities)
    
    def _best_action_with_reward_tie_break(self, state_key: Tuple, 
                                          available_actions: List[int]) -> int:
        """Select best action, breaking ties with node rewards"""
        q_values = [(a, self.Q[(state_key, a)]) for a in available_actions]
        max_q = max(q for _, q in q_values)
        
        # Get all actions with max Q-value
        best_actions = [a for a, q in q_values if abs(q - max_q) < 1e-6]
        
        if len(best_actions) == 1:
            return best_actions[0]
        
        # Break ties by preferring higher rewards
        best_action = max(best_actions, key=lambda a: self.node_rewards.get(a, 0))
        return best_action
    
    def train_episode(self) -> Tuple[float, int, bool]:
        """Run one training episode with NDTS update"""
        current_node = 0
        battery = self.env.max_battery
        time_remaining = self.env.time_limit
        visited = set()
        
        trajectory = []
        total_reward = 0
        steps = 0
        done = False
        success = False
        
        while not done and steps < 1000:
            state_key = self.get_state_key(current_node, battery, time_remaining, visited)
            
            available_actions = self.env.get_available_actions(
                current_node, battery, time_remaining, visited
            )
            
            if not available_actions:
                next_node = 1
            else:
                # Use reward-biased action selection
                next_node = self.select_action_reward_biased(
                    state_key, available_actions, current_node, battery, time_remaining
                )
            
            reward, new_battery, new_time, done = self.env.step(
                current_node, next_node, battery, time_remaining
            )
            
            new_visited = visited.copy()
            if self.env.nodes[next_node].node_type == 'service':
                new_visited.add(next_node)
            
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
            
            if done and reward >= 0:
                success = True
            
            current_node = next_node
            battery = new_battery
            time_remaining = new_time
            visited = new_visited
            
            if reward < 0:
                done = True
                break
        
        # NDTS update
        self._ndts_update(trajectory)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return total_reward, steps, success
    
    def _ndts_update(self, trajectory: List[Dict]):
        """Non-Decreasing Tree Search Q-value update"""
        if not trajectory:
            return
        
        for i in range(len(trajectory) - 1, -1, -1):
            exp = trajectory[i]
            state = exp['state']
            action = exp['action']
            reward = exp['reward']
            next_state = exp['next_state']
            done = exp['done']
            
            current_q = self.Q[(state, action)]
            
            if done:
                target_q = reward
            else:
                next_node = exp['next_node']
                next_visited = set()
                for j in range(i + 1):
                    if trajectory[j]['reward'] > 0:
                        next_visited.add(trajectory[j]['action'])
                
                next_available = [a for a in range(len(self.env.nodes)) 
                                if a not in next_visited or 
                                self.env.nodes[a].node_type in ['charging', 'depot']]
                
                if next_available:
                    max_next_q = max([self.Q[(next_state, a)] for a in next_available])
                else:
                    max_next_q = 0
                
                target_q = reward + self.gamma * max_next_q
            
            # Non-decreasing update
            if target_q > current_q:
                self.Q[(state, action)] = current_q + self.alpha * (target_q - current_q)
    
    def train(self, n_episodes: int, verbose: bool = True) -> Dict:
        """Train the agent"""
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
    
    def get_route(self, max_steps: int = 100, greedy: bool = True) -> Tuple[List[int], float]:
        """Get route using learned policy with optional reward-biased selection"""
        current_node = 0
        battery = self.env.max_battery
        time_remaining = self.env.time_limit
        visited = set()
        
        route = [current_node]
        total_reward = 0
        steps = 0
        done = False
        
        # Use very low epsilon for deployment (still slightly exploratory)
        temp_epsilon = self.epsilon
        self.epsilon = 0.0 if greedy else 0.05
        
        while not done and steps < max_steps:
            state_key = self.get_state_key(current_node, battery, time_remaining, visited)
            available_actions = self.env.get_available_actions(
                current_node, battery, time_remaining, visited
            )
            
            if not available_actions:
                break
            
            # Use reward-biased selection even during deployment
            next_node = self._best_action_with_reward_tie_break(state_key, available_actions)
            
            reward, battery, time_remaining, done = self.env.step(
                current_node, next_node, battery, time_remaining
            )
            
            if reward < 0:
                break
            
            route.append(next_node)
            total_reward += reward
            
            if self.env.nodes[next_node].node_type == 'service':
                visited.add(next_node)
            
            current_node = next_node
            steps += 1
        
        self.epsilon = temp_epsilon
        return route, total_reward
    
    def save(self, filename: str):
        """Save Q-table to file"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'Q': dict(self.Q),
                'battery_bins': self.battery_bins,
                'time_bins': self.time_bins,
                'episode_rewards': self.episode_rewards
            }, f)
    
    def load(self, filename: str):
        """Load Q-table from file"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.Q = defaultdict(float, data['Q'])
            self.battery_bins = data['battery_bins']
            self.time_bins = data['time_bins']
            self.episode_rewards = data['episode_rewards']