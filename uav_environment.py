import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Set
import json

@dataclass
class Node:
    """Represents a node in the network"""
    id: int
    x: float
    y: float
    reward: float
    node_type: str  # 'depot', 'service', 'charging'
    
    def distance_to(self, other: 'Node') -> float:
        """Calculate Euclidean distance to another node"""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

class UAVEnvironment:
    """Environment for UAV Team Orienteering Problem with Charging Stations"""
    
    def __init__(self, 
                 n_service_nodes: int = 20,
                 n_charging_stations: int = 2,
                 map_size: int = 100,
                 time_limit: float = 100.0,
                 battery_limit: float = 50.0,
                 seed: int = None):
        """
        Initialize the UAV environment
        
        Args:
            n_service_nodes: Number of service nodes to visit
            n_charging_stations: Number of charging stations
            map_size: Size of the square map
            time_limit: Maximum time for each UAV
            battery_limit: Maximum battery capacity
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.n_service_nodes = n_service_nodes
        self.n_charging_stations = n_charging_stations
        self.map_size = map_size
        self.time_limit = time_limit
        self.battery_limit = battery_limit
        self.max_battery = battery_limit
        
        # Generate nodes
        self.nodes = self._generate_nodes()
        self.depot = self.nodes[0]
        self.destination = self.nodes[1]
        self.service_nodes = [n for n in self.nodes if n.node_type == 'service']
        self.charging_stations = [n for n in self.nodes if n.node_type == 'charging']
        
        # Precompute distance and cost matrices
        self.n_total_nodes = len(self.nodes)
        self.distance_matrix = self._compute_distance_matrix()
        self.time_cost_matrix = self.distance_matrix.copy()  # Assume speed = 1
        self.battery_cost_matrix = self.distance_matrix.copy()  # Assume battery consumption = distance
        
    def _generate_nodes(self) -> List[Node]:
        """Generate random nodes in the map"""
        nodes = []
        
        # Depot (origin) - center of map
        nodes.append(Node(
            id=0,
            x=self.map_size / 2,
            y=self.map_size / 2,
            reward=0,
            node_type='depot'
        ))
        
        # Destination - same as depot for closed tours
        nodes.append(Node(
            id=1,
            x=self.map_size / 2,
            y=self.map_size / 2,
            reward=0,
            node_type='depot'
        ))
        
        # Service nodes - random positions with random rewards
        for i in range(self.n_service_nodes):
            nodes.append(Node(
                id=2 + i,
                x=np.random.uniform(5, self.map_size - 5),
                y=np.random.uniform(5, self.map_size - 5),
                reward=np.random.randint(10, 50),
                node_type='service'
            ))
        
        # Charging stations - random positions
        for i in range(self.n_charging_stations):
            nodes.append(Node(
                id=2 + self.n_service_nodes + i,
                x=np.random.uniform(10, self.map_size - 10),
                y=np.random.uniform(10, self.map_size - 10),
                reward=0,
                node_type='charging'
            ))
        
        return nodes
    
    def _compute_distance_matrix(self) -> np.ndarray:
        """Compute distance matrix between all nodes"""
        n = len(self.nodes)
        dist_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist_matrix[i, j] = self.nodes[i].distance_to(self.nodes[j])
        
        return dist_matrix
    
    def get_state_features(self, current_node_id: int, battery: float, 
                          time_remaining: float, visited: Set[int]) -> dict:
        """Get state features for RL agent"""
        return {
            'current_node': current_node_id,
            'battery': battery,
            'time_remaining': time_remaining,
            'visited': visited.copy(),
            'battery_normalized': battery / self.max_battery,
            'time_normalized': time_remaining / self.time_limit
        }
    
    def get_available_actions(self, current_node_id: int, battery: float,
                             time_remaining: float, visited: Set[int]) -> List[int]:
        """Get list of feasible actions from current state"""
        available = []
        
        for node in self.nodes:
            # Skip already visited service nodes
            if node.node_type == 'service' and node.id in visited:
                continue
            
            # Skip current node
            if node.id == current_node_id:
                continue
            
            # Check if we can reach this node and return to depot
            time_to_node = self.time_cost_matrix[current_node_id, node.id]
            battery_to_node = self.battery_cost_matrix[current_node_id, node.id]
            
            if node.node_type == 'depot' and node.id == 1:  # destination
                # Can always try to return to depot if we have enough resources
                if battery_to_node <= battery and time_to_node <= time_remaining:
                    available.append(node.id)
            
            elif node.node_type == 'charging':
                # Can visit charging station if reachable
                time_back = self.time_cost_matrix[node.id, 1]
                battery_back = self.battery_cost_matrix[node.id, 1]
                
                if (time_to_node + time_back <= time_remaining and 
                    battery_to_node <= battery):
                    available.append(node.id)
            
            elif node.node_type == 'service':
                # For service nodes, check if we can visit and return
                time_back = self.time_cost_matrix[node.id, 1]
                battery_back = self.battery_cost_matrix[node.id, 1]
                
                if (time_to_node + time_back <= time_remaining and 
                    battery_to_node + battery_back <= battery):
                    available.append(node.id)
        
        return available
    
    def step(self, current_node_id: int, next_node_id: int, 
             battery: float, time_remaining: float) -> Tuple[float, float, float, bool]:
        """
        Take a step in the environment
        
        Returns:
            reward: Immediate reward received
            new_battery: Battery level after the step
            new_time: Time remaining after the step
            done: Whether episode is complete
        """
        time_cost = self.time_cost_matrix[current_node_id, next_node_id]
        battery_cost = self.battery_cost_matrix[current_node_id, next_node_id]
        
        new_time = time_remaining - time_cost
        new_battery = battery - battery_cost
        
        # Get reward
        next_node = self.nodes[next_node_id]
        reward = next_node.reward
        
        # Recharge at charging stations
        if next_node.node_type == 'charging':
            new_battery = self.max_battery
        
        # Check if done (returned to depot destination)
        done = (next_node_id == 1)
        
        # Penalize infeasible actions
        if new_battery < 0 or new_time < 0:
            reward = -1000
            done = True
        
        return reward, new_battery, new_time, done
    
    def save_to_file(self, filename: str):
        """Save environment configuration to file"""
        data = {
            'n_service_nodes': self.n_service_nodes,
            'n_charging_stations': self.n_charging_stations,
            'map_size': self.map_size,
            'time_limit': self.time_limit,
            'battery_limit': self.battery_limit,
            'nodes': [
                {
                    'id': n.id,
                    'x': n.x,
                    'y': n.y,
                    'reward': n.reward,
                    'type': n.node_type
                }
                for n in self.nodes
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filename: str) -> 'UAVEnvironment':
        """Load environment configuration from file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        env = cls(
            n_service_nodes=data['n_service_nodes'],
            n_charging_stations=data['n_charging_stations'],
            map_size=data['map_size'],
            time_limit=data['time_limit'],
            battery_limit=data['battery_limit']
        )
        
        # Override generated nodes with loaded ones
        env.nodes = [
            Node(
                id=n['id'],
                x=n['x'],
                y=n['y'],
                reward=n['reward'],
                node_type=n['type']
            )
            for n in data['nodes']
        ]
        
        env.depot = env.nodes[0]
        env.destination = env.nodes[1]
        env.service_nodes = [n for n in env.nodes if n.node_type == 'service']
        env.charging_stations = [n for n in env.nodes if n.node_type == 'charging']
        env.distance_matrix = env._compute_distance_matrix()
        env.time_cost_matrix = env.distance_matrix.copy()
        env.battery_cost_matrix = env.distance_matrix.copy()
        
        return env
    
    def visualize(self, routes: List[List[int]] = None, title: str = "UAV Environment"):
        """Visualize the environment and optionally routes"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot depot
        ax.scatter(self.depot.x, self.depot.y, c='red', s=300, marker='s', 
                  label='Depot', zorder=5, edgecolors='black', linewidth=2)
        
        # Plot service nodes
        service_x = [n.x for n in self.service_nodes]
        service_y = [n.y for n in self.service_nodes]
        service_rewards = [n.reward for n in self.service_nodes]
        
        scatter = ax.scatter(service_x, service_y, c=service_rewards, s=200, 
                           cmap='YlOrRd', label='Service Nodes', zorder=4,
                           edgecolors='black', linewidth=1)
        plt.colorbar(scatter, ax=ax, label='Reward')
        
        # Plot charging stations
        charging_x = [n.x for n in self.charging_stations]
        charging_y = [n.y for n in self.charging_stations]
        ax.scatter(charging_x, charging_y, c='green', s=250, marker='^',
                  label='Charging Stations', zorder=5, edgecolors='black', linewidth=2)
        
        # Add node labels
        for node in self.nodes:
            if node.node_type != 'depot':
                ax.annotate(f'{node.id}', (node.x, node.y), 
                          fontsize=8, ha='center', va='center')
        
        # Plot routes if provided
        if routes:
            colors = plt.cm.tab10(np.linspace(0, 1, len(routes)))
            for i, route in enumerate(routes):
                if len(route) > 1:
                    route_x = [self.nodes[node_id].x for node_id in route]
                    route_y = [self.nodes[node_id].y for node_id in route]
                    ax.plot(route_x, route_y, 'o-', linewidth=2, markersize=8,
                           color=colors[i], alpha=0.6, label=f'UAV {i+1}')
        
        ax.set_xlim(0, self.map_size)
        ax.set_ylim(0, self.map_size)
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_title(title)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        return fig


if __name__ == "__main__":
    # Create sample environments
    print("Creating sample environments...")
    
    # Small instance (20 nodes)
    env_20 = UAVEnvironment(n_service_nodes=20, n_charging_stations=2, seed=42)
    env_20.save_to_file('env_20.json')
    print(f"Created environment with {env_20.n_service_nodes} service nodes")
    
    # Medium instance (50 nodes)
    env_50 = UAVEnvironment(n_service_nodes=50, n_charging_stations=5, 
                           time_limit=150, seed=43)
    env_50.save_to_file('env_50.json')
    print(f"Created environment with {env_50.n_service_nodes} service nodes")
    
    # Large instance (100 nodes)
    env_100 = UAVEnvironment(n_service_nodes=100, n_charging_stations=10,
                            time_limit=200, seed=44)
    env_100.save_to_file('env_100.json')
    print(f"Created environment with {env_100.n_service_nodes} service nodes")
    
    # Visualize the small environment
    fig = env_20.visualize(title="TOPCS-20: Sample Environment")
    plt.savefig('env_20_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nEnvironment files saved successfully!")