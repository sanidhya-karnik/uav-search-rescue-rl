import numpy as np
from sklearn.cluster import KMeans
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from q_learning_ndts import QLearningNDTS
from uav_environment import UAVEnvironment, Node
import copy

class TwoPhaseApproach:
    """Cluster First Q-learning Second (CFQS) approach for TOP"""
    
    def __init__(self, env: UAVEnvironment, n_uavs: int):
        """
        Initialize two-phase approach
        
        Args:
            env: Original UAVEnvironment
            n_uavs: Number of UAVs (clusters)
        """
        self.env = env
        self.n_uavs = n_uavs
        self.clusters = None
        self.cluster_envs = []
        self.agents = []
        self.routes = []
        self.cluster_assignments = None
    
    def phase1_clustering(self, method: str = 'kmeans'):
        """
        Phase 1: Cluster service nodes
        
        Args:
            method: Clustering method ('kmeans', 'random', etc.)
        """
        print(f"Phase 1: Clustering {len(self.env.service_nodes)} nodes into {self.n_uavs} clusters...")
        
        # Extract service node coordinates
        service_coords = np.array([[n.x, n.y] for n in self.env.service_nodes])
        
        if method == 'kmeans':
            # K-means clustering based on Euclidean distance
            kmeans = KMeans(n_clusters=self.n_uavs, random_state=42, n_init=10)
            labels = kmeans.fit_predict(service_coords)
            self.cluster_assignments = labels
            
        elif method == 'random':
            # Random assignment
            self.cluster_assignments = np.random.randint(0, self.n_uavs, len(self.env.service_nodes))
        
        # Create clusters
        self.clusters = [[] for _ in range(self.n_uavs)]
        for i, node in enumerate(self.env.service_nodes):
            cluster_id = self.cluster_assignments[i]
            self.clusters[cluster_id].append(node)
        
        print(f"Cluster sizes: {[len(c) for c in self.clusters]}")
        
        # Create sub-environments for each cluster
        self._create_cluster_environments()
    
    def _create_cluster_environments(self):
        """Create a separate environment for each cluster"""
        self.cluster_envs = []
        self.cluster_node_mappings = []  # Track original node IDs
        
        for i, cluster_nodes in enumerate(self.clusters):
            # Create new environment with cluster nodes + all charging stations + depot
            cluster_env = copy.deepcopy(self.env)
            
            # Keep mapping from new ID to original ID
            node_mapping = {}
            
            # Keep only depot, destination, cluster service nodes, and all charging stations
            new_nodes = []
            
            # Add depot (always ID 0 in cluster)
            new_nodes.append(cluster_env.depot)
            node_mapping[0] = 0
            
            # Add destination (always ID 1 in cluster)
            new_nodes.append(cluster_env.destination)
            node_mapping[1] = 1
            
            # Add cluster service nodes
            for node in cluster_nodes:
                new_id = len(new_nodes)
                new_node = copy.deepcopy(node)
                new_node.id = new_id
                node_mapping[new_id] = node.id  # Map new ID to original ID
                new_nodes.append(new_node)
            
            # Add all charging stations (they can be used by any UAV)
            for station in cluster_env.charging_stations:
                new_id = len(new_nodes)
                new_station = copy.deepcopy(station)
                new_station.id = new_id
                node_mapping[new_id] = station.id  # Map new ID to original ID
                new_nodes.append(new_station)
            
            cluster_env.nodes = new_nodes
            cluster_env.service_nodes = [n for n in new_nodes if n.node_type == 'service']
            cluster_env.charging_stations = [n for n in new_nodes if n.node_type == 'charging']
            cluster_env.n_total_nodes = len(new_nodes)
            
            # Recompute distance matrices
            cluster_env.distance_matrix = cluster_env._compute_distance_matrix()
            cluster_env.time_cost_matrix = cluster_env.distance_matrix.copy()
            cluster_env.battery_cost_matrix = cluster_env.distance_matrix.copy()
            
            self.cluster_envs.append(cluster_env)
            self.cluster_node_mappings.append(node_mapping)
            
            print(f"Cluster {i}: {len(cluster_env.service_nodes)} service nodes, "
                  f"{len(cluster_env.charging_stations)} charging stations")
    
    def phase2_solve_clusters(self, n_episodes: int = 10000, 
                              battery_bins: int = 10, 
                              time_bins: int = 10,
                              verbose: bool = True):
        """
        Phase 2: Solve each cluster independently using Q-learning
        
        Args:
            n_episodes: Number of training episodes per cluster
            battery_bins: Battery discretization
            time_bins: Time discretization
            verbose: Show progress
        """
        print(f"\nPhase 2: Solving each cluster with Q-learning NDTS...")
        
        self.agents = []
        self.routes = []
        total_reward = 0
        
        for i, cluster_env in enumerate(self.cluster_envs):
            print(f"\n=== Solving Cluster {i+1}/{self.n_uavs} ===")
            
            # Create and train agent for this cluster
            agent = QLearningNDTS(
                cluster_env,
                battery_bins=battery_bins,
                time_bins=time_bins
            )
            
            agent.train(n_episodes=n_episodes, verbose=verbose)
            
            # Get best route
            route, reward = agent.get_route()
            
            # Convert back to original node IDs using cluster index
            original_route = self._convert_route_to_original_ids(route, cluster_env, i)
            
            self.agents.append(agent)
            self.routes.append(original_route)
            total_reward += reward
            
            print(f"Cluster {i+1} - Route: {original_route[:10]}{'...' if len(original_route) > 10 else ''}")
            print(f"Cluster {i+1} - Reward: {reward}")
        
        print(f"\n=== Total Reward: {total_reward} ===")
        return total_reward
    
    def phase2_solve_clusters_improved(self, n_episodes: int = 10000, 
                                      battery_bins: int = 10, 
                                      time_bins: int = 10,
                                      verbose: bool = True):
        """
        Phase 2: Solve each cluster using Improved Q-learning with reward-biased exploration
        """
        from improved_q_learning import ImprovedQLearningNDTS
        
        print(f"\nPhase 2: Solving each cluster with Improved Q-learning...")
        
        self.agents = []
        self.routes = []
        total_reward = 0
        
        for i, cluster_env in enumerate(self.cluster_envs):
            print(f"\n=== Solving Cluster {i+1}/{self.n_uavs} with Improved QL ===")
            
            # Create and train improved agent
            agent = ImprovedQLearningNDTS(
                cluster_env,
                battery_bins=battery_bins,
                time_bins=time_bins,
                reward_bias=0.7  # High bias towards rewards
            )
            
            agent.train(n_episodes=n_episodes, verbose=verbose)
            
            # Get best route
            route, reward = agent.get_route()
            
            # Convert back to original node IDs
            original_route = self._convert_route_to_original_ids(route, cluster_env, i)
            
            self.agents.append(agent)
            self.routes.append(original_route)
            total_reward += reward
            
            print(f"Cluster {i+1} - Route: {original_route[:10]}{'...' if len(original_route) > 10 else ''}")
            print(f"Cluster {i+1} - Reward: {reward}")
        
        print(f"\n=== Total Reward: {total_reward} ===")
        return total_reward
    
    def _convert_route_to_original_ids(self, route: List[int], cluster_env: UAVEnvironment, cluster_idx: int) -> List[int]:
        """Convert cluster-local node IDs back to original environment IDs"""
        original_route = []
        node_mapping = self.cluster_node_mappings[cluster_idx]
        
        for local_id in route:
            # Use the mapping to get original ID
            original_id = node_mapping[local_id]
            original_route.append(original_id)
        
        return original_route
    
    def visualize_clusters(self, save_path: str = None):
        """Visualize the clustering result"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot depot
        ax.scatter(self.env.depot.x, self.env.depot.y, c='red', s=400, marker='s',
                  label='Depot', zorder=5, edgecolors='black', linewidth=2)
        
        # Plot charging stations
        for station in self.env.charging_stations:
            ax.scatter(station.x, station.y, c='green', s=300, marker='^',
                      zorder=4, edgecolors='black', linewidth=2)
        ax.scatter([], [], c='green', s=300, marker='^', label='Charging Stations',
                  edgecolors='black', linewidth=2)
        
        # Plot clustered service nodes
        colors = plt.cm.tab10(np.linspace(0, 1, self.n_uavs))
        
        for cluster_id in range(self.n_uavs):
            cluster_nodes = self.clusters[cluster_id]
            if cluster_nodes:
                x = [n.x for n in cluster_nodes]
                y = [n.y for n in cluster_nodes]
                ax.scatter(x, y, c=[colors[cluster_id]], s=200, 
                          label=f'Cluster {cluster_id+1}', alpha=0.7,
                          edgecolors='black', linewidth=1)
        
        ax.set_xlim(0, self.env.map_size)
        ax.set_ylim(0, self.env.map_size)
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_title(f'Service Node Clustering ({self.n_uavs} clusters)')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def visualize_routes(self, save_path: str = None):
        """Visualize all UAV routes"""
        fig = self.env.visualize(routes=self.routes, 
                                title=f"CFQS Routes ({self.n_uavs} UAVs)")
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def get_solution_summary(self) -> Dict:
        """Get summary statistics of the solution"""
        total_reward = 0
        total_service_nodes = 0
        
        for i, route in enumerate(self.routes):
            reward = sum(self.env.nodes[node_id].reward 
                        for node_id in route 
                        if self.env.nodes[node_id].node_type == 'service')
            n_service = sum(1 for node_id in route 
                          if self.env.nodes[node_id].node_type == 'service')
            
            total_reward += reward
            total_service_nodes += n_service
        
        return {
            'n_uavs': self.n_uavs,
            'total_reward': total_reward,
            'total_service_nodes_visited': total_service_nodes,
            'routes': self.routes,
            'route_lengths': [len(r) for r in self.routes]
        }
    
    def validate_cluster_assignment(self):
        """Validate that each UAV only visits nodes from its assigned cluster"""
        for uav_idx, route in enumerate(self.routes):
            cluster_node_ids = set(node.id for node in self.clusters[uav_idx])
            
            visited_service = [node_id for node_id in route 
                             if self.env.nodes[node_id].node_type == 'service']
            
            violations = [nid for nid in visited_service if nid not in cluster_node_ids]
            
            if violations:
                print(f"⚠️  WARNING: UAV {uav_idx+1} visited nodes outside its cluster: {violations}")
            else:
                print(f"✓ UAV {uav_idx+1} correctly stayed within cluster {uav_idx+1}")


def compare_different_n_uavs(env: UAVEnvironment, 
                             uav_counts: List[int] = [1, 2, 3],
                             n_episodes: int = 10000):
    """Compare performance with different numbers of UAVs"""
    results = []
    
    for n_uavs in uav_counts:
        print(f"\n{'='*60}")
        print(f"Testing with {n_uavs} UAV(s)")
        print(f"{'='*60}")
        
        cfqs = TwoPhaseApproach(env, n_uavs=n_uavs)
        cfqs.phase1_clustering()
        total_reward = cfqs.phase2_solve_clusters(n_episodes=n_episodes, verbose=True)
        
        summary = cfqs.get_solution_summary()
        summary['cfqs_object'] = cfqs
        results.append(summary)
        
        # Visualize
        cfqs.visualize_clusters(save_path=f'clusters_u{n_uavs}.png')
        cfqs.visualize_routes(save_path=f'routes_u{n_uavs}.png')
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    n_uavs_list = [r['n_uavs'] for r in results]
    rewards = [r['total_reward'] for r in results]
    nodes_visited = [r['total_service_nodes_visited'] for r in results]
    
    axes[0].bar(n_uavs_list, rewards, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Number of UAVs')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Total Reward vs Number of UAVs')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].bar(n_uavs_list, nodes_visited, color='coral', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Number of UAVs')
    axes[1].set_ylabel('Service Nodes Visited')
    axes[1].set_title('Coverage vs Number of UAVs')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('uav_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return results


if __name__ == "__main__":
    from uav_environment import UAVEnvironment
    
    # Load environment
    print("Loading environment...")
    env = UAVEnvironment.load_from_file('env_20.json')
    
    # Test with different numbers of UAVs
    results = compare_different_n_uavs(
        env, 
        uav_counts=[1, 2, 3],
        n_episodes=10000
    )
    
    # Print summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    for result in results:
        print(f"\n{result['n_uavs']} UAV(s):")
        print(f"  Total Reward: {result['total_reward']}")
        print(f"  Service Nodes Visited: {result['total_service_nodes_visited']}")
        print(f"  Route Lengths: {result['route_lengths']}")