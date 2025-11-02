"""
Greedy Baseline Solver for Comparison
Always selects the highest reward feasible node
"""

import numpy as np
from typing import List, Tuple, Set
from uav_environment import UAVEnvironment

class GreedySolver:
    """Greedy heuristic that prioritizes high-reward nodes"""
    
    def __init__(self, env: UAVEnvironment):
        self.env = env
    
    def solve_single_uav(self) -> Tuple[List[int], float]:
        """
        Solve for single UAV using greedy reward-first strategy
        
        Returns:
            route: List of node IDs
            total_reward: Total reward collected
        """
        current_node = 0
        battery = self.env.max_battery
        time_remaining = self.env.time_limit
        visited = set()
        
        route = [current_node]
        total_reward = 0
        
        while True:
            available_actions = self.env.get_available_actions(
                current_node, battery, time_remaining, visited
            )
            
            if not available_actions or 1 in available_actions and len(available_actions) == 1:
                # Only depot available or no actions - return home
                if 1 in available_actions:
                    route.append(1)
                break
            
            # Remove depot from consideration (we'll return later)
            actions_to_consider = [a for a in available_actions if a != 1]
            
            if not actions_to_consider:
                # Only depot available
                route.append(1)
                break
            
            # Select highest reward node among available
            best_action = self._select_best_node(
                actions_to_consider, current_node, battery, time_remaining
            )
            
            # Take action
            reward, battery, time_remaining, done = self.env.step(
                current_node, best_action, battery, time_remaining
            )
            
            if reward < 0:  # Infeasible
                break
            
            route.append(best_action)
            total_reward += reward
            
            if self.env.nodes[best_action].node_type == 'service':
                visited.add(best_action)
            
            current_node = best_action
            
            # Check if we should head back
            if self._should_return_to_depot(current_node, battery, time_remaining, visited):
                available = self.env.get_available_actions(
                    current_node, battery, time_remaining, visited
                )
                if 1 in available:
                    route.append(1)
                break
        
        return route, total_reward
    
    def _select_best_node(self, available_actions: List[int], 
                         current_node: int, battery: float, 
                         time_remaining: float) -> int:
        """
        Select best node considering:
        1. Reward
        2. Distance efficiency (reward per distance)
        3. Battery/time margin
        """
        best_score = -float('inf')
        best_action = available_actions[0]
        
        for action in available_actions:
            node = self.env.nodes[action]
            
            # Skip non-service nodes in selection (charging stations are fallback only)
            if node.node_type != 'service':
                continue
            
            # Calculate reward-to-distance ratio
            distance = self.env.distance_matrix[current_node, action]
            distance_back = self.env.distance_matrix[action, 1]
            total_distance = distance + distance_back
            
            if total_distance < 0.1:
                continue
            
            reward_efficiency = node.reward / total_distance
            
            # Bonus for high absolute reward
            reward_bonus = node.reward / 50.0
            
            # Penalty for low remaining resources
            resource_margin = min(
                battery - self.env.battery_cost_matrix[current_node, action],
                time_remaining - self.env.time_cost_matrix[current_node, action]
            )
            resource_factor = 1.0 + (resource_margin / 100.0)
            
            # Combined score
            score = (reward_efficiency + reward_bonus) * resource_factor
            
            if score > best_score:
                best_score = score
                best_action = action
        
        # If no service nodes selected, pick nearest charging or depot
        if best_action == available_actions[0] and self.env.nodes[best_action].node_type != 'service':
            # Find nearest charging station or return to depot
            charging_actions = [a for a in available_actions 
                              if self.env.nodes[a].node_type == 'charging']
            
            if charging_actions:
                # Go to nearest charging station
                best_action = min(charging_actions, 
                                key=lambda a: self.env.distance_matrix[current_node, a])
            else:
                # Return to depot
                best_action = 1 if 1 in available_actions else available_actions[0]
        
        return best_action
    
    def _should_return_to_depot(self, current_node: int, battery: float,
                               time_remaining: float, visited: Set[int]) -> bool:
        """Decide if UAV should return to depot"""
        # Check if we can reach any unvisited service node
        for node in self.env.service_nodes:
            if node.id in visited:
                continue
            
            dist_to_node = self.env.distance_matrix[current_node, node.id]
            dist_node_to_depot = self.env.distance_matrix[node.id, 1]
            
            if (dist_to_node + dist_node_to_depot < battery and
                dist_to_node + dist_node_to_depot < time_remaining):
                return False  # Can still visit more nodes
        
        return True  # Should return
    
    def solve_multi_uav(self, n_uavs: int) -> Tuple[List[List[int]], float]:
        """
        Solve for multiple UAVs using greedy assignment
        
        Args:
            n_uavs: Number of UAVs
            
        Returns:
            routes: List of routes for each UAV
            total_reward: Combined reward
        """
        # Cluster nodes first
        from sklearn.cluster import KMeans
        
        service_coords = np.array([[n.x, n.y] for n in self.env.service_nodes])
        kmeans = KMeans(n_clusters=n_uavs, random_state=42, n_init=10)
        labels = kmeans.fit_predict(service_coords)
        
        # Create clusters
        clusters = [[] for _ in range(n_uavs)]
        for i, node in enumerate(self.env.service_nodes):
            clusters[labels[i]].append(node)
        
        # Solve each cluster with greedy
        routes = []
        total_reward = 0
        
        for cluster_nodes in clusters:
            # Create temporary environment for this cluster
            cluster_solver = GreedyClusterSolver(self.env, cluster_nodes)
            route, reward = cluster_solver.solve()
            
            routes.append(route)
            total_reward += reward
        
        return routes, total_reward


class GreedyClusterSolver:
    """Greedy solver for a single cluster"""
    
    def __init__(self, env: UAVEnvironment, cluster_nodes: List):
        self.env = env
        self.cluster_nodes = set(n.id for n in cluster_nodes)
    
    def solve(self) -> Tuple[List[int], float]:
        """Solve cluster with greedy approach"""
        current_node = 0
        battery = self.env.max_battery
        time_remaining = self.env.time_limit
        visited = set()
        
        route = [current_node]
        total_reward = 0
        
        while True:
            # Get available actions in this cluster
            available = self.env.get_available_actions(
                current_node, battery, time_remaining, visited
            )
            
            # Filter to only cluster nodes + charging + depot
            cluster_available = [
                a for a in available
                if (a in self.cluster_nodes or 
                    self.env.nodes[a].node_type in ['charging', 'depot'])
            ]
            
            if not cluster_available or (1 in cluster_available and len(cluster_available) == 1):
                if 1 in cluster_available:
                    route.append(1)
                break
            
            # Remove depot from selection
            actions = [a for a in cluster_available if a != 1]
            
            if not actions:
                route.append(1)
                break
            
            # Select highest reward service node
            best_action = None
            best_score = -1
            
            for action in actions:
                node = self.env.nodes[action]
                if node.node_type == 'service':
                    distance = self.env.distance_matrix[current_node, action]
                    score = node.reward / (distance + 1)  # Reward per distance
                    if score > best_score:
                        best_score = score
                        best_action = action
            
            # If no service node, pick charging or depot
            if best_action is None:
                best_action = actions[0]
            
            # Take action
            reward, battery, time_remaining, done = self.env.step(
                current_node, best_action, battery, time_remaining
            )
            
            if reward < 0:
                break
            
            route.append(best_action)
            total_reward += reward
            
            if self.env.nodes[best_action].node_type == 'service':
                visited.add(best_action)
            
            current_node = best_action
        
        return route, total_reward


if __name__ == "__main__":
    from uav_environment import UAVEnvironment
    import matplotlib.pyplot as plt
    
    # Test greedy solver
    env = UAVEnvironment(n_service_nodes=20, n_charging_stations=2, seed=42)
    
    print("Testing Greedy Solver...")
    solver = GreedySolver(env)
    
    # Single UAV
    route, reward = solver.solve_single_uav()
    print(f"\nGreedy Single UAV:")
    print(f"  Route: {route}")
    print(f"  Reward: {reward}")
    
    # Multi UAV
    routes, total_reward = solver.solve_multi_uav(n_uavs=2)
    print(f"\nGreedy Multi-UAV (2 UAVs):")
    print(f"  Total Reward: {total_reward}")
    for i, r in enumerate(routes):
        print(f"  UAV {i+1}: {r}")
    
    # Visualize
    fig = env.visualize(routes=routes, title=f"Greedy Solution (Reward: {total_reward})")
    plt.show()