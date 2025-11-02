import pulp
import numpy as np
from typing import List, Tuple, Optional
import time
from uav_environment import UAVEnvironment

class MILPSolver:
    """MILP Solver for Team Orienteering Problem with Charging Stations"""
    
    def __init__(self, env: UAVEnvironment, n_uavs: int, time_limit: int = 3600):
        """
        Initialize MILP solver
        
        Args:
            env: UAVEnvironment instance
            n_uavs: Number of UAVs
            time_limit: Time limit in seconds (default 1 hour)
        """
        self.env = env
        self.n_uavs = n_uavs
        self.time_limit = time_limit
        self.model = None
        self.solution_routes = []
        self.solution_reward = 0
        self.solve_time = 0
        self.status = None
    
    def build_model(self):
        """Build the MILP model according to the paper's formulation"""
        print("Building MILP model...")
        
        # Create problem
        self.model = pulp.LpProblem("TOP_with_Charging", pulp.LpMaximize)
        
        # Sets
        n = len(self.env.service_nodes)
        m = len(self.env.charging_stations)
        
        # Node indices
        C = list(range(2, 2 + n))  # Service nodes: 2 to n+1
        R = list(range(2 + n, 2 + n + m))  # Charging stations
        O = [0, 1] + C + R  # All nodes
        
        # Parameters
        U = self.n_uavs
        Tmax = self.env.time_limit
        Bmax = self.env.max_battery
        M = 10000  # Big M
        
        # Decision variables
        # x[i,j] = 1 if UAV goes from node i to j
        x = pulp.LpVariable.dicts("x", 
                                   [(i, j) for i in O for j in O if i != j],
                                   cat='Binary')
        
        # y[i,j] = cumulative time when arriving at j from i
        y = pulp.LpVariable.dicts("y",
                                   [(i, j) for i in O for j in O if i != j],
                                   lowBound=0,
                                   upBound=Tmax,
                                   cat='Continuous')
        
        # z[i] = battery level at node i
        z = pulp.LpVariable.dicts("z",
                                   O,
                                   lowBound=0,
                                   upBound=Bmax,
                                   cat='Continuous')
        
        # Objective: maximize total rewards
        rewards = [self.env.nodes[i].reward for i in O]
        self.model += pulp.lpSum([rewards[i] * x[(i, j)] 
                                  for i in O for j in O 
                                  if i != j and i in C])
        
        # Constraint (1): At most U UAVs depart from origin
        self.model += (pulp.lpSum([x[(0, i)] for i in O if i != 0]) <= U,
                      "max_uavs_depart")
        self.model += (pulp.lpSum([x[(i, 1)] for i in O if i != 1]) <= U,
                      "max_uavs_return")
        
        # Constraints (2) and (3): Each service node visited at most once
        for i in C:
            self.model += (pulp.lpSum([x[(j, i)] for j in O if j != i]) <= 1,
                          f"visit_once_in_{i}")
            self.model += (pulp.lpSum([x[(i, j)] for j in O if j != i]) <= 1,
                          f"visit_once_out_{i}")
        
        # Constraint (4): Flow conservation
        for k in O:
            if k not in [0, 1]:
                self.model += (pulp.lpSum([x[(i, k)] for i in O if i != k]) ==
                              pulp.lpSum([x[(k, j)] for j in O if j != k]),
                              f"flow_conservation_{k}")
        
        # Constraint (5): Initial time from depot
        for i in O:
            if i != 0 and i != 1:
                t0i = self.env.time_cost_matrix[0, i]
                self.model += (y[(0, i)] == t0i * x[(0, i)],
                              f"initial_time_{i}")
        
        # Constraint (6): Time limit for each arc
        for i in O:
            if i != 1:
                for j in O:
                    if j != 0 and j != i:
                        self.model += (y[(i, j)] <= Tmax * x[(i, j)],
                                      f"time_limit_{i}_{j}")
        
        # Constraint (7): Time accumulation and subtour elimination
        for i in O:
            if i not in [0, 1]:
                tij_sum_out = pulp.lpSum([self.env.time_cost_matrix[i, j] * x[(i, j)]
                                          for j in O if j != 0 and j != i])
                yji_sum_in = pulp.lpSum([y[(j, i)] for j in O if j != 1 and j != i])
                yij_sum_out = pulp.lpSum([y[(i, j)] for j in O if j != 0 and j != i])
                
                self.model += (yij_sum_out == yji_sum_in + tij_sum_out,
                              f"time_accumulation_{i}")
        
        # Constraint (8): Full battery at depot and charging stations
        self.model += (z[0] == Bmax, "depot_battery")
        self.model += (z[1] == Bmax, "destination_battery")
        for r in R:
            self.model += (z[r] == Bmax, f"charging_battery_{r}")
        
        # Constraints (9) and (10): Battery consumption
        for i in O:
            for j in C:
                if i != j:
                    bij = self.env.battery_cost_matrix[i, j]
                    # Constraint (9)
                    self.model += (z[i] - z[j] >= (bij + M) * x[(i, j)] - M,
                                  f"battery_lower_{i}_{j}")
                    # Constraint (10)
                    self.model += (z[i] - z[j] <= M + (bij - M) * x[(i, j)],
                                  f"battery_upper_{i}_{j}")
        
        print(f"Model built with {len(x)} binary variables and {len(y) + len(z)} continuous variables")
    
    def solve(self, solver_name: str = 'PULP_CBC_CMD', verbose: bool = True):
        """
        Solve the MILP model
        
        Args:
            solver_name: Solver to use ('PULP_CBC_CMD', 'GUROBI', etc.)
            verbose: Print solver output
        """
        if self.model is None:
            self.build_model()
        
        print(f"\nSolving with {solver_name} (time limit: {self.time_limit}s)...")
        
        # Select solver
        if solver_name == 'GUROBI':
            try:
                solver = pulp.GUROBI_CMD(timeLimit=self.time_limit, msg=verbose)
            except:
                print("Gurobi not available, falling back to CBC")
                solver = pulp.PULP_CBC_CMD(timeLimit=self.time_limit, msg=verbose)
        else:
            solver = pulp.PULP_CBC_CMD(timeLimit=self.time_limit, msg=verbose)
        
        # Solve
        start_time = time.time()
        self.status = self.model.solve(solver)
        self.solve_time = time.time() - start_time
        
        # Extract solution
        if self.status == pulp.LpStatusOptimal or self.status == pulp.LpStatusNotSolved:
            self._extract_solution()
        
        # Print results
        status_name = pulp.LpStatus[self.status]
        print(f"\nStatus: {status_name}")
        print(f"Solve time: {self.solve_time:.2f} seconds")
        print(f"Objective value: {self.solution_reward:.2f}")
        
        return self.status
    
    def _extract_solution(self):
        """Extract routes from solution"""
        self.solution_reward = pulp.value(self.model.objective)
        
        if self.solution_reward is None:
            print("No feasible solution found")
            return
        
        # Get all edges that are used (x[i,j] = 1)
        edges = []
        for v in self.model.variables():
            if v.name.startswith('x_') and pulp.value(v) > 0.5:
                # Parse variable name: x_(i,_j)
                parts = v.name.split('_')
                i = int(parts[1].strip('(,'))
                j = int(parts[2].strip('),'))
                edges.append((i, j))
        
        # Reconstruct routes
        self.solution_routes = self._reconstruct_routes(edges)
    
    def _reconstruct_routes(self, edges: List[Tuple[int, int]]) -> List[List[int]]:
        """Reconstruct UAV routes from edges"""
        routes = []
        remaining_edges = edges.copy()
        
        # Find routes starting from depot (node 0)
        while remaining_edges:
            # Start from depot
            current = 0
            route = [current]
            
            # Follow edges until return to depot (node 1) or no more edges
            max_steps = 1000
            steps = 0
            while current != 1 and steps < max_steps:
                # Find next edge
                next_edge = None
                for edge in remaining_edges:
                    if edge[0] == current:
                        next_edge = edge
                        break
                
                if next_edge is None:
                    break
                
                current = next_edge[1]
                route.append(current)
                remaining_edges.remove(next_edge)
                steps += 1
            
            if len(route) > 1:
                routes.append(route)
            
            if not remaining_edges:
                break
        
        return routes
    
    def get_solution_summary(self) -> dict:
        """Get summary of the solution"""
        total_service_nodes = 0
        for route in self.solution_routes:
            for node_id in route:
                if self.env.nodes[node_id].node_type == 'service':
                    total_service_nodes += 1
        
        return {
            'status': pulp.LpStatus[self.status] if self.status else 'Not solved',
            'objective': self.solution_reward,
            'solve_time': self.solve_time,
            'n_routes': len(self.solution_routes),
            'routes': self.solution_routes,
            'service_nodes_visited': total_service_nodes
        }


def solve_with_clustering(env: UAVEnvironment, n_uavs: int, 
                         time_limit_per_cluster: int = 300) -> Tuple[List[List[int]], float, float]:
    """
    Solve using cluster-first-solver-second approach
    
    Returns:
        routes: List of routes
        total_reward: Total reward
        total_time: Total computation time
    """
    from two_phase_cfqs import TwoPhaseApproach
    
    print(f"\n=== Cluster-First-Solver-Second (MILP) ===")
    print(f"Clustering into {n_uavs} clusters...")
    
    # Phase 1: Clustering
    cfqs = TwoPhaseApproach(env, n_uavs)
    cfqs.phase1_clustering()
    
    # Phase 2: Solve each cluster with MILP
    routes = []
    total_reward = 0
    total_time = 0
    
    for i, cluster_env in enumerate(cfqs.cluster_envs):
        print(f"\n--- Solving Cluster {i+1}/{n_uavs} with MILP ---")
        
        solver = MILPSolver(cluster_env, n_uavs=1, time_limit=time_limit_per_cluster)
        solver.solve(verbose=False)
        
        if solver.solution_routes:
            # Convert to original IDs using cluster index
            for route in solver.solution_routes:
                original_route = cfqs._convert_route_to_original_ids(route, cluster_env, i)
                routes.append(original_route)
            
            total_reward += solver.solution_reward
            total_time += solver.solve_time
        
        print(f"Cluster {i+1} - Reward: {solver.solution_reward:.2f}, Time: {solver.solve_time:.2f}s")
    
    print(f"\nTotal Reward: {total_reward:.2f}")
    print(f"Total Time: {total_time:.2f}s")
    
    return routes, total_reward, total_time


if __name__ == "__main__":
    # Test MILP solver
    print("Loading environment...")
    env = UAVEnvironment.load_from_file('env_20.json')
    
    # Solve directly with MILP
    print("\n" + "="*60)
    print("Testing Direct MILP Solver")
    print("="*60)
    
    solver = MILPSolver(env, n_uavs=2, time_limit=60)
    solver.solve(verbose=True)
    
    summary = solver.get_solution_summary()
    print("\nSolution Summary:")
    for key, value in summary.items():
        if key != 'routes':
            print(f"  {key}: {value}")
    
    # Visualize
    if solver.solution_routes:
        fig = env.visualize(routes=solver.solution_routes, 
                          title=f"MILP Solution (Reward: {solver.solution_reward:.1f})")
        plt.savefig('milp_solution.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # Test cluster-first-solver-second
    print("\n" + "="*60)
    print("Testing Cluster-First-MILP-Second")
    print("="*60)
    
    routes, reward, solve_time = solve_with_clustering(env, n_uavs=2, time_limit_per_cluster=60)
    
    fig = env.visualize(routes=routes, 
                       title=f"Cluster+MILP Solution (Reward: {reward:.1f})")
    plt.savefig('cluster_milp_solution.png', dpi=150, bbox_inches='tight')
    plt.show()