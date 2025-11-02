import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import time
import json

from uav_environment import UAVEnvironment
from q_learning_ndts import QLearningNDTS
from dqn_agent import DQNAgent
from two_phase_cfqs import TwoPhaseApproach
from milp_solver import MILPSolver, solve_with_clustering


class ComprehensiveComparison:
    """Compare all solution approaches"""
    
    def __init__(self, env_files: List[str], uav_counts: List[int]):
        """
        Initialize comparison
        
        Args:
            env_files: List of environment files to test
            uav_counts: List of UAV counts to test
        """
        self.env_files = env_files
        self.uav_counts = uav_counts
        self.results = []
    
    def run_all_comparisons(self):
        """Run all comparison experiments"""
        for env_file in self.env_files:
            print(f"\n{'='*80}")
            print(f"Testing on: {env_file}")
            print(f"{'='*80}")
            
            env = UAVEnvironment.load_from_file(env_file)
            
            for n_uavs in self.uav_counts:
                print(f"\n{'='*60}")
                print(f"Testing with {n_uavs} UAV(s)")
                print(f"{'='*60}")
                
                result = {
                    'env_file': env_file,
                    'n_nodes': env.n_service_nodes,
                    'n_uavs': n_uavs
                }
                
                # 1. MILP Direct (only for small instances)
                if env.n_service_nodes <= 20:
                    result.update(self._test_milp_direct(env, n_uavs))
                else:
                    result['milp_direct_reward'] = None
                    result['milp_direct_time'] = None
                    result['milp_direct_status'] = 'Skipped'
                
                # 2. Cluster + MILP
                result.update(self._test_cluster_milp(env, n_uavs))
                
                # 3. Cluster + Q-learning (CFQS)
                result.update(self._test_cfqs(env, n_uavs))
                
                # 4. Cluster + DQN (if not too large)
                if env.n_service_nodes <= 50:
                    result.update(self._test_cluster_dqn(env, n_uavs))
                else:
                    result['dqn_reward'] = None
                    result['dqn_time'] = None
                
                self.results.append(result)
        
        return self.results
    
    def _test_milp_direct(self, env: UAVEnvironment, n_uavs: int) -> Dict:
        """Test MILP solver directly"""
        print("\n--- MILP Direct ---")
        try:
            solver = MILPSolver(env, n_uavs, time_limit=3600)
            status = solver.solve(verbose=False)
            
            return {
                'milp_direct_reward': solver.solution_reward,
                'milp_direct_time': solver.solve_time,
                'milp_direct_status': pulp.LpStatus[status] if status else 'Failed',
                'milp_direct_nodes': solver.get_solution_summary()['service_nodes_visited']
            }
        except Exception as e:
            print(f"MILP Direct failed: {e}")
            return {
                'milp_direct_reward': None,
                'milp_direct_time': None,
                'milp_direct_status': 'Error'
            }
    
    def _test_cluster_milp(self, env: UAVEnvironment, n_uavs: int) -> Dict:
        """Test clustering + MILP approach"""
        print("\n--- Cluster + MILP ---")
        try:
            start = time.time()
            routes, reward, solve_time = solve_with_clustering(env, n_uavs, time_limit_per_cluster=600)
            total_time = time.time() - start
            
            nodes_visited = sum(1 for route in routes for node_id in route 
                              if env.nodes[node_id].node_type == 'service')
            
            return {
                'cluster_milp_reward': reward,
                'cluster_milp_time': total_time,
                'cluster_milp_nodes': nodes_visited
            }
        except Exception as e:
            print(f"Cluster+MILP failed: {e}")
            return {
                'cluster_milp_reward': None,
                'cluster_milp_time': None
            }
    
    def _test_cfqs(self, env: UAVEnvironment, n_uavs: int) -> Dict:
        """Test CFQS (Cluster First Q-learning Second)"""
        print("\n--- CFQS (Q-learning) ---")
        try:
            start = time.time()
            
            cfqs = TwoPhaseApproach(env, n_uavs)
            cfqs.phase1_clustering()
            
            # Adjust episodes based on problem size
            if env.n_service_nodes <= 20:
                n_episodes = 100000
            elif env.n_service_nodes <= 50:
                n_episodes = 200000
            else:
                n_episodes = 500000
            
            cfqs.phase2_solve_clusters(n_episodes=n_episodes, verbose=False)
            
            total_time = time.time() - start
            summary = cfqs.get_solution_summary()
            
            return {
                'cfqs_reward': summary['total_reward'],
                'cfqs_time': total_time,
                'cfqs_nodes': summary['total_service_nodes_visited']
            }
        except Exception as e:
            print(f"CFQS failed: {e}")
            return {
                'cfqs_reward': None,
                'cfqs_time': None
            }
    
    def _test_cluster_dqn(self, env: UAVEnvironment, n_uavs: int) -> Dict:
        """Test clustering + DQN approach"""
        print("\n--- Cluster + DQN ---")
        try:
            from two_phase_cfqs import TwoPhaseApproach
            
            start = time.time()
            
            # Cluster nodes
            cfqs = TwoPhaseApproach(env, n_uavs)
            cfqs.phase1_clustering()
            
            # Train DQN on each cluster
            total_reward = 0
            routes = []
            
            for i, cluster_env in enumerate(cfqs.cluster_envs):
                agent = DQNAgent(cluster_env)
                agent.train(n_episodes=5000, verbose=False)
                
                route, reward = agent.get_route()
                original_route = cfqs._convert_route_to_original_ids(route, cluster_env)
                
                routes.append(original_route)
                total_reward += reward
            
            total_time = time.time() - start
            
            nodes_visited = sum(1 for route in routes for node_id in route 
                              if env.nodes[node_id].node_type == 'service')
            
            return {
                'dqn_reward': total_reward,
                'dqn_time': total_time,
                'dqn_nodes': nodes_visited
            }
        except Exception as e:
            print(f"DQN failed: {e}")
            return {
                'dqn_reward': None,
                'dqn_time': None
            }
    
    def create_results_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame"""
        return pd.DataFrame(self.results)
    
    def plot_results(self, save_path: str = 'comparison_results.png'):
        """Plot comparison results"""
        df = self.create_results_dataframe()
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Reward comparison
        ax1 = fig.add_subplot(gs[0, :])
        
        methods = []
        colors_list = []
        
        for idx, row in df.iterrows():
            label = f"{row['n_nodes']} nodes, {row['n_uavs']} UAV(s)"
            x_pos = idx * 5
            
            if row['milp_direct_reward'] is not None:
                ax1.bar(x_pos, row['milp_direct_reward'], color='blue', alpha=0.7, width=0.8)
                if idx == 0:
                    methods.append('MILP Direct')
                    colors_list.append('blue')
            
            if row['cluster_milp_reward'] is not None:
                ax1.bar(x_pos + 1, row['cluster_milp_reward'], color='green', alpha=0.7, width=0.8)
                if idx == 0:
                    methods.append('Cluster+MILP')
                    colors_list.append('green')
            
            if row['cfqs_reward'] is not None:
                ax1.bar(x_pos + 2, row['cfqs_reward'], color='orange', alpha=0.7, width=0.8)
                if idx == 0:
                    methods.append('CFQS')
                    colors_list.append('orange')
            
            if row['dqn_reward'] is not None:
                ax1.bar(x_pos + 3, row['dqn_reward'], color='red', alpha=0.7, width=0.8)
                if idx == 0:
                    methods.append('DQN')
                    colors_list.append('red')
            
            ax1.text(x_pos + 1.5, -20, label, ha='center', rotation=0, fontsize=9)
        
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Solution Quality Comparison', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.legend(methods, loc='upper left')
        ax1.set_xticks([])
        
        # 2. Computation time comparison
        ax2 = fig.add_subplot(gs[1, 0])
        
        for idx, row in df.iterrows():
            label = f"{row['n_nodes']}n,{row['n_uavs']}u"
            x_pos = idx * 5
            
            if row['milp_direct_time'] is not None:
                ax2.bar(x_pos, row['milp_direct_time'], color='blue', alpha=0.7, width=0.8)
            if row['cluster_milp_time'] is not None:
                ax2.bar(x_pos + 1, row['cluster_milp_time'], color='green', alpha=0.7, width=0.8)
            if row['cfqs_time'] is not None:
                ax2.bar(x_pos + 2, row['cfqs_time'], color='orange', alpha=0.7, width=0.8)
            if row['dqn_time'] is not None:
                ax2.bar(x_pos + 3, row['dqn_time'], color='red', alpha=0.7, width=0.8)
            
            ax2.text(x_pos + 1.5, -5, label, ha='center', rotation=45, fontsize=8)
        
        ax2.set_ylabel('Computation Time (seconds)')
        ax2.set_title('Computational Efficiency', fontsize=12, fontweight='bold')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_xticks([])
        
        # 3. Gap analysis (compared to MILP direct when available)
        ax3 = fig.add_subplot(gs[1, 1])
        
        gaps = {'Cluster+MILP': [], 'CFQS': [], 'DQN': []}
        labels_gap = []
        
        for idx, row in df.iterrows():
            if row['milp_direct_reward'] is not None and row['milp_direct_reward'] > 0:
                baseline = row['milp_direct_reward']
                labels_gap.append(f"{row['n_nodes']}n,{row['n_uavs']}u")
                
                if row['cluster_milp_reward'] is not None:
                    gap = (baseline - row['cluster_milp_reward']) / baseline * 100
                    gaps['Cluster+MILP'].append(gap)
                else:
                    gaps['Cluster+MILP'].append(None)
                
                if row['cfqs_reward'] is not None:
                    gap = (baseline - row['cfqs_reward']) / baseline * 100
                    gaps['CFQS'].append(gap)
                else:
                    gaps['CFQS'].append(None)
                
                if row['dqn_reward'] is not None:
                    gap = (baseline - row['dqn_reward']) / baseline * 100
                    gaps['DQN'].append(gap)
                else:
                    gaps['DQN'].append(None)
        
        if labels_gap:
            x = np.arange(len(labels_gap))
            width = 0.25
            
            for i, (method, gap_values) in enumerate(gaps.items()):
                valid_gaps = [g if g is not None else 0 for g in gap_values]
                ax3.bar(x + i*width, valid_gaps, width, label=method, alpha=0.7)
            
            ax3.set_ylabel('Optimality Gap (%)')
            ax3.set_title('Gap vs MILP Direct', fontsize=12, fontweight='bold')
            ax3.set_xticks(x + width)
            ax3.set_xticklabels(labels_gap, rotation=45)
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')
            ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        
        # 4. Service nodes visited
        ax4 = fig.add_subplot(gs[2, :])
        
        for idx, row in df.iterrows():
            label = f"{row['n_nodes']}n,{row['n_uavs']}u"
            x_pos = idx * 5
            
            if row.get('milp_direct_nodes') is not None:
                ax4.bar(x_pos, row['milp_direct_nodes'], color='blue', alpha=0.7, width=0.8)
            if row.get('cluster_milp_nodes') is not None:
                ax4.bar(x_pos + 1, row['cluster_milp_nodes'], color='green', alpha=0.7, width=0.8)
            if row.get('cfqs_nodes') is not None:
                ax4.bar(x_pos + 2, row['cfqs_nodes'], color='orange', alpha=0.7, width=0.8)
            if row.get('dqn_nodes') is not None:
                ax4.bar(x_pos + 3, row['dqn_nodes'], color='red', alpha=0.7, width=0.8)
            
            ax4.text(x_pos + 1.5, -0.5, label, ha='center', rotation=45, fontsize=8)
        
        ax4.set_ylabel('Service Nodes Visited')
        ax4.set_title('Coverage Comparison', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_xticks([])
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nResults plot saved to: {save_path}")
        
        return fig
    
    def save_results(self, filename: str = 'comparison_results.json'):
        """Save results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to: {filename}")
    
    def print_summary_table(self):
        """Print a formatted summary table"""
        df = self.create_results_dataframe()
        
        print("\n" + "="*100)
        print("COMPREHENSIVE COMPARISON SUMMARY")
        print("="*100)
        
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        
        print(df.to_string(index=False))
        print("="*100)


def run_paper_experiments():
    """Run the experiments as described in the paper"""
    
    print("="*80)
    print("RUNNING PAPER EXPERIMENTS")
    print("="*80)
    
    # Test configurations matching the paper
    env_files = ['env_20.json', 'env_50.json']
    uav_counts = [1, 2, 3]
    
    comparison = ComprehensiveComparison(env_files, uav_counts)
    results = comparison.run_all_comparisons()
    
    # Print summary
    comparison.print_summary_table()
    
    # Plot results
    comparison.plot_results('paper_comparison_results.png')
    
    # Save results
    comparison.save_results('paper_comparison_results.json')
    
    return comparison


if __name__ == "__main__":
    import pulp
    
    # Run experiments
    comparison = run_paper_experiments()
    
    plt.show()