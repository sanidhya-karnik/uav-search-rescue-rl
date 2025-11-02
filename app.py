"""
UAV Search and Rescue - Streamlit Dashboard
Interactive visualization and control interface
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import time
from pathlib import Path

# Import your modules
from uav_environment import UAVEnvironment, Node
from two_phase_cfqs import TwoPhaseApproach
from q_learning_ndts import QLearningNDTS
from improved_q_learning import ImprovedQLearningNDTS
from greedy_baseline import GreedySolver

# Page configuration
st.set_page_config(
    page_title="UAV Search & Rescue Simulator",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    h1 {
        color: #1f77b4;
    }
    .legend-item {
        display: inline-block;
        margin-right: 20px;
        padding: 5px 10px;
        border-radius: 5px;
        background-color: #f0f2f6;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# HELPER FUNCTIONS - Define ALL functions BEFORE using them
# ============================================================================

# Define consistent color palette for all visualizations
ROUTE_COLORS = [
    '#2196F3',  # Blue
    '#FF6B6B',  # Red
    '#4ECDC4',  # Teal
    '#FFD93D',  # Yellow
    '#A855F7',  # Purple
    '#FB923C',  # Orange
    '#10B981',  # Green
    '#EC4899',  # Pink
]

def create_route_plot(env, routes):
    """Create interactive route visualization"""
    fig = go.Figure()
    
    # Service nodes
    service_nodes = env.service_nodes
    fig.add_trace(go.Scatter(
        x=[n.x for n in service_nodes],
        y=[n.y for n in service_nodes],
        mode='markers',
        name='Service Nodes',
        marker=dict(
            size=[10 + n.reward/3 for n in service_nodes],
            color=[n.reward for n in service_nodes],
            colorscale='YlOrRd',
            showscale=True,
            colorbar=dict(
                title="Reward",
                x=1.15,
                len=0.5
            ),
            line=dict(color='black', width=1)
        ),
        text=[f"Node {n.id}<br>Reward: {n.reward}" for n in service_nodes],
        hovertemplate='<b>%{text}</b><br>Position: (%{x:.1f}, %{y:.1f})<extra></extra>'
    ))
    
    # Charging stations
    charging = env.charging_stations
    fig.add_trace(go.Scatter(
        x=[n.x for n in charging],
        y=[n.y for n in charging],
        mode='markers',
        name='Charging Stations',
        marker=dict(
            size=20,
            color='#66bb6a',
            symbol='triangle-up',
            line=dict(color='black', width=2)
        ),
        text=[f"Charging Station {n.id}" for n in charging],
        hovertemplate='<b>%{text}</b><br>Position: (%{x:.1f}, %{y:.1f})<extra></extra>'
    ))
    
    # Depot
    depot = env.depot
    fig.add_trace(go.Scatter(
        x=[depot.x],
        y=[depot.y],
        mode='markers',
        name='Depot',
        marker=dict(
            size=25,
            color='#ef5350',
            symbol='square',
            line=dict(color='black', width=2)
        ),
        text=['Depot'],
        hovertemplate='<b>Depot</b><br>Position: (%{x:.1f}, %{y:.1f})<extra></extra>'
    ))
    
    # Routes with distinct colors
    for idx, route in enumerate(routes):
        if len(route) > 1:
            route_nodes = [env.nodes[node_id] for node_id in route]
            fig.add_trace(go.Scatter(
                x=[n.x for n in route_nodes],
                y=[n.y for n in route_nodes],
                mode='lines+markers',
                name=f'UAV {idx + 1}',
                line=dict(
                    color=ROUTE_COLORS[idx % len(ROUTE_COLORS)], 
                    width=4,
                    dash='solid'
                ),
                marker=dict(
                    size=12,
                    color=ROUTE_COLORS[idx % len(ROUTE_COLORS)],
                    line=dict(color='white', width=2),
                    symbol='circle'
                ),
                text=[f"UAV {idx + 1} - Step {i}" for i in range(len(route_nodes))],
                hovertemplate='<b>%{text}</b><br>Position: (%{x:.1f}, %{y:.1f})<extra></extra>'
            ))
    
    fig.update_layout(
        title={
            'text': "UAV Routes Visualization",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#1f77b4'}
        },
        xaxis_title="X Coordinate",
        yaxis_title="Y Coordinate",
        hovermode='closest',
        height=700,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1,
            font=dict(size=11)
        ),
        plot_bgcolor='#f8f9fa',
        xaxis=dict(
            range=[0, env.map_size],
            gridcolor='lightgray',
            showgrid=True
        ),
        yaxis=dict(
            range=[0, env.map_size], 
            scaleanchor="x", 
            scaleratio=1,
            gridcolor='lightgray',
            showgrid=True
        ),
        margin=dict(r=200)
    )
    
    return fig


def create_reward_chart(env, routes):
    """Create reward per UAV bar chart"""
    rewards = []
    colors_list = []
    
    for idx, route in enumerate(routes):
        reward = sum(env.nodes[node_id].reward for node_id in route 
                    if env.nodes[node_id].node_type == 'service')
        rewards.append({'UAV': f'UAV {idx+1}', 'Reward': reward})
        colors_list.append(ROUTE_COLORS[idx % len(ROUTE_COLORS)])
    
    df = pd.DataFrame(rewards)
    
    fig = go.Figure(data=[
        go.Bar(
            x=df['UAV'],
            y=df['Reward'],
            marker_color=colors_list,
            marker_line_color='black',
            marker_line_width=1.5,
            text=df['Reward'],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Reward per UAV',
        xaxis_title='UAV',
        yaxis_title='Total Reward',
        height=400,
        plot_bgcolor='#f8f9fa',
        showlegend=False
    )
    
    return fig


def create_node_heatmap(env, routes):
    """Create heatmap of visited vs unvisited nodes"""
    visited_nodes = set()
    for route in routes:
        for node_id in route:
            if env.nodes[node_id].node_type == 'service':
                visited_nodes.add(node_id)
    
    data = []
    for node in env.service_nodes:
        data.append({
            'Node ID': node.id,
            'Reward': node.reward,
            'Status': 'Visited' if node.id in visited_nodes else 'Unvisited'
        })
    
    df = pd.DataFrame(data)
    
    fig = px.scatter(
        df,
        x='Node ID',
        y='Reward',
        color='Status',
        title='Service Nodes: Visited vs Unvisited',
        color_discrete_map={'Visited': '#2196F3', 'Unvisited': '#ff9800'},
        size='Reward',
        size_max=20
    )
    
    fig.update_layout(
        height=400,
        plot_bgcolor='#f8f9fa'
    )
    
    return fig


def create_battery_chart(env, routes):
    """Create battery consumption chart"""
    fig = go.Figure()
    
    for idx, route in enumerate(routes):
        battery_levels = [env.battery_limit]
        steps = [0]
        current_battery = env.battery_limit
        
        for i in range(len(route) - 1):
            from_node = env.nodes[route[i]]
            to_node = env.nodes[route[i + 1]]
            
            distance = np.sqrt((to_node.x - from_node.x)**2 + (to_node.y - from_node.y)**2)
            
            if to_node.node_type == 'charging':
                current_battery = env.battery_limit
            else:
                current_battery = max(0, current_battery - distance)
            
            battery_levels.append(current_battery)
            steps.append(i + 1)
        
        fig.add_trace(go.Scatter(
            x=steps,
            y=battery_levels,
            mode='lines+markers',
            name=f'UAV {idx + 1}',
            line=dict(color=ROUTE_COLORS[idx % len(ROUTE_COLORS)], width=3),
            marker=dict(size=8, color=ROUTE_COLORS[idx % len(ROUTE_COLORS)])
        ))
    
    fig.add_hline(
        y=env.battery_limit,
        line_dash="dash",
        line_color="red",
        annotation_text="Max Battery",
        annotation_position="right"
    )
    
    fig.update_layout(
        title='Battery Level Throughout Mission',
        xaxis_title='Step',
        yaxis_title='Battery Level',
        height=500,
        plot_bgcolor='#f8f9fa',
        hovermode='x unified'
    )
    
    return fig


def calculate_battery_stats(env, routes):
    """Calculate battery statistics for each UAV"""
    stats = []
    
    for idx, route in enumerate(routes):
        current_battery = env.battery_limit
        min_battery = env.battery_limit
        recharges = 0
        total_distance = 0
        
        for i in range(len(route) - 1):
            from_node = env.nodes[route[i]]
            to_node = env.nodes[route[i + 1]]
            
            distance = np.sqrt((to_node.x - from_node.x)**2 + (to_node.y - from_node.y)**2)
            total_distance += distance
            
            if to_node.node_type == 'charging':
                recharges += 1
                current_battery = env.battery_limit
            else:
                current_battery -= distance
                min_battery = min(min_battery, current_battery)
        
        stats.append({
            'UAV': f'UAV {idx + 1}',
            'Total Distance': f'{total_distance:.1f}',
            'Min Battery': f'{min_battery:.1f}',
            'Recharges': recharges,
            'Final Battery': f'{current_battery:.1f}'
        })
    
    return pd.DataFrame(stats)


def create_cluster_plot(cfqs):
    """Create clustering visualization with route overlay"""
    env = cfqs.env
    fig = go.Figure()
    
    # Plot depot
    fig.add_trace(go.Scatter(
        x=[env.depot.x],
        y=[env.depot.y],
        mode='markers',
        name='Depot',
        marker=dict(size=25, color='#ef5350', symbol='square', 
                   line=dict(color='black', width=2)),
        showlegend=True
    ))
    
    # Plot charging stations
    charging = env.charging_stations
    fig.add_trace(go.Scatter(
        x=[n.x for n in charging],
        y=[n.y for n in charging],
        mode='markers',
        name='Charging Stations',
        marker=dict(size=18, color='#66bb6a', symbol='triangle-up',
                   line=dict(color='black', width=2)),
        showlegend=True
    ))
    
    # Plot clustered service nodes with matching route colors
    for cluster_id, cluster_nodes in enumerate(cfqs.clusters):
        if cluster_nodes:
            fig.add_trace(go.Scatter(
                x=[n.x for n in cluster_nodes],
                y=[n.y for n in cluster_nodes],
                mode='markers',
                name=f'Cluster {cluster_id + 1} (UAV {cluster_id + 1})',
                marker=dict(
                    size=14,
                    color=ROUTE_COLORS[cluster_id % len(ROUTE_COLORS)],
                    opacity=0.6,
                    line=dict(color='black', width=2)
                ),
                text=[f"Node {n.id}<br>Reward: {n.reward}" for n in cluster_nodes],
                hovertemplate='<b>%{text}</b><br>Cluster %{cluster_id}<extra></extra>'.replace('%{cluster_id}', str(cluster_id + 1))
            ))
    
    fig.update_layout(
        title={
            'text': 'K-Means Clustering (Each cluster assigned to one UAV)',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='X Coordinate',
        yaxis_title='Y Coordinate',
        height=600,
        plot_bgcolor='#f8f9fa',
        xaxis=dict(range=[0, env.map_size], gridcolor='lightgray', showgrid=True),
        yaxis=dict(range=[0, env.map_size], scaleanchor="x", scaleratio=1, 
                  gridcolor='lightgray', showgrid=True)
    )
    
    return fig


def get_cluster_statistics(cfqs):
    """Get statistics for each cluster"""
    stats = []
    
    for idx, cluster_nodes in enumerate(cfqs.clusters):
        total_reward = sum(n.reward for n in cluster_nodes)
        avg_reward = total_reward / len(cluster_nodes) if cluster_nodes else 0
        
        stats.append({
            'Cluster': f'Cluster {idx + 1}',
            'Nodes': len(cluster_nodes),
            'Total Reward': f'{total_reward:.0f}',
            'Avg Reward': f'{avg_reward:.1f}'
        })
    
    return pd.DataFrame(stats)


def create_route_dataframe(env, route, uav_id):
    """Create detailed dataframe for a route"""
    data = []
    
    for step, node_id in enumerate(route):
        node = env.nodes[node_id]
        data.append({
            'Step': step,
            'Node ID': node_id,
            'Type': node.node_type.capitalize(),
            'X': f'{node.x:.1f}',
            'Y': f'{node.y:.1f}',
            'Reward': node.reward if node.node_type == 'service' else '-'
        })
    
    return pd.DataFrame(data)


def analyze_missed_opportunities(env, routes):
    """Analyze high-reward nodes that were not visited"""
    visited_service_nodes = set()
    for route in routes:
        for node_id in route:
            if env.nodes[node_id].node_type == 'service':
                visited_service_nodes.add(node_id)
    
    # Find unvisited nodes
    unvisited = []
    for node in env.service_nodes:
        if node.id not in visited_service_nodes:
            unvisited.append({
                'Node ID': node.id,
                'Reward': node.reward,
                'Position': f'({node.x:.1f}, {node.y:.1f})'
            })
    
    # Sort by reward (highest first)
    unvisited.sort(key=lambda x: x['Reward'], reverse=True)
    
    return pd.DataFrame(unvisited) if unvisited else None


# ============================================================================
# MAIN APPLICATION
# ============================================================================

# Initialize session state
if 'environment' not in st.session_state:
    st.session_state.environment = None
if 'routes' not in st.session_state:
    st.session_state.routes = []
if 'solution_summary' not in st.session_state:
    st.session_state.solution_summary = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = None

# Title and description
st.title("🚁 UAV Search & Rescue Simulator")
st.markdown("**Team Orienteering Problem with Charging Stations using Reinforcement Learning**")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Environment Settings")
    n_service_nodes = st.selectbox(
        "Service Nodes",
        options=[10, 20, 30, 50, 100],
        index=1,
        help="Number of survivor locations to visit"
    )
    
    n_uavs = st.selectbox(
        "Number of UAVs",
        options=[1, 2, 3, 4, 5],
        index=1,
        help="Number of drones available"
    )
    
    n_charging_stations = st.selectbox(
        "Charging Stations",
        options=[2, 3, 5, 10],
        index=0,
        help="Number of battery recharge points"
    )
    
    map_size = st.slider(
        "Map Size",
        min_value=50,
        max_value=200,
        value=100,
        step=10,
        help="Size of search area"
    )
    
    time_limit = st.slider(
        "Time Limit",
        min_value=50,
        max_value=300,
        value=100,
        step=10,
        help="Maximum mission duration per UAV"
    )
    
    battery_limit = st.slider(
        "Battery Capacity",
        min_value=25,
        max_value=150,
        value=50,
        step=5,
        help="Maximum battery capacity"
    )
    
    seed = st.number_input(
        "Random Seed",
        min_value=0,
        max_value=9999,
        value=42,
        help="Seed for reproducibility"
    )
    
    st.divider()
    
    st.subheader("Algorithm Settings")
    
    algorithm = st.selectbox(
        "Algorithm",
        options=["Original Q-Learning (NDTS)", "Improved Q-Learning (Reward-Biased)", "Greedy Baseline"],
        index=1,
        help="Choose the routing algorithm"
    )
    
    n_episodes = st.selectbox(
        "Training Episodes",
        options=[5000, 10000, 20000, 50000, 100000],
        index=2,  # Default to 20,000 for better results
        help="Number of training iterations"
    )
    
    st.divider()
    
    # Action buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Generate", type="primary", use_container_width=True):
            with st.spinner("Generating environment..."):
                st.session_state.environment = UAVEnvironment(
                    n_service_nodes=n_service_nodes,
                    n_charging_stations=n_charging_stations,
                    map_size=map_size,
                    time_limit=time_limit,
                    battery_limit=battery_limit,
                    seed=seed
                )
                st.session_state.routes = []
                st.session_state.solution_summary = None
                st.session_state.training_history = None
                st.success("✅ Environment generated!")
    
    with col2:
        if st.button("🚀 Solve", type="secondary", use_container_width=True):
            if st.session_state.environment is None:
                st.error("⚠️ Generate environment first!")
            else:
                if algorithm == "Greedy Baseline":
                    # Use greedy solver
                    with st.spinner(f"Running Greedy solver for {n_uavs} UAV(s)..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        solver = GreedySolver(st.session_state.environment)
                        
                        status_text.text("Phase 1: Clustering...")
                        progress_bar.progress(30)
                        
                        routes, total_reward = solver.solve_multi_uav(n_uavs=n_uavs)
                        
                        progress_bar.progress(100)
                        status_text.text("Greedy solution complete ✓")
                        
                        st.session_state.routes = routes
                        st.session_state.solution_summary = {
                            'n_uavs': n_uavs,
                            'total_reward': total_reward,
                            'total_service_nodes_visited': sum(
                                1 for route in routes for nid in route
                                if st.session_state.environment.nodes[nid].node_type == 'service'
                            ),
                            'routes': routes,
                            'route_lengths': [len(r) for r in routes]
                        }
                        st.session_state.algorithm_used = "Greedy"
                        
                        time.sleep(0.5)
                        progress_bar.empty()
                        status_text.empty()
                        st.success("✅ Greedy solution found!")
                        st.rerun()
                
                else:
                    # Use Q-Learning (original or improved)
                    with st.spinner(f"Training {n_uavs} UAV(s) with Q-Learning..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        cfqs = TwoPhaseApproach(st.session_state.environment, n_uavs=n_uavs)
                        cfqs.phase1_clustering()
                        
                        status_text.text("Phase 1: Clustering complete ✓")
                        progress_bar.progress(30)
                        
                        status_text.text(f"Phase 2: Training with {algorithm}...")
                        
                        # Choose algorithm
                        if algorithm == "Improved Q-Learning (Reward-Biased)":
                            cfqs.phase2_solve_clusters_improved(n_episodes=n_episodes, verbose=False)
                        else:
                            cfqs.phase2_solve_clusters(n_episodes=n_episodes, verbose=False)
                        
                        cfqs.validate_cluster_assignment()
                        
                        progress_bar.progress(100)
                        status_text.text("Training complete ✓")
                        
                        st.session_state.routes = cfqs.routes
                        st.session_state.solution_summary = cfqs.get_solution_summary()
                        st.session_state.cfqs = cfqs
                        st.session_state.algorithm_used = algorithm
                        
                        time.sleep(0.5)
                        progress_bar.empty()
                        status_text.empty()
                        st.success("✅ Solution found!")
                        st.rerun()

# Main content area
if st.session_state.environment is None:
    st.info("👈 Configure settings in the sidebar and click **Generate** to create an environment")
    
    st.subheader("Example Scenario")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Service Nodes", "20", help="Locations to visit")
    with col2:
        st.metric("UAVs", "2", help="Number of drones")
    with col3:
        st.metric("Charging Stations", "2", help="Recharge points")
    
    st.markdown("""
    ### How it works:
    1. **Generate Environment**: Creates a random search area with service nodes and charging stations
    2. **Solve**: Uses Reinforcement Learning to find optimal UAV routes
    3. **Visualize**: See routes, rewards, and battery consumption
    
    ### Features:
    - 🎯 Maximize survivor rescue rewards
    - 🔋 Battery management with charging stations
    - ⏱️ Time-constrained missions
    - 🤖 Q-Learning with Non-Decreasing Tree Search (NDTS)
    """)

else:
    env = st.session_state.environment
    
    # Display metrics
    if st.session_state.solution_summary is not None:
        summary = st.session_state.solution_summary
        
        # Show algorithm used
        if hasattr(st.session_state, 'algorithm_used'):
            st.info(f"🤖 Algorithm: **{st.session_state.algorithm_used}**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Reward",
                f"{summary['total_reward']:.0f}",
                help="Total reward collected from all UAVs"
            )
        
        with col2:
            st.metric(
                "Nodes Visited",
                f"{summary['total_service_nodes_visited']}/{len(env.service_nodes)}",
                help="Service nodes visited out of total available"
            )
        
        with col3:
            coverage = (summary['total_service_nodes_visited'] / len(env.service_nodes)) * 100
            st.metric(
                "Coverage",
                f"{coverage:.1f}%",
                help="Percentage of area covered"
            )
        
        with col4:
            efficiency = summary['total_reward'] / n_uavs
            st.metric(
                "Efficiency Score",
                f"{efficiency:.1f}",
                help="Average reward per UAV"
            )
    else:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Service Nodes", len(env.service_nodes))
        with col2:
            st.metric("Charging Stations", len(env.charging_stations))
        with col3:
            st.metric("Time Limit", f"{env.time_limit}")
        with col4:
            st.metric("Battery Capacity", f"{env.battery_limit}")
    
    st.divider()
    
    # Legend
    st.markdown("""
    <div style='text-align: center; padding: 10px; background-color: #f0f2f6; border-radius: 10px; margin-bottom: 20px;'>
        <span class='legend-item' style='background-color: #ef5350;'>🔴 Depot</span>
        <span class='legend-item' style='background-color: #ffa726;'>🟠 Service Nodes</span>
        <span class='legend-item' style='background-color: #66bb6a;'>🟢 Charging Stations</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Visualization tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Route Visualization", "📊 Reward Analysis", "🔋 Battery Analysis", "📈 Clustering"])
    
    with tab1:
        if st.session_state.routes:
            try:
                fig = create_route_plot(env, st.session_state.routes)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Could not create route plot")
            except Exception as e:
                st.error(f"Error creating route plot: {str(e)}")
        else:
            st.info("Click 'Solve' in the sidebar to generate routes")
    
    with tab2:
        if st.session_state.routes:
            col1, col2 = st.columns(2)
            
            with col1:
                try:
                    fig_rewards = create_reward_chart(env, st.session_state.routes)
                    if fig_rewards is not None:
                        st.plotly_chart(fig_rewards, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating reward chart: {str(e)}")
            
            with col2:
                try:
                    fig_heatmap = create_node_heatmap(env, st.session_state.routes)
                    if fig_heatmap is not None:
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating heatmap: {str(e)}")
        else:
            st.info("Solve the problem first to see reward analysis")
    
    with tab3:
        if st.session_state.routes:
            try:
                fig_battery = create_battery_chart(env, st.session_state.routes)
                if fig_battery is not None:
                    st.plotly_chart(fig_battery, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating battery chart: {str(e)}")
            
            st.subheader("Battery Statistics")
            try:
                battery_stats = calculate_battery_stats(env, st.session_state.routes)
                st.dataframe(battery_stats, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating battery stats: {str(e)}")
        else:
            st.info("Solve the problem first to see battery analysis")
    
    with tab4:
        if hasattr(st.session_state, 'cfqs'):
            try:
                fig_cluster = create_cluster_plot(st.session_state.cfqs)
                if fig_cluster is not None:
                    st.plotly_chart(fig_cluster, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating cluster plot: {str(e)}")
            
            st.subheader("Cluster Statistics")
            try:
                cluster_stats = get_cluster_statistics(st.session_state.cfqs)
                st.dataframe(cluster_stats, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating cluster stats: {str(e)}")
        else:
            st.info("Solve the problem first to see clustering analysis")
    
    # Route details
    if st.session_state.routes:
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Route Details")
            
            for i, route in enumerate(st.session_state.routes):
                with st.expander(f"UAV {i+1} - Route Details"):
                    route_df = create_route_dataframe(env, route, i+1)
                    st.dataframe(route_df, use_container_width=True)
        
        with col2:
            st.subheader("⚠️ Missed Opportunities")
            
            missed_df = analyze_missed_opportunities(env, st.session_state.routes)
            
            if missed_df is not None and len(missed_df) > 0:
                st.warning(f"**{len(missed_df)} high-reward nodes were not visited!**")
                
                # Highlight top 5 missed nodes
                top_missed = missed_df.head(5)
                st.dataframe(top_missed, use_container_width=True)
                
                if len(missed_df) > 5:
                    with st.expander(f"See all {len(missed_df)} missed nodes"):
                        st.dataframe(missed_df, use_container_width=True)
                
                # Calculate potential gain
                potential_reward = missed_df['Reward'].sum()
                current_reward = st.session_state.solution_summary['total_reward']
                st.metric(
                    "Potential Additional Reward",
                    f"{potential_reward:.0f}",
                    delta=f"+{(potential_reward/current_reward*100):.1f}%"
                )
            else:
                st.success("✅ All service nodes visited!")


if __name__ == "__main__":
    pass