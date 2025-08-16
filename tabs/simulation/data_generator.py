import streamlit as st
import pandas as pd
from utils import simulate_responses

class DataGenerator:
    """Component for handling data generation logic"""
    
    def __init__(self):
        self.updated_weights = None
    
    def prepare_weights_for_simulation(self, df, base_weights, clusters):
        """
        Prepare and augment weights for simulation, ensuring all clustered items are included.
        
        Parameters:
        - df: Original DataFrame
        - base_weights: Base weights from session state
        - clusters: Cluster information from session state
        
        Returns:
        - Updated weights dictionary
        """
        self.updated_weights = base_weights.copy()
        
        # Check for missing items from clusters that need to be included
        if clusters:
            # Get all unique items from clusters
            all_cluster_items = set()
            for items in clusters.values():
                all_cluster_items.update(items)
            
            # Add any missing items with default distribution weights
            for item in all_cluster_items:
                if item not in self.updated_weights and item in df.columns:
                    # Use the actual distribution from the data
                    try:
                        counts = df[item].value_counts().sort_index()
                        values = counts.index.values
                        counts_values = counts.values
                        
                        # Normalize to sum to 1
                        if sum(counts_values) > 0:
                            normalized_weights = counts_values / sum(counts_values)
                            weight_dict = {val: weight for val, weight in zip(values, normalized_weights)}
                            
                            self.updated_weights[item] = {
                                'is_distribution': True,
                                'weights': weight_dict
                            }
                        else:
                            # If no data, use equal weights
                            self.updated_weights[item] = {
                                'is_distribution': False,
                                'weight': 0.5
                            }
                    except Exception as e:
                        st.warning(f"Could not create weights for {item}: {str(e)}")
                        self.updated_weights[item] = {
                            'is_distribution': False,
                            'weight': 0.5
                        }
        
        return self.updated_weights
    
    def generate_simulated_data(self, weights, num_simulations, noise_level, likert_items):
        """
        Generate simulated data and reorder columns to match original data order.
        
        Parameters:
        - weights: Weights dictionary for simulation
        - num_simulations: Number of responses to simulate
        - noise_level: Noise level for simulation
        - likert_items: Original order of likert items
        
        Returns:
        - DataFrame with simulated data
        """
        # Generate simulated data
        sim_data = simulate_responses(weights, num_simulations, noise_level)
        
        # Reorder columns to match original data order
        original_order = likert_items
        
        # Ensure all columns from original order exist in simulated data
        available_cols = [col for col in original_order if col in sim_data.columns]
        missing_cols = [col for col in original_order if col not in sim_data.columns]
        
        if missing_cols:
            st.warning(f"Some columns missing from simulation: {missing_cols[:5]}{'...' if len(missing_cols) > 5 else ''}")
        
        # Reorder the simulated data to match original column order
        if available_cols:
            sim_data = sim_data[available_cols]
            st.success(f"✅ Simulated data reordered to match original column sequence")
        
        return sim_data
    
    def render_simulation_settings(self):
        """Render basic simulation settings UI"""
        st.subheader("Simulate New Responses")
        
        # Base simulation settings
        col1, col2 = st.columns(2)
        
        with col1:
            noise_level = st.slider("Noise Level", 0.0, 1.0, 0.1, 0.05, 
                                   help="Higher values create more variable responses")
        
        with col2:
            num_simulations = st.number_input("Number of responses to simulate", 
                                           min_value=10, 
                                           max_value=10000, 
                                           value=100,
                                           step=10,
                                           help="Enter the number of simulated responses you want to generate")
        
        return {
            'noise_level': noise_level,
            'num_simulations': num_simulations
        }
    
    def render_data_preview(self, sim_data):
        """Render simulated data preview"""
        st.subheader("Simulated Data Preview")
        with st.expander("View simulated data", expanded=True):
            # Allow user to select how many rows to display
            total_rows = len(sim_data)
            num_rows = st.slider("Number of rows to display", 5, min(total_rows, 1000), min(10, total_rows))
            
            # Show data preview with selected number of rows
            st.dataframe(sim_data.head(num_rows))
            
            # Show summary statistics
            st.subheader("Summary Statistics")
            st.dataframe(sim_data.describe())
