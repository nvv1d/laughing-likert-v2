
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from utils import simulate_responses, cronbach_alpha

def render_simulation_tab(df):
    """Render the Response Simulation tab"""
    st.header("Response Simulation")
    
    if not st.session_state.weights:
        st.warning("Please extract item weights in the Pattern Extraction tab first")
    else:
        st.subheader("Simulate New Responses")
        
        noise_level = st.slider("Noise Level", 0.0, 1.0, 0.1, 0.05, 
                               help="Higher values create more variable responses")
        
        # Allow user to specify number of simulations (default 100)
        num_simulations = st.number_input("Number of responses to simulate", 
                                       min_value=10, 
                                       max_value=10000, 
                                       value=100,
                                       step=10,
                                       help="Enter the number of simulated responses you want to generate")
        
        # Bias control options
        st.markdown("### Response Bias Control")
        
        enable_bias = st.checkbox("Enable response bias toward higher values", 
                                 value=False,
                                 help="When enabled, simulated responses will be biased toward higher scale values")
        
        bias_strength = 1.0
        bias_percentage = 0.0
        
        if enable_bias:
            col1, col2 = st.columns(2)
            
            with col1:
                bias_strength = st.slider(
                    "Bias Strength", 
                    0.1, 3.0, 1.5, 0.1,
                    help="How much to bias responses upward (1.0 = no bias, >1.0 = higher bias)"
                )
            
            with col2:
                bias_percentage = st.slider(
                    "High-Achiever Percentage", 
                    0.0, 100.0, 30.0, 5.0,
                    help="Percentage of responses to apply strong upward bias (simulates high-achievers)"
                )
            
            st.info(f"📈 Bias active: {bias_percentage:.0f}% of responses will be strongly biased toward higher values (strength: {bias_strength:.1f}x)")
        else:
            st.info("📊 Using original data patterns without bias")
        
        if st.button("Simulate Responses"):
            with st.spinner(f"Simulating {num_simulations} responses..."):
                # Make sure all clustered items are in the weights
                updated_weights = st.session_state.weights.copy()
                
                # Check for missing items from clusters that need to be included
                if st.session_state.clusters:
                    # Get all unique items from clusters
                    all_cluster_items = set()
                    for items in st.session_state.clusters.values():
                        all_cluster_items.update(items)
                    
                    # Add any missing items with default distribution weights
                    for item in all_cluster_items:
                        if item not in updated_weights and item in df.columns:
                            # Use the actual distribution from the data
                            try:
                                counts = df[item].value_counts().sort_index()
                                values = counts.index.values
                                counts_values = counts.values
                                
                                # Normalize to sum to 1
                                if sum(counts_values) > 0:
                                    normalized_weights = counts_values / sum(counts_values)
                                    weight_dict = {val: weight for val, weight in zip(values, normalized_weights)}
                                    
                                    updated_weights[item] = {
                                        'is_distribution': True,
                                        'weights': weight_dict
                                    }
                                else:
                                    # If no data, use equal weights
                                    updated_weights[item] = {
                                        'is_distribution': False,
                                        'weight': 0.5
                                    }
                            except Exception as e:
                                st.warning(f"Could not create weights for {item}: {str(e)}")
                                updated_weights[item] = {
                                    'is_distribution': False,
                                    'weight': 0.5
                                }
                
                # Now simulate with complete weights and bias parameters
                sim_data = simulate_responses(updated_weights, num_simulations, noise_level, 
                                            enable_bias=enable_bias, 
                                            bias_strength=bias_strength, 
                                            bias_percentage=bias_percentage)
                
                # FIX: Reorder columns to match original data order
                # Use the original Likert items order from the session state
                original_order = st.session_state.likert_items
                
                # Ensure all columns from original order exist in simulated data
                available_cols = [col for col in original_order if col in sim_data.columns]
                missing_cols = [col for col in original_order if col not in sim_data.columns]
                
                if missing_cols:
                    st.warning(f"Some columns missing from simulation: {missing_cols[:5]}{'...' if len(missing_cols) > 5 else ''}")
                
                # Reorder the simulated data to match original column order
                if available_cols:
                    sim_data = sim_data[available_cols]
                    st.success(f"✅ Simulated data reordered to match original column sequence")
                
                st.session_state.sim_data = sim_data
                
                # Update the session state weights with the augmented weights to ensure consistency
                st.session_state.weights = updated_weights
        
        if st.session_state.sim_data is not None:
            st.success(f"Generated {len(st.session_state.sim_data)} simulated responses")
            
            # Show preview of simulated data
            st.subheader("Simulated Data Preview")
            with st.expander("View simulated data", expanded=True):
                # Allow user to select how many rows to display (limit to the number of simulated responses)
                total_rows = len(st.session_state.sim_data)
                num_rows = st.slider("Number of rows to display", 5, min(total_rows, 1000), min(10, total_rows))
                
                # Show data preview with selected number of rows (respect user selection)
                st.dataframe(st.session_state.sim_data.head(num_rows))
                
                # Show summary statistics
                st.subheader("Summary Statistics")
                st.dataframe(st.session_state.sim_data.describe())
            
            # Compare original vs simulated data
            st.subheader("Original vs Simulated Distributions")
            
            # Option to select multiple items to compare or single item
            compare_type = st.radio(
                "Comparison type", 
                ["Show one item in detail", "Show multiple items side by side"]
            )
            
            items = list(st.session_state.weights.keys())
            
            if compare_type == "Show one item in detail":
                # Select a single item to visualize in detail
                selected_item = st.selectbox("Select item to visualize", items)
                
                if selected_item:
                    # Create histograms with overlay
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=df[selected_item],
                        name="Original",
                        opacity=0.7,
                        marker_color="blue"
                    ))
                    fig.add_trace(go.Histogram(
                        x=st.session_state.sim_data[selected_item],
                        name="Simulated",
                        opacity=0.7,
                        marker_color="red"
                    ))
                    fig.update_layout(
                        title=f"Distribution Comparison: {selected_item}",
                        xaxis_title="Response Value",
                        yaxis_title="Count",
                        barmode="overlay"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show percentages side by side
                    st.subheader(f"Response Percentages: {selected_item}")
                    
                    # Create a DataFrame with percentages for comparison
                    orig_counts = df[selected_item].value_counts(normalize=True).sort_index() * 100
                    sim_counts = st.session_state.sim_data[selected_item].value_counts(normalize=True).sort_index() * 100
                    
                    comparison_df = pd.DataFrame({
                        'Response': sorted(set(orig_counts.index) | set(sim_counts.index)),
                        'Original (%)': [orig_counts.get(i, 0) for i in sorted(set(orig_counts.index) | set(sim_counts.index))],
                        'Simulated (%)': [sim_counts.get(i, 0) for i in sorted(set(orig_counts.index) | set(sim_counts.index))],
                    })
                    
                    # Format as percentages with 1 decimal point
                    comparison_df['Original (%)'] = comparison_df['Original (%)'].apply(lambda x: f"{x:.1f}%")
                    comparison_df['Simulated (%)'] = comparison_df['Simulated (%)'].apply(lambda x: f"{x:.1f}%")
                    
                    st.dataframe(comparison_df)
                    
            else:  # Show multiple items
                # Select multiple items to compare
                selected_items = st.multiselect(
                    "Select items to compare (max 6 recommended)",
                    options=items,
                    default=items[:min(3, len(items))]
                )
                
                if selected_items:
                    # Create a grid of subplots
                    cols = min(2, len(selected_items))
                    rows = (len(selected_items) + 1) // 2
                    
                    fig = make_subplots(rows=rows, cols=cols, 
                                      subplot_titles=[f"Item: {item}" for item in selected_items])
                    
                    # Add traces for each item
                    for i, item in enumerate(selected_items):
                        row = i // cols + 1
                        col = i % cols + 1
                        
                        # Original data
                        fig.add_trace(
                            go.Histogram(
                                x=df[item],
                                name=f"Original {item}",
                                opacity=0.7,
                                marker_color="blue",
                                showlegend=(i == 0)  # Only show legend once
                            ),
                            row=row, col=col
                        )
                        
                        # Simulated data
                        fig.add_trace(
                            go.Histogram(
                                x=st.session_state.sim_data[item],
                                name=f"Simulated {item}",
                                opacity=0.7,
                                marker_color="red",
                                showlegend=(i == 0)  # Only show legend once
                            ),
                            row=row, col=col
                        )
                        
                        # Update layout for each subplot
                        fig.update_xaxes(title_text="Response Value", row=row, col=col)
                        if col == 1:  # Only for left column
                            fig.update_yaxes(title_text="Count", row=row, col=col)
                    
                    # Update overall layout
                    fig.update_layout(
                        title="Distribution Comparison: Multiple Items",
                        height=300 * rows,
                        barmode="overlay"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Download simulated data
            sim_csv = st.session_state.sim_data.to_csv(index=False)
            st.download_button(
                "Download Simulated Data as CSV",
                sim_csv,
                file_name=f"simulated_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
            # Statistical comparison functionality (abbreviated for space)
            _render_statistical_comparison(df)

def _render_statistical_comparison(df):
    """Render statistical comparison section"""
    st.markdown("---")
    st.subheader("Statistical Comparison: Real vs Simulated Data")
    
    analyze_button = st.button("Compare Real vs Simulated Data")
    
    if analyze_button or 'show_stat_analysis' in st.session_state:
        st.session_state.show_stat_analysis = True
        
        try:
            # Basic descriptive statistics comparison
            with st.expander("Descriptive Statistics Comparison", expanded=True):
                real_desc = df[st.session_state.likert_items].describe().T
                sim_desc = st.session_state.sim_data[st.session_state.likert_items].describe().T
                
                # Calculate additional statistics
                real_desc['var'] = df[st.session_state.likert_items].var()
                sim_desc['var'] = st.session_state.sim_data[st.session_state.likert_items].var()
                
                real_desc['skew'] = df[st.session_state.likert_items].skew()
                sim_desc['skew'] = st.session_state.sim_data[st.session_state.likert_items].skew()
                
                # Calculate mean absolute differences
                stats_diff = {}
                metrics = ['mean', 'std', 'var', 'skew', 'min', '25%', '50%', '75%', 'max']
                for metric in metrics:
                    if metric in real_desc.columns and metric in sim_desc.columns:
                        diff = abs(real_desc[metric] - sim_desc[metric])
                        stats_diff[metric] = diff.mean()
                
                # Overall similarity score
                overall_similarity = sum(max(0, 100 - 100 * stats_diff[k]) for k in stats_diff.keys()) / len(stats_diff)
                st.metric("Overall Statistical Similarity", f"{overall_similarity:.2f}%")
            
            # Correlation structure comparison
            with st.expander("Correlation Structure Comparison", expanded=True):
                real_corr = df[st.session_state.likert_items].corr()
                sim_corr = st.session_state.sim_data[st.session_state.likert_items].corr()
                
                # Calculate correlation matrix difference
                corr_diff = abs(real_corr - sim_corr)
                mean_diff = corr_diff.values[np.triu_indices_from(corr_diff.values, k=1)].mean()
                corr_similarity = max(0, 100 - (mean_diff * 100))
                st.metric("Correlation Structure Similarity", f"{corr_similarity:.2f}%")
            
            # Reliability comparison
            if st.session_state.clusters:
                with st.expander("Reliability Comparison", expanded=True):
                    alpha_data = []
                    for sc, items in st.session_state.clusters.items():
                        if len(items) > 1 and all(item in st.session_state.sim_data.columns for item in items):
                            orig_alpha = st.session_state.alphas.get(sc, 0)
                            sim_alpha = cronbach_alpha(st.session_state.sim_data, items)
                            alpha_diff = abs(orig_alpha - sim_alpha)
                            alpha_similarity = max(0, 100 - (alpha_diff * 100))
                            
                            alpha_data.append({
                                'Cluster': sc,
                                'Original Alpha': orig_alpha,
                                'Simulated Alpha': sim_alpha,
                                'Similarity (%)': alpha_similarity
                            })
                    
                    if alpha_data:
                        alpha_df = pd.DataFrame(alpha_data)
                        st.dataframe(alpha_df)
                        reliability_similarity = alpha_df['Similarity (%)'].mean()
                        st.metric("Overall Reliability Similarity", f"{reliability_similarity:.2f}%")
            
        except Exception as e:
            st.error(f"Error performing statistical comparison: {str(e)}")
