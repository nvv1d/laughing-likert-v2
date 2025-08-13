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
                    
                    # FIX: Handle mixed data types when combining indices
                    try:
                        # Get all unique response values
                        all_responses = set(orig_counts.index) | set(sim_counts.index)
                        
                        # Convert all to strings for consistent sorting, then convert back if they're all numeric
                        str_responses = [str(x) for x in all_responses]
                        
                        # Try to sort as numbers if possible, otherwise sort as strings
                        try:
                            # Check if all can be converted to numbers
                            numeric_responses = [float(x) for x in str_responses]
                            sorted_responses = sorted(all_responses, key=lambda x: float(x))
                        except (ValueError, TypeError):
                            # If not all numeric, sort as strings
                            sorted_responses = sorted(all_responses, key=str)
                        
                        comparison_df = pd.DataFrame({
                            'Response': sorted_responses,
                            'Original (%)': [orig_counts.get(i, 0) for i in sorted_responses],
                            'Simulated (%)': [sim_counts.get(i, 0) for i in sorted_responses],
                        })
                        
                        # Format as percentages with 1 decimal point
                        comparison_df['Original (%)'] = comparison_df['Original (%)'].apply(lambda x: f"{x:.1f}%")
                        comparison_df['Simulated (%)'] = comparison_df['Simulated (%)'].apply(lambda x: f"{x:.1f}%")
                        
                        st.dataframe(comparison_df)
                        
                    except Exception as e:
                        st.error(f"Error creating response percentage comparison: {str(e)}")
                        # Fallback: show basic statistics
                        st.write("**Original Data Distribution:**")
                        st.write(orig_counts.to_dict())
                        st.write("**Simulated Data Distribution:**")
                        st.write(sim_counts.to_dict())
                    
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
                # Filter to only numeric columns for statistical analysis
                numeric_items = []
                for item in st.session_state.likert_items:
                    if item in df.columns and item in st.session_state.sim_data.columns:
                        # Check if both original and simulated data for this item are numeric
                        try:
                            pd.to_numeric(df[item], errors='raise')
                            pd.to_numeric(st.session_state.sim_data[item], errors='raise')
                            numeric_items.append(item)
                        except (ValueError, TypeError):
                            continue
                
                if not numeric_items:
                    st.warning("No numeric items found for statistical comparison.")
                    return
                
                # Convert to numeric to ensure consistent data types
                real_numeric = df[numeric_items].apply(pd.to_numeric, errors='coerce')
                sim_numeric = st.session_state.sim_data[numeric_items].apply(pd.to_numeric, errors='coerce')
                
                # Remove any remaining NaN values
                real_numeric = real_numeric.dropna()
                sim_numeric = sim_numeric.dropna()
                
                if real_numeric.empty or sim_numeric.empty:
                    st.warning("No valid numeric data available for comparison after cleaning.")
                    return
                
                real_desc = real_numeric.describe().T
                sim_desc = sim_numeric.describe().T
                
                # Calculate additional statistics
                real_desc['var'] = real_numeric.var()
                sim_desc['var'] = sim_numeric.var()
                
                real_desc['skew'] = real_numeric.skew()
                sim_desc['skew'] = sim_numeric.skew()
                
                # Calculate mean absolute differences
                stats_diff = {}
                metrics = ['mean', 'std', 'var', 'skew', 'min', '25%', '50%', '75%', 'max']
                
                for metric in metrics:
                    if metric in real_desc.columns and metric in sim_desc.columns:
                        # Ensure both are numeric before subtraction
                        real_metric = pd.to_numeric(real_desc[metric], errors='coerce')
                        sim_metric = pd.to_numeric(sim_desc[metric], errors='coerce')
                        
                        # Only calculate difference for items present in both datasets
                        common_items = real_metric.index.intersection(sim_metric.index)
                        if len(common_items) > 0:
                            diff = abs(real_metric[common_items] - sim_metric[common_items])
                            stats_diff[metric] = diff.mean()
                
                if stats_diff:
                    # Overall similarity score
                    overall_similarity = sum(max(0, 100 - 100 * stats_diff[k]) for k in stats_diff.keys()) / len(stats_diff)
                    st.metric("Overall Statistical Similarity", f"{overall_similarity:.2f}%")
                    
                    # Show detailed comparison table
                    comparison_data = []
                    for item in common_items:
                        row_data = {'Item': item}
                        for metric in ['mean', 'std', 'var']:
                            if metric in real_desc.columns and metric in sim_desc.columns:
                                real_val = pd.to_numeric(real_desc.loc[item, metric], errors='coerce')
                                sim_val = pd.to_numeric(sim_desc.loc[item, metric], errors='coerce')
                                row_data[f'Real_{metric}'] = real_val
                                row_data[f'Sim_{metric}'] = sim_val
                                row_data[f'Diff_{metric}'] = abs(real_val - sim_val) if pd.notna(real_val) and pd.notna(sim_val) else np.nan
                        comparison_data.append(row_data)
                    
                    if comparison_data:
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df.round(3))
                else:
                    st.warning("Could not calculate statistical differences.")
            
            # Correlation structure comparison
            with st.expander("Correlation Structure Comparison", expanded=True):
                try:
                    # Use only numeric items for correlation analysis
                    if len(numeric_items) < 2:
                        st.warning("Need at least 2 numeric items for correlation analysis.")
                    else:
                        real_corr = real_numeric.corr()
                        sim_corr = sim_numeric.corr()
                        
                        # Calculate correlation matrix difference
                        corr_diff = abs(real_corr - sim_corr)
                        
                        # Get upper triangle indices (excluding diagonal)
                        mask = np.triu(np.ones_like(corr_diff.values, dtype=bool), k=1)
                        upper_triangle_diff = corr_diff.values[mask]
                        
                        if len(upper_triangle_diff) > 0:
                            mean_diff = np.nanmean(upper_triangle_diff)
                            corr_similarity = max(0, 100 - (mean_diff * 100))
                            st.metric("Correlation Structure Similarity", f"{corr_similarity:.2f}%")
                            
                            # Show correlation comparison
                            st.write("**Average Correlations:**")
                            avg_real_corr = np.nanmean(real_corr.values[mask])
                            avg_sim_corr = np.nanmean(sim_corr.values[mask])
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Real Data Avg Correlation", f"{avg_real_corr:.3f}")
                            with col2:
                                st.metric("Simulated Data Avg Correlation", f"{avg_sim_corr:.3f}")
                        else:
                            st.warning("Could not calculate correlation differences.")
                            
                except Exception as corr_error:
                    st.warning(f"Correlation analysis failed: {str(corr_error)}")
            
            # Reliability comparison
            if st.session_state.clusters:
                with st.expander("Reliability Comparison", expanded=True):
                    try:
                        alpha_data = []
                        for sc, items in st.session_state.clusters.items():
                            # Filter to only numeric items in this cluster
                            numeric_cluster_items = [item for item in items if item in numeric_items]
                            
                            if len(numeric_cluster_items) > 1 and all(item in st.session_state.sim_data.columns for item in numeric_cluster_items):
                                orig_alpha = st.session_state.alphas.get(sc, 0)
                                
                                # Calculate Cronbach's alpha for simulated data
                                try:
                                    sim_alpha = cronbach_alpha(sim_numeric, numeric_cluster_items)
                                    alpha_diff = abs(orig_alpha - sim_alpha)
                                    alpha_similarity = max(0, 100 - (alpha_diff * 100))
                                    
                                    alpha_data.append({
                                        'Cluster': sc,
                                        'Items_Count': len(numeric_cluster_items),
                                        'Original Alpha': round(orig_alpha, 3),
                                        'Simulated Alpha': round(sim_alpha, 3),
                                        'Difference': round(alpha_diff, 3),
                                        'Similarity (%)': round(alpha_similarity, 1)
                                    })
                                except Exception as alpha_error:
                                    st.warning(f"Could not calculate alpha for cluster {sc}: {str(alpha_error)}")
                        
                        if alpha_data:
                            alpha_df = pd.DataFrame(alpha_data)
                            st.dataframe(alpha_df)
                            reliability_similarity = alpha_df['Similarity (%)'].mean()
                            st.metric("Overall Reliability Similarity", f"{reliability_similarity:.2f}%")
                        else:
                            st.warning("No reliability comparisons could be calculated.")
                            
                    except Exception as reliability_error:
                        st.warning(f"Reliability analysis failed: {str(reliability_error)}")
            
        except Exception as e:
            st.error(f"Error performing statistical comparison: {str(e)}")
            st.write("**Debug Info:**")
            st.write(f"Available items: {st.session_state.likert_items[:5]}...")
            st.write(f"Data types in original: {df[st.session_state.likert_items[:3]].dtypes.to_dict()}")
            st.write(f"Data types in simulated: {st.session_state.sim_data[st.session_state.likert_items[:3]].dtypes.to_dict()}")
