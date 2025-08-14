
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from utils import simulate_responses, cronbach_alpha

def detect_scales_by_pattern(columns):
    """
    Automatically detect scales based on naming patterns like A1, A2, A3, B1, B2, B3, etc.
    Returns a dictionary with scale names as keys and lists of items as values.
    """
    import re
    from collections import defaultdict
    
    scales = defaultdict(list)

    # Pattern 1: Letter + Number (A1, A2, B1, B2, etc.)
    pattern1 = re.compile(r'^([A-Za-z]+)(\d+)$')

    # Pattern 2: Word + Number (Scale1_1, Scale1_2, Scale2_1, etc.)
    pattern2 = re.compile(r'^([A-Za-z_]+?)(\d+)(?:_(\d+))?$')

    # Pattern 3: Common prefixes (att1, att2, sat1, sat2, etc.)
    pattern3 = re.compile(r'^([A-Za-z]+)(\d+)$')

    for col in columns:
        # Try pattern 1: A1, A2, B1, B2
        match1 = pattern1.match(col)
        if match1:
            prefix = match1.group(1).upper()
            number = int(match1.group(2))
            scales[f"Scale_{prefix}"].append((col, number))
            continue

        # Try pattern 2: Scale1_1, Scale1_2, Scale2_1
        match2 = pattern2.match(col)
        if match2:
            prefix = match2.group(1)
            if match2.group(3):  # Has underscore number
                scale_num = int(match2.group(2))
                item_num = int(match2.group(3))
                scales[f"Scale_{prefix}_{scale_num}"].append((col, item_num))
            else:
                number = int(match2.group(2))
                scales[f"Scale_{prefix}"].append((col, number))
            continue

    # Sort items within each scale by their numbers and return just the column names
    sorted_scales = {}
    for scale_name, items in scales.items():
        if len(items) >= 2:  # Only keep scales with at least 2 items
            sorted_items = sorted(items, key=lambda x: x[1])
            sorted_scales[scale_name] = [item[0] for item in sorted_items]

    return sorted_scales

def apply_bias_to_weights(weights, bias_type, bias_strength, bias_percentage):
    """
    Apply bias to existing weights to simulate high-achievers or low-achievers.
    
    Parameters:
    - weights: Original weights dictionary
    - bias_type: 'high' or 'low'
    - bias_strength: float 0-1, how strong the bias is
    - bias_percentage: float 0-1, what percentage of responses should be biased
    
    Returns:
    - Modified weights dictionary
    """
    biased_weights = weights.copy()

    for item, w_data in biased_weights.items():
        if isinstance(w_data, dict) and w_data.get('is_distribution', False):
            original_dist = w_data['weights'].copy()

            # Create biased distribution
            biased_dist = {}

            # Get all possible values and sort them
            values = sorted([int(k) for k in original_dist.keys()])

            if bias_type == 'high':
                # Shift probability mass toward higher values
                target_values = values[-2:]  # Top 2 values
            else:  # bias_type == 'low'
                # Shift probability mass toward lower values
                target_values = values[:2]  # Bottom 2 values

            # Calculate bias adjustment
            for val_str, prob in original_dist.items():
                val = int(val_str)

                if val in target_values:
                    # Increase probability for target values
                    bias_multiplier = 1 + (bias_strength * bias_percentage)
                    biased_dist[val_str] = prob * bias_multiplier
                else:
                    # Decrease probability for other values
                    bias_multiplier = 1 - (bias_strength * bias_percentage * 0.5)
                    biased_dist[val_str] = prob * bias_multiplier

            # Normalize to ensure probabilities sum to 1
            total_prob = sum(biased_dist.values())
            if total_prob > 0:
                biased_dist = {k: v/total_prob for k, v in biased_dist.items()}

            # Update the weights
            biased_weights[item] = {
                'is_distribution': True,
                'weights': biased_dist
            }

    return biased_weights

def render_simulation_tab(df):
    """Render the Response Simulation tab with enhanced bias options and comprehensive analysis"""
    st.header("Response Simulation")
    
    if not st.session_state.weights:
        st.warning("Please extract item weights in the Pattern Extraction tab first")
    else:
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
        
        # Enhanced Bias Options
        st.markdown("---")
        st.subheader("🎯 Response Bias Options")
        
        enable_bias = st.checkbox(
            "Enable Response Bias", 
            value=False,
            help="Apply systematic bias to simulate specific response patterns"
        )
        
        if enable_bias:
            st.info("🔧 Configure bias settings to simulate different respondent types (e.g., high-achievers, pessimists)")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                bias_type = st.selectbox(
                    "Bias Direction",
                    options=["high", "low"],
                    format_func=lambda x: "High Bias (optimistic/high-achievers)" if x == "high" else "Low Bias (pessimistic/critical)",
                    help="Direction of bias: high = toward maximum scale values, low = toward minimum scale values"
                )
            
            with col2:
                bias_strength = st.slider(
                    "Bias Strength",
                    min_value=0.1,
                    max_value=2.0,
                    value=0.5,
                    step=0.1,
                    help="How strong the bias is (higher = more extreme bias)"
                )
            
            with col3:
                bias_percentage = st.slider(
                    "Percentage Affected",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.3,
                    step=0.05,
                    format_func=lambda x: f"{x*100:.0f}%",
                    help="What percentage of responses should be affected by bias"
                )
            
            # Bias explanation and preview
            with st.expander("🔍 Bias Configuration Preview", expanded=True):
                if bias_type == "high":
                    st.success(f"**High Bias Configuration:**")
                    st.write(f"- **Effect**: {bias_percentage*100:.0f}% of responses will be biased toward higher values")
                    st.write(f"- **Strength**: {bias_strength:.1f}x increase in probability for top 2 scale values")
                    st.write(f"- **Use case**: Simulate high-achievers, optimistic respondents, or positive response bias")
                else:
                    st.warning(f"**Low Bias Configuration:**")
                    st.write(f"- **Effect**: {bias_percentage*100:.0f}% of responses will be biased toward lower values")
                    st.write(f"- **Strength**: {bias_strength:.1f}x increase in probability for bottom 2 scale values")
                    st.write(f"- **Use case**: Simulate critical respondents, pessimistic views, or negative response bias")
                
                st.write(f"- **Unbiased responses**: {(1-bias_percentage)*100:.0f}% will follow original patterns")
                
                # Show example of how bias would affect a 5-point scale
                st.write("**Example Effect on 5-point Scale (1-5):**")
                example_original = {1: 0.1, 2: 0.2, 3: 0.4, 4: 0.2, 5: 0.1}
                
                if bias_type == "high":
                    target_values = [4, 5]
                else:
                    target_values = [1, 2]
                
                example_biased = {}
                for val, prob in example_original.items():
                    if val in target_values:
                        example_biased[val] = prob * (1 + (bias_strength * bias_percentage))
                    else:
                        example_biased[val] = prob * (1 - (bias_strength * bias_percentage * 0.5))
                
                # Normalize
                total_prob = sum(example_biased.values())
                example_biased = {k: v/total_prob for k, v in example_biased.items()}
                
                comparison_df = pd.DataFrame({
                    'Scale Value': [1, 2, 3, 4, 5],
                    'Original': [example_original[i] for i in [1, 2, 3, 4, 5]],
                    'Biased': [example_biased[i] for i in [1, 2, 3, 4, 5]]
                })
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=comparison_df['Scale Value'],
                    y=comparison_df['Original'],
                    name='Original Distribution',
                    marker_color='lightblue'
                ))
                fig.add_trace(go.Bar(
                    x=comparison_df['Scale Value'],
                    y=comparison_df['Biased'],
                    name='Biased Distribution',
                    marker_color='orange'
                ))
                fig.update_layout(
                    title="Example: Bias Effect on Response Distribution",
                    xaxis_title="Scale Value",
                    yaxis_title="Probability",
                    barmode='group',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Simulation button
        if st.button("🚀 Simulate Responses"):
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
                
                # Apply bias if enabled
                if enable_bias:
                    st.info(f"Applying {bias_type} bias (strength: {bias_strength}, affected: {bias_percentage*100:.0f}%)")
                    updated_weights = apply_bias_to_weights(
                        updated_weights, bias_type, bias_strength, bias_percentage
                    )
                
                # Now simulate with complete weights
                sim_data = simulate_responses(updated_weights, num_simulations, noise_level)
                
                # Reorder columns to match original data order
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
                st.success(f"🎯 Successfully generated {num_simulations} responses!")
        
        if st.session_state.sim_data is not None:
            st.success(f"Generated {len(st.session_state.sim_data)} simulated responses")
            
            # Show preview of simulated data
            st.subheader("Simulated Data Preview")
            with st.expander("View simulated data", expanded=True):
                # Allow user to select how many rows to display
                total_rows = len(st.session_state.sim_data)
                num_rows = st.slider("Number of rows to display", 5, min(total_rows, 1000), min(10, total_rows))
                
                # Show data preview with selected number of rows
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
                    
                    # Handle mixed data types when combining indices
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
            
            # Statistical comparison functionality - enhanced version from uploaded file
            _render_comprehensive_statistical_comparison(df)

def _render_comprehensive_statistical_comparison(df):
    """Render comprehensive statistical comparison section matching uploaded file functionality"""
    st.markdown("---")
    st.subheader("Statistical Comparison: Real vs Simulated Data")
    
    # Add a button to perform comprehensive statistical comparison
    analyze_button = st.button("Compare Real vs Simulated Data")
    
    if analyze_button or 'show_stat_analysis' in st.session_state:
        st.session_state.show_stat_analysis = True
        
        try:
            # 1. Basic descriptive statistics comparison
            with st.expander("Descriptive Statistics Comparison", expanded=True):
                # Calculate descriptive statistics
                real_desc = df[st.session_state.likert_items].describe().T
                sim_desc = st.session_state.sim_data[st.session_state.likert_items].describe().T
                
                # Calculate additional statistics
                real_desc['var'] = df[st.session_state.likert_items].var()
                sim_desc['var'] = st.session_state.sim_data[st.session_state.likert_items].var()
                
                real_desc['skew'] = df[st.session_state.likert_items].skew()
                sim_desc['skew'] = st.session_state.sim_data[st.session_state.likert_items].skew()
                
                real_desc['kurtosis'] = df[st.session_state.likert_items].kurtosis()
                sim_desc['kurtosis'] = st.session_state.sim_data[st.session_state.likert_items].kurtosis()
                
                # Calculate mean absolute differences between real and simulated statistics
                stats_diff = {}
                metrics = ['mean', 'std', 'var', 'skew', 'kurtosis', 'min', '25%', '50%', '75%', 'max']
                for metric in metrics:
                    if metric in real_desc.columns and metric in sim_desc.columns:
                        diff = abs(real_desc[metric] - sim_desc[metric])
                        stats_diff[metric] = diff.mean()
                
                # Create comparison charts for key metrics
                selected_stats = st.multiselect(
                    "Select statistics to compare", 
                    options=list(stats_diff.keys()),
                    default=['mean', 'std', 'var']
                )
                
                for stat in selected_stats:
                    st.subheader(f"Comparison of {stat.capitalize()}")
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=real_desc.index,
                        y=real_desc[stat],
                        name="Original",
                        marker_color='blue'
                    ))
                    fig.add_trace(go.Bar(
                        x=sim_desc.index,
                        y=sim_desc[stat],
                        name="Simulated",
                        marker_color='red'
                    ))
                    fig.update_layout(
                        title=f"{stat.capitalize()} Comparison",
                        barmode='group',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Overall similarity scores
                st.subheader("Similarity Metrics")
                similarity_df = pd.DataFrame({
                    'Statistic': list(stats_diff.keys()),
                    'Mean Absolute Difference': [stats_diff[k] for k in stats_diff.keys()],
                    'Similarity Score (%)': [max(0, 100 - 100 * stats_diff[k]) for k in stats_diff.keys()]
                })
                st.dataframe(similarity_df.sort_values('Similarity Score (%)', ascending=False))
                
                # Overall similarity score
                overall_similarity = similarity_df['Similarity Score (%)'].mean()
                st.metric("Overall Statistical Similarity", f"{overall_similarity:.2f}%")
            
            # 2. Correlation structure comparison
            with st.expander("Correlation Structure Comparison", expanded=True):
                # Calculate correlation matrices
                real_corr = df[st.session_state.likert_items].corr()
                sim_corr = st.session_state.sim_data[st.session_state.likert_items].corr()
                
                # Calculate correlation matrix difference
                corr_diff = abs(real_corr - sim_corr)
                
                # Display side-by-side correlation heatmaps
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("Original Correlation Matrix")
                    fig = px.imshow(
                        real_corr, 
                        title="Original Data Correlations",
                        color_continuous_scale="Blues",
                        zmin=-1, zmax=1
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.write("Simulated Correlation Matrix")
                    fig = px.imshow(
                        sim_corr, 
                        title="Simulated Data Correlations",
                        color_continuous_scale="Reds",
                        zmin=-1, zmax=1
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col3:
                    st.write("Difference Matrix")
                    fig = px.imshow(
                        corr_diff, 
                        title="Correlation Difference (Absolute)",
                        color_continuous_scale="Greens",
                        zmin=0, zmax=2
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Calculate overall correlation similarity
                mean_diff = corr_diff.values[np.triu_indices_from(corr_diff.values, k=1)].mean()
                corr_similarity = max(0, 100 - (mean_diff * 100))
                st.metric("Correlation Structure Similarity", f"{corr_similarity:.2f}%")
            
            # 3. Distribution comparison
            with st.expander("Distribution Comparison", expanded=True):
                # Select items to compare
                dist_items = st.multiselect(
                    "Select items for distribution comparison",
                    options=st.session_state.likert_items,
                    default=st.session_state.likert_items[:min(3, len(st.session_state.likert_items))]
                )
                
                if dist_items:
                    # Calculate KL divergence for each item
                    kl_divergences = {}
                    js_distances = {}
                    
                    for item in dist_items:
                        # Calculate distribution proportions
                        real_dist = df[item].value_counts(normalize=True).sort_index()
                        sim_dist = st.session_state.sim_data[item].value_counts(normalize=True).sort_index()
                        
                        # Ensure both distributions have the same categories
                        all_values = sorted(set(real_dist.index) | set(sim_dist.index))
                        real_probs = np.array([real_dist.get(val, 0) for val in all_values])
                        sim_probs = np.array([sim_dist.get(val, 0) for val in all_values])
                        
                        # Avoid zero probabilities for KL divergence
                        real_probs = np.clip(real_probs, 1e-10, 1.0)
                        sim_probs = np.clip(sim_probs, 1e-10, 1.0)
                        
                        # Normalize to sum to 1
                        real_probs = real_probs / real_probs.sum()
                        sim_probs = sim_probs / sim_probs.sum()
                        
                        try:
                            # Calculate KL divergence: D_KL(P||Q)
                            kl_div = np.sum(real_probs * np.log(real_probs / sim_probs))
                            kl_divergences[item] = kl_div
                            
                            # Calculate Jensen-Shannon distance
                            m_dist = 0.5 * (real_probs + sim_probs)
                            js_div = 0.5 * np.sum(real_probs * np.log(real_probs / m_dist)) + 0.5 * np.sum(sim_probs * np.log(sim_probs / m_dist))
                            js_distances[item] = np.sqrt(js_div)  # JS distance is sqrt of JS divergence
                        except Exception as e:
                            st.warning(f"Could not calculate divergence for item {item}: {str(e)}")
                    
                    # Display the divergence measures
                    divergence_df = pd.DataFrame({
                        'Item': list(kl_divergences.keys()),
                        'KL Divergence': list(kl_divergences.values()),
                        'JS Distance': list(js_distances.values()),
                        'Distribution Similarity (%)': [max(0, 100 - (js * 100)) for js in js_distances.values()]
                    })
                    st.dataframe(divergence_df.sort_values('Distribution Similarity (%)', ascending=False))
                    
                    # Overall distribution similarity
                    dist_similarity = divergence_df['Distribution Similarity (%)'].mean()
                    st.metric("Overall Distribution Similarity", f"{dist_similarity:.2f}%")
            
            # 4. Reliability Measures (Cronbach's Alpha comparison)
            if st.session_state.clusters:
                with st.expander("Reliability Comparison", expanded=True):
                    # Calculate alphas for original and simulated data by cluster
                    sim_alphas = {}
                    
                    # First verify all required columns exist in simulated data
                    available_items = set(st.session_state.sim_data.columns)
                    
                    # Make sure all items exist in the simulated data
                    valid_items = {}
                    for sc, items in st.session_state.clusters.items():
                        # Check if all items in this cluster exist in the simulated data
                        valid_cluster_items = [item for item in items if item in st.session_state.sim_data.columns]
                        if len(valid_cluster_items) > 1:  # Need at least 2 items for reliability
                            valid_items[sc] = valid_cluster_items
                    
                    if valid_items:
                        alpha_data = []
                        
                        for sc, items in valid_items.items():
                            try:
                                orig_alpha = st.session_state.alphas.get(sc, 0)
                                sim_alpha = cronbach_alpha(st.session_state.sim_data, items)
                                
                                alpha_diff = abs(orig_alpha - sim_alpha)
                                alpha_similarity = max(0, 100 - (alpha_diff * 100))
                                
                                alpha_data.append({
                                    'Cluster': sc,
                                    'Items': len(items),
                                    'Original Alpha': orig_alpha,
                                    'Simulated Alpha': sim_alpha,
                                    'Difference': alpha_diff,
                                    'Similarity (%)': alpha_similarity
                                })
                            except Exception as e:
                                st.warning(f"Could not calculate reliability for cluster {sc}: {str(e)}")
                        
                        # Display results table
                        alpha_df = pd.DataFrame(alpha_data)
                        if not alpha_df.empty:
                            st.dataframe(alpha_df.sort_values('Similarity (%)', ascending=False))
                            
                            # Overall reliability similarity
                            reliability_similarity = alpha_df['Similarity (%)'].mean()
                            st.metric("Overall Reliability Similarity", f"{reliability_similarity:.2f}%")
                            
                            # Display clusters with reliability issues
                            problem_clusters = alpha_df[alpha_df['Similarity (%)'] < 75]
                            if not problem_clusters.empty:
                                st.warning("The following clusters show significant reliability differences:")
                                st.dataframe(problem_clusters[['Cluster', 'Original Alpha', 'Simulated Alpha', 'Similarity (%)']])
                                
                                # Suggestions for improvement
                                st.info("💡 Suggestions: Use a larger dataset or try adjusting noise level or using different item weights extraction methods.")
            
            # Final combined score
            overall_metrics = []
            
            if 'overall_similarity' in locals():
                overall_metrics.append(('Statistical Descriptives', overall_similarity))
            
            if 'corr_similarity' in locals():
                overall_metrics.append(('Correlation Structure', corr_similarity))
            
            if 'dist_similarity' in locals():
                overall_metrics.append(('Distribution Similarity', dist_similarity))
            
            if 'reliability_similarity' in locals():
                overall_metrics.append(('Reliability Metrics', reliability_similarity))
            
            if overall_metrics:
                st.markdown("---")
                st.subheader("Overall Similarity Assessment")
                
                final_score = sum(score for _, score in overall_metrics) / len(overall_metrics)
                
                # Create a gauge chart for final score
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = final_score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Overall Simulation Quality"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "royalblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "red"},
                            {'range': [50, 75], 'color': "orange"},
                            {'range': [75, 90], 'color': "yellow"},
                            {'range': [90, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': final_score
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Individual score breakdown
                score_df = pd.DataFrame(overall_metrics, columns=['Metric', 'Score (%)'])
                fig = px.bar(
                    score_df, 
                    x='Score (%)', 
                    y='Metric', 
                    orientation='h',
                    title="Component Similarity Scores",
                    color='Score (%)',
                    color_continuous_scale=px.colors.sequential.Viridis,
                    range_color=[0, 100]
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Quality assessment
                if final_score >= 90:
                    st.success("🌟 Excellent simulation quality! The simulated data closely matches the original data across all metrics.")
                elif final_score >= 80:
                    st.success("✅ Good simulation quality. The simulated data captures most patterns in the original data.")
                elif final_score >= 70:
                    st.warning("⚠️ Fair simulation quality. The simulated data captures some patterns but has notable differences.")
                else:
                    st.error("❌ Poor simulation quality. Consider adjusting parameters to improve results.")
                
        except Exception as e:
            st.error(f"Error performing statistical comparison: {str(e)}")
    else:
        st.info("Click the 'Compare Real vs Simulated Data' button above for a comprehensive statistical comparison between your original and simulated data.")
