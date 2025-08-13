
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import identify_likert_columns, detect_reverse_items, reverse_code, check_sampling, cronbach_alpha, bootstrap_alpha

def render_data_preparation_tab(df, min_cats, max_cats, reverse_threshold):
    """Render the Data Preparation tab"""
    st.header("Data Preparation")
    
    # Show data summary first
    st.subheader("Data Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", df.shape[0])
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        st.metric("Numeric Columns", numeric_cols)
    
    # Enhanced Likert item detection
    st.subheader("Likert Item Detection")
    
    # Add detection method selection
    detection_method = st.radio(
        "Detection Method",
        ["Automatic (Smart Detection)", "Manual Selection", "All Numeric Columns"],
        help="Choose how to identify Likert scale items"
    )
    
    if detection_method == "Automatic (Smart Detection)":
        col1, col2 = st.columns(2)
        with col1:
            min_cats_display = st.slider("Minimum Categories", 2, 10, min_cats,
                                key="min_cats_display",
                                help="Minimum number of unique values to identify Likert items")
        with col2:
            max_cats_display = st.slider("Maximum Categories", min_cats_display, 15, max_cats,
                                key="max_cats_display", 
                                help="Maximum number of unique values to identify Likert items")
        
        # Re-identify Likert items with new parameters
        if st.button("Re-detect Likert Items") or len(st.session_state.likert_items) == 0:
            with st.spinner("Analyzing columns for Likert scale characteristics..."):
                detected_items = identify_likert_columns(df, min_cats_display, max_cats_display)
                st.session_state.likert_items = detected_items
        
        # Show detection results with detailed breakdown
        st.subheader("Detection Results")
        
        # Analyze all numeric columns and show why some were excluded
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        detection_details = []
        
        for col in numeric_columns:
            unique_vals = df[col].nunique()
            min_val = df[col].min()
            max_val = df[col].max()
            has_negatives = min_val < 0
            is_detected = col in st.session_state.likert_items
            
            # Determine why it was included/excluded
            if is_detected:
                reason = "✅ Detected as Likert item"
            elif unique_vals < min_cats_display:
                reason = f"❌ Too few categories ({unique_vals} < {min_cats_display})"
            elif unique_vals > max_cats_display:
                reason = f"❌ Too many categories ({unique_vals} > {max_cats_display})"
            elif has_negatives:
                reason = f"⚠️ Contains negative values (min: {min_val})"
            else:
                reason = "❌ Other exclusion criteria"
            
            detection_details.append({
                'Column': col,
                'Unique Values': unique_vals,
                'Range': f"{min_val} - {max_val}",
                'Status': reason,
                'Include': is_detected
            })
        
        # Display the detection table
        detection_df = pd.DataFrame(detection_details)
        
        # Show summary
        detected_count = len(st.session_state.likert_items)
        excluded_count = len(numeric_columns) - detected_count
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Detected Items", detected_count, delta=f"+{detected_count} included")
        with col2:
            st.metric("Excluded Items", excluded_count, delta=f"-{excluded_count} excluded")
        with col3:
            detection_rate = (detected_count / len(numeric_columns) * 100) if numeric_columns else 0
            st.metric("Detection Rate", f"{detection_rate:.1f}%")
        
        # Show detailed breakdown in expandable section
        with st.expander("View Detection Details", expanded=True):
            # Add filter options
            filter_option = st.selectbox("Filter by:", ["All columns", "Detected only", "Excluded only"])
            
            if filter_option == "Detected only":
                display_df = detection_df[detection_df['Include'] == True]
            elif filter_option == "Excluded only":
                display_df = detection_df[detection_df['Include'] == False]
            else:
                display_df = detection_df
            
            st.dataframe(display_df, use_container_width=True)
            
            # Show excluded items that might be recoverable
            potentially_recoverable = detection_df[
                (detection_df['Include'] == False) & 
                (detection_df['Unique Values'] >= 2) &
                (~detection_df['Status'].str.contains('negative'))
            ]
            
            if len(potentially_recoverable) > 0:
                st.warning(f"Found {len(potentially_recoverable)} potentially recoverable items that were excluded due to category count limits.")
                recoverable_items = st.multiselect(
                    "Select items to include anyway:",
                    options=potentially_recoverable['Column'].tolist(),
                    help="These items were excluded due to category limits but might still be valid Likert items"
                )
                
                if recoverable_items:
                    # Add recovered items to the Likert items list
                    updated_items = list(set(st.session_state.likert_items + recoverable_items))
                    st.session_state.likert_items = updated_items
                    st.success(f"Added {len(recoverable_items)} recovered items. Total Likert items: {len(updated_items)}")
                    st.rerun()
    
    elif detection_method == "Manual Selection":
        # Let user manually select from all columns
        st.write("Select which columns represent Likert scale items:")
        
        # Get all numeric columns as candidates
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Show column info to help with selection
        col_info = []
        for col in numeric_columns:
            unique_vals = df[col].nunique()
            min_val = df[col].min()
            max_val = df[col].max()
            col_info.append(f"{col} (Range: {min_val}-{max_val}, {unique_vals} categories)")
        
        # Multi-select with current selection as default
        current_selection = st.session_state.get('likert_items', [])
        selected_items = st.multiselect(
            "Choose Likert scale columns:",
            options=numeric_columns,
            default=[item for item in current_selection if item in numeric_columns],
            format_func=lambda x: next((info for info in col_info if info.startswith(x)), x),
            help="Select all columns that contain Likert scale responses"
        )
        
        st.session_state.likert_items = selected_items
        
        if selected_items:
            st.success(f"Selected {len(selected_items)} Likert items")
        else:
            st.warning("No items selected")
    
    else:  # All Numeric Columns
        # Use all numeric columns as Likert items
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        st.session_state.likert_items = numeric_columns
        
        st.info(f"Using all {len(numeric_columns)} numeric columns as Likert items")
        
        # Show the columns that will be used
        with st.expander("View all numeric columns", expanded=False):
            col_details = []
            for col in numeric_columns:
                unique_vals = df[col].nunique()
                min_val = df[col].min()
                max_val = df[col].max()
                col_details.append({
                    'Column': col,
                    'Categories': unique_vals,
                    'Min': min_val,
                    'Max': max_val,
                    'Range': f"{min_val} - {max_val}"
                })
            
            st.dataframe(pd.DataFrame(col_details))
    
    # Final confirmation and overview
    likert_items = st.session_state.likert_items
    
    if len(likert_items) == 0:
        st.error("No Likert scale items identified. Please adjust your detection settings or manually select items.")
    else:
        # Show final selection with option to fine-tune
        st.subheader("Final Likert Items Selection")
        st.success(f"✅ **{len(likert_items)} Likert items** ready for analysis")
        
        # Allow final adjustments
        with st.expander("Fine-tune selection (optional)", expanded=False):
            adjusted_items = st.multiselect(
                "Adjust final selection:",
                options=df.select_dtypes(include=[np.number]).columns.tolist(),
                default=likert_items,
                help="Make final adjustments to your Likert items selection"
            )
            
            if adjusted_items != likert_items:
                st.session_state.likert_items = adjusted_items
                likert_items = adjusted_items
                st.success(f"Updated selection: {len(adjusted_items)} items")
                st.rerun()
        
        # Show distribution of selected items
        if likert_items:
            with st.expander("Item Distributions", expanded=True):
                st.subheader("Item Distributions")
                
                # Create a dropdown to select an item to visualize
                selected_item = st.selectbox(
                    "Select an item to view its distribution",
                    options=likert_items
                )
                
                # Show the distribution of the selected item
                fig = px.histogram(
                    df, x=selected_item, 
                    nbins=len(df[selected_item].unique()),
                    title=f"Distribution of {selected_item}",
                    color_discrete_sequence=['#3366CC']
                )
                fig.update_layout(bargap=0.1)
                st.plotly_chart(fig, use_container_width=True)
                
                # Create a multiselect to choose which items to display
                items_to_show = st.multiselect(
                    "Select items to compare distributions",
                    options=likert_items,
                    default=likert_items[:min(3, len(likert_items))]
                )
                
                if items_to_show:
                    # Create a comparison figure
                    fig = go.Figure()
                    for item in items_to_show:
                        counts = df[item].value_counts().sort_index()
                        fig.add_trace(go.Bar(
                            x=counts.index,
                            y=counts.values,
                            name=item
                        ))
                    
                    fig.update_layout(
                        title="Item Distribution Comparison",
                        xaxis_title="Response Value",
                        yaxis_title="Count",
                        barmode='group'
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Detect and handle reverse-coded items
        st.subheader("Reverse-Coded Items Detection")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            reverse_threshold_display = st.slider(
                "Reverse Item Detection Threshold", 
                -1.0, 0.0, reverse_threshold, 0.05,
                key="reverse_threshold_display",
                help="Correlation threshold to identify reverse-coded items"
            )
        
        with col2:
            detect_button = st.button("Detect Reverse-Coded Items")
        
        # Create dedicated containers that will persist on the page
        reverse_count_container = st.empty()
        reverse_items_list = st.empty()
        
        if detect_button:
            with st.spinner("Analyzing item correlations to detect reverse-coded items..."):
                # Add more detailed analysis here
                reverse_items = detect_reverse_items(df, likert_items, reverse_threshold_display)
                st.session_state.reverse_items = reverse_items
                
                # Get correlation details for reverse items
                corr_details = {}
                corr_matrix = df[likert_items].corr()
                
                for item in reverse_items:
                    # Find items with highest negative correlation to this reverse item
                    negative_correlations = corr_matrix[item].sort_values().head(3)
                    corr_details[item] = negative_correlations.to_dict()
                
                st.session_state.reverse_details = corr_details
                
                # Show results immediately
                if len(reverse_items) > 0:
                    reverse_count_container.success(f"📋 Detected {len(reverse_items)} reverse-coded items")
                    reverse_items_list.markdown(f"### Reverse-coded items:\n**{', '.join(reverse_items)}**")
                else:
                    reverse_count_container.info("No reverse-coded items detected in this dataset")
        
        # Always show the results, even if not from current button press
        elif 'reverse_items' in st.session_state and st.session_state.reverse_items:
            reverse_count_container.success(f"📋 Detected {len(st.session_state.reverse_items)} reverse-coded items")
            reverse_items_list.markdown(f"### Reverse-coded items:\n**{', '.join(st.session_state.reverse_items)}**")
        
        # Detailed reverse items container (expandable analysis)
        if 'reverse_items' in st.session_state and st.session_state.reverse_items:
            st.markdown("---")
            st.subheader("Reverse-Coded Items Analysis")
            
            # Show detailed information about each reverse item
            for i, item in enumerate(st.session_state.reverse_items):
                with st.expander(f"Reverse item {i+1}: {item}", expanded=i==0):
                    st.write("This item appears to be reverse-coded compared to other items.")
                    
                    if 'reverse_details' in st.session_state:
                        st.write("#### Strongest negative correlations with other items:")
                        
                        details = st.session_state.reverse_details.get(item, {})
                        for other_item, corr_value in details.items():
                            st.metric(
                                f"Correlation with {other_item}", 
                                f"{corr_value:.3f}",
                                delta="Negative correlation confirms reverse coding"
                            )
                    
                    # Show before/after histograms if we apply reverse coding
                    if len(st.session_state.reverse_items) > 0:
                        col1, col2 = st.columns(2)
                        
                        # Left: before reverse coding
                        with col1:
                            st.write("##### Original Distribution")
                            counts = df[item].value_counts().sort_index()
                            fig = px.bar(
                                x=counts.index, 
                                y=counts.values,
                                title="Before Reverse Coding"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Right: after reverse coding  
                        with col2:
                            st.write("##### After Reverse Coding (Preview)")
                            scale_min = int(df[likert_items].min().min())
                            scale_max = int(df[likert_items].max().max())
                            
                            # Reverse code this single item for preview
                            reversed_values = scale_min + scale_max - df[item]
                            counts = reversed_values.value_counts().sort_index()
                            
                            fig = px.bar(
                                x=counts.index, 
                                y=counts.values,
                                title="After Reverse Coding"
                            )
                            st.plotly_chart(fig, use_container_width=True)
            
            # Moved the button outside of the expanders for better visibility
            if st.button("Apply Reverse Coding to All Items"):
                scale_min = int(df[likert_items].min().min())
                scale_max = int(df[likert_items].max().max())
                df = reverse_code(df, st.session_state.reverse_items, scale_min, scale_max)
                st.success(f"Reverse coding applied to {len(st.session_state.reverse_items)} items")
                st.rerun()
        
        # Cronbach's Alpha Reliability Analysis
        st.subheader("Scale Reliability Analysis (Cronbach's Alpha)")
        
        # Automatic scale detection based on naming patterns
        def detect_scales(items):
            """Detect scales based on naming patterns like A1, A2, B1, B2, etc."""
            import re
            scales = {}
            
            for item in items:
                # Extract prefix (letters) and suffix (numbers)
                match = re.match(r'^([A-Za-z]+)(\d+)$', str(item))
                if match:
                    prefix = match.group(1).upper()
                    if prefix not in scales:
                        scales[prefix] = []
                    scales[prefix].append(item)
                else:
                    # For items without clear pattern, try common prefixes
                    item_upper = str(item).upper()
                    found_scale = False
                    for existing_prefix in scales.keys():
                        if item_upper.startswith(existing_prefix):
                            scales[existing_prefix].append(item)
                            found_scale = True
                            break
                    
                    if not found_scale:
                        # Create individual scale for items that don't match patterns
                        single_scale = f"SINGLE_{item}"
                        scales[single_scale] = [item]
            
            # Filter out single-item scales for reliability analysis
            multi_item_scales = {k: v for k, v in scales.items() if len(v) > 1}
            return multi_item_scales, scales
        
        if likert_items:
            with st.spinner("Detecting scales and computing reliability..."):
                multi_item_scales, all_scales = detect_scales(likert_items)
                
                if multi_item_scales:
                    st.success(f"✅ Detected {len(multi_item_scales)} multi-item scales for reliability analysis")
                    
                    # Store detected scales in session state
                    st.session_state.detected_scales = multi_item_scales
                    
                    # Calculate Cronbach's alpha for each scale
                    reliability_results = []
                    
                    for scale_name, items in multi_item_scales.items():
                        alpha = cronbach_alpha(df, items)
                        alpha_ci = bootstrap_alpha(df, items)
                        
                        # Reliability interpretation
                        if alpha >= 0.9:
                            interpretation = "Excellent"
                            color = "green"
                        elif alpha >= 0.8:
                            interpretation = "Good"
                            color = "green"
                        elif alpha >= 0.7:
                            interpretation = "Acceptable"
                            color = "orange"
                        elif alpha >= 0.6:
                            interpretation = "Questionable"
                            color = "orange"
                        else:
                            interpretation = "Poor"
                            color = "red"
                        
                        reliability_results.append({
                            'Scale': scale_name,
                            'Items': ', '.join(items),
                            'N Items': len(items),
                            'Cronbach α': f"{alpha:.3f}",
                            '95% CI': f"[{alpha_ci[0]:.3f}, {alpha_ci[1]:.3f}]",
                            'Interpretation': interpretation,
                            'Color': color
                        })
                    
                    # Display results in a nice table
                    reliability_df = pd.DataFrame(reliability_results)
                    
                    # Show summary metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        excellent_count = sum(1 for r in reliability_results if r['Interpretation'] == 'Excellent')
                        st.metric("Excellent Scales (α ≥ 0.9)", excellent_count)
                    with col2:
                        good_count = sum(1 for r in reliability_results if r['Interpretation'] in ['Good', 'Excellent'])
                        st.metric("Good+ Scales (α ≥ 0.8)", good_count)
                    with col3:
                        acceptable_count = sum(1 for r in reliability_results if r['Interpretation'] in ['Acceptable', 'Good', 'Excellent'])
                        st.metric("Acceptable+ Scales (α ≥ 0.7)", acceptable_count)
                    
                    # Display detailed results
                    with st.expander("Detailed Reliability Results", expanded=True):
                        for i, result in enumerate(reliability_results):
                            with st.container():
                                col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
                                
                                with col1:
                                    st.write(f"**{result['Scale']}**")
                                    st.caption(f"Items: {result['Items']}")
                                
                                with col2:
                                    st.metric("α", result['Cronbach α'])
                                
                                with col3:
                                    st.metric("Items", result['N Items'])
                                
                                with col4:
                                    if result['Color'] == 'green':
                                        st.success(f"{result['Interpretation']}")
                                    elif result['Color'] == 'orange':
                                        st.warning(f"{result['Interpretation']}")
                                    else:
                                        st.error(f"{result['Interpretation']}")
                                    st.caption(f"95% CI: {result['95% CI']}")
                                
                                if i < len(reliability_results) - 1:
                                    st.markdown("---")
                    
                    # Additional analysis options
                    with st.expander("Scale Analysis Options"):
                        selected_scale = st.selectbox(
                            "Select scale for detailed analysis:",
                            options=list(multi_item_scales.keys())
                        )
                        
                        if selected_scale:
                            scale_items = multi_item_scales[selected_scale]
                            
                            # Item-total correlations
                            st.subheader(f"Item Analysis for {selected_scale}")
                            
                            # Calculate item-total correlations
                            scale_data = df[scale_items].dropna()
                            scale_total = scale_data.sum(axis=1)
                            
                            item_stats = []
                            for item in scale_items:
                                # Item-total correlation (corrected)
                                item_total_corr = scale_data[item].corr(scale_total - scale_data[item])
                                
                                # Alpha if item deleted
                                remaining_items = [i for i in scale_items if i != item]
                                alpha_if_deleted = cronbach_alpha(df, remaining_items) if len(remaining_items) > 1 else 0
                                
                                item_stats.append({
                                    'Item': item,
                                    'Item-Total Correlation': f"{item_total_corr:.3f}",
                                    'Alpha if Deleted': f"{alpha_if_deleted:.3f}",
                                    'Should Delete?': 'Consider' if alpha_if_deleted > reliability_results[list(multi_item_scales.keys()).index(selected_scale)]['Cronbach α'].replace('α=', '') else 'No'
                                })
                            
                            st.dataframe(pd.DataFrame(item_stats))
                            
                            # Scale distribution
                            st.subheader(f"Scale Score Distribution: {selected_scale}")
                            scale_scores = df[scale_items].sum(axis=1)
                            
                            fig = px.histogram(
                                x=scale_scores,
                                nbins=20,
                                title=f"{selected_scale} Total Score Distribution",
                                labels={'x': 'Total Score', 'y': 'Frequency'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Basic descriptive statistics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Mean", f"{scale_scores.mean():.2f}")
                            with col2:
                                st.metric("Std Dev", f"{scale_scores.std():.2f}")
                            with col3:
                                st.metric("Min Score", f"{scale_scores.min():.0f}")
                            with col4:
                                st.metric("Max Score", f"{scale_scores.max():.0f}")
                
                else:
                    st.info("No multi-item scales detected based on naming patterns. Cronbach's alpha requires at least 2 items per scale.")
                    
                    # Show what was detected
                    if all_scales:
                        st.write("**Items detected by pattern:**")
                        for scale_name, items in all_scales.items():
                            if len(items) == 1:
                                st.caption(f"• {scale_name}: {items[0]} (single item)")
                            else:
                                st.write(f"• {scale_name}: {', '.join(items)}")
        
        # Sampling adequacy tests
        st.subheader("Sampling Adequacy")
        with st.spinner("Checking sampling adequacy..."):
            sampling = check_sampling(df, likert_items)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("KMO", f"{sampling['kmo']:.3f}", 
                          delta="+Good" if sampling['kmo'] > 0.6 else "-Poor")
            with col2:
                st.metric("Bartlett's p-value", f"{sampling['bartlett_p']:.3e}", 
                          delta="+Significant" if sampling['bartlett_p'] < 0.05 else "-Not Significant")
    
    return df
