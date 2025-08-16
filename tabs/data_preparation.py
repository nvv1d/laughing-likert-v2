import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer
from scipy.stats import pearsonr
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re
from collections import defaultdict
import json
from utils import identify_likert_columns, detect_reverse_items, reverse_code, check_sampling, cronbach_alpha, bootstrap_alpha, clean_dataframe_for_display

def detect_scales_by_pattern(columns):
    """
    Automatically detect scales based on naming patterns like A1, A2, A3, B1, B2, B3, etc.
    Returns a dictionary with scale names as keys and lists of items as values.
    """
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

def render_data_preparation_tab(df, min_cats, max_cats, reverse_threshold):
    """Render the Data Preparation tab with enhanced reliability analysis"""
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

        # NEW FEATURE: Automatic Scale Detection and Reliability Analysis
        st.markdown("---")
        st.subheader("🔍 Automatic Scale Detection & Reliability Analysis")

        # Detect scales based on naming patterns
        if st.button("🔎 Detect Scales by Naming Pattern"):
            with st.spinner("Analyzing column names for scale patterns..."):
                detected_scales = detect_scales_by_pattern(likert_items)
                st.session_state.detected_scales = detected_scales

                if detected_scales:
                    st.success(f"🎯 Detected {len(detected_scales)} scales with clear naming patterns!")
                else:
                    st.warning("⚠️ No clear naming patterns detected. Items may not follow standard naming conventions (e.g., A1, A2, B1, B2).")

        # Show detected scales if available
        if st.session_state.get('detected_scales', {}):
            st.subheader("📋 Detected Scales")

            # Calculate reliability for each detected scale
            scale_reliability = {}
            reliability_details = []

            for scale_name, scale_items in st.session_state.get('detected_scales', {}).items():
                # Filter items that actually exist in the data
                valid_items = [item for item in scale_items if item in df.columns]

                if len(valid_items) >= 2:
                    try:
                        alpha = cronbach_alpha(df, valid_items)
                        alpha_ci = bootstrap_alpha(df, valid_items, n_bootstrap=100)

                        scale_reliability[scale_name] = {
                            'items': valid_items,
                            'n_items': len(valid_items),
                            'alpha': alpha,
                            'alpha_ci': alpha_ci,
                            'quality': 'Excellent' if alpha >= 0.9 else
                                      'Good' if alpha >= 0.8 else
                                      'Acceptable' if alpha >= 0.7 else
                                      'Questionable' if alpha >= 0.6 else 'Poor'
                        }

                        reliability_details.append({
                            'Scale': scale_name,
                            'Items': len(valid_items),
                            'Cronbach Alpha': f"{alpha:.3f}",
                            '95% CI': f"[{alpha_ci[0]:.3f}, {alpha_ci[1]:.3f}]",
                            'Quality': scale_reliability[scale_name]['quality'],
                            'Item List': ', '.join(valid_items)
                        })
                    except Exception as e:
                        st.warning(f"Could not calculate reliability for scale {scale_name}: {str(e)}")

            # Display reliability results in a nice format
            if reliability_details:
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)

                excellent_scales = len([s for s in scale_reliability.values() if s['alpha'] >= 0.9])
                good_scales = len([s for s in scale_reliability.values() if 0.8 <= s['alpha'] < 0.9])
                acceptable_scales = len([s for s in scale_reliability.values() if 0.7 <= s['alpha'] < 0.8])
                poor_scales = len([s for s in scale_reliability.values() if s['alpha'] < 0.7])

                with col1:
                    st.metric("Excellent (≥0.9)", excellent_scales, delta="✅")
                with col2:
                    st.metric("Good (≥0.8)", good_scales, delta="👍")
                with col3:
                    st.metric("Acceptable (≥0.7)", acceptable_scales, delta="⚠️")
                with col4:
                    st.metric("Poor (<0.7)", poor_scales, delta="❌")

                # Detailed results table
                reliability_df = pd.DataFrame(reliability_details)
                st.dataframe(reliability_df, use_container_width=True)

                # Option to export reliability results
                if st.button("📥 Export Reliability Analysis"):
                    # Create comprehensive export data
                    export_data = {
                        'analysis_timestamp': datetime.now().isoformat(),
                        'total_scales': len(scale_reliability),
                        'scale_details': []
                    }

                    for scale_name, scale_info in scale_reliability.items():
                        item_stats = []
                        for item in scale_info['items']:
                            other_items = [i for i in scale_info['items'] if i != item]
                            other_total = df[other_items].sum(axis=1)
                            corr = df[item].corr(other_total)

                            item_stats.append({
                                'item': item,
                                'mean': float(df[item].mean()),
                                'std': float(df[item].std()),
                                'corrected_item_total_correlation': float(corr)
                            })

                        export_data['scale_details'].append({
                            'scale_name': scale_name,
                            'n_items': scale_info['n_items'],
                            'cronbach_alpha': float(scale_info['alpha']),
                            'alpha_ci_lower': float(scale_info['alpha_ci'][0]),
                            'alpha_ci_upper': float(scale_info['alpha_ci'][1]),
                            'quality': scale_info['quality'],
                            'items': scale_info['items'],
                            'item_statistics': item_stats
                        })

                    # Convert to JSON and create download
                    export_json = json.dumps(export_data, indent=2)

                    st.download_button(
                        "Download Reliability Analysis (JSON)",
                        export_json,
                        file_name=f"scale_reliability_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )

                    # Also create a CSV summary
                    csv_data = []
                    for scale_name, scale_info in scale_reliability.items():
                        csv_data.append({
                            'Scale_Name': scale_name,
                            'N_Items': scale_info['n_items'],
                            'Cronbach_Alpha': scale_info['alpha'],
                            'CI_Lower': scale_info['alpha_ci'][0],
                            'CI_Upper': scale_info['alpha_ci'][1],
                            'Quality': scale_info['quality'],
                            'Items': '; '.join(scale_info['items'])
                        })

                    csv_df = pd.DataFrame(csv_data)
                    csv_string = csv_df.to_csv(index=False)

                    st.download_button(
                        "Download Reliability Summary (CSV)",
                        csv_string,
                        file_name=f"scale_reliability_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

                # Expandable detailed analysis for each scale
                st.subheader("📊 Detailed Scale Analysis")

                for scale_name, scale_info in scale_reliability.items():
                    with st.expander(f"📈 {scale_name} (α = {scale_info['alpha']:.3f}, {scale_info['quality']})",
                       expanded=bool(scale_info['alpha'] < 0.7)):  # Auto-expand poor scales

                        col1, col2 = st.columns([2, 1])

                        with col1:
                            # Scale statistics
                            st.write("**Scale Information:**")
                            st.write(f"- **Items**: {', '.join(scale_info['items'])}")
                            st.write(f"- **Number of items**: {scale_info['n_items']}")
                            st.write(f"- **Cronbach's Alpha**: {scale_info['alpha']:.3f}")
                            st.write(f"- **95% Confidence Interval**: [{scale_info['alpha_ci'][0]:.3f}, {scale_info['alpha_ci'][1]:.3f}]")
                            st.write(f"- **Reliability Quality**: {scale_info['quality']}")

                            # Item-total correlations
                            st.write("**Item-Total Correlations:**")
                            scale_data = df[scale_info['items']]
                            scale_total = scale_data.sum(axis=1)

                            correlations = []
                            for item in scale_info['items']:
                                # Corrected item-total correlation (excluding the item itself)
                                other_items = [i for i in scale_info['items'] if i != item]
                                other_total = df[other_items].sum(axis=1)
                                corr = df[item].corr(other_total)
                                correlations.append({
                                    'Item': item,
                                    'Corrected Item-Total r': f"{corr:.3f}"
                                })

                            corr_df = pd.DataFrame(correlations)
                            st.dataframe(corr_df, use_container_width=True)

                        with col2:
                            # Reliability interpretation
                            st.write("**Interpretation:**")
                            if scale_info['alpha'] >= 0.9:
                                st.success("🌟 Excellent internal consistency")
                            elif scale_info['alpha'] >= 0.8:
                                st.success("✅ Good internal consistency")
                            elif scale_info['alpha'] >= 0.7:
                                st.warning("⚠️ Acceptable internal consistency")
                            elif scale_info['alpha'] >= 0.6:
                                st.warning("⚡ Questionable internal consistency")
                            else:
                                st.error("❌ Poor internal consistency")

                            # Recommendations
                            st.write("**Recommendations:**")
                            if scale_info['alpha'] < 0.7:
                                st.write("- Consider removing poor-performing items")
                                st.write("- Check for reverse-coded items")
                                st.write("- Examine item wording for clarity")
                            elif scale_info['alpha'] < 0.8:
                                st.write("- Scale is usable but could be improved")
                                st.write("- Consider adding more items")
                            else:
                                st.write("- Scale shows good reliability")
                                st.write("- Suitable for research use")

                        # Inter-item correlation matrix for this scale
                        if len(scale_info['items']) <= 10:  # Only show for smaller scales
                            st.write("**Inter-item Correlation Matrix:**")
                            corr_matrix = df[scale_info['items']].corr()

                            fig = px.imshow(
                                corr_matrix,
                                title=f"Inter-item Correlations: {scale_name}",
                                color_continuous_scale="RdBu_r",
                                zmin=-1, zmax=1,
                                aspect="auto"
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No scales with sufficient items (≥2) found for reliability analysis.")

        else:
            # Manual scale definition option
            st.info("💡 **Tip**: If automatic detection didn't work, you can manually group your items into scales for reliability analysis.")

            with st.expander("🔧 Manual Scale Definition", expanded=False):
                st.write("Define scales manually by grouping related items:")

                # Initialize manual scales in session state
                if 'manual_scales' not in st.session_state:
                    st.session_state.manual_scales = {}

                # Add new scale
                new_scale_name = st.text_input("New Scale Name (e.g., 'Satisfaction', 'Attitude'):")

                if new_scale_name:
                    scale_items = st.multiselect(
                        f"Select items for '{new_scale_name}':",
                        options=likert_items,
                        key=f"manual_scale_{new_scale_name}"
                    )

                    if len(scale_items) >= 2:
                        if st.button(f"Add '{new_scale_name}' Scale"):
                            st.session_state.manual_scales[new_scale_name] = scale_items
                            st.success(f"Added scale '{new_scale_name}' with {len(scale_items)} items")
                            st.rerun()

                # Show existing manual scales
                if st.session_state.manual_scales:
                    st.write("**Defined Scales:**")
                    for scale_name, items in st.session_state.manual_scales.items():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"- **{scale_name}**: {', '.join(items)}")
                        with col2:
                            if st.button(f"Remove", key=f"remove_{scale_name}"):
                                del st.session_state.manual_scales[scale_name]
                                st.rerun()

                    # Calculate reliability for manual scales
                    if st.button("Calculate Reliability for Manual Scales"):
                        manual_reliability = {}
                        for scale_name, scale_items in st.session_state.manual_scales.items():
                            if len(scale_items) >= 2:
                                try:
                                    alpha = cronbach_alpha(df, scale_items)
                                    alpha_ci = bootstrap_alpha(df, scale_items, n_bootstrap=100)
                                    manual_reliability[scale_name] = {
                                        'alpha': alpha,
                                        'alpha_ci': alpha_ci,
                                        'items': scale_items
                                    }
                                except Exception as e:
                                    st.warning(f"Could not calculate reliability for {scale_name}: {str(e)}")

                        # Display results
                        if manual_reliability:
                            st.subheader("Manual Scale Reliability Results")
                            for scale_name, results in manual_reliability.items():
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric(f"{scale_name}", f"α = {results['alpha']:.3f}")
                                with col2:
                                    st.write(f"95% CI: [{results['alpha_ci'][0]:.3f}, {results['alpha_ci'][1]:.3f}]")
                                with col3:
                                    quality = ('Excellent' if results['alpha'] >= 0.9 else
                                             'Good' if results['alpha'] >= 0.8 else
                                             'Acceptable' if results['alpha'] >= 0.7 else
                                             'Questionable' if results['alpha'] >= 0.6 else 'Poor')
                                    st.write(f"Quality: {quality}")

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
            st.write("") # Placeholder for alignment
            st.write("") # Placeholder for alignment
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
                                delta="Negative correlation"
                            )

                    if len(st.session_state.reverse_items) > 0:
                        col1, col2 = st.columns(2)

                        with col1:
                            st.write("##### Original Distribution")
                            counts = df[item].value_counts().sort_index()
                            fig = px.bar(
                                x=counts.index,
                                y=counts.values,
                                title="Before Reverse Coding"
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        with col2:
                            st.write("##### After Reverse Coding (Preview)")
                            scale_min = int(df[likert_items].min().min())
                            scale_max = int(df[likert_items].max().max())

                            reversed_values = scale_min + scale_max - df[item]
                            counts = reversed_values.value_counts().sort_index()

                            fig = px.bar(
                                x=counts.index,
                                y=counts.values,
                                title="After Reverse Coding"
                            )
                            st.plotly_chart(fig, use_container_width=True)

            if st.button("Apply Reverse Coding to All Items"):
                scale_min = int(df[likert_items].min().min())
                scale_max = int(df[likert_items].max().max())
                df = reverse_code(df, st.session_state.reverse_items, scale_min, scale_max)
                st.success(f"Reverse coding applied to {len(st.session_state.reverse_items)} items")
                st.rerun()

        # Sampling adequacy tests
        st.subheader("Sampling Adequacy")
        with st.spinner("Checking sampling adequacy..."):
            sampling = check_sampling(df, likert_items)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("KMO", f"{sampling['kmo']:.3f}",
                          delta="Good" if sampling['kmo'] > 0.6 else "Poor")
            with col2:
                st.metric("Bartlett's p-value", f"{sampling['bartlett_p']:.3e}",
                          delta="Significant" if sampling['bartlett_p'] < 0.05 else "Not Significant")

    # Clean the dataframe for display after all operations
    cleaned_df = clean_dataframe_for_display(df)
    st.subheader("Processed Data Preview")
    st.dataframe(cleaned_df, use_container_width=True)

    return df
