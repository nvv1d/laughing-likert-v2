
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from utils import determine_factors, extract_weights

def render_pattern_extraction_tab(df, n_factors):
    """Render the Pattern Extraction tab"""
    st.header("Pattern Extraction")
    
    if not st.session_state.clusters:
        st.warning("Please cluster items in the Item Analysis tab first")
    else:
        # Determine factors
        if 'n_factors_detected' not in st.session_state:
            with st.spinner("Determining optimal number of factors..."):
                n_factors_detected = determine_factors(df, st.session_state.likert_items)
                st.session_state.n_factors_detected = n_factors_detected
        
        st.info(f"Optimal number of factors: {st.session_state.n_factors_detected}")
        
        # Extract weights
        if st.button("Extract Item Weights"):
            with st.spinner("Extracting item weights..."):
                weights = extract_weights(
                    df, 
                    st.session_state.clusters,
                    n_factors if n_factors > 0 else st.session_state.n_factors_detected
                )
                st.session_state.weights = weights
        
        if st.session_state.weights:
            st.success("Item weights extracted successfully")
            
            # Visualize weights by cluster
            for sc, items in st.session_state.clusters.items():
                # Extract weights in a format suitable for visualization
                vis_weights = {}
                for item in items:
                    if item in st.session_state.weights:
                        w_data = st.session_state.weights[item]
                        if isinstance(w_data, dict):
                            if w_data.get('is_distribution', False):
                                # For distribution-based weights, use average value
                                if 'weights' in w_data:
                                    dist = w_data['weights']
                                    # Calculate weighted average
                                    try:
                                        avg = sum(float(k) * v for k, v in dist.items()) / sum(dist.values())
                                        vis_weights[item] = avg
                                    except:
                                        vis_weights[item] = 0.5  # Fallback
                            else:
                                # For factor analysis weights
                                vis_weights[item] = w_data.get('weight', 0.5)
                        else:
                            # Legacy format (simple value)
                            vis_weights[item] = w_data
                
                if vis_weights:
                    fig = px.bar(
                        x=list(vis_weights.keys()),
                        y=list(vis_weights.values()),
                        title=f"Item Weights (Cluster {sc})"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Create a more detailed weights table showing all information
            weights_rows = []
            for item, w_data in st.session_state.weights.items():
                if isinstance(w_data, dict):
                    if w_data.get('is_distribution', False):
                        # Distribution-based
                        weights_rows.append({
                            'Item': item,
                            'Weight Type': 'Distribution', 
                            'Weight': str(w_data.get('weights', {}))[:30] + '...' if len(str(w_data.get('weights', {}))) > 30 else str(w_data.get('weights', {}))
                        })
                    else:
                        # Factor-based
                        weights_rows.append({
                            'Item': item,
                            'Weight Type': 'Factor Loading', 
                            'Weight': w_data.get('weight', 0.0)
                        })
                else:
                    # Legacy format
                    weights_rows.append({
                        'Item': item,
                        'Weight Type': 'Simple', 
                        'Weight': w_data
                    })
            
            # Display detailed weights table
            weights_df = pd.DataFrame(weights_rows)
            st.dataframe(weights_df)
            
            # Download weights - create a simpler version for download
            simple_weights = {}
            for item, w_data in st.session_state.weights.items():
                if isinstance(w_data, dict):
                    if w_data.get('is_distribution', False):
                        # For distribution, use the weights dict
                        simple_weights[item] = str(w_data.get('weights', {}))
                    else:
                        # For factor analysis weights
                        simple_weights[item] = w_data.get('weight', 0.0)
                else:
                    # Legacy format
                    simple_weights[item] = w_data
            
            # Create simple CSV for download
            download_df = pd.DataFrame({
                'Item': simple_weights.keys(),
                'Weight': simple_weights.values()
            })
            weights_csv = download_df.to_csv(index=False)
            
            st.download_button(
                "Download Weights as CSV",
                weights_csv,
                file_name=f"likert_weights_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
