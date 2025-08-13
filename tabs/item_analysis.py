
import streamlit as st
import plotly.express as px
from utils import cluster_items, cronbach_alpha, bootstrap_alpha, create_network_graph

def render_item_analysis_tab(df, n_clusters):
    """Render the Item Analysis tab"""
    st.header("Item Analysis")
    
    if len(st.session_state.likert_items) == 0:
        st.warning("Please identify Likert items in the Data Preparation tab first")
    else:
        # Cluster the items
        st.subheader("Item Clustering")
        
        if st.button("Cluster Items"):
            with st.spinner("Clustering items..."):
                clusters = cluster_items(
                    df, st.session_state.likert_items, 
                    n_clusters if n_clusters > 0 else None
                )
                st.session_state.clusters = clusters
        
        if st.session_state.clusters:
            st.success(f"Identified {len(st.session_state.clusters)} clusters")
            
            # Calculate reliability for each cluster
            alphas = {}
            alpha_ci = {}
            for sc, items in st.session_state.clusters.items():
                if len(items) > 1:  # Need at least 2 items for reliability
                    alphas[sc] = cronbach_alpha(df, items)
                    alpha_ci[sc] = bootstrap_alpha(df, items)
            
            st.session_state.alphas = alphas
            st.session_state.alpha_ci = alpha_ci
            
            # Display each cluster and its reliability
            for sc, items in st.session_state.clusters.items():
                with st.expander(f"Cluster {sc} ({len(items)} items)"):
                    st.write(", ".join(items))
                    
                    if sc in alphas:
                        st.metric(
                            "Cronbach's Alpha", 
                            f"{alphas[sc]:.3f}",
                            delta=f"CI: [{alpha_ci[sc][0]:.3f}, {alpha_ci[sc][1]:.3f}]"
                        )
                        
                        # Item correlations within cluster
                        corr = df[items].corr()
                        fig = px.imshow(
                            corr, 
                            title=f"Item Correlations (Cluster {sc})",
                            color_continuous_scale="Blues"
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        # Create enhanced network graph of item relationships
        st.subheader("Item Relationships Network")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            # Add layout selection
            layout_option = st.selectbox(
                "Graph Layout",
                options=["force", "circular", "kamada_kawai", "spectral"],
                index=0,
                help="Select the algorithm used to position nodes in the graph"
            )
        
        with col2:
            # Add correlation threshold slider
            corr_threshold = st.slider(
                "Correlation Threshold",
                min_value=0.1,
                max_value=0.9,
                value=0.3,
                step=0.05,
                help="Minimum correlation required to display a connection between items"
            )
        
        with col3:
            # Add button to generate the graph
            generate_graph = st.button("Generate Network Graph")
        
        # Create the graph when the button is clicked
        if generate_graph:
            with st.spinner("Creating enhanced network visualization..."):
                fig = create_network_graph(
                    df, 
                    st.session_state.likert_items,
                    threshold=corr_threshold,
                    layout=layout_option
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("""
                **How to read this graph:**
                - **Nodes** represent survey items, sized by their importance (higher correlation sum = larger node)
                - **Colors** indicate communities of related items detected by clustering
                - **Lines** represent correlations between items, with thickness proportional to correlation strength
                - Hover over nodes to see details about each item and its strongest connections
                - Drag nodes to reposition them for better viewing
                """)
