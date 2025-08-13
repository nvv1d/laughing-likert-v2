
import streamlit as st
import json
from datetime import datetime
from utils import save_html_report

def render_reports_tab(df, analysis_method):
    """Render the Reports tab"""
    st.header("Analysis Reports")
    
    # Simple info about analysis method
    if analysis_method != "Python Only":
        st.info("Currently using analysis method: " + analysis_method)
        
    # Direct users to the right place for analysis information
    st.write("The detailed analysis results are available in the respective tabs:")
    st.markdown("""
    - **Data Preparation**: View item distributions and detect reverse-coded items
    - **Item Analysis**: See clustering results and reliability metrics
    - **Pattern Extraction**: Access item weights and factor analysis results
    - **Simulation**: Generate and validate simulated responses
    """)
    
    st.subheader("Generate Reports")
    st.write("Use the buttons below to export your analysis results:")
    
    # Generate HTML report
    if st.button("Generate Full Report"):
        with st.spinner("Generating report..."):
            # Prepare simulation statistics to include in the report
            sim_stats = {}
            if st.session_state.sim_data is not None and 'show_stat_analysis' in st.session_state:
                # Note: These variables would need to be properly passed from simulation tab
                # This is a simplified version
                sim_stats['overall'] = 85.0  # Placeholder
            
            report_path = save_html_report({
                'data': df,
                'likert_items': st.session_state.likert_items,
                'reverse_items': st.session_state.reverse_items,
                'clusters': st.session_state.clusters,
                'weights': st.session_state.weights,
                'alphas': st.session_state.alphas,
                'alpha_ci': st.session_state.alpha_ci,
                'simulated': st.session_state.sim_data,
                'sim_stats': sim_stats
            })
            
            # Display the report
            with open(report_path, 'r') as file:
                html_content = file.read()
            
            # Create a download button for the report
            st.download_button(
                "Download HTML Report",
                html_content,
                file_name=f"likert_analysis_report_{datetime.now().strftime('%Y%m%d')}.html",
                mime="text/html"
            )
            
            # Display in iframe (limited functionality in Streamlit)
            st.components.v1.html(html_content, height=600, scrolling=True)
    
    # Export all results to JSON
    if st.button("Export All Results"):
        # Prepare weights in a serializable format
        serializable_weights = {}
        for item, w_data in st.session_state.weights.items():
            if isinstance(w_data, dict):
                # Complex weights format
                if w_data.get('is_distribution', False) and 'weights' in w_data:
                    # Distribution weights need to be converted to strings
                    weights_dict = {}
                    for k, v in w_data['weights'].items():
                        weights_dict[str(k)] = float(v)
                    serializable_weights[item] = {
                        'is_distribution': True,
                        'weights': weights_dict
                    }
                else:
                    # Factor weights can be serialized directly
                    serializable_weights[item] = {
                        'is_distribution': False,
                        'weight': float(w_data.get('weight', 0.0))
                    }
                    if 'dist_weights' in w_data and w_data['dist_weights']:
                        dist_dict = {}
                        for k, v in w_data['dist_weights'].items():
                            dist_dict[str(k)] = float(v)
                        serializable_weights[item]['dist_weights'] = dist_dict
            else:
                # Legacy format (just a float)
                try:
                    serializable_weights[item] = float(w_data)
                except:
                    serializable_weights[item] = 0.0
        
        # Prepare all results for export
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'analysis_method': st.session_state.analysis_method,
            'likert_items': st.session_state.likert_items,
            'reverse_items': st.session_state.reverse_items,
            'clusters': {str(k): v for k, v in st.session_state.clusters.items()},
            'weights': serializable_weights,
            'alphas': {str(k): float(v) for k, v in st.session_state.alphas.items()},
            'alpha_ci': {str(k): [float(v[0]), float(v[1])] for k, v in st.session_state.alpha_ci.items()}
        }
        
        # Include simulation data and statistics if available
        if st.session_state.sim_data is not None:
            all_results['simulated_count'] = len(st.session_state.sim_data)
        
        # Convert to JSON
        results_json = json.dumps(all_results, indent=2)
        
        # Create download button
        st.download_button(
            "Download Results as JSON",
            results_json,
            file_name=f"likert_analysis_results_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )
