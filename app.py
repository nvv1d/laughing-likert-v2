import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from auth import init_auth_session_state, render_login_page, is_authenticated, update_last_activity, render_logout_button, get_current_username
from utils import load_data
from tabs.data_preparation import render_data_preparation_tab
from tabs.item_analysis import render_item_analysis_tab
from tabs.pattern_extraction import render_pattern_extraction_tab
from tabs.simulation import render_simulation_tab  # Now imports from the modular simulation package
from tabs.reports import render_reports_tab

# Page configuration
st.set_page_config(
    page_title="Likert Scale Pattern Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize authentication
init_auth_session_state()

# Display login if not logged in
if not is_authenticated():
    render_login_page()
    st.stop()

# Update last activity when logged in
update_last_activity()

# Main Application (only shown if logged in)
st.title("Likert Scale Pattern Analysis")
st.markdown("""
This application helps you analyze Likert scale survey data, extract response patterns, 
and generate simulated data based on those patterns. Upload your data file to get started.
""")

# Sidebar for inputs and controls
with st.sidebar:
    st.write(f"👋 Welcome, {get_current_username()}!")
    render_logout_button()

    st.header("Upload Data")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])

    st.header("Analysis Settings")

    # Initialize analysis method in session state if not present (default to Hybrid)
    if 'analysis_method' not in st.session_state:
        st.session_state.analysis_method = "Hybrid (Python + R)"

    # Create the selectbox WITHOUT using the session state key to avoid the warning
    # Instead, we'll update session state manually when it changes
    analysis_method = st.selectbox(
        "Analysis Method",
        ["Python Only", "R Only", "Hybrid (Python + R)"],
        index=["Python Only", "R Only", "Hybrid (Python + R)"].index(st.session_state.analysis_method),
        key="analysis_method_selector",  # Use a different key
        help="Select which analysis engine to use"
    )

    # Update session state if the value changed
    if analysis_method != st.session_state.analysis_method:
        st.session_state.analysis_method = analysis_method

    # Display message when analysis method changes
    if st.session_state.get('_previous_analysis_method') != analysis_method:
        if 'clusters' in st.session_state or 'weights' in st.session_state:
            st.warning("Analysis method changed! Results will be recalculated with the new method.")
            # Clear previous analysis results but keep data
            for key in ['clusters', 'weights', 'alphas', 'alpha_ci', 'sim_data', 'n_factors_detected']:
                if key in st.session_state:
                    del st.session_state[key]
        st.session_state['_previous_analysis_method'] = analysis_method

    if uploaded_file is not None:
        # Advanced settings (collapsed by default)
        with st.expander("Advanced Settings"):
            min_cats = st.slider("Minimum Categories", 2, 10, 3,
                                 help="Minimum number of categories to identify Likert items")
            max_cats = st.slider("Maximum Categories", min_cats, 15, 10,
                                 help="Maximum number of categories to identify Likert items")
            reverse_threshold = st.slider("Reverse Item Threshold", -1.0, 0.0, -0.2, 0.05,
                                         help="Correlation threshold to identify reverse-coded items")

            if analysis_method in ["Python Only", "Hybrid (Python + R)"]:
                n_clusters = st.number_input("Number of Clusters", min_value=0, value=0, 
                                           help="Number of item clusters (0 for automatic)")
                n_factors = st.number_input("Number of Factors", min_value=0, value=0,
                                          help="Number of factors (0 for automatic)")
            else:
                # Set defaults for R Only method
                n_clusters = 0
                n_factors = 0

# Only show the rest if a file is uploaded
if uploaded_file is not None:
    # Load the data
    try:
        df = load_data(uploaded_file)

        # Show the data preview
        st.header("Data Preview")
        st.write(f"Total rows: {df.shape[0]}, Total columns: {df.shape[1]}")
        st.dataframe(df)

        # Show data information
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Number of Responses", df.shape[0])
        with col2:
            st.metric("Number of Variables", df.shape[1])

        # Analysis tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Data Preparation", 
            "Item Analysis", 
            "Pattern Extraction", 
            "Simulation", 
            "Reports"
        ])

        # Initialize session state variables
        if 'likert_items' not in st.session_state:
            st.session_state.likert_items = []
        if 'reverse_items' not in st.session_state:
            st.session_state.reverse_items = []
        if 'clusters' not in st.session_state:
            st.session_state.clusters = {}
        if 'weights' not in st.session_state:
            st.session_state.weights = {}
        if 'alphas' not in st.session_state:
            st.session_state.alphas = {}
        if 'alpha_ci' not in st.session_state:
            st.session_state.alpha_ci = {}
        if 'sim_data' not in st.session_state:
            st.session_state.sim_data = None

        # For debugging, add a button to clear the session state
        with st.sidebar:
            if st.button("Reset Analysis"):
                for key in ['likert_items', 'reverse_items', 'clusters', 'weights', 
                            'alphas', 'alpha_ci', 'sim_data', 'n_factors_detected']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

        # Render each tab using the modular functions
        with tab1:
            df = render_data_preparation_tab(df, min_cats, max_cats, reverse_threshold)

        with tab2:
            render_item_analysis_tab(df, n_clusters)

        with tab3:
            render_pattern_extraction_tab(df, n_factors)

        with tab4:
            render_simulation_tab(df)  # This now uses the modular simulation components

        with tab5:
            render_reports_tab(df, analysis_method)

    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.exception(e)
else:
    # Show instructions when no file is uploaded
    st.info("Please upload a CSV or Excel file containing Likert scale survey data.")

    # Sample instruction guide
    with st.expander("How to prepare your data"):
        st.markdown("""
        ### Data Format Requirements

        - Your data should be in CSV or Excel format
        - Each row represents a respondent
        - Each column represents a survey item or demographic variable
        - Likert scale items should have integer values (e.g., 1-5, 1-7)
        - Missing values should be represented as empty cells or standard missing value codes (Better results if data is screened/cleaned first)

        ### Typical Data Structure

        | Respondent | Gender | Age | Item1 | Item2 | Item3 | ... |
        |------------|--------|-----|-------|-------|-------|-----|
        | 1          | F      | 25  | 4     | 5     | 3     | ... |
        | 2          | M      | 31  | 2     | 4     | 4     | ... |
        | ...        | ...    | ... | ...   | ...   | ...   | ... |

        ### Naming Conventions

        For best results, name your items with a common prefix to indicate which scale they belong to:

        - X1, X2, X3 (items for X scale)
        - Y1, Y2, Y3 (items for Y scale)
        - etc.
        """)

# Footer
st.markdown("---")
st.markdown(
    "Likert Scale Pattern Analysis Tool | "
    f"Version 1.0 | {datetime.now().year}"
)
