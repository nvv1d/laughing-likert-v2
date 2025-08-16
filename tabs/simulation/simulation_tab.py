import streamlit as st
from .bias_config import EnhancedBiasConfigComponent, apply_enhanced_bias_to_weights
from .data_generator import DataGenerator
from .visualization import VisualizationManager
from .statistical_analysis import StatisticalAnalyzer

def render_simulation_tab(df):
    """Main function to render the Response Simulation tab with modular components"""
    st.header("Response Simulation")
    
    if not st.session_state.weights:
        st.warning("Please extract item weights in the Pattern Extraction tab first")
        return
    
    # Initialize components
    bias_config = EnhancedBiasConfigComponent()
    data_generator = DataGenerator()
    viz_manager = VisualizationManager()
    stat_analyzer = StatisticalAnalyzer()
    
    # 1. Render basic simulation settings
    simulation_settings = data_generator.render_simulation_settings()
    
    # 2. Render bias configuration
    bias_settings = bias_config.render_bias_config()
    
    # 3. Simulation button and data generation
    if st.button("🚀 Simulate Responses"):
        _perform_simulation(
            df, data_generator, bias_settings, simulation_settings
        )
    
    # 4. Display results if simulation data exists
    if st.session_state.sim_data is not None:
        st.success(f"Generated {len(st.session_state.sim_data)} simulated responses")
        
        # Show data preview
        data_generator.render_data_preview(st.session_state.sim_data)
        
        # Show distribution comparisons
        viz_manager.render_distribution_comparison(
            df, st.session_state.sim_data, st.session_state.weights
        )
        
        # Download button
        viz_manager.render_download_button(st.session_state.sim_data)
        
        # Statistical comparison
        stat_analyzer.render_comprehensive_statistical_comparison(
            df, 
            st.session_state.sim_data, 
            st.session_state.likert_items,
            st.session_state.clusters,
            st.session_state.alphas
        )

def _perform_simulation(df, data_generator, bias_settings, simulation_settings):
    """Perform the actual simulation with bias application"""
    with st.spinner(f"Simulating {simulation_settings['num_simulations']} responses..."):
        # Prepare weights for simulation
        updated_weights = data_generator.prepare_weights_for_simulation(
            df, st.session_state.weights, st.session_state.clusters
        )
        
        # Apply bias if enabled
        if bias_settings['enable_bias']:
            st.info(f"Applying {bias_settings['bias_type']} bias "
                   f"(strength: {bias_settings['bias_strength']}, "
                   f"affected: {bias_settings['bias_percentage']*100:.0f}%)")
            
            updated_weights = apply_bias_to_weights(
                updated_weights, 
                bias_settings['bias_type'], 
                bias_settings['bias_strength'], 
                bias_settings['bias_percentage']
            )
        
        # Generate simulated data
        sim_data = data_generator.generate_simulated_data(
            updated_weights,
            simulation_settings['num_simulations'],
            simulation_settings['noise_level'],
            st.session_state.likert_items
        )
        
        # Store results in session state
        st.session_state.sim_data = sim_data
        st.session_state.weights = updated_weights  # Update with augmented weights
        
        st.success(f"🎯 Successfully generated {simulation_settings['num_simulations']} responses!")
