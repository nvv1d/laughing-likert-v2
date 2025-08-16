import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Union
import json

class EnhancedBiasConfigComponent:
    """Enhanced component for handling comprehensive bias configuration UI and logic"""
    
    def __init__(self):
        self.enable_bias = False
        self.bias_profiles = []
        self.dataset_generation_config = {}
        self.available_items = []
        self.available_scales = []
        
    def set_available_items_and_scales(self, items: List[str], scales: List[str]):
        """Set available items and scales for selection"""
        self.available_items = items
        self.available_scales = scales
    
    def render_bias_config(self):
        """Render the enhanced bias configuration UI and return settings"""
        st.markdown("---")
        st.header("🎯 Advanced Response Bias Configuration")
        
        # Main bias enable/disable
        self.enable_bias = st.checkbox(
            "Enable Response Bias System", 
            value=False,
            help="Apply systematic bias patterns to simulate different respondent behaviors"
        )
        
        if not self.enable_bias:
            return {
            'enable_bias': False,
            'bias_profiles': [],
            'scale_specific_config': {},
            'distribution_controls': {},
            'sem_optimization': False
        }
        
        # Bias configuration tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎭 Bias Profiles", 
            "🎯 Scale-Specific Bias", 
            "📊 Distribution Controls", 
            "🔬 SEM Dataset Generation"
        ])
        
        with tab1:
            self._render_bias_profiles_tab()
        
        with tab2:
            self._render_scale_specific_tab()
        
        with tab3:
            self._render_distribution_controls_tab()
        
        with tab4:
            self._render_sem_dataset_tab()
        
        # Global preview and validation
        self._render_global_preview()
        
        return self._compile_configuration()
    
    def _render_bias_profiles_tab(self):
        """Render bias profiles configuration"""
        st.subheader("👥 Response Bias Profiles")
        st.info("Create multiple bias profiles to simulate different respondent types simultaneously")
        
        # Profile management
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            new_profile_name = st.text_input("New Profile Name", placeholder="e.g., High Achievers, Pessimists")
        with col2:
            if st.button("➕ Add Profile", disabled=not new_profile_name.strip()):
                self._add_bias_profile(new_profile_name.strip())
        with col3:
            if st.button("🗑️ Clear All") and self.bias_profiles:
                self.bias_profiles = []
                st.rerun()
        
        # Display existing profiles
        if not self.bias_profiles:
            st.warning("No bias profiles configured. Add a profile to begin.")
            return
        
        for i, profile in enumerate(self.bias_profiles):
            with st.expander(f"📋 {profile['name']} Profile", expanded=i == 0):
                self._render_profile_config(i, profile)
    
    def _render_profile_config(self, profile_idx: int, profile: dict):
        """Render individual profile configuration"""
        col1, col2 = st.columns([3, 1])
        
        with col2:
            if st.button(f"🗑️ Delete", key=f"delete_profile_{profile_idx}"):
                self.bias_profiles.pop(profile_idx)
                st.rerun()
        
        with col1:
            # Basic profile settings
            profile['bias_type'] = st.selectbox(
                "Bias Direction",
                options=["high", "mid", "low", "extreme_high", "extreme_low", "central_tendency", "random_response"],
                format_func=self._format_bias_type,
                key=f"bias_type_{profile_idx}",
                help="Direction and pattern of response bias"
            )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            profile['bias_strength'] = st.slider(
                "Bias Strength",
                min_value=0.1, max_value=3.0, value=profile.get('bias_strength', 0.5),
                step=0.1, key=f"strength_{profile_idx}",
                help="Intensity of the bias effect"
            )
        
        with col2:
            profile['population_percentage'] = st.slider(
                "Population %",
                min_value=0.05, max_value=1.0, value=profile.get('population_percentage', 0.3),
                step=0.05, key=f"population_{profile_idx}",
                help="Percentage of total respondents with this profile"
            )
            st.caption(f"{profile['population_percentage']*100:.0f}% of respondents")
        
        with col3:
            profile['consistency'] = st.slider(
                "Response Consistency",
                min_value=0.1, max_value=1.0, value=profile.get('consistency', 0.8),
                step=0.1, key=f"consistency_{profile_idx}",
                help="How consistently this bias is applied (1.0 = always, 0.1 = rarely)"
            )
        
        # Advanced profile settings
        with st.expander("🔧 Advanced Settings", expanded=False):
            profile['temporal_drift'] = st.checkbox(
                "Enable Temporal Drift",
                value=profile.get('temporal_drift', False),
                key=f"temporal_{profile_idx}",
                help="Allow bias strength to change over time/items"
            )
            
            if profile['temporal_drift']:
                col1, col2 = st.columns(2)
                with col1:
                    profile['drift_direction'] = st.selectbox(
                        "Drift Direction",
                        options=["increasing", "decreasing", "cyclical"],
                        key=f"drift_dir_{profile_idx}"
                    )
                with col2:
                    profile['drift_rate'] = st.slider(
                        "Drift Rate",
                        min_value=0.01, max_value=0.2, value=0.05,
                        key=f"drift_rate_{profile_idx}"
                    )
            
            profile['fatigue_effect'] = st.checkbox(
                "Survey Fatigue Effect",
                value=profile.get('fatigue_effect', False),
                key=f"fatigue_{profile_idx}",
                help="Simulate decreasing response quality over survey length"
            )
            
            profile['social_desirability'] = st.slider(
                "Social Desirability Bias",
                min_value=0.0, max_value=1.0, value=profile.get('social_desirability', 0.0),
                key=f"social_{profile_idx}",
                help="Tendency to give socially desirable responses"
            )
        
        # Profile preview
        self._render_profile_preview(profile)
    
    def _render_scale_specific_tab(self):
        """Render scale-specific bias configuration"""
        st.subheader("🎯 Scale & Item-Specific Bias")
        st.info("Apply different bias patterns to specific scales or individual items")
        
        if not self.available_items and not self.available_scales:
            st.warning("No items or scales detected. Please ensure your survey data is loaded.")
            return
        
        # Scale selection mode
        selection_mode = st.radio(
            "Selection Mode",
            options=["scales", "items", "mixed"],
            format_func=lambda x: {
                "scales": "📊 By Scale Type",
                "items": "📝 By Individual Items", 
                "mixed": "🔀 Mixed Selection"
            }[x],
            help="Choose how to select which survey elements to apply bias to"
        )
        
        # Initialize scale_specific_config if not exists
        if not hasattr(self, 'scale_specific_config'):
            self.scale_specific_config = {}
        
        if selection_mode == "scales":
            self._render_scale_selection()
        elif selection_mode == "items":
            self._render_item_selection()
        else:
            self._render_mixed_selection()
        
        # Preview scale-specific effects
        if self.scale_specific_config:
            self._render_scale_specific_preview()
    
    def _render_scale_selection(self):
        """Render scale-based selection interface"""
        if not self.available_scales:
            st.warning("No scales detected")
            return
            
        st.write("**Select scales to apply bias:**")
        for scale in self.available_scales:
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                enabled = st.checkbox(f"📊 {scale}", key=f"scale_enable_{scale}")
            
            if enabled:
                if scale not in self.scale_specific_config:
                    self.scale_specific_config[scale] = {
                        'type': 'scale',
                        'bias_type': 'high',
                        'strength': 0.5,
                        'percentage': 0.3
                    }
                
                with col2:
                    self.scale_specific_config[scale]['bias_type'] = st.selectbox(
                        "Bias", options=["high", "mid", "low"],
                        key=f"scale_bias_{scale}", index=0
                    )
                with col3:
                    self.scale_specific_config[scale]['strength'] = st.slider(
                        "Strength", 0.1, 2.0, 0.5, key=f"scale_strength_{scale}"
                    )
            elif scale in self.scale_specific_config:
                del self.scale_specific_config[scale]
    
    def _render_item_selection(self):
        """Render item-based selection interface"""
        if not self.available_items:
            st.warning("No items detected")
            return
        
        # Search functionality
        search_term = st.text_input("🔍 Search items", placeholder="Type to filter items...")
        filtered_items = [item for item in self.available_items 
                         if search_term.lower() in item.lower()] if search_term else self.available_items
        
        # Batch operations
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("✅ Select All Filtered"):
                for item in filtered_items:
                    if item not in self.scale_specific_config:
                        self.scale_specific_config[item] = {
                            'type': 'item', 'bias_type': 'high', 'strength': 0.5, 'percentage': 0.3
                        }
        with col2:
            if st.button("❌ Deselect All"):
                self.scale_specific_config = {}
        with col3:
            batch_bias = st.selectbox("Batch Bias Type", ["high", "mid", "low"], key="batch_bias")
        
        # Item selection
        st.write(f"**Items ({len(filtered_items)} shown):**")
        for item in filtered_items[:20]:  # Limit display for performance
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                enabled = st.checkbox(f"📝 {item[:50]}{'...' if len(item) > 50 else ''}", 
                                    key=f"item_enable_{item}")
            
            if enabled:
                if item not in self.scale_specific_config:
                    self.scale_specific_config[item] = {
                        'type': 'item', 'bias_type': batch_bias, 'strength': 0.5, 'percentage': 0.3
                    }
                
                with col2:
                    self.scale_specific_config[item]['bias_type'] = st.selectbox(
                        "Bias", ["high", "mid", "low"], key=f"item_bias_{item}", 
                        index=["high", "mid", "low"].index(self.scale_specific_config[item]['bias_type'])
                    )
                with col3:
                    self.scale_specific_config[item]['strength'] = st.slider(
                        "Str", 0.1, 2.0, self.scale_specific_config[item]['strength'], 
                        key=f"item_strength_{item}"
                    )
            elif item in self.scale_specific_config:
                del self.scale_specific_config[item]
        
        if len(filtered_items) > 20:
            st.info(f"Showing first 20 of {len(filtered_items)} items. Use search to narrow down.")
    
    def _render_mixed_selection(self):
        """Render mixed scale/item selection interface"""
        st.write("**Mixed Selection Mode - Choose scales and/or individual items:**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**📊 Available Scales:**")
            for scale in self.available_scales[:10]:
                if st.checkbox(f"{scale}", key=f"mixed_scale_{scale}"):
                    if scale not in self.scale_specific_config:
                        self.scale_specific_config[scale] = {
                            'type': 'scale', 'bias_type': 'high', 'strength': 0.5
                        }
        
        with col2:
            st.write("**📝 Available Items:**")
            for item in self.available_items[:10]:
                if st.checkbox(f"{item[:30]}{'...' if len(item) > 30 else ''}", 
                              key=f"mixed_item_{item}"):
                    if item not in self.scale_specific_config:
                        self.scale_specific_config[item] = {
                            'type': 'item', 'bias_type': 'high', 'strength': 0.5
                        }
    
    def _render_distribution_controls_tab(self):
        """Render distribution control settings"""
        st.subheader("📊 Distribution Control & Quality Settings")
        
        # Distribution shape controls
        st.write("**🎲 Distribution Shape Controls**")
        col1, col2 = st.columns(2)
        
        with col1:
            self.skewness_control = st.slider(
                "Target Skewness",
                min_value=-2.0, max_value=2.0, value=0.0, step=0.1,
                help="Control distribution skewness: negative=left-skewed, positive=right-skewed"
            )
            
            self.kurtosis_control = st.slider(
                "Target Kurtosis",
                min_value=-1.0, max_value=3.0, value=0.0, step=0.1,
                help="Control distribution peakedness: negative=flatter, positive=more peaked"
            )
        
        with col2:
            self.missing_data_rate = st.slider(
                "Missing Data Rate",
                min_value=0.0, max_value=0.3, value=0.02, step=0.01,
                help="Proportion of responses to set as missing (realistic survey conditions)"
            )
            
            self.response_quality = st.selectbox(
                "Overall Response Quality",
                options=["high", "medium", "low"],
                help="Affects consistency, attention, and response patterns"
            )
        
        # Correlation structure
        st.write("**🔗 Inter-item Correlation Controls**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            self.base_correlation = st.slider(
                "Base Correlation Strength",
                min_value=0.1, max_value=0.9, value=0.4, step=0.05,
                help="Average correlation between related items"
            )
        
        with col2:
            self.correlation_variation = st.slider(
                "Correlation Variation",
                min_value=0.0, max_value=0.3, value=0.1, step=0.02,
                help="How much correlations vary around the base level"
            )
        
        with col3:
            self.factor_loading_strength = st.slider(
                "Factor Loading Strength",
                min_value=0.3, max_value=0.9, value=0.7, step=0.05,
                help="Strength of factor loadings for SEM compatibility"
            )
        
        # Advanced distribution controls
        with st.expander("🔬 Advanced Distribution Controls"):
            col1, col2 = st.columns(2)
            
            with col1:
                self.multicollinearity_control = st.slider(
                    "Multicollinearity Prevention",
                    min_value=0.0, max_value=1.0, value=0.8, step=0.1,
                    help="Prevent excessive correlations that could cause SEM issues"
                )
                
                self.heteroscedasticity = st.slider(
                    "Heteroscedasticity Level",
                    min_value=0.0, max_value=0.5, value=0.1, step=0.05,
                    help="Introduce varying response variance (0 = homoscedastic)"
                )
            
            with col2:
                self.outlier_rate = st.slider(
                    "Outlier Rate",
                    min_value=0.0, max_value=0.1, value=0.02, step=0.005,
                    help="Proportion of responses that are statistical outliers"
                )
                
                self.response_set_bias = st.slider(
                    "Response Set Bias",
                    min_value=0.0, max_value=0.3, value=0.05, step=0.01,
                    help="Tendency for some respondents to use same response pattern"
                )
    
    def _render_sem_dataset_tab(self):
        """Render SEM dataset generation configuration"""
        st.subheader("🔬 SEM/AMOS Dataset Generation")
        st.info("Generate datasets optimized for Structural Equation Modeling analysis")
        
        # Enable SEM optimization
        self.enable_sem_optimization = st.checkbox(
            "Enable SEM Optimization",
            help="Optimize dataset characteristics for structural equation modeling"
        )
        
        if not self.enable_sem_optimization:
            st.warning("SEM optimization disabled. Standard bias patterns will be applied.")
            return
        
        # SEM-specific configuration
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📊 Model Specification**")
            self.measurement_model = st.selectbox(
                "Measurement Model Type",
                options=["cfa", "hierarchical", "bifactor", "custom"],
                format_func=lambda x: {
                    "cfa": "Confirmatory Factor Analysis",
                    "hierarchical": "Higher-order Factor Model", 
                    "bifactor": "Bifactor Model",
                    "custom": "Custom Specification"
                }[x]
            )
            
            self.n_factors = st.number_input(
                "Number of Factors", min_value=1, max_value=10, value=3,
                help="Primary factors in your measurement model"
            )
            
            self.items_per_factor = st.number_input(
                "Items per Factor", min_value=3, max_value=15, value=5,
                help="Recommended: 3-7 items per factor for identification"
            )
        
        with col2:
            st.write("**🎯 Model Fit Optimization**")
            self.target_fit_indices = st.multiselect(
                "Optimize for Fit Indices",
                options=["CFI", "TLI", "RMSEA", "SRMR", "GFI", "AGFI"],
                default=["CFI", "RMSEA", "SRMR"],
                help="Which fit indices to optimize for"
            )
            
            self.target_cfi = st.slider("Target CFI", 0.90, 0.99, 0.95, 0.01) if "CFI" in self.target_fit_indices else 0.95
            self.target_rmsea = st.slider("Target RMSEA", 0.01, 0.08, 0.05, 0.005) if "RMSEA" in self.target_fit_indices else 0.05
            
            self.ensure_identification = st.checkbox(
                "Ensure Model Identification", 
                value=True,
                help="Guarantee sufficient constraints for model identification"
            )
        
        # Advanced SEM settings
        with st.expander("🔧 Advanced SEM Settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                self.cross_loadings = st.slider(
                    "Cross-loading Intensity",
                    min_value=0.0, max_value=0.3, value=0.1, step=0.02,
                    help="Strength of secondary factor loadings (adds realism)"
                )
                
                self.method_effects = st.checkbox(
                    "Include Method Effects",
                    help="Add correlated errors for similar item formats"
                )
                
                self.measurement_invariance = st.selectbox(
                    "Measurement Invariance Level",
                    options=["configural", "metric", "scalar", "strict"],
                    help="Level of measurement invariance to maintain across groups"
                )
            
            with col2:
                self.reliability_targets = st.text_area(
                    "Factor Reliability Targets",
                    value="0.80, 0.85, 0.90",
                    help="Comma-separated reliability targets for each factor (Cronbach's α)"
                )
                
                self.error_correlation_structure = st.selectbox(
                    "Error Correlation Structure",
                    options=["independent", "method_based", "content_based", "mixed"],
                    help="Pattern of correlated measurement errors"
                )
                
                self.normality_enforcement = st.slider(
                    "Multivariate Normality Strength",
                    min_value=0.5, max_value=1.0, value=0.8, step=0.05,
                    help="How strictly to enforce multivariate normality"
                )
        
        # Dataset export options
        st.write("**💾 Export Options**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            self.include_sem_syntax = st.checkbox(
                "Include AMOS Syntax", 
                value=True,
                help="Generate AMOS model syntax file"
            )
        
        with col2:
            self.include_factor_scores = st.checkbox(
                "Include Factor Scores",
                help="Calculate and include estimated factor scores"
            )
        
        with col3:
            self.validation_split = st.checkbox(
                "Create Validation Split",
                help="Split data for cross-validation (70/30)"
            )
    
    def _render_global_preview(self):
        """Render global configuration preview and validation"""
        st.markdown("---")
        st.subheader("📋 Configuration Summary & Validation")
        
        # Configuration summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🎭 Active Profiles:**")
            if self.bias_profiles:
                total_population = sum(p.get('population_percentage', 0) for p in self.bias_profiles)
                for profile in self.bias_profiles:
                    pct = profile.get('population_percentage', 0) * 100
                    st.write(f"• {profile['name']}: {pct:.1f}% ({profile.get('bias_type', 'unknown')})")
                
                if abs(total_population - 1.0) > 0.01:
                    st.error(f"⚠️ Population percentages sum to {total_population*100:.1f}% (should be 100%)")
                else:
                    st.success(f"✅ Population coverage: {total_population*100:.1f}%")
            else:
                st.info("No bias profiles configured")
        
        with col2:
            st.write("**🎯 Scale-Specific Settings:**")
            if hasattr(self, 'scale_specific_config') and self.scale_specific_config:
                for item, config in self.scale_specific_config.items():
                    st.write(f"• {item[:30]}{'...' if len(item) > 30 else ''}: {config['bias_type']}")
            else:
                st.info("No scale-specific bias configured")
        
        # Validation warnings
        self._render_validation_warnings()
        
        # Advanced preview
        if st.checkbox("📊 Show Advanced Preview", help="Display detailed bias effect predictions"):
            self._render_advanced_preview()
    
    def _render_validation_warnings(self):
        """Render configuration validation warnings"""
        warnings = []
        
        # Check population coverage
        if self.bias_profiles:
            total_pop = sum(p.get('population_percentage', 0) for p in self.bias_profiles)
            if total_pop > 1.01:
                warnings.append("Population percentages exceed 100% - profiles will overlap")
            elif total_pop < 0.99:
                warnings.append(f"Population coverage incomplete ({total_pop*100:.1f}%) - some respondents unbiased")
        
        # Check SEM compatibility
        if hasattr(self, 'enable_sem_optimization') and self.enable_sem_optimization:
            if not hasattr(self, 'n_factors') or self.n_factors < 2:
                warnings.append("SEM optimization requires at least 2 factors")
            
            total_items = getattr(self, 'n_factors', 3) * getattr(self, 'items_per_factor', 5)
            if total_items < 9:
                warnings.append("SEM model may be under-identified with fewer than 9 items")
        
        # Display warnings
        if warnings:
            st.warning("⚠️ **Configuration Warnings:**")
            for warning in warnings:
                st.write(f"• {warning}")
    
    def _render_advanced_preview(self):
        """Render advanced preview with charts and statistics"""
        if not self.bias_profiles:
            st.info("Add bias profiles to see advanced preview")
            return
        
        # Create distribution comparison chart
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Response Distribution Comparison", "Population Breakdown", 
                          "Bias Strength by Profile", "Quality Metrics"),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "scatter"}, {"type": "indicator"}]]
        )
        
        # Sample distribution comparison
        x_values = list(range(1, 6))
        baseline_dist = [0.1, 0.2, 0.4, 0.2, 0.1]
        
        fig.add_trace(
            go.Bar(x=x_values, y=baseline_dist, name="Baseline", marker_color="lightblue"),
            row=1, col=1
        )
        
        # Calculate composite biased distribution
        composite_dist = self._calculate_composite_distribution(baseline_dist)
        fig.add_trace(
            go.Bar(x=x_values, y=composite_dist, name="With Bias", marker_color="orange"),
            row=1, col=1
        )
        
        # Population pie chart
        profile_names = [p['name'] for p in self.bias_profiles]
        profile_pcts = [p.get('population_percentage', 0) for p in self.bias_profiles]
        
        fig.add_trace(
            go.Pie(labels=profile_names, values=profile_pcts, name="Population"),
            row=1, col=2
        )
        
        # Bias strength scatter
        bias_strengths = [p.get('bias_strength', 0.5) for p in self.bias_profiles]
        consistencies = [p.get('consistency', 0.8) for p in self.bias_profiles]
        
        fig.add_trace(
            go.Scatter(x=bias_strengths, y=consistencies, mode='markers+text',
                      text=profile_names, textposition="top center", name="Profiles"),
            row=2, col=1
        )
        
        fig.update_layout(height=800, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
    def _calculate_composite_distribution(self, baseline_dist):
        """Calculate composite distribution with all bias effects"""
        composite = baseline_dist.copy()
        
        for profile in self.bias_profiles:
            bias_type = profile.get('bias_type', 'high')
            strength = profile.get('bias_strength', 0.5)
            percentage = profile.get('population_percentage', 0.3)
            
            # Apply bias effect (simplified calculation)
            if bias_type == 'high':
                composite[-2:] = [c * (1 + strength * percentage) for c in composite[-2:]]
                composite[:-2] = [c * (1 - strength * percentage * 0.3) for c in composite[:-2]]
            elif bias_type == 'low':
                composite[:2] = [c * (1 + strength * percentage) for c in composite[:2]]
                composite[2:] = [c * (1 - strength * percentage * 0.3) for c in composite[2:]]
            elif bias_type == 'mid':
                composite[2] = composite[2] * (1 + strength * percentage)
                # Fix the problematic line using list comprehension and enumerate
                for i, c in enumerate(composite):
                    if i < 2 or i > 2:  # All elements except index 2
                        composite[i] = c * (1 - strength * percentage * 0.2)
        
        # Normalize
        total = sum(composite)
        return [c / total for c in composite] if total > 0 else composite
    
    def _add_bias_profile(self, name: str):
        """Add a new bias profile"""
        new_profile = {
            'name': name,
            'bias_type': 'high',
            'bias_strength': 0.5,
            'population_percentage': 0.3,
            'consistency': 0.8,
            'temporal_drift': False,
            'fatigue_effect': False,
            'social_desirability': 0.0
        }
        self.bias_profiles.append(new_profile)
    
    def _format_bias_type(self, bias_type: str) -> str:
        """Format bias type for display"""
        formats = {
            "high": "🔺 High Bias (toward maximum values)",
            "mid": "🔶 Mid Bias (toward middle values)", 
            "low": "🔻 Low Bias (toward minimum values)",
            "extreme_high": "⬆️ Extreme High (maximum values only)",
            "extreme_low": "⬇️ Extreme Low (minimum values only)",
            "central_tendency": "🎯 Central Tendency (avoid extremes)",
            "random_response": "🎲 Random Response Pattern"
        }
        return formats.get(bias_type, bias_type)
    
    def _render_profile_preview(self, profile: dict):
        """Render individual profile preview"""
        st.write("**Profile Effect Preview:**")
        
        # Create sample distribution
        baseline = {1: 0.1, 2: 0.2, 3: 0.4, 4: 0.2, 5: 0.1}
        biased = self._apply_profile_bias(baseline, profile)
        
        # Create comparison chart
        fig = go.Figure()
        
        values = list(baseline.keys())
        fig.add_trace(go.Bar(
            x=values, y=list(baseline.values()),
            name="Baseline", marker_color="lightblue", opacity=0.7
        ))
        fig.add_trace(go.Bar(
            x=values, y=list(biased.values()),
            name=f"{profile['name']}", marker_color="orange", opacity=0.8
        ))
        
        fig.update_layout(
            title=f"Distribution Effect: {profile['name']}",
            xaxis_title="Scale Value",
            yaxis_title="Probability",
            barmode='group',
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            baseline_mean = sum(k*v for k,v in baseline.items())
            biased_mean = sum(k*v for k,v in biased.items())
            st.metric("Mean Shift", f"{biased_mean:.2f}", f"{biased_mean-baseline_mean:+.2f}")
        
        with col2:
            baseline_var = sum(k*k*v for k,v in baseline.items()) - baseline_mean**2
            biased_var = sum(k*k*v for k,v in biased.items()) - biased_mean**2
            st.metric("Variance", f"{biased_var:.3f}", f"{biased_var-baseline_var:+.3f}")
        
        with col3:
            effect_size = abs(biased_mean - baseline_mean) / (baseline_var**0.5)
            st.metric("Effect Size", f"{effect_size:.2f}")
    
    def _apply_profile_bias(self, baseline_dist: dict, profile: dict) -> dict:
        """Apply a single profile's bias to a distribution"""
        biased = baseline_dist.copy()
        bias_type = profile.get('bias_type', 'high')
        strength = profile.get('bias_strength', 0.5)
        
        values = sorted(biased.keys())
        
        if bias_type == 'high':
            target_vals = values[-2:]
        elif bias_type == 'low':
            target_vals = values[:2]
        elif bias_type == 'mid':
            mid_idx = len(values) // 2
            target_vals = [values[mid_idx]]
        elif bias_type == 'extreme_high':
            target_vals = [values[-1]]
        elif bias_type == 'extreme_low':
            target_vals = [values[0]]
        elif bias_type == 'central_tendency':
            # Boost middle values, reduce extremes
            mid_idx = len(values) // 2
            target_vals = values[mid_idx-1:mid_idx+2] if mid_idx > 0 else [values[mid_idx]]
        elif bias_type == 'random_response':
            # Flatten distribution
            uniform_prob = 1.0 / len(values)
            return {k: uniform_prob for k in values}
        else:
            target_vals = values[-2:]  # Default to high
        
        # Apply bias
        for val in biased:
            if val in target_vals:
                biased[val] *= (1 + strength)
            else:
                biased[val] *= (1 - strength * 0.3)
        
        # Normalize
        total = sum(biased.values())
        return {k: v/total for k, v in biased.items()} if total > 0 else biased
    
    def _render_scale_specific_preview(self):
        """Render preview of scale-specific bias effects"""
        st.write("**Scale-Specific Bias Preview:**")
        
        if not self.scale_specific_config:
            st.info("No scale-specific bias configured")
            return
        
        # Summary table
        preview_data = []
        for item, config in self.scale_specific_config.items():
            preview_data.append({
                'Item/Scale': item[:40] + '...' if len(item) > 40 else item,
                'Type': config['type'].title(),
                'Bias Direction': config['bias_type'].title(),
                'Strength': f"{config['strength']:.1f}x",
                'Expected Effect': self._describe_bias_effect(config['bias_type'], config['strength'])
            })
        
        if preview_data:
            st.dataframe(pd.DataFrame(preview_data), use_container_width=True)
    
    def _describe_bias_effect(self, bias_type: str, strength: float) -> str:
        """Describe the expected effect of bias configuration"""
        intensity = "Strong" if strength > 1.5 else "Moderate" if strength > 1.0 else "Mild"
        
        effects = {
            'high': f"{intensity} shift toward high values",
            'mid': f"{intensity} central tendency",
            'low': f"{intensity} shift toward low values"
        }
        return effects.get(bias_type, f"{intensity} bias effect")
    
    def _get_empty_config(self) -> dict:
        """Return empty configuration when bias is disabled"""
        return {
            'enable_bias': False,
            'bias_profiles': [],
            'scale_specific_config': {},
            'distribution_controls': {},
            'sem_optimization': False
        }
    
    def _compile_configuration(self) -> dict:
        """Compile all configuration settings into a single dictionary"""
        config = {
            'enable_bias': self.enable_bias,
            'bias_profiles': self.bias_profiles,
            'scale_specific_config': getattr(self, 'scale_specific_config', {}),
            'distribution_controls': {
                'skewness_control': getattr(self, 'skewness_control', 0.0),
                'kurtosis_control': getattr(self, 'kurtosis_control', 0.0),
                'missing_data_rate': getattr(self, 'missing_data_rate', 0.02),
                'response_quality': getattr(self, 'response_quality', 'medium'),
                'base_correlation': getattr(self, 'base_correlation', 0.4),
                'correlation_variation': getattr(self, 'correlation_variation', 0.1),
                'factor_loading_strength': getattr(self, 'factor_loading_strength', 0.7),
                'multicollinearity_control': getattr(self, 'multicollinearity_control', 0.8),
                'heteroscedasticity': getattr(self, 'heteroscedasticity', 0.1),
                'outlier_rate': getattr(self, 'outlier_rate', 0.02),
                'response_set_bias': getattr(self, 'response_set_bias', 0.05)
            },
            'sem_optimization': {
                'enabled': getattr(self, 'enable_sem_optimization', False),
                'measurement_model': getattr(self, 'measurement_model', 'cfa'),
                'n_factors': getattr(self, 'n_factors', 3),
                'items_per_factor': getattr(self, 'items_per_factor', 5),
                'target_fit_indices': getattr(self, 'target_fit_indices', ['CFI', 'RMSEA']),
                'target_cfi': getattr(self, 'target_cfi', 0.95),
                'target_rmsea': getattr(self, 'target_rmsea', 0.05),
                'ensure_identification': getattr(self, 'ensure_identification', True),
                'cross_loadings': getattr(self, 'cross_loadings', 0.1),
                'method_effects': getattr(self, 'method_effects', False),
                'measurement_invariance': getattr(self, 'measurement_invariance', 'configural'),
                'reliability_targets': getattr(self, 'reliability_targets', "0.80, 0.85, 0.90"),
                'error_correlation_structure': getattr(self, 'error_correlation_structure', 'independent'),
                'normality_enforcement': getattr(self, 'normality_enforcement', 0.8),
                'include_sem_syntax': getattr(self, 'include_sem_syntax', True),
                'include_factor_scores': getattr(self, 'include_factor_scores', False),
                'validation_split': getattr(self, 'validation_split', False)
            }
        }
        return config


def apply_enhanced_bias_to_weights(weights: dict, config: dict, item_metadata: dict = None) -> dict:
    """
    Apply enhanced bias configuration to weights with multiple bias types and SEM optimization.
    
    Parameters:
    - weights: Original weights dictionary
    - config: Configuration from EnhancedBiasConfigComponent
    - item_metadata: Optional metadata about items (scales, factors, etc.)
    
    Returns:
    - Modified weights dictionary with applied biases
    """
    if not config.get('enable_bias', False):
        return weights
    
    biased_weights = weights.copy()
    
    # Apply profile-based biases
    if config.get('bias_profiles'):
        biased_weights = _apply_profile_biases(biased_weights, config['bias_profiles'])
    
    # Apply scale-specific biases
    if config.get('scale_specific_config'):
        biased_weights = _apply_scale_specific_biases(
            biased_weights, config['scale_specific_config'], item_metadata
        )
    
    # Apply distribution controls
    if config.get('distribution_controls'):
        biased_weights = _apply_distribution_controls(
            biased_weights, config['distribution_controls']
        )
    
    # Apply SEM optimizations
    if config.get('sem_optimization', {}).get('enabled', False):
        biased_weights = _apply_sem_optimizations(
            biased_weights, config['sem_optimization'], item_metadata
        )
    
    return biased_weights


def _apply_profile_biases(weights: dict, profiles: list) -> dict:
    """Apply multiple bias profiles to weights"""
    modified_weights = weights.copy()
    
    for item, w_data in modified_weights.items():
        if isinstance(w_data, dict) and w_data.get('is_distribution', False):
            original_dist = w_data['weights'].copy()
            composite_dist = original_dist.copy()
            
            # Apply each profile's effect proportionally
            for profile in profiles:
                profile_effect = _calculate_profile_effect(
                    original_dist, profile
                )
                
                # Blend profile effect based on population percentage
                pop_pct = profile.get('population_percentage', 0.3)
                for val_str in composite_dist:
                    baseline_prob = composite_dist[val_str]
                    profile_prob = profile_effect.get(val_str, baseline_prob)
                    composite_dist[val_str] = (
                        baseline_prob * (1 - pop_pct) + 
                        profile_prob * pop_pct
                    )
            
            # Normalize
            total_prob = sum(composite_dist.values())
            if total_prob > 0:
                composite_dist = {k: v/total_prob for k, v in composite_dist.items()}
            
            modified_weights[item] = {
                'is_distribution': True,
                'weights': composite_dist
            }
    
    return modified_weights


def _calculate_profile_effect(original_dist: dict, profile: dict) -> dict:
    """Calculate the effect of a single bias profile"""
    biased_dist = original_dist.copy()
    
    bias_type = profile.get('bias_type', 'high')
    strength = profile.get('bias_strength', 0.5)
    consistency = profile.get('consistency', 0.8)
    
    # Apply consistency factor
    effective_strength = strength * consistency
    
    values = sorted([int(k) for k in original_dist.keys()])
    
    # Define target values based on bias type
    if bias_type == 'high':
        target_vals = values[-2:]
    elif bias_type == 'low':
        target_vals = values[:2]
    elif bias_type == 'mid':
        mid_idx = len(values) // 2
        target_vals = [values[mid_idx]]
        if len(values) > 3:  # Include neighbors for better mid bias
            if mid_idx > 0:
                target_vals.append(values[mid_idx - 1])
            if mid_idx < len(values) - 1:
                target_vals.append(values[mid_idx + 1])
    elif bias_type == 'extreme_high':
        target_vals = [values[-1]]
    elif bias_type == 'extreme_low':
        target_vals = [values[0]]
    elif bias_type == 'central_tendency':
        # Avoid extremes, favor middle range
        if len(values) >= 5:
            target_vals = values[1:-1]
        else:
            target_vals = values[1:] if len(values) > 2 else values
    elif bias_type == 'random_response':
        # Flatten distribution
        uniform_prob = 1.0 / len(values)
        return {str(v): uniform_prob for v in values}
    else:
        target_vals = values[-2:]  # Default to high
    
    # Apply bias transformation
    for val_str, prob in biased_dist.items():
        val = int(val_str)
        if val in target_vals:
            biased_dist[val_str] = prob * (1 + effective_strength)
        else:
            reduction_factor = effective_strength * 0.5 / len([v for v in values if v not in target_vals])
            biased_dist[val_str] = prob * (1 - reduction_factor)
    
    # Handle temporal drift
    if profile.get('temporal_drift', False):
        drift_direction = profile.get('drift_direction', 'increasing')
        drift_rate = profile.get('drift_rate', 0.05)
        # Note: Temporal effects would be applied during response generation
        # This is a placeholder for the configuration
    
    # Handle fatigue effects
    if profile.get('fatigue_effect', False):
        # Fatigue typically increases central tendency and reduces variance
        # This would be applied during response generation based on item position
        pass
    
    # Apply social desirability bias
    social_desirability = profile.get('social_desirability', 0.0)
    if social_desirability > 0:
        # Generally shifts toward higher values for positive constructs
        # This is a simplified implementation
        for val_str in biased_dist:
            val = int(val_str)
            if val >= values[-2]:  # Higher values
                biased_dist[val_str] *= (1 + social_desirability * 0.5)
            elif val <= values[1]:  # Lower values
                biased_dist[val_str] *= (1 - social_desirability * 0.3)
    
    # Normalize
    total = sum(biased_dist.values())
    return {k: v/total for k, v in biased_dist.items()} if total > 0 else biased_dist


def _apply_scale_specific_biases(weights: dict, scale_config: dict, item_metadata: dict = None) -> dict:
    """Apply scale or item-specific bias configurations"""
    modified_weights = weights.copy()
    
    for target, bias_config in scale_config.items():
        config_type = bias_config.get('type', 'item')
        bias_type = bias_config.get('bias_type', 'high')
        strength = bias_config.get('strength', 0.5)
        
        # Determine which items to affect
        if config_type == 'scale' and item_metadata:
            affected_items = [
                item for item, meta in item_metadata.items() 
                if meta.get('scale') == target
            ]
        else:
            affected_items = [target] if target in weights else []
        
        # Apply bias to affected items
        for item in affected_items:
            if item in modified_weights:
                w_data = modified_weights[item]
                if isinstance(w_data, dict) and w_data.get('is_distribution', False):
                    original_dist = w_data['weights'].copy()
                    
                    # Create single-profile config for consistency
                    profile_config = {
                        'bias_type': bias_type,
                        'bias_strength': strength,
                        'consistency': 1.0,
                        'population_percentage': 1.0
                    }
                    
                    biased_dist = _calculate_profile_effect(original_dist, profile_config)
                    
                    modified_weights[item] = {
                        'is_distribution': True,
                        'weights': biased_dist
                    }
    
    return modified_weights


def _apply_distribution_controls(weights: dict, dist_controls: dict) -> dict:
    """Apply distribution shape and quality controls"""
    modified_weights = weights.copy()
    
    skewness_target = dist_controls.get('skewness_control', 0.0)
    kurtosis_target = dist_controls.get('kurtosis_control', 0.0)
    response_quality = dist_controls.get('response_quality', 'medium')
    
    # Quality-based adjustments
    quality_factors = {
        'high': {'consistency': 0.9, 'attention': 0.95},
        'medium': {'consistency': 0.75, 'attention': 0.85},
        'low': {'consistency': 0.6, 'attention': 0.7}
    }
    
    quality_params = quality_factors.get(response_quality, quality_factors['medium'])
    
    for item, w_data in modified_weights.items():
        if isinstance(w_data, dict) and w_data.get('is_distribution', False):
            dist = w_data['weights'].copy()
            
            # Apply skewness adjustment
            if abs(skewness_target) > 0.1:
                dist = _adjust_distribution_skewness(dist, skewness_target)
            
            # Apply kurtosis adjustment
            if abs(kurtosis_target) > 0.1:
                dist = _adjust_distribution_kurtosis(dist, kurtosis_target)
            
            # Apply quality adjustments
            if quality_params['consistency'] < 0.8:
                # Reduce consistency by adding some randomness
                dist = _add_response_inconsistency(dist, 1 - quality_params['consistency'])
            
            modified_weights[item] = {
                'is_distribution': True,
                'weights': dist
            }
    
    return modified_weights


def _adjust_distribution_skewness(dist: dict, target_skewness: float) -> dict:
    """Adjust distribution to achieve target skewness"""
    values = sorted([int(k) for k in dist.keys()])
    probs = [dist[str(v)] for v in values]
    
    if target_skewness > 0:  # Right skew
        # Shift probability mass toward lower values
        for i, val in enumerate(values):
            if val <= values[len(values)//2]:
                dist[str(val)] *= (1 + abs(target_skewness) * 0.5)
            else:
                dist[str(val)] *= (1 - abs(target_skewness) * 0.3)
    elif target_skewness < 0:  # Left skew
        # Shift probability mass toward higher values
        for i, val in enumerate(values):
            if val >= values[len(values)//2]:
                dist[str(val)] *= (1 + abs(target_skewness) * 0.5)
            else:
                dist[str(val)] *= (1 - abs(target_skewness) * 0.3)
    
    # Normalize
    total = sum(dist.values())
    return {k: v/total for k, v in dist.items()} if total > 0 else dist


def _adjust_distribution_kurtosis(dist: dict, target_kurtosis: float) -> dict:
    """Adjust distribution to achieve target kurtosis"""
    values = sorted([int(k) for k in dist.keys()])
    
    if target_kurtosis > 0:  # More peaked
        # Increase center, decrease tails
        mid_idx = len(values) // 2
        for i, val in enumerate(values):
            if abs(i - mid_idx) <= 1:  # Center values
                dist[str(val)] *= (1 + target_kurtosis * 0.5)
            else:  # Tail values
                dist[str(val)] *= (1 - target_kurtosis * 0.3)
    elif target_kurtosis < 0:  # Flatter
        # Decrease center, increase tails
        mid_idx = len(values) // 2
        for i, val in enumerate(values):
            if abs(i - mid_idx) <= 1:  # Center values
                dist[str(val)] *= (1 - abs(target_kurtosis) * 0.3)
            else:  # Tail values
                dist[str(val)] *= (1 + abs(target_kurtosis) * 0.4)
    
    # Normalize
    total = sum(dist.values())
    return {k: v/total for k, v in dist.items()} if total > 0 else dist


def _add_response_inconsistency(dist: dict, inconsistency_level: float) -> dict:
    """Add response inconsistency by introducing randomness"""
    modified_dist = dist.copy()
    
    # Add small random variations
    import random
    for key in modified_dist:
        random_factor = 1 + (random.random() - 0.5) * inconsistency_level * 0.2
        modified_dist[key] *= random_factor
    
    # Normalize
    total = sum(modified_dist.values())
    return {k: v/total for k, v in modified_dist.items()} if total > 0 else modified_dist


def _apply_sem_optimizations(weights: dict, sem_config: dict, item_metadata: dict = None) -> dict:
    """Apply SEM-specific optimizations to ensure good model fit"""
    modified_weights = weights.copy()
    
    # SEM optimizations would include:
    # 1. Ensuring sufficient correlation structure
    # 2. Avoiding multicollinearity
    # 3. Maintaining factor structure integrity
    # 4. Optimizing for target fit indices
    
    # This is a placeholder for complex SEM optimization logic
    # In practice, this would involve sophisticated statistical adjustments
    
    target_cfi = sem_config.get('target_cfi', 0.95)
    target_rmsea = sem_config.get('target_rmsea', 0.05)
    factor_loading_strength = sem_config.get('cross_loadings', 0.1)
    
    # Apply cross-loading adjustments
    if factor_loading_strength > 0 and item_metadata:
        # Introduce subtle cross-loadings to increase model realism
        # This would be implemented based on factor structure
        pass
    
    # Ensure measurement invariance requirements
    invariance_level = sem_config.get('measurement_invariance', 'configural')
    if invariance_level in ['metric', 'scalar', 'strict']:
        # Apply constraints to maintain invariance
        # This would involve constraining certain distributional properties
        pass
    
    return modified_weights


def generate_amos_syntax(config: dict, item_names: list, factor_structure: dict = None) -> str:
    """
    Generate AMOS syntax for the configured measurement model.
    
    Parameters:
    - config: SEM configuration dictionary
    - item_names: List of item variable names
    - factor_structure: Dictionary mapping factors to items
    
    Returns:
    - String containing AMOS model syntax
    """
    if not config.get('sem_optimization', {}).get('enabled', False):
        return ""
    
    sem_config = config['sem_optimization']
    model_type = sem_config.get('measurement_model', 'cfa')
    n_factors = sem_config.get('n_factors', 3)
    items_per_factor = sem_config.get('items_per_factor', 5)
    
    syntax_lines = []
    syntax_lines.append("# AMOS Syntax Generated by Enhanced Bias Configuration")
    syntax_lines.append("# Measurement Model: " + model_type.upper())
    syntax_lines.append("")
    
    # Generate factor loadings
    if model_type == 'cfa':
        syntax_lines.append("# Confirmatory Factor Analysis Model")
        for factor_num in range(1, n_factors + 1):
            factor_name = f"Factor{factor_num}"
            syntax_lines.append(f"\n{factor_name} =~ \\")
            
            start_idx = (factor_num - 1) * items_per_factor
            end_idx = min(start_idx + items_per_factor, len(item_names))
            factor_items = item_names[start_idx:end_idx]
            
            for i, item in enumerate(factor_items):
                connector = " + \\" if i < len(factor_items) - 1 else ""
                syntax_lines.append(f"  {item}{connector}")
    
    elif model_type == 'hierarchical':
        syntax_lines.append("# Higher-order Factor Analysis Model")
        # First-order factors
        for factor_num in range(1, n_factors + 1):
            factor_name = f"F{factor_num}"
            syntax_lines.append(f"\n{factor_name} =~ \\")
            
            start_idx = (factor_num - 1) * items_per_factor
            end_idx = min(start_idx + items_per_factor, len(item_names))
            factor_items = item_names[start_idx:end_idx]
            
            for i, item in enumerate(factor_items):
                connector = " + \\" if i < len(factor_items) - 1 else ""
                syntax_lines.append(f"  {item}{connector}")
        
        # Second-order factor
        syntax_lines.append(f"\nGeneral =~ \\")
        for factor_num in range(1, n_factors + 1):
            connector = " + \\" if factor_num < n_factors else ""
            syntax_lines.append(f"  F{factor_num}{connector}")
    
    elif model_type == 'bifactor':
        syntax_lines.append("# Bifactor Model")
        # General factor
        syntax_lines.append(f"\nGeneral =~ \\")
        for i, item in enumerate(item_names):
            connector = " + \\" if i < len(item_names) - 1 else ""
            syntax_lines.append(f"  {item}{connector}")
        
        # Specific factors
        for factor_num in range(1, n_factors + 1):
            factor_name = f"Specific{factor_num}"
            syntax_lines.append(f"\n{factor_name} =~ \\")
            
            start_idx = (factor_num - 1) * items_per_factor
            end_idx = min(start_idx + items_per_factor, len(item_names))
            factor_items = item_names[start_idx:end_idx]
            
            for i, item in enumerate(factor_items):
                connector = " + \\" if i < len(factor_items) - 1 else ""
                syntax_lines.append(f"  {item}{connector}")
    
    # Add method effects if requested
    if sem_config.get('method_effects', False):
        syntax_lines.append("\n# Method Effects (Correlated Errors)")
        syntax_lines.append("# Add correlated errors for similar item formats")
        syntax_lines.append("# Example: item1 ~~ item2")
    
    # Add cross-loadings if specified
    cross_loadings = sem_config.get('cross_loadings', 0.0)
    if cross_loadings > 0:
        syntax_lines.append("\n# Cross-loadings (if theoretically justified)")
        syntax_lines.append("# Example: Factor2 =~ item1")
    
    # Add model constraints for invariance
    invariance = sem_config.get('measurement_invariance', 'configural')
    if invariance != 'configural':
        syntax_lines.append(f"\n# Measurement Invariance: {invariance}")
        if invariance in ['metric', 'scalar', 'strict']:
            syntax_lines.append("# Add appropriate equality constraints")
    
    # Add fit index targets as comments
    syntax_lines.append("\n# Target Fit Indices:")
    for fit_index in sem_config.get('target_fit_indices', []):
        if fit_index == 'CFI':
            syntax_lines.append(f"# CFI >= {sem_config.get('target_cfi', 0.95)}")
        elif fit_index == 'RMSEA':
            syntax_lines.append(f"# RMSEA <= {sem_config.get('target_rmsea', 0.05)}")
    
    return "\n".join(syntax_lines)
