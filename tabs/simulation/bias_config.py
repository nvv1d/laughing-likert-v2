import streamlit as st
import pandas as pd
import plotly.graph_objects as go

class BiasConfigComponent:
    """Component for handling bias configuration UI and logic"""
    
    def __init__(self):
        self.bias_type = None
        self.bias_strength = None
        self.bias_percentage = None
        self.enable_bias = False
    
    def render_bias_config(self):
        """Render the bias configuration UI and return settings"""
        st.markdown("---")
        st.subheader("🎯 Response Bias Options")
        
        self.enable_bias = st.checkbox(
            "Enable Response Bias", 
            value=False,
            help="Apply systematic bias to simulate specific response patterns"
        )
        
        if self.enable_bias:
            st.info("🔧 Configure bias settings to simulate different respondent types (e.g., high-achievers, pessimists)")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                self.bias_type = st.selectbox(
                    "Bias Direction",
                    options=["high", "low"],
                    format_func=lambda x: "High Bias (optimistic/high-achievers)" if x == "high" else "Low Bias (pessimistic/critical)",
                    help="Direction of bias: high = toward maximum scale values, low = toward minimum scale values"
                )
            
            with col2:
                self.bias_strength = st.slider(
                    "Bias Strength",
                    min_value=0.1,
                    max_value=2.0,
                    value=0.5,
                    step=0.1,
                    help="How strong the bias is (higher = more extreme bias)"
                )
            
            with col3:
                self.bias_percentage = st.slider(
                    "Percentage Affected",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.3,
                    step=0.05,
                    help="What percentage of responses should be affected by bias"
                )
                st.caption(f"Selected: {self.bias_percentage*100:.0f}%")
            
            # Show bias preview
            self._render_bias_preview()
        
        return {
            'enable_bias': self.enable_bias,
            'bias_type': self.bias_type,
            'bias_strength': self.bias_strength,
            'bias_percentage': self.bias_percentage
        }
    
    def _render_bias_preview(self):
        """Render bias configuration preview and example"""
        with st.expander("🔍 Bias Configuration Preview", expanded=True):
            if self.bias_type == "high":
                st.success(f"**High Bias Configuration:**")
                st.write(f"- **Effect**: {self.bias_percentage*100:.0f}% of responses will be biased toward higher values")
                st.write(f"- **Strength**: {self.bias_strength:.1f}x increase in probability for top 2 scale values")
                st.write(f"- **Use case**: Simulate high-achievers, optimistic respondents, or positive response bias")
            else:
                st.warning(f"**Low Bias Configuration:**")
                st.write(f"- **Effect**: {self.bias_percentage*100:.0f}% of responses will be biased toward lower values")
                st.write(f"- **Strength**: {self.bias_strength:.1f}x increase in probability for bottom 2 scale values")
                st.write(f"- **Use case**: Simulate critical respondents, pessimistic views, or negative response bias")
            
            st.write(f"- **Unbiased responses**: {(1-self.bias_percentage)*100:.0f}% will follow original patterns")
            
            # Show example effect
            self._show_bias_example()
    
    def _show_bias_example(self):
        """Show example of how bias would affect a 5-point scale"""
        st.write("**Example Effect on 5-point Scale (1-5):**")
        example_original = {1: 0.1, 2: 0.2, 3: 0.4, 4: 0.2, 5: 0.1}
        
        if self.bias_type == "high":
            target_values = [4, 5]
        else:
            target_values = [1, 2]
        
        example_biased = {}
        for val, prob in example_original.items():
            if val in target_values:
                example_biased[val] = prob * (1 + (self.bias_strength * self.bias_percentage))
            else:
                example_biased[val] = prob * (1 - (self.bias_strength * self.bias_percentage * 0.5))
        
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
