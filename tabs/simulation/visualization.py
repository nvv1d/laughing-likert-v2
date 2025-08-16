import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

class VisualizationManager:
    """Component for handling all visualization and comparison plots"""
    
    def render_distribution_comparison(self, df, sim_data, weights):
        """Render original vs simulated distribution comparison"""
        st.subheader("Original vs Simulated Distributions")
        
        # Option to select multiple items to compare or single item
        compare_type = st.radio(
            "Comparison type", 
            ["Show one item in detail", "Show multiple items side by side"]
        )
        
        items = list(weights.keys())
        
        if compare_type == "Show one item in detail":
            self._render_single_item_comparison(df, sim_data, weights, items)
        else:
            self._render_multiple_items_comparison(df, sim_data, items)
    
    def _render_single_item_comparison(self, df, sim_data, weights, items):
        """Render detailed comparison for a single item"""
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
                x=sim_data[selected_item],
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
            self._render_response_percentages(df, sim_data, selected_item, weights)
    
    def _render_response_percentages(self, df, sim_data, selected_item, weights):
        """Render response percentages comparison table"""
        st.subheader(f"Response Percentages: {selected_item}")
        
        try:
            orig_series = pd.to_numeric(df[selected_item], errors='coerce').dropna()
            sim_series = pd.to_numeric(sim_data[selected_item], errors='coerce').dropna()
            
            orig_counts = orig_series.value_counts(normalize=True).sort_index() * 100
            sim_counts = sim_series.value_counts(normalize=True).sort_index() * 100
            
            # Get all unique response values
            all_responses = set(orig_counts.index) | set(sim_counts.index)
            
            # Sort responses numerically
            sorted_responses = sorted(all_responses)
            
            comparison_data = []
            for response in sorted_responses:
                comparison_data.append({
                    'Response': int(response),  # Ensure integer type
                    'Weight': float(weights.get(selected_item, {}).get('weights', {}).get(str(response), 0.0)),  # Ensure float type
                    'Original (%)': f"{orig_counts.get(response, 0):.1f}%",
                    'Simulated (%)': f"{sim_counts.get(response, 0):.1f}%"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df)
            
        except Exception as e:
            st.error(f"Error creating response percentage comparison: {str(e)}")
            # Fallback: show basic statistics
            self._render_fallback_distributions(df, sim_data, selected_item)
    
    def _render_fallback_distributions(self, df, sim_data, selected_item):
        """Fallback distribution display when main comparison fails"""
        st.write("**Original Data Distribution:**")
        try:
            orig_dist = df[selected_item].value_counts(normalize=True) * 100
            for val, pct in orig_dist.items():
                st.write(f"  {val}: {pct:.1f}%")
        except:
            st.write("Could not calculate original distribution")
        
        st.write("**Simulated Data Distribution:**")
        try:
            sim_dist = sim_data[selected_item].value_counts(normalize=True) * 100
            for val, pct in sim_dist.items():
                st.write(f"  {val}: {pct:.1f}%")
        except:
            st.write("Could not calculate simulated distribution")
    
    def _render_multiple_items_comparison(self, df, sim_data, items):
        """Render comparison for multiple items side by side"""
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
                        x=sim_data[item],
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
    
    def render_download_button(self, sim_data):
        """Render download button for simulated data"""
        sim_csv = sim_data.to_csv(index=False)
        st.download_button(
            "Download Simulated Data as CSV",
            sim_csv,
            file_name=f"simulated_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    def create_correlation_heatmaps(self, real_corr, sim_corr, corr_diff):
        """Create side-by-side correlation heatmaps"""
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
    
    def create_similarity_gauge(self, final_score):
        """Create gauge chart for overall similarity score"""
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
    
    def create_component_scores_bar_chart(self, overall_metrics):
        """Create bar chart for component similarity scores"""
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
