import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import cronbach_alpha
from .visualization import VisualizationManager

class StatisticalAnalyzer:
    """Component for comprehensive statistical analysis and comparison"""
    
    def __init__(self):
        self.viz_manager = VisualizationManager()
    
    def render_comprehensive_statistical_comparison(self, df, sim_data, likert_items, clusters, alphas):
        """Render comprehensive statistical comparison section"""
        st.markdown("---")
        st.subheader("Statistical Comparison: Real vs Simulated Data")
        
        # Add a button to perform comprehensive statistical comparison
        analyze_button = st.button("Compare Real vs Simulated Data")
        
        if analyze_button or 'show_stat_analysis' in st.session_state:
            st.session_state.show_stat_analysis = True
            
            try:
                # Calculate all similarity metrics
                overall_similarity = self._analyze_descriptive_statistics(df, sim_data, likert_items)
                corr_similarity = self._analyze_correlation_structure(df, sim_data, likert_items)
                dist_similarity = self._analyze_distributions(df, sim_data, likert_items)
                reliability_similarity = self._analyze_reliability(df, sim_data, clusters, alphas)
                
                # Final combined score and assessment
                self._render_final_assessment(overall_similarity, corr_similarity, dist_similarity, reliability_similarity)
                
            except Exception as e:
                st.error(f"Error performing statistical comparison: {str(e)}")
                import traceback
                st.error(f"Detailed error: {traceback.format_exc()}")
        else:
            st.info("Click the 'Compare Real vs Simulated Data' button above for a comprehensive statistical comparison between your original and simulated data.")
    
    def _analyze_descriptive_statistics(self, df, sim_data, likert_items):
        """Analyze and compare descriptive statistics"""
        with st.expander("Descriptive Statistics Comparison", expanded=True):
            # Ensure all likert items are numeric to prevent type errors
            real_data_numeric = df[likert_items].apply(pd.to_numeric, errors='coerce')
            sim_data_numeric = sim_data[likert_items].apply(pd.to_numeric, errors='coerce')
            
            # Calculate descriptive statistics
            real_desc = real_data_numeric.describe().T
            sim_desc = sim_data_numeric.describe().T
            
            # Calculate additional statistics with error handling
            try:
                real_desc['var'] = real_data_numeric.var()
                sim_desc['var'] = sim_data_numeric.var()
                
                real_desc['skew'] = real_data_numeric.skew()
                sim_desc['skew'] = sim_data_numeric.skew()
                
                real_desc['kurtosis'] = real_data_numeric.kurtosis()
                sim_desc['kurtosis'] = sim_data_numeric.kurtosis()
            except Exception as e:
                st.warning(f"Could not calculate some advanced statistics: {str(e)}")
            
            # Calculate mean absolute differences between real and simulated statistics
            stats_diff = {}
            metrics = ['mean', 'std', 'var', 'skew', 'kurtosis', 'min', '25%', '50%', '75%', 'max']
            for metric in metrics:
                if metric in real_desc.columns and metric in sim_desc.columns:
                    try:
                        # Ensure both are numeric before subtraction
                        real_vals = pd.to_numeric(real_desc[metric], errors='coerce')
                        sim_vals = pd.to_numeric(sim_desc[metric], errors='coerce')
                        
                        # Remove NaN values before calculation
                        valid_mask = ~(real_vals.isna() | sim_vals.isna())
                        if valid_mask.any():
                            diff = abs(real_vals[valid_mask] - sim_vals[valid_mask])
                            stats_diff[metric] = diff.mean()
                    except Exception as e:
                        st.warning(f"Could not calculate difference for {metric}: {str(e)}")
                        continue
            
            # Create comparison charts for key metrics
            self._render_statistical_comparison_charts(real_desc, sim_desc, stats_diff)
            
            # Calculate overall similarity
            if stats_diff:
                similarity_df = pd.DataFrame({
                    'Statistic': list(stats_diff.keys()),
                    'Mean Absolute Difference': [stats_diff[k] for k in stats_diff.keys()],
                    'Similarity Score (%)': [max(0, 100 - 100 * stats_diff[k]) for k in stats_diff.keys()]
                })
                st.dataframe(similarity_df.sort_values('Similarity Score (%)', ascending=False))
                
                overall_similarity = similarity_df['Similarity Score (%)'].mean()
            else:
                st.warning("No valid statistics could be compared.")
                overall_similarity = 0
            
            st.metric("Overall Statistical Similarity", f"{overall_similarity:.2f}%")
            return overall_similarity
    
    def _render_statistical_comparison_charts(self, real_desc, sim_desc, stats_diff):
        """Render statistical comparison charts"""
        available_stats = list(stats_diff.keys())
        if available_stats:
            default_stats = [stat for stat in ['mean', 'std', 'var'] if stat in available_stats]
            selected_stats = st.multiselect(
                "Select statistics to compare", 
                options=available_stats,
                default=default_stats[:3] if default_stats else available_stats[:3]
            )
            
            for stat in selected_stats:
                st.subheader(f"Comparison of {stat.capitalize()}")
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=real_desc.index,
                    y=pd.to_numeric(real_desc[stat], errors='coerce'),
                    name="Original",
                    marker_color='blue'
                ))
                fig.add_trace(go.Bar(
                    x=sim_desc.index,
                    y=pd.to_numeric(sim_desc[stat], errors='coerce'),
                    name="Simulated",
                    marker_color='red'
                ))
                fig.update_layout(
                    title=f"{stat.capitalize()} Comparison",
                    barmode='group',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No statistics could be calculated. Please check your data types.")
    
    def _analyze_correlation_structure(self, df, sim_data, likert_items):
        """Analyze and compare correlation structures"""
        with st.expander("Correlation Structure Comparison", expanded=True):
            try:
                # Calculate correlation matrices with numeric data only
                real_data_numeric = df[likert_items].apply(pd.to_numeric, errors='coerce')
                sim_data_numeric = sim_data[likert_items].apply(pd.to_numeric, errors='coerce')
                
                real_corr = real_data_numeric.corr()
                sim_corr = sim_data_numeric.corr()
                
                # Calculate correlation matrix difference
                corr_diff = abs(real_corr - sim_corr)
                
                # Display side-by-side correlation heatmaps
                self.viz_manager.create_correlation_heatmaps(real_corr, sim_corr, corr_diff)
                
                # Calculate overall correlation similarity
                mean_diff = corr_diff.values[np.triu_indices_from(corr_diff.values, k=1)].mean()
                corr_similarity = max(0, 100 - (mean_diff * 100))
                st.metric("Correlation Structure Similarity", f"{corr_similarity:.2f}%")
                return corr_similarity
            except Exception as e:
                st.error(f"Error calculating correlations: {str(e)}")
                return 0
    
    def _analyze_distributions(self, df, sim_data, likert_items):
        """Analyze and compare distributions using KL divergence and JS distance"""
        with st.expander("Distribution Comparison", expanded=True):
            # Select items to compare
            dist_items = st.multiselect(
                "Select items for distribution comparison",
                options=likert_items,
                default=likert_items[:min(3, len(likert_items))]
            )
            
            if dist_items:
                # Calculate KL divergence for each item
                kl_divergences = {}
                js_distances = {}
                
                for item in dist_items:
                    try:
                        # Calculate distribution proportions with proper data type handling
                        real_series = pd.to_numeric(df[item], errors='coerce').dropna()
                        sim_series = pd.to_numeric(sim_data[item], errors='coerce').dropna()
                        
                        real_dist = real_series.value_counts(normalize=True).sort_index()
                        sim_dist = sim_series.value_counts(normalize=True).sort_index()
                        
                        # Ensure both distributions have the same categories
                        all_values = sorted(set(real_dist.index) | set(sim_dist.index))
                        real_probs = np.array([real_dist.get(val, 0) for val in all_values])
                        sim_probs = np.array([sim_dist.get(val, 0) for val in all_values])
                        
                        # Avoid zero probabilities for KL divergence
                        real_probs = np.clip(real_probs, 1e-10, 1.0)
                        sim_probs = np.clip(sim_probs, 1e-10, 1.0)
                        
                        # Normalize to sum to 1
                        real_probs = real_probs / real_probs.sum()
                        sim_probs = sim_probs / sim_probs.sum()
                        
                        # Calculate KL divergence: D_KL(P||Q)
                        kl_div = np.sum(real_probs * np.log(real_probs / sim_probs))
                        kl_divergences[item] = kl_div
                        
                        # Calculate Jensen-Shannon distance
                        m_dist = 0.5 * (real_probs + sim_probs)
                        js_div = 0.5 * np.sum(real_probs * np.log(real_probs / m_dist)) + 0.5 * np.sum(sim_probs * np.log(sim_probs / m_dist))
                        js_distances[item] = np.sqrt(js_div)  # JS distance is sqrt of JS divergence
                    except Exception as e:
                        st.warning(f"Could not calculate divergence for item {item}: {str(e)}")
                
                if kl_divergences and js_distances:
                    # Display the divergence measures
                    divergence_df = pd.DataFrame({
                        'Item': list(kl_divergences.keys()),
                        'KL Divergence': list(kl_divergences.values()),
                        'JS Distance': list(js_distances.values()),
                        'Distribution Similarity (%)': [max(0, 100 - (js * 100)) for js in js_distances.values()]
                    })
                    st.dataframe(divergence_df.sort_values('Distribution Similarity (%)', ascending=False))
                    
                    # Overall distribution similarity
                    dist_similarity = divergence_df['Distribution Similarity (%)'].mean()
                    st.metric("Overall Distribution Similarity", f"{dist_similarity:.2f}%")
                    return dist_similarity
                else:
                    st.warning("Could not calculate distribution similarities for selected items.")
                    return 0
            else:
                return 0
    
    def _analyze_reliability(self, df, sim_data, clusters, alphas):
        """Analyze and compare reliability measures (Cronbach's Alpha)"""
        if not clusters:
            return 0
        
        with st.expander("Reliability Comparison", expanded=True):
            # Calculate alphas for original and simulated data by cluster
            # First verify all required columns exist in simulated data
            available_items = set(sim_data.columns)
            
            # Make sure all items exist in the simulated data
            valid_items = {}
            for sc, items in clusters.items():
                # Check if all items in this cluster exist in the simulated data
                valid_cluster_items = [item for item in items if item in sim_data.columns]
                if len(valid_cluster_items) > 1:  # Need at least 2 items for reliability
                    valid_items[sc] = valid_cluster_items
            
            if valid_items:
                alpha_data = []
                
                for sc, items in valid_items.items():
                    try:
                        orig_alpha = alphas.get(sc, 0)
                        sim_alpha = cronbach_alpha(sim_data, items)
                        
                        alpha_diff = abs(orig_alpha - sim_alpha)
                        alpha_similarity = max(0, 100 - (alpha_diff * 100))
                        
                        alpha_data.append({
                            'Cluster': sc,
                            'Items': len(items),
                            'Original Alpha': orig_alpha,
                            'Simulated Alpha': sim_alpha,
                            'Difference': alpha_diff,
                            'Similarity (%)': alpha_similarity
                        })
                    except Exception as e:
                        st.warning(f"Could not calculate reliability for cluster {sc}: {str(e)}")
                
                # Display results table
                alpha_df = pd.DataFrame(alpha_data)
                if not alpha_df.empty:
                    st.dataframe(alpha_df.sort_values('Similarity (%)', ascending=False))
                    
                    # Overall reliability similarity
                    reliability_similarity = alpha_df['Similarity (%)'].mean()
                    st.metric("Overall Reliability Similarity", f"{reliability_similarity:.2f}%")
                    
                    # Display clusters with reliability issues
                    problem_clusters = alpha_df[alpha_df['Similarity (%)'] < 75]
                    if not problem_clusters.empty:
                        st.warning("The following clusters show significant reliability differences:")
                        st.dataframe(problem_clusters[['Cluster', 'Original Alpha', 'Simulated Alpha', 'Similarity (%)']])
                        
                        # Suggestions for improvement
                        st.info("💡 Suggestions: Use a larger dataset or try adjusting noise level or using different item weights extraction methods.")
                    
                    return reliability_similarity
                else:
                    return 0
            else:
                st.warning("No valid clusters found for reliability comparison.")
                return 0
    
    def _render_final_assessment(self, overall_similarity, corr_similarity, dist_similarity, reliability_similarity):
        """Render final combined score and quality assessment"""
        # Final combined score
        overall_metrics = []
        
        if overall_similarity is not None:
            overall_metrics.append(('Statistical Descriptives', overall_similarity))
        
        if corr_similarity is not None:
            overall_metrics.append(('Correlation Structure', corr_similarity))
        
        if dist_similarity is not None:
            overall_metrics.append(('Distribution Similarity', dist_similarity))
        
        if reliability_similarity is not None:
            overall_metrics.append(('Reliability Metrics', reliability_similarity))
        
        if overall_metrics:
            st.markdown("---")
            st.subheader("Overall Similarity Assessment")
            
            final_score = sum(score for _, score in overall_metrics) / len(overall_metrics)
            
            # Create a gauge chart for final score
            self.viz_manager.create_similarity_gauge(final_score)
            
            # Individual score breakdown
            self.viz_manager.create_component_scores_bar_chart(overall_metrics)
            
            # Quality assessment
            self._provide_quality_assessment(final_score)
    
    def _provide_quality_assessment(self, final_score):
        """Provide quality assessment based on final score"""
        if final_score >= 90:
            st.success("🌟 Excellent simulation quality! The simulated data closely matches the original data across all metrics.")
        elif final_score >= 80:
            st.success("✅ Good simulation quality. The simulated data captures most patterns in the original data.")
        elif final_score >= 70:
            st.warning("⚠️ Fair simulation quality. The simulated data captures some patterns but has notable differences.")
        else:
            st.error("❌ Poor simulation quality. Consider adjusting parameters to improve results.")
