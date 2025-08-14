, expanded=True):
                # Select items to compare
                dist_items = st.multiselect(
                    "Select items for distribution comparison",
                    options=st.session_state.likert_items,
                    default=st.session_state.likert_items[:min(3, len(st.session_state.likert_items))]
                )

                if dist_items:
                    # Calculate KL divergence for each item
                    kl_divergences = {}
                    js_distances = {}

                    for item in dist_items:
                        # Calculate distribution proportions
                        real_dist = df[item].value_counts(normalize=True).sort_index()
                        sim_dist = st.session_state.sim_data[item].value_counts(normalize=True).sort_index()

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

                        try:
                            # Calculate KL divergence: D_KL(P||Q)
                            kl_div = np.sum(real_probs * np.log(real_probs / sim_probs))
                            kl_divergences[item] = kl_div

                            # Calculate Jensen-Shannon distance
                            m_dist = 0.5 * (real_probs + sim_probs)
                            js_div = 0.5 * np.sum(real_probs * np.log(real_probs / m_dist)) + 0.5 * np.sum(sim_probs * np.log(sim_probs / m_dist))
                            js_distances[item] = np.sqrt(js_div)  # JS distance is sqrt of JS divergence
                        except Exception as e:
                            st.warning(f"Could not calculate divergence for item {item}: {str(e)}")

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

            # 4. Reliability Measures (Cronbach's Alpha comparison)
            if st.session_state.clusters:
                with st.expander("Reliability Comparison", expanded=True):
                    # Calculate alphas for original and simulated data by cluster
                    sim_alphas = {}

                    # First verify all required columns exist in simulated data
                    available_items = set(st.session_state.sim_data.columns)

                    # Make sure all items exist in the simulated data
                    valid_items = {}
                    for sc, items in st.session_state.clusters.items():
                        # Check if all items in this cluster exist in the simulated data
                        valid_cluster_items = [item for item in items if item in st.session_state.sim_data.columns]
                        if len(valid_cluster_items) > 1:  # Need at least 2 items for reliability
                            valid_items[sc] = valid_cluster_items

                    if valid_items:
                        alpha_data = []

                        for sc, items in valid_items.items():
                            try:
                                orig_alpha = st.session_state.alphas.get(sc, 0)
                                sim_alpha = cronbach_alpha(st.session_state.sim_data, items)

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

            # Final combined score
            overall_metrics = []

            if 'overall_similarity' in locals():
                overall_metrics.append(('Statistical Descriptives', overall_similarity))

            if 'corr_similarity' in locals():
                overall_metrics.append(('Correlation Structure', corr_similarity))

            if 'dist_similarity' in locals():
                overall_metrics.append(('Distribution Similarity', dist_similarity))

            if 'reliability_similarity' in locals():
                overall_metrics.append(('Reliability Metrics', reliability_similarity))

            if overall_metrics:
                st.markdown("---")
                st.subheader("Overall Similarity Assessment")

                final_score = sum(score for _, score in overall_metrics) / len(overall_metrics)

                # Create a gauge chart for final score
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

                # Individual score breakdown
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

                # Quality assessment
                if final_score >= 90:
                    st.success("🌟 Excellent simulation quality! The simulated data closely matches the original data across all metrics.")
                elif final_score >= 80:
                    st.success("✅ Good simulation quality. The simulated data captures most patterns in the original data.")
                elif final_score >= 70:
                    st.warning("⚠️ Fair simulation quality. The simulated data captures some patterns but has notable differences.")
                else:
                    st.error("❌ Poor simulation quality. Consider adjusting parameters to improve results.")

        except Exception as e:
            st.error(f"Error performing statistical comparison: {str(e)}")
    else:
        st.info("Click the 'Compare Real vs Simulated Data' button above for a comprehensive statistical comparison between your original and simulated data.")
