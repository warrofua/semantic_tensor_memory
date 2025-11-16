#!/usr/bin/env python3
"""
Explainability Engine for Semantic Tensor Memory

Provides clear explanations for analysis results and auto-adjusts parameters
based on data quality and analysis outcomes.
"""

import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.metrics import silhouette_score
import plotly.express as px

@dataclass
class AnalysisExplanation:
    """Structured explanation of analysis results."""
    analysis_type: str
    quality_score: float
    confidence_level: str  # "high", "medium", "low"
    what_it_means: str
    why_these_results: str
    what_to_do_next: List[str]
    technical_details: Dict[str, Any]
    alternative_approaches: List[str]

@dataclass
class AutoAdjustment:
    """Auto-adjustment recommendation with rationale."""
    parameter_name: str
    current_value: Any
    suggested_value: Any
    reason: str
    expected_improvement: str
    confidence: float  # 0-1

class ExplainabilityEngine:
    """Engine for providing explanations and auto-adjustments."""
    
    def __init__(self):
        self.analysis_history = []
        self.adjustment_history = []
    
    def explain_pca_results(self, pca_results: Dict[str, Any], dataset_info: Dict[str, Any]) -> AnalysisExplanation:
        """Provide detailed explanation of PCA results."""
        
        explained_variance = pca_results.get('cumulative_variance', [0])[-1]
        n_components = len(pca_results.get('explained_variance_ratio', []))
        n_samples = dataset_info.get('n_samples', 0)
        n_features = dataset_info.get('n_features', 0)
        
        # Determine quality and confidence
        if explained_variance > 0.7:
            quality_score = 0.9
            confidence = "high"
            what_it_means = f"🎯 **Excellent PCA Results**: The {n_components} principal components capture {explained_variance:.1%} of your data's variance, which is excellent. This means the 2D/3D visualization represents your conversations very well."
            why_these_results = "Your conversations have clear, distinct patterns that can be effectively captured in lower dimensions. This suggests strong thematic structure in your data."
        elif explained_variance > 0.5:
            quality_score = 0.7
            confidence = "medium"
            what_it_means = f"✅ **Good PCA Results**: The {n_components} components capture {explained_variance:.1%} of variance. This is reasonable for high-dimensional semantic data, though some nuance is lost."
            why_these_results = "Semantic embeddings are naturally high-dimensional, so 50-70% explained variance is common. Your conversations have moderate structure."
        elif explained_variance > 0.3:
            quality_score = 0.4
            confidence = "low"
            what_it_means = f"⚠️ **Moderate PCA Results**: Only {explained_variance:.1%} of variance is captured. The visualization shows some patterns but misses significant information."
            why_these_results = "Your conversations may be very diverse with complex, high-dimensional relationships that don't compress well into 2D/3D."
        else:
            quality_score = 0.2
            confidence = "low"
            what_it_means = f"❌ **Poor PCA Results**: Only {explained_variance:.1%} variance captured. The 2D/3D visualization may be misleading."
            why_these_results = "Your data may be too high-dimensional or noisy for PCA. The conversations might not have clear linear patterns."
        
        # Generate actionable next steps
        what_to_do_next = []
        alternative_approaches = []
        
        if explained_variance < 0.5:
            what_to_do_next.append("🔄 Try UMAP or t-SNE for non-linear dimensionality reduction")
            what_to_do_next.append("📊 Increase number of PCA components to capture more variance")
            alternative_approaches.extend(["UMAP", "t-SNE", "Autoencoders"])
        
        if n_samples < n_features * 5:
            what_to_do_next.append("📈 Consider collecting more data for better PCA stability")
        
        if explained_variance > 0.8:
            what_to_do_next.append("🎯 Results are excellent - proceed with confidence to clustering analysis")
        
        return AnalysisExplanation(
            analysis_type="Principal Component Analysis",
            quality_score=quality_score,
            confidence_level=confidence,
            what_it_means=what_it_means,
            why_these_results=why_these_results,
            what_to_do_next=what_to_do_next,
            technical_details={
                'explained_variance': explained_variance,
                'n_components': n_components,
                'sample_to_feature_ratio': n_samples / max(1, n_features),
                'individual_variances': pca_results.get('explained_variance_ratio', [])
            },
            alternative_approaches=alternative_approaches
        )
    
    def explain_clustering_results(self, cluster_labels: np.ndarray = None, embeddings: np.ndarray = None, 
                                 n_clusters: int = None, method: str = "K-means", 
                                 n_sessions: int = None, cluster_quality_score: float = None) -> AnalysisExplanation:
        """Provide detailed explanation of clustering results."""
        
        # Handle both calling patterns - with actual cluster data or just summary stats
        if cluster_labels is not None and embeddings is not None:
            return self._explain_clustering_with_data(cluster_labels, embeddings, n_clusters, method)
        elif n_clusters is not None and n_sessions is not None:
            return self._explain_clustering_with_stats(n_clusters, n_sessions, cluster_quality_score)
        else:
            return AnalysisExplanation(
                analysis_type="Clustering Analysis",
                quality_score=0.5,
                confidence_level="low",
                what_it_means="⚠️ **Insufficient Information**: Not enough data to assess clustering quality.",
                why_these_results="Missing cluster data or summary statistics.",
                what_to_do_next=["Provide cluster labels and embeddings for detailed analysis"],
                technical_details={'error': 'insufficient_data'},
                alternative_approaches=["Run clustering analysis first"]
            )
    
    def _explain_clustering_with_data(self, cluster_labels: np.ndarray, embeddings: np.ndarray, 
                                    n_clusters: int, method: str) -> AnalysisExplanation:
        """Explain clustering results using actual cluster data."""
        try:
            silhouette = silhouette_score(embeddings, cluster_labels)
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            cluster_balance = np.min(counts) / np.max(counts)
            
        except Exception as e:
            return AnalysisExplanation(
                analysis_type=f"{method} Clustering",
                quality_score=0.1,
                confidence_level="low",
                what_it_means="❌ **Clustering Failed**: Could not assess clustering quality due to data issues.",
                why_these_results=f"Technical error: {str(e)}",
                what_to_do_next=["Check data quality", "Try different preprocessing"],
                technical_details={'error': str(e)},
                alternative_approaches=["DBSCAN", "Hierarchical clustering"]
            )
        
        # Assess clustering quality
        if silhouette > 0.5:
            quality_score = 0.9
            confidence = "high"
            what_it_means = f"🎯 **Excellent Clustering**: Silhouette score of {silhouette:.2f} indicates very distinct, well-separated conversation clusters."
            why_these_results = f"Your {n_clusters} conversation themes are clearly distinct from each other. The {method} algorithm found natural groupings."
        elif silhouette > 0.3:
            quality_score = 0.7
            confidence = "medium"
            what_it_means = f"✅ **Good Clustering**: Silhouette score of {silhouette:.2f} shows moderately distinct clusters."
            why_these_results = f"There are {n_clusters} identifiable conversation themes, though some overlap exists between themes."
        elif silhouette > 0.1:
            quality_score = 0.4
            confidence = "low"
            what_it_means = f"⚠️ **Moderate Clustering**: Silhouette score of {silhouette:.2f} suggests weak cluster separation."
            why_these_results = "The conversation themes may not be very distinct, or the number of clusters might not be optimal."
        else:
            quality_score = 0.2
            confidence = "low"
            what_it_means = f"❌ **Poor Clustering**: Silhouette score of {silhouette:.2f} indicates poorly separated clusters."
            why_these_results = "Your conversations may not have clear thematic groups, or clustering parameters need adjustment."
        
        # Generate recommendations
        what_to_do_next = []
        alternative_approaches = []
        
        if silhouette < 0.3:
            what_to_do_next.append(f"🔢 Try different numbers of clusters (current: {n_clusters})")
            what_to_do_next.append("🔄 Experiment with different clustering algorithms")
            alternative_approaches.extend(["DBSCAN", "Hierarchical Clustering", "Gaussian Mixture Models"])
        
        if cluster_balance < 0.2:
            what_to_do_next.append("⚖️ Clusters are imbalanced - consider adjusting cluster count")
        
        if silhouette > 0.5:
            what_to_do_next.append("🎯 Excellent clusters! Proceed to analyze cluster themes and evolution")
        
        return AnalysisExplanation(
            analysis_type=f"{method} Clustering",
            quality_score=quality_score,
            confidence_level=confidence,
            what_it_means=what_it_means,
            why_these_results=why_these_results,
            what_to_do_next=what_to_do_next,
            technical_details={
                'silhouette_score': silhouette,
                'n_clusters': n_clusters,
                'cluster_sizes': counts.tolist(),
                'cluster_balance': cluster_balance,
                'largest_cluster_pct': np.max(counts) / len(cluster_labels) * 100
            },
            alternative_approaches=alternative_approaches
        )
    
    def _explain_clustering_with_stats(self, n_clusters: int, n_sessions: int, 
                                     cluster_quality_score: float = None) -> AnalysisExplanation:
        """Explain clustering results using summary statistics."""
        
        # Use provided quality score or estimate based on cluster/session ratio
        if cluster_quality_score is None:
            # Heuristic: optimal cluster count is roughly sqrt(n_sessions/2)
            optimal_clusters = max(2, int(np.sqrt(n_sessions / 2)))
            cluster_ratio = abs(n_clusters - optimal_clusters) / optimal_clusters
            quality_score = max(0.3, 1.0 - cluster_ratio)  # Rough quality estimate
        else:
            quality_score = cluster_quality_score
        
        # Assess quality
        if quality_score > 0.7:
            confidence = "high"
            what_it_means = f"🎯 **Good Clustering Setup**: {n_clusters} clusters for {n_sessions} sessions appears well-balanced."
            why_these_results = f"The cluster-to-session ratio ({n_sessions/n_clusters:.1f} sessions per cluster) suggests meaningful groupings."
        elif quality_score > 0.5:
            confidence = "medium"
            what_it_means = f"✅ **Reasonable Clustering**: {n_clusters} clusters for {n_sessions} sessions is workable."
            why_these_results = f"With {n_sessions/n_clusters:.1f} sessions per cluster on average, you should see some thematic groupings."
        else:
            confidence = "low"
            what_it_means = f"⚠️ **Suboptimal Clustering**: {n_clusters} clusters may not be ideal for {n_sessions} sessions."
            why_these_results = f"The ratio of {n_sessions/n_clusters:.1f} sessions per cluster suggests either too many or too few clusters."
        
        # Generate recommendations
        recommendations = []
        sessions_per_cluster = n_sessions / n_clusters
        
        if sessions_per_cluster < 3:
            recommendations.append(f"🔢 Consider fewer clusters - current ratio is only {sessions_per_cluster:.1f} sessions per cluster")
            recommendations.append("💡 Try 3-8 clusters for better thematic coherence")
        elif sessions_per_cluster > 20:
            recommendations.append(f"🔢 Consider more clusters - current ratio is {sessions_per_cluster:.1f} sessions per cluster")
            recommendations.append("💡 More clusters might reveal finer-grained themes")
        else:
            recommendations.append("✅ Current cluster count looks reasonable for your dataset size")
            recommendations.append("🎯 Proceed to analyze cluster themes and evolution patterns")
        
        # Add general recommendations
        if n_sessions < 20:
            recommendations.append("📊 Consider gathering more sessions for more robust clustering")
        
        return AnalysisExplanation(
            analysis_type="Clustering Configuration Analysis",
            quality_score=quality_score,
            confidence_level=confidence,
            what_it_means=what_it_means,
            why_these_results=why_these_results,
            what_to_do_next=recommendations,
            technical_details={
                'n_clusters': n_clusters,
                'n_sessions': n_sessions,
                'sessions_per_cluster': sessions_per_cluster,
                'estimated_quality': quality_score
            },
            alternative_approaches=["Hierarchical clustering", "DBSCAN", "Different cluster counts"]
        )
    
    def explain_processing_quality(self, success_rate: float, memory_efficiency: float, 
                                 processing_speed: float, estimated_quality: float) -> AnalysisExplanation:
        """Provide detailed explanation of processing quality metrics."""
        
        # Determine overall assessment
        if estimated_quality > 0.8:
            confidence = "high"
            what_it_means = f"🎯 **Excellent Processing Quality**: Your data processed smoothly with {success_rate:.1%} success rate. The system efficiently handled your dataset."
            quality_assessment = "excellent"
        elif estimated_quality > 0.6:
            confidence = "medium"
            what_it_means = f"✅ **Good Processing Quality**: Solid processing with {success_rate:.1%} success rate. Some room for optimization."
            quality_assessment = "good"
        else:
            confidence = "low"
            what_it_means = f"⚠️ **Processing Issues Detected**: {success_rate:.1%} success rate indicates some problems during processing."
            quality_assessment = "needs_attention"
        
        # Detailed reasoning
        reasoning_parts = []
        
        if success_rate < 0.9:
            reasoning_parts.append(f"**Success Rate ({success_rate:.1%})**: Some sessions failed to process, likely due to data quality issues, encoding problems, or insufficient text content.")
        else:
            reasoning_parts.append(f"**Success Rate ({success_rate:.1%})**: Excellent - almost all sessions processed successfully.")
        
        if memory_efficiency < 1.0:
            reasoning_parts.append(f"**Memory Efficiency ({memory_efficiency:.1f} sessions/MB)**: Lower efficiency suggests either very large embeddings or memory management issues.")
        elif memory_efficiency > 3.0:
            reasoning_parts.append(f"**Memory Efficiency ({memory_efficiency:.1f} sessions/MB)**: Excellent memory usage - compact, efficient processing.")
        else:
            reasoning_parts.append(f"**Memory Efficiency ({memory_efficiency:.1f} sessions/MB)**: Good memory usage within normal range.")
        
        if processing_speed < 0.5:
            reasoning_parts.append(f"**Processing Speed ({processing_speed:.1f}/s)**: Slower processing likely due to large text size, complex embeddings, or system resource constraints.")
        elif processing_speed > 2.0:
            reasoning_parts.append(f"**Processing Speed ({processing_speed:.1f}/s)**: Fast processing indicates good system performance and optimal data size.")
        else:
            reasoning_parts.append(f"**Processing Speed ({processing_speed:.1f}/s)**: Normal processing speed for semantic embeddings.")
        
        why_these_results = " ".join(reasoning_parts)
        
        # Generate recommendations
        recommendations = []
        
        if success_rate < 0.9:
            recommendations.append("Clean your data: remove empty rows, fix encoding issues, ensure sufficient text per session")
        
        if memory_efficiency < 1.5:
            recommendations.append("For large datasets, try intelligent sampling to reduce memory usage")
        
        if processing_speed < 1.0:
            recommendations.append("Close other applications to free up system resources")
            recommendations.append("Consider processing smaller batches or using sampling for initial exploration")
        
        if estimated_quality > 0.8:
            recommendations.append("Your processing quality is excellent - proceed with confidence to analysis")
        else:
            recommendations.append("Consider re-processing with data cleaning or different settings")
        
        return AnalysisExplanation(
            analysis_type="Processing Quality Assessment",
            quality_score=estimated_quality,
            confidence_level=confidence,
            what_it_means=what_it_means,
            why_these_results=why_these_results,
            what_to_do_next=recommendations,
            technical_details={
                'success_rate': success_rate,
                'memory_efficiency': memory_efficiency,
                'processing_speed': processing_speed,
                'overall_quality': estimated_quality,
                'quality_assessment': quality_assessment
            },
            alternative_approaches=["Intelligent sampling", "Data preprocessing", "Batch size optimization"]
        )
    
    def explain_complexity_score(self, complexity_score: float, processing_strategy: str) -> AnalysisExplanation:
        """Provide detailed explanation of dataset complexity and chosen strategy."""
        
        # Assess complexity level
        if complexity_score > 0.7:
            complexity_level = "high"
            confidence = "high"
            what_it_means = f"🔴 **High Complexity Dataset** (Score: {complexity_score:.2f}/1.0): Large, diverse dataset with significant computational requirements."
        elif complexity_score > 0.4:
            complexity_level = "medium"
            confidence = "medium"
            what_it_means = f"🟡 **Medium Complexity Dataset** (Score: {complexity_score:.2f}/1.0): Moderate size and diversity, manageable with standard processing."
        else:
            complexity_level = "low"
            confidence = "high"
            what_it_means = f"🟢 **Low Complexity Dataset** (Score: {complexity_score:.2f}/1.0): Small, focused dataset ideal for comprehensive analysis."
        
        # Explain strategy choice
        strategy_explanations = {
            "progressive_sampling": "Applied intelligent sampling to handle large dataset while preserving semantic diversity. This reduces computational load while maintaining analysis quality.",
            "smart_batching": "Used adaptive batching to optimize memory usage and processing speed while handling the full dataset.",
            "full_processing": "Processed the complete dataset directly since it's within optimal size limits for comprehensive analysis."
        }
        
        strategy_explanation = strategy_explanations.get(processing_strategy, "Standard processing applied.")
        
        why_these_results = f"**Complexity Factors**: Dataset size, text diversity, embedding dimensionality, and system resources all contribute to complexity. **Strategy Choice**: {strategy_explanation}"
        
        # Generate recommendations based on complexity and strategy
        recommendations = []
        
        if complexity_score > 0.7:
            recommendations.append("For future uploads, consider pre-sampling large datasets")
            recommendations.append("Focus on specific time periods or topics for deeper analysis")
            recommendations.append("Use progressive analysis: start with samples, then expand")
        elif complexity_score > 0.4:
            recommendations.append("Current dataset is well-sized for comprehensive analysis")
            recommendations.append("Try multiple analysis methods to compare results")
        else:
            recommendations.append("Perfect size for trying all analysis features")
            recommendations.append("Consider expanding dataset for more robust patterns")
        
        if processing_strategy == "progressive_sampling":
            recommendations.append("The sampling preserved key patterns - results are representative")
            recommendations.append("For full dataset analysis, consider cloud processing or larger system")
        
        return AnalysisExplanation(
            analysis_type="Dataset Complexity Analysis",
            quality_score=1.0 - complexity_score,  # Lower complexity = higher quality for processing
            confidence_level=confidence,
            what_it_means=what_it_means,
            why_these_results=why_these_results,
            what_to_do_next=recommendations,
            technical_details={
                'complexity_score': complexity_score,
                'complexity_level': complexity_level,
                'processing_strategy': processing_strategy,
                'strategy_explanation': strategy_explanation
            },
            alternative_approaches=["Manual sampling", "Time-based chunking", "Topic-based filtering"]
        )

class AutoParameterTuner:
    """Automatically adjust analysis parameters based on data characteristics and results."""
    
    def __init__(self):
        self.tuning_history = []
    
    def suggest_pca_adjustments(self, pca_results: Dict, dataset_info: Dict) -> List[AutoAdjustment]:
        """Suggest PCA parameter adjustments based on results."""
        
        adjustments = []
        explained_variance = pca_results.get('cumulative_variance', [0])[-1]
        n_components = len(pca_results.get('explained_variance_ratio', []))
        n_samples = dataset_info.get('n_samples', 0)
        
        # Adjust number of components
        if explained_variance < 0.5 and n_components < min(n_samples, 10):
            new_components = min(n_samples - 1, n_components * 2, 10)
            adjustments.append(AutoAdjustment(
                parameter_name="n_components",
                current_value=n_components,
                suggested_value=new_components,
                reason=f"Low explained variance ({explained_variance:.1%}) suggests more components needed",
                expected_improvement="Capture more data variance, better representation",
                confidence=0.8
            ))
        
        # Suggest preprocessing changes
        if explained_variance < 0.3:
            adjustments.append(AutoAdjustment(
                parameter_name="preprocessing_method",
                current_value="StandardScaler",
                suggested_value="RobustScaler + feature_selection",
                reason="Very low explained variance may indicate noisy or inappropriate features",
                expected_improvement="Better feature quality, more stable results",
                confidence=0.6
            ))
        
        # Suggest alternative methods
        if explained_variance < 0.4:
            adjustments.append(AutoAdjustment(
                parameter_name="dimensionality_method",
                current_value="PCA",
                suggested_value="UMAP",
                reason="Linear PCA may not capture non-linear relationships in your data",
                expected_improvement="Better representation of complex semantic relationships",
                confidence=0.7
            ))
        
        return adjustments

def create_explanation_dashboard(explanations: List[AnalysisExplanation], 
                               adjustments: List[AutoAdjustment]) -> None:
    """Create an interactive explanation dashboard."""
    
    st.subheader("🔍 Analysis Explanation Dashboard")
    
    # Overall quality summary
    avg_quality = np.mean([exp.quality_score for exp in explanations])
    quality_color = "green" if avg_quality > 0.7 else "orange" if avg_quality > 0.4 else "red"
    
    st.markdown(f"""
    <div style='padding: 10px; border-left: 4px solid {quality_color}; background-color: rgba(0,0,0,0.1);'>
    <h4>Overall Analysis Quality: {avg_quality:.1%}</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Individual analysis explanations
    for i, explanation in enumerate(explanations):
        with st.expander(f"🔍 {explanation.analysis_type} - {explanation.confidence_level.title()} Confidence", 
                        expanded=i == 0):
            
            # Main explanation
            st.markdown(explanation.what_it_means)
            st.markdown(f"**Why these results:** {explanation.why_these_results}")
            
            # Action items
            if explanation.what_to_do_next:
                st.markdown("**🎯 What to do next:**")
                for action in explanation.what_to_do_next:
                    st.markdown(f"- {action}")
            
            # Technical details in collapsible section
            with st.expander("📊 Technical Details", expanded=False):
                st.json(explanation.technical_details)
            
            # Alternative approaches
            if explanation.alternative_approaches:
                st.info(f"**🔄 Alternative methods to try:** {', '.join(explanation.alternative_approaches)}")
    
    # Auto-adjustment recommendations
    if adjustments:
        st.subheader("🔧 Auto-Adjustment Recommendations")
        
        for adj in adjustments:
            with st.expander(f"⚙️ {adj.parameter_name} (Confidence: {adj.confidence:.1%})", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Current Value", str(adj.current_value))
                    st.metric("Suggested Value", str(adj.suggested_value))
                
                with col2:
                    st.markdown(f"**Reason:** {adj.reason}")
                    st.markdown(f"**Expected Improvement:** {adj.expected_improvement}")
                
                if st.button(f"Apply {adj.parameter_name} adjustment", key=f"apply_{adj.parameter_name}"):
                    st.success(f"✅ Applied {adj.parameter_name} adjustment!")
    
    # Interactive quality visualization
    st.subheader("📈 Quality Metrics Visualization")
    
    quality_data = pd.DataFrame([
        {
            'Analysis': exp.analysis_type,
            'Quality Score': exp.quality_score,
            'Confidence': exp.confidence_level
        }
        for exp in explanations
    ])
    
    fig = px.bar(quality_data, x='Analysis', y='Quality Score', 
                 color='Confidence', 
                 title="Analysis Quality by Method",
                 color_discrete_map={'high': 'green', 'medium': 'orange', 'low': 'red'})
    
    st.plotly_chart(fig, use_container_width=True)

# Integration function
def integrate_explainability_with_existing_app():
    """Integration points for the existing app."""
    
    # This would be integrated at key points in the main app:
    # 1. After PCA analysis
    # 2. After clustering analysis  
    # 3. After concept drift analysis
    # 4. Before major parameter changes
    
    pass 
