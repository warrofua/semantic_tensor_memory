#!/usr/bin/env python3
"""
Performance Optimization Module for Semantic Tensor Memory

Intelligent data management and adaptive processing for large datasets.
"""

import streamlit as st
import numpy as np
import torch
import psutil
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
import gc

@dataclass
class DatasetProfile:
    """Profile of dataset characteristics for adaptive processing."""
    total_sessions: int
    avg_tokens_per_session: int
    total_tokens: int
    estimated_memory_mb: float
    complexity_score: float  # 0-1, higher = more complex
    recommended_batch_size: int
    recommended_method: str
    processing_strategy: str

@dataclass
class PerformanceMetrics:
    """Track performance during processing."""
    memory_usage_mb: float
    processing_time_sec: float
    quality_score: float  # 0-1, higher = better results
    warnings: List[str]
    recommendations: List[str]

class AdaptiveDataProcessor:
    """Intelligent data processor that adapts to dataset size and system capabilities."""
    
    def __init__(self):
        self.system_memory_gb = psutil.virtual_memory().total / (1024**3)
        self.available_memory_gb = psutil.virtual_memory().available / (1024**3)
        self.cpu_count = psutil.cpu_count()
        
    def profile_dataset(self, session_data: List[Dict]) -> DatasetProfile:
        """Analyze dataset characteristics to determine optimal processing strategy."""
        
        total_sessions = len(session_data)
        
        # Estimate token counts and memory usage
        sample_size = min(50, total_sessions)
        sample_tokens = []
        
        for i in range(0, total_sessions, max(1, total_sessions // sample_size)):
            text = session_data[i].get('text', '')
            # Rough token estimate: 4 chars per token
            estimated_tokens = len(text) // 4
            sample_tokens.append(estimated_tokens)
        
        avg_tokens = np.mean(sample_tokens) if sample_tokens else 100
        total_tokens = int(avg_tokens * total_sessions)
        
        # Estimate memory usage (rough: 4 bytes per float * 384 dims * tokens)
        estimated_memory_mb = (total_tokens * 384 * 4) / (1024**2)
        
        # Calculate complexity score
        complexity_factors = []
        complexity_factors.append(min(1.0, total_sessions / 2000))  # Session count factor
        complexity_factors.append(min(1.0, avg_tokens / 500))       # Token density factor
        complexity_factors.append(min(1.0, estimated_memory_mb / (self.available_memory_gb * 1024 * 0.3)))  # Memory pressure
        
        complexity_score = np.mean(complexity_factors)
        
        # Determine processing strategy
        if complexity_score < 0.3:
            strategy = "full_processing"
            batch_size = total_sessions
            method = "standard_pca"
        elif complexity_score < 0.6:
            strategy = "smart_batching"
            batch_size = min(500, max(50, int(total_sessions / 4)))
            method = "incremental_pca"
        else:
            strategy = "progressive_sampling"
            batch_size = min(200, max(25, int(total_sessions / 8)))
            method = "mini_batch_clustering"
        
        return DatasetProfile(
            total_sessions=total_sessions,
            avg_tokens_per_session=int(avg_tokens),
            total_tokens=total_tokens,
            estimated_memory_mb=estimated_memory_mb,
            complexity_score=complexity_score,
            recommended_batch_size=batch_size,
            recommended_method=method,
            processing_strategy=strategy
        )
    
    def apply_intelligent_sampling(self, session_data: List[Dict], target_size: int) -> Tuple[List[Dict], List[int]]:
        """Apply intelligent sampling to reduce dataset size while preserving semantic diversity."""
        
        if len(session_data) <= target_size:
            return session_data, list(range(len(session_data)))
        
        # Strategy 1: Temporal stratification (ensure coverage across time)
        temporal_samples = []
        total_sessions = len(session_data)
        
        # Take samples from different time periods
        for i in range(target_size // 2):
            idx = int((i / (target_size // 2)) * total_sessions)
            temporal_samples.append(idx)
        
        # Strategy 2: Content diversity sampling (simple text length diversity)
        text_lengths = [len(session.get('text', '')) for session in session_data]
        length_percentiles = np.percentile(text_lengths, [10, 25, 50, 75, 90])
        
        diversity_samples = []
        for percentile in length_percentiles:
            # Find closest session to this percentile
            closest_idx = np.argmin(np.abs(np.array(text_lengths) - percentile))
            diversity_samples.append(closest_idx)
        
        # Strategy 3: Random sampling for remaining slots
        remaining_slots = target_size - len(set(temporal_samples + diversity_samples))
        if remaining_slots > 0:
            excluded_indices = set(temporal_samples + diversity_samples)
            available_indices = [i for i in range(total_sessions) if i not in excluded_indices]
            random_samples = np.random.choice(available_indices, 
                                            size=min(remaining_slots, len(available_indices)), 
                                            replace=False).tolist()
        else:
            random_samples = []
        
        # Combine all sampling strategies
        selected_indices = sorted(set(temporal_samples + diversity_samples + random_samples))
        selected_data = [session_data[i] for i in selected_indices]
        
        return selected_data, selected_indices

class QualityAssessment:
    """Assess the quality of analysis results and provide auto-adjustment recommendations."""
    
    @staticmethod
    def assess_pca_quality(results: Dict[str, Any]) -> PerformanceMetrics:
        """Assess PCA quality and provide recommendations."""
        warnings = []
        recommendations = []
        
        explained_variance = results.get('cumulative_variance', [0])[-1]
        condition_number = results.get('condition_number', 1)
        n_samples = results.get('n_samples', 0)
        n_features = results.get('n_features', 0)
        
        # Quality assessment
        quality_factors = []
        
        # Factor 1: Explained variance
        if explained_variance > 0.7:
            quality_factors.append(1.0)
        elif explained_variance > 0.5:
            quality_factors.append(0.7)
            recommendations.append("Consider using more PCA components or alternative methods like UMAP")
        elif explained_variance > 0.3:
            quality_factors.append(0.4)
            warnings.append("Low explained variance - results may not be reliable")
            recommendations.append("Switch to UMAP or autoencoder-based dimensionality reduction")
        else:
            quality_factors.append(0.1)
            warnings.append("Very low explained variance - PCA may not be suitable for this data")
            recommendations.append("Use non-linear methods like t-SNE or UMAP instead")
        
        # Factor 2: Condition number (multicollinearity)
        if condition_number < 1e6:
            quality_factors.append(1.0)
        elif condition_number < 1e9:
            quality_factors.append(0.6)
            warnings.append("Moderate multicollinearity detected")
        else:
            quality_factors.append(0.2)
            warnings.append("High multicollinearity - results may be unstable")
            recommendations.append("Apply feature selection or regularization")
        
        # Factor 3: Sample size adequacy
        samples_per_feature = n_samples / max(1, n_features)
        if samples_per_feature > 10:
            quality_factors.append(1.0)
        elif samples_per_feature > 5:
            quality_factors.append(0.7)
        elif samples_per_feature > 2:
            quality_factors.append(0.4)
            warnings.append("Low sample-to-feature ratio")
        else:
            quality_factors.append(0.1)
            warnings.append("Very low sample-to-feature ratio - results unreliable")
            recommendations.append("Collect more data or reduce dimensionality")
        
        quality_score = np.mean(quality_factors)
        
        return PerformanceMetrics(
            memory_usage_mb=psutil.Process().memory_info().rss / 1024 / 1024,
            processing_time_sec=0,  # Would be set by caller
            quality_score=quality_score,
            warnings=warnings,
            recommendations=recommendations
        )
    
    @staticmethod
    def assess_clustering_quality(cluster_labels: np.ndarray, embeddings: np.ndarray) -> PerformanceMetrics:
        """Assess clustering quality using multiple metrics."""
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
        
        warnings = []
        recommendations = []
        quality_factors = []
        
        try:
            # Silhouette score
            silhouette = silhouette_score(embeddings, cluster_labels)
            if silhouette > 0.5:
                quality_factors.append(1.0)
            elif silhouette > 0.3:
                quality_factors.append(0.7)
            elif silhouette > 0.1:
                quality_factors.append(0.4)
                warnings.append("Moderate clustering quality")
                recommendations.append("Try different number of clusters or clustering algorithm")
            else:
                quality_factors.append(0.1)
                warnings.append("Poor clustering quality")
                recommendations.append("Data may not have natural clusters - consider different approach")
            
            # Calinski-Harabasz score (higher is better)
            ch_score = calinski_harabasz_score(embeddings, cluster_labels)
            # Normalize by typical ranges (this is dataset-dependent)
            ch_normalized = min(1.0, ch_score / 100)
            quality_factors.append(ch_normalized)
            
            # Check cluster balance
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            balance_ratio = np.min(counts) / np.max(counts)
            if balance_ratio > 0.3:
                quality_factors.append(1.0)
            elif balance_ratio > 0.1:
                quality_factors.append(0.6)
            else:
                quality_factors.append(0.2)
                warnings.append("Imbalanced clusters detected")
                recommendations.append("Adjust number of clusters or use different algorithm")
            
        except Exception as e:
            quality_factors.append(0.1)
            warnings.append(f"Could not assess clustering quality: {e}")
        
        quality_score = np.mean(quality_factors) if quality_factors else 0.1
        
        return PerformanceMetrics(
            memory_usage_mb=psutil.Process().memory_info().rss / 1024 / 1024,
            processing_time_sec=0,
            quality_score=quality_score,
            warnings=warnings,
            recommendations=recommendations
        )

class ProgressiveAnalyzer:
    """Progressive analysis that builds up complexity based on data quality."""
    
    def __init__(self):
        self.processor = AdaptiveDataProcessor()
        self.quality_assessor = QualityAssessment()
    
    def run_progressive_analysis(self, session_data: List[Dict]) -> Dict[str, Any]:
        """Run analysis with progressive complexity based on results quality."""
        
        # Step 1: Profile the dataset
        profile = self.processor.profile_dataset(session_data)
        
        st.info(f"📊 **Dataset Profile**: {profile.total_sessions} sessions, "
                f"~{profile.avg_tokens_per_session} tokens/session, "
                f"Complexity: {profile.complexity_score:.2f}")
        
        # Step 2: Apply intelligent sampling if needed
        if profile.processing_strategy == "progressive_sampling":
            processed_data, selected_indices = self.processor.apply_intelligent_sampling(
                session_data, profile.recommended_batch_size
            )
            st.warning(f"🎯 **Smart Sampling**: Using {len(processed_data)}/{profile.total_sessions} sessions "
                      f"for optimal performance (temporal + diversity sampling)")
        else:
            processed_data = session_data
            selected_indices = list(range(len(session_data)))
        
        # Step 3: Run basic analysis and assess quality
        results = {'profile': profile, 'selected_indices': selected_indices}
        
        # Add progressive complexity based on quality
        analysis_chain = [
            ('basic_stats', self._analyze_basic_stats),
            ('embeddings', self._analyze_embeddings),
            ('dimensionality_reduction', self._analyze_dimensionality),
            ('clustering', self._analyze_clustering),
            ('temporal_patterns', self._analyze_temporal_patterns),
        ]
        
        overall_quality = 1.0
        
        for step_name, step_function in analysis_chain:
            if overall_quality < 0.3:
                st.warning(f"⚠️ Skipping {step_name} due to poor data quality ({overall_quality:.2f})")
                break
            
            step_results = step_function(processed_data, results)
            results[step_name] = step_results
            
            # Update overall quality based on step results
            step_quality = step_results.get('quality_metrics', {}).get('quality_score', 0.5)
            overall_quality = (overall_quality + step_quality) / 2
        
        results['overall_quality'] = overall_quality
        return results
    
    def _analyze_basic_stats(self, data: List[Dict], context: Dict) -> Dict[str, Any]:
        """Basic statistical analysis."""
        text_lengths = [len(session.get('text', '')) for session in data]
        
        return {
            'text_length_stats': {
                'mean': np.mean(text_lengths),
                'median': np.median(text_lengths),
                'std': np.std(text_lengths),
                'range': [np.min(text_lengths), np.max(text_lengths)]
            },
            'quality_metrics': PerformanceMetrics(
                memory_usage_mb=0,
                processing_time_sec=0,
                quality_score=1.0,  # Basic stats always succeed
                warnings=[],
                recommendations=[]
            )
        }
    
    def _analyze_embeddings(self, data: List[Dict], context: Dict) -> Dict[str, Any]:
        """Embedding analysis with quality assessment."""
        # This would integrate with existing embedding pipeline
        # Placeholder for actual implementation
        return {
            'embedding_quality': 'high',
            'quality_metrics': PerformanceMetrics(
                memory_usage_mb=0,
                processing_time_sec=0,
                quality_score=0.8,
                warnings=[],
                recommendations=[]
            )
        }
    
    def _analyze_dimensionality(self, data: List[Dict], context: Dict) -> Dict[str, Any]:
        """Dimensionality reduction with adaptive method selection."""
        # This would integrate with existing PCA pipeline
        # Placeholder for actual implementation
        return {
            'method_used': 'adaptive_pca',
            'quality_metrics': PerformanceMetrics(
                memory_usage_mb=0,
                processing_time_sec=0,
                quality_score=0.7,
                warnings=[],
                recommendations=[]
            )
        }
    
    def _analyze_clustering(self, data: List[Dict], context: Dict) -> Dict[str, Any]:
        """Clustering with quality assessment."""
        # This would integrate with existing clustering pipeline
        # Placeholder for actual implementation
        return {
            'clusters_found': 5,
            'quality_metrics': PerformanceMetrics(
                memory_usage_mb=0,
                processing_time_sec=0,
                quality_score=0.6,
                warnings=[],
                recommendations=[]
            )
        }
    
    def _analyze_temporal_patterns(self, data: List[Dict], context: Dict) -> Dict[str, Any]:
        """Temporal pattern analysis."""
        # This would integrate with existing drift analysis
        # Placeholder for actual implementation
        return {
            'drift_patterns': [],
            'quality_metrics': PerformanceMetrics(
                memory_usage_mb=0,
                processing_time_sec=0,
                quality_score=0.8,
                warnings=[],
                recommendations=[]
            )
        }

def create_performance_dashboard(results: Dict[str, Any]) -> None:
    """Create a performance and quality dashboard."""
    
    st.subheader("🚀 Performance & Quality Dashboard")
    
    profile = results.get('profile')
    overall_quality = results.get('overall_quality', 0.5)
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall Quality", f"{overall_quality:.1%}")
    with col2:
        st.metric("Complexity Score", f"{profile.complexity_score:.2f}")
    with col3:
        st.metric("Processing Strategy", profile.processing_strategy.replace('_', ' ').title())
    with col4:
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
        st.metric("Memory Usage", f"{memory_usage:.0f}MB")
    
    # Quality assessment per analysis step
    st.subheader("📊 Analysis Quality Breakdown")
    
    analysis_steps = ['basic_stats', 'embeddings', 'dimensionality_reduction', 'clustering', 'temporal_patterns']
    
    for step in analysis_steps:
        step_results = results.get(step, {})
        quality_metrics = step_results.get('quality_metrics')
        
        if quality_metrics:
            with st.expander(f"🔍 {step.replace('_', ' ').title()}", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Quality Score", f"{quality_metrics.quality_score:.1%}")
                    st.metric("Memory Impact", f"{quality_metrics.memory_usage_mb:.0f}MB")
                
                with col2:
                    if quality_metrics.warnings:
                        st.warning("⚠️ **Warnings:**")
                        for warning in quality_metrics.warnings:
                            st.write(f"• {warning}")
                    
                    if quality_metrics.recommendations:
                        st.info("💡 **Recommendations:**")
                        for rec in quality_metrics.recommendations:
                            st.write(f"• {rec}")

# Integration functions for existing app
def enhance_existing_analysis_with_performance_optimization():
    """Enhance existing analysis functions with performance optimization."""
    
    # This would be integrated into the main app
    # Key integration points:
    # 1. Replace direct PCA calls with adaptive processing
    # 2. Add quality assessment to all analysis steps
    # 3. Implement progressive complexity
    # 4. Add performance monitoring dashboard
    
    pass 