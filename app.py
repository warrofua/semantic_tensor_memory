"""
Semantic Tensor Analysis Application

A comprehensive Streamlit application for analyzing semantic evolution across multiple modalities.
"""

import streamlit as st
import streamlit.components.v1 as components
import time
import pandas as pd
import csv
import io
import os
import warnings
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple, Union
try:
    import torch
except ModuleNotFoundError as _torch_exc:  # pragma: no cover - runtime safeguard
    torch = None  # type: ignore[assignment]
    _missing_torch_error = _torch_exc
else:
    _missing_torch_error = None
import gc
import psutil
from pathlib import Path
import sys
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from semantic_tensor_analysis.storage.manager import StorageManager
from semantic_tensor_analysis.memory.universal_core import UniversalEmbedding

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))  # Ensure local package is importable when running without installation
DEMO_DATA_PATH = BASE_DIR / "data" / "ultimate_demo_dataset.csv"
COMMON_APP_DIR = BASE_DIR / "src" / "semantic_tensor_analysis" / "app"
if COMMON_APP_DIR.exists() and str(COMMON_APP_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_APP_DIR))
ASSETS_DIR = COMMON_APP_DIR / "assets"
LOGO_PATH = ASSETS_DIR / "semantic_tensor_art_logo.png"

# Fix PyTorch/Streamlit compatibility issues
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Set environment variables to prevent PyTorch/Streamlit conflicts (if torch is present)
os.environ['TORCH_USE_CUDA_DSA'] = '0'
os.environ['TORCH_DISABLE_WARN'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Fix for Universal STM

if torch is not None:
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        if hasattr(torch, '_C') and hasattr(torch._C, '_disable_jit_profiling'):
            torch._C._disable_jit_profiling()
    except Exception:
        pass

# Cache recent Plotly figures for vision analysis
_orig_plotly_chart = st.plotly_chart
def _plotly_chart_with_cache(fig, *args, **kwargs):
    try:
        figs = st.session_state.get("recent_plotly_figs", [])
        figs.append(fig)
        st.session_state["recent_plotly_figs"] = figs[-5:]  # keep last 5
    except Exception:
        pass
    return _orig_plotly_chart(fig, *args, **kwargs)
st.plotly_chart = _plotly_chart_with_cache

# Import Semantic Tensor Analysis modules
from semantic_tensor_analysis import (
    UniversalMemoryStore,
    Modality,
    create_universal_embedder,
    embed_text,
    create_text_embedding,
    embed_sentence,
)
from semantic_tensor_analysis.memory import get_text_embedder
from semantic_tensor_analysis.memory.store import save, append  # Keep legacy store for compatibility
from semantic_tensor_analysis.memory.drift import drift_series  # Keep legacy drift analysis

# Import visualization modules
from semantic_tensor_analysis.streamlit import (
    initialize_session_state,
    robust_pca_pipeline,
    plot_drift_plotly,
    plot_heatmap_plotly,
    create_pca_visualization,
    create_animated_pca_trajectory,
    create_temporal_heatmap,
    create_variance_evolution_animation,
    plot_ridgeline_plotly,
    plot_enhanced_ridgeline_altair,
    plot_ridgeline_altair,
    create_pca_timeline_animation,
    create_4d_semantic_space_visualization,
    create_liminal_tunnel_visualization,
)
from semantic_tensor_analysis.visualization import (
    render_semantic_drift_river_analysis,
    render_holistic_semantic_analysis,
    token_alignment_heatmap,
    generate_narrative_summary,
)
from semantic_tensor_analysis.visualization.tools.concept_visualizer import (
    visualize_concept_evolution,
)
from semantic_tensor_analysis.chat.analysis import render_comprehensive_chat_analysis
# Chat history analysis (unified with main processing)
from semantic_tensor_analysis.chat.history_analyzer import ChatHistoryParser

from semantic_tensor_analysis.analytics.trajectory import (
    calculate_semantic_trajectory_data,
    create_3d_trajectory_plot,
    display_trajectory_analysis_table,
)
from semantic_tensor_analysis.analytics.dimensionality import (
    compare_dimensionality_methods,
    create_alternative_visualization,
)

# Add performance optimizer import at the top
from semantic_tensor_analysis.optimization import (
    AdaptiveDataProcessor,
    ProgressiveAnalyzer,
    create_performance_dashboard,
    DatasetProfile,
    PerformanceMetrics,
)
from semantic_tensor_analysis.explainability import (
    ExplainabilityEngine,
    create_explanation_dashboard,
)

# Set page config
st.set_page_config(
    page_title="Semantic Tensor Analysis",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Memory management utilities
def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    except:
        return 0.0

def cleanup_memory():
    """Force memory cleanup and garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Sidebar control utilities
def _collapse_sidebar_via_js():
    """Attempt to collapse the Streamlit sidebar using a tiny JS hack."""
    components.html(
        """
        <script>
        (function() {
          const doc = window.parent.document;
          const btn = doc.querySelector('[data-testid="stSidebarCollapseButton"]');
          const sidebar = doc.querySelector('section[data-testid="stSidebar"]');
          if (btn && sidebar && sidebar.offsetWidth > 0) {
            btn.click();
          }
        })();
        </script>
        """,
        height=0,
        width=0,
    )

def collapse_sidebar_once_after_load():
    """Collapse the sidebar exactly once after data has been loaded."""
    if not st.session_state.get('sidebar_minimized_after_load', False):
        _collapse_sidebar_via_js()
        st.session_state['sidebar_minimized_after_load'] = True

# CRITICAL PERFORMANCE FIX: Model caching
@st.cache_resource
def get_cached_text_embedder():
    """Get cached TextEmbedder to prevent reloading 1GB models for every CSV import."""
    st.info("🔄 Loading embedding models (this happens once per session)...")
    embedder = get_text_embedder()
    st.success("✅ Models loaded and cached!")
    return embedder

@st.cache_resource  
def get_cached_universal_store():
    """Get cached UniversalMemoryStore."""
    return UniversalMemoryStore()

# Initialize session state with Semantic Tensor Analysis support and memory management
def initialize_universal_session_state():
    """Initialize session state with Semantic Tensor Analysis support and memory management."""
    initialize_session_state()  # Call original initialization
    
    # Memory monitoring
    if 'initial_memory' not in st.session_state:
        st.session_state.initial_memory = get_memory_usage()
    
    # Add Universal STM components with size limits (use cached version)
    if 'universal_store' not in st.session_state:
        st.session_state.universal_store = get_cached_universal_store()
    
    if 'active_modalities' not in st.session_state:
        st.session_state.active_modalities = set()
    
    if 'modality_sessions' not in st.session_state:
        st.session_state.modality_sessions = {modality: [] for modality in Modality}
    
    # Periodic memory cleanup
    current_memory = get_memory_usage()
    if current_memory > st.session_state.initial_memory + 2000:  # 2GB threshold
        st.warning(f"🧠 Memory usage high ({current_memory:.0f}MB). Running cleanup...")
        cleanup_memory()

initialize_universal_session_state()


def detect_file_type_and_content(uploaded_file):
    """
    Intelligently detect file type and content format.
    
    Returns:
        tuple: (file_type, content_type) where:
        - file_type: 'csv', 'json', 'txt' 
        - content_type: 'ai_conversation', 'csv_sessions', 'unknown'
    """
    file_name = uploaded_file.name.lower()
    
    # Read content to analyze
    content = uploaded_file.read()
    uploaded_file.seek(0)  # Reset file pointer
    
    try:
        content_str = content.decode('utf-8')
    except:
        return 'unknown', 'unknown'
    
    # File extension detection
    if file_name.endswith('.csv'):
        # Check if it's a CSV with text column (traditional) or conversation-like
        try:
            import csv
            import io
            csv_reader = csv.DictReader(io.StringIO(content_str))
            first_row = next(csv_reader, {})
            
            if 'text' in first_row:
                return 'csv', 'csv_sessions'
            else:
                return 'csv', 'unknown'
        except:
            return 'csv', 'unknown'
    
    elif file_name.endswith('.json'):
        # Check if it looks like AI conversation JSON
        try:
            import json
            data = json.loads(content_str)
            
            # ChatGPT format detection
            if isinstance(data, list) or ('mapping' in data if isinstance(data, dict) else False):
                return 'json', 'ai_conversation'
            else:
                return 'json', 'unknown'
        except:
            return 'json', 'unknown'
    
    elif file_name.endswith('.txt'):
        # Analyze text content for conversation patterns
        lines = content_str.split('\n')
        conversation_indicators = 0
        
        # Look for conversation patterns
        for line in lines[:20]:  # Check first 20 lines
            line_lower = line.strip().lower()
            if any(pattern in line_lower for pattern in [
                'you:', 'user:', 'human:', 'assistant:', 'ai:', 'chatgpt:', 'claude:',
                '**you**:', '**assistant**:', '> '
            ]):
                conversation_indicators += 1
        
        if conversation_indicators >= 2:
            return 'txt', 'ai_conversation'
        else:
            return 'txt', 'unknown'
    
    return 'unknown', 'unknown'


def convert_ai_conversation_to_sessions(messages):
    """
    Convert AI conversation messages to multi-resolution temporal format.
    
    NEW: Now supports both legacy format and multi-resolution analysis.
    
    Args:
        messages: List of ChatMessage objects
        
    Returns:
        List of dict objects compatible with CSV processing (legacy mode)
        OR multi-resolution data if enabled
    """
    from temporal_resolution_manager import TemporalResolutionManager, TemporalResolution
    
    # Check if multi-resolution is enabled in session state
    use_multi_resolution = st.session_state.get('enable_multi_resolution', True)
    
    if use_multi_resolution:
        # NEW MULTI-RESOLUTION PROCESSING
        temporal_manager = TemporalResolutionManager()
        processing_results = temporal_manager.process_conversation_messages(messages)
        
        # Store temporal manager in session state for UI access
        st.session_state.temporal_manager = temporal_manager
        st.session_state.temporal_processing_results = processing_results
        
        # Get current resolution data (start with turn-level for maximum granularity)
        resolution_data = temporal_manager.zoom_to_resolution(TemporalResolution.TURN)
        
        # Convert to legacy session format for compatibility
        sessions = []
        for i, meta in enumerate(resolution_data['metadata']):
            session_data = {
                'text': meta.get('user_text', ''),
                'session_id': meta.get('turn_id', f'turn_{i}'),
                'timestamp': meta.get('timestamp', '').isoformat() if meta.get('timestamp') else None,
                'conversation_id': meta.get('conversation_id', 'unknown'),
                'message_id': meta.get('turn_id', f'turn_{i}'),
                'source_type': 'ai_conversation_multi_res',
                'role': 'user',
                'resolution': 'turn',
                'turn_index': meta.get('turn_index', i),
                'has_assistant_response': meta.get('has_assistant_message', False)
            }
            sessions.append(session_data)
        
        # Add processing summary to session state
        st.session_state.multi_res_summary = {
            'total_turns': processing_results['turns_created'],
            'total_conversations': processing_results['conversations_created'],
            'total_days': processing_results['days_covered'],
            'temporal_span_days': processing_results['temporal_span_days'],
            'current_resolution': 'turn'
        }
        
        return sessions
    
    else:
        # LEGACY PROCESSING (original behavior)
        sessions = []
        
        # Filter to user messages only (focus on user's evolution)
        user_messages = [msg for msg in messages if msg.role == 'user']
        
        for i, msg in enumerate(user_messages):
            session_data = {
                'text': msg.content,
                'session_id': i,
                'timestamp': msg.timestamp.isoformat() if msg.timestamp else None,
                'conversation_id': msg.conversation_id or 'unknown',
                'message_id': msg.message_id or f'msg_{i}',
                'source_type': 'ai_conversation',
                'role': msg.role
            }
            sessions.append(session_data)
        
        return sessions


def handle_unified_upload(uploaded_file):
    """
    Unified handler that processes both AI conversations and CSV data intelligently.
    """
    try:
        # Detect what we're dealing with
        file_type, content_type = detect_file_type_and_content(uploaded_file)
        
        st.info(f"🔍 Detected: {file_type.upper()} file with {content_type.replace('_', ' ')}")
        
        if content_type == 'ai_conversation':
            # Process as AI conversation
            with st.spinner("🤖 Processing AI conversation data..."):
                file_content = uploaded_file.read().decode('utf-8')
                
                # Parse messages using existing chat parser
                from semantic_tensor_analysis.chat.history_analyzer import ChatHistoryParser
                messages = ChatHistoryParser.auto_detect_format(file_content)
                
                if not messages:
                    st.error("❌ No conversation messages found in the uploaded file")
                    return False
                
                # Convert to session format
                session_data = convert_ai_conversation_to_sessions(messages)
                
                if len(session_data) < 2:
                    st.error("❌ Need at least 2 user messages for analysis")
                    return False
                
                st.success(f"✅ Extracted {len(session_data)} user messages from conversation")
                
        elif content_type == 'csv_sessions':
            # Process as traditional CSV
            with st.spinner("📊 Processing CSV session data..."):
                csv_data = uploaded_file.read().decode('utf-8')
                import csv
                import io
                reader = csv.DictReader(io.StringIO(csv_data))
                session_data = list(reader)
                
                if not session_data:
                    st.error("❌ The CSV file appears to be empty")
                    return False
                    
                # Validate text column
                text_found = any('text' in row and row['text'].strip() for row in session_data)
                if not text_found:
                    st.error("❌ No 'text' column with content found in CSV")
                    return False
                    
                st.success(f"✅ Loaded {len(session_data)} sessions from CSV")
                
        else:
            st.error(f"❌ Unsupported file format. Please upload:")
            st.markdown("""
            - **CSV files** with a 'text' column
            - **JSON files** from ChatGPT exports  
            - **TXT files** with conversation format
            """)
            return False
        
        # Now process the unified session data
        return process_unified_sessions(session_data, uploaded_file.name, content_type)
        
    except Exception as e:
        st.error(f"❌ Failed to process {uploaded_file.name}: {str(e)}")
        return False


def process_unified_sessions(session_data, filename, content_type):
    """
    Process unified session data with adaptive performance optimization.
    """
    try:
        # ENHANCED: Use adaptive data processor
        adaptive_processor = AdaptiveDataProcessor()
        
        # Profile the dataset first
        profile = adaptive_processor.profile_dataset(session_data)
        
        # Show dataset profile to user
        st.info(f"""
        📊 **Dataset Analysis**:
        - **{profile.total_sessions:,} sessions** (~{profile.avg_tokens_per_session} tokens each)
        - **Complexity Score**: {profile.complexity_score:.2f}/1.0
        - **Strategy**: {profile.processing_strategy.replace('_', ' ').title()}
        - **Estimated Memory**: {profile.estimated_memory_mb:.0f}MB
        """)
        
        # Apply intelligent processing strategy
        if profile.processing_strategy == "progressive_sampling":
            st.warning(f"🎯 **Large Dataset Detected**: Using intelligent sampling for optimal performance")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(f"🔬 Smart Sample ({profile.recommended_batch_size} sessions)", type="primary"):
                    session_data, selected_indices = adaptive_processor.apply_intelligent_sampling(
                        session_data, profile.recommended_batch_size
                    )
                    st.success(f"✅ Applied intelligent sampling: {len(session_data)} sessions selected")
                else:
                    return False
            
            with col2:
                if st.button(f"📊 Process First {profile.recommended_batch_size}"):
                    session_data = session_data[:profile.recommended_batch_size]
                    selected_indices = list(range(profile.recommended_batch_size))
                    st.info(f"✅ Processing first {len(session_data)} sessions")
                else:
                    return False
            
            with col3:
                if st.button("🔄 Cancel & Try Smaller File"):
                    st.info("Consider splitting your data or using the sampling option")
                    return False
            
            return False
        
        elif profile.processing_strategy == "smart_batching":
            st.info(f"🔄 **Smart Batching**: Processing in optimized batches of {profile.recommended_batch_size}")
        
        else:
            st.success(f"✅ **Full Processing**: Dataset size is optimal for direct processing")
        
        # Initialize performance tracking
        start_time = time.time()
        start_memory = get_memory_usage()
        
        memory = []  # Legacy format for backward compatibility
        meta = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        performance_metrics = st.empty()
        
        # PERFORMANCE FIX: Use cached models
        text_embedder = get_cached_text_embedder()
        universal_store = get_cached_universal_store()
        
        # Clear any existing data to prevent accumulation
        universal_store.clear()
        
        valid_sessions = 0
        skipped_sessions = 0
        
        # Use adaptive batch size
        batch_size = min(profile.recommended_batch_size, 50)  # Cap at 50 for UI responsiveness
        
        # Enhanced memory monitoring
        memory_threshold = min(2000, int(adaptive_processor.available_memory_gb * 1024 * 0.5))  # Use 50% of available memory
        
        # Processing quality tracking
        processing_quality = {
            'successful_embeddings': 0,
            'failed_embeddings': 0,
            'avg_processing_time': 0,
            'memory_efficiency': 0
        }
        
        for batch_start in range(0, len(session_data), batch_size):
            batch_end = min(batch_start + batch_size, len(session_data))
            batch = session_data[batch_start:batch_end]
            
            batch_start_time = time.time()
            
            # ENHANCED MEMORY CHECK with adaptive thresholds
            current_memory = get_memory_usage()
            memory_usage_pct = (current_memory / (adaptive_processor.system_memory_gb * 1024)) * 100
            
            if current_memory > memory_threshold or memory_usage_pct > 70:
                st.warning(f"⚠️ High memory usage: {current_memory:.0f}MB ({memory_usage_pct:.1f}% of system)")
                cleanup_memory()
                
                # If still high, suggest reducing batch size
                if get_memory_usage() > memory_threshold * 0.9:
                    batch_size = max(5, batch_size // 2)
                    st.info(f"🔧 Auto-reducing batch size to {batch_size} for memory efficiency")
            
            # Process batch with error resilience
            batch_successes = 0
            for i, row in enumerate(batch):
                global_i = batch_start + i
                text = row.get('text', '').strip()
                
                if text:
                    try:
                        # PERFORMANCE FIX: Single embedding call
                        session_id = f"session_{valid_sessions}"
                        universal_embedding = text_embedder.process_raw_data(text, session_id=session_id)
                        
                        # Extract legacy format from universal embedding
                        legacy_emb = universal_embedding.event_embeddings
                        meta_row = dict(row)
                        meta_row['tokens'] = legacy_emb.shape[0]
                        meta_row['content_type'] = content_type
                        meta_row['processing_batch'] = batch_start // batch_size
                        
                        # Store both formats
                        memory.append(legacy_emb)
                        meta.append(meta_row)
                        universal_store.add_session(universal_embedding)
                        
                        # Track modality
                        st.session_state.active_modalities.add(Modality.TEXT)
                        st.session_state.modality_sessions[Modality.TEXT].append({
                            'session_id': session_id,
                            'text': text,
                            'meta': meta_row,
                            'index': valid_sessions
                        })
                        
                        valid_sessions += 1
                        batch_successes += 1
                        processing_quality['successful_embeddings'] += 1
                        
                    except Exception as e:
                        skipped_sessions += 1
                        processing_quality['failed_embeddings'] += 1
                        if skipped_sessions <= 5:
                            status_text.text(f"⚠️ Skipping row {global_i + 1}: {str(e)}")
                        continue
                else:
                    skipped_sessions += 1
            
            # Calculate batch performance metrics
            batch_time = time.time() - batch_start_time
            batch_efficiency = batch_successes / len(batch) if batch else 0
            
            # Update progress with enhanced metrics
            progress_pct = batch_end / len(session_data)
            progress_bar.progress(progress_pct)
            
            current_memory = get_memory_usage()
            elapsed_time = time.time() - start_time
            estimated_total_time = elapsed_time / progress_pct if progress_pct > 0 else 0
            eta = estimated_total_time - elapsed_time if estimated_total_time > elapsed_time else 0
            
            # Enhanced status display
            status_text.markdown(f"""
            **Processing Batch {batch_start // batch_size + 1}**: {batch_start + 1}-{batch_end} of {len(session_data)}
            - ✅ **Sessions**: {valid_sessions} processed, {skipped_sessions} skipped
            - 🧠 **Memory**: {current_memory:.0f}MB ({(current_memory/start_memory-1)*100:+.0f}%)
            - ⏱️ **Time**: {elapsed_time:.1f}s elapsed, ~{eta:.0f}s remaining
            - 📊 **Batch Efficiency**: {batch_efficiency:.1%}
            """)
            
            # Real-time performance metrics display
            with performance_metrics.container():
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Progress", f"{progress_pct:.1%}")
                with col2:
                    st.metric("Memory Usage", f"{current_memory:.0f}MB", 
                             f"{current_memory-start_memory:+.0f}MB")
                with col3:
                    st.metric("Processing Rate", f"{valid_sessions/elapsed_time:.1f}/s")
                with col4:
                    success_rate = processing_quality['successful_embeddings'] / max(1, processing_quality['successful_embeddings'] + processing_quality['failed_embeddings'])
                    st.metric("Success Rate", f"{success_rate:.1%}")
            
            time.sleep(0.01)  # Brief pause for UI responsiveness
        
        # Clear progress displays
        progress_bar.empty()
        status_text.empty()
        performance_metrics.empty()
        
        if memory:
            # Calculate final performance metrics
            total_time = time.time() - start_time
            final_memory = get_memory_usage()
            memory_efficiency = len(memory) / (final_memory - start_memory) if final_memory > start_memory else 1
            
            # Update session state
            st.session_state.memory = memory
            st.session_state.meta = meta
            st.session_state.universal_store = universal_store
            
            def _token_count(session):
                if isinstance(session, UniversalEmbedding):
                    if session.event_embeddings is not None:
                        return session.event_embeddings.shape[0]
                    return 0
                try:
                    return session.shape[0]
                except Exception:
                    return 0

            # Enhanced dataset info with performance metrics
            st.session_state.dataset_info = {
                'source': content_type,
                'filename': filename,
                'upload_timestamp': time.time(),
                'session_count': len(memory),
                'total_tokens': sum(_token_count(m) for m in memory),
                'universal_sessions': len(universal_store.embeddings),
                'active_modalities': list(st.session_state.active_modalities),
                'memory_usage': final_memory,
                'processing_time': total_time,
                'processing_strategy': profile.processing_strategy,
                'complexity_score': profile.complexity_score,
                'performance_metrics': {
                    'sessions_per_second': len(memory) / total_time,
                    'memory_efficiency': memory_efficiency,
                    'success_rate': processing_quality['successful_embeddings'] / max(1, valid_sessions + skipped_sessions),
                    'estimated_quality': min(1.0, success_rate * memory_efficiency)
                }
            }
            
            # Save the data (legacy)
            if memory and not isinstance(memory[0], UniversalEmbedding):
                save(memory, meta)
            else:
                st.info("💾 Sessions stored via Universal Memory; legacy tensor save skipped.")
            
            # Show performance summary
            st.success(f"""
            🎉 **Processing Complete!**
            - **Processed**: {valid_sessions:,} sessions in {total_time:.1f}s
            - **Performance**: {len(memory)/total_time:.1f} sessions/second
            - **Memory**: {final_memory:.0f}MB final ({final_memory-start_memory:+.0f}MB change)
            - **Quality**: {processing_quality['successful_embeddings']/(valid_sessions+skipped_sessions):.1%} success rate
            """)
            st.session_state["upload_success"] = True
            if st.button("➡️ Go to analysis", type="primary"):
                st.rerun()
            
            # Show processing strategy results
            if profile.processing_strategy != "full_processing":
                st.info(f"🎯 **Strategy: {profile.processing_strategy.replace('_', ' ').title()}** - Optimized for your system capabilities")
            
            # Performance warning if needed
            memory_increase = final_memory - start_memory
            if memory_increase > 1000:  # 1GB increase
                st.warning(f"📊 High memory usage: +{memory_increase:.0f}MB. Consider using sampling for larger datasets.")
            elif processing_quality['successful_embeddings'] / max(1, valid_sessions + skipped_sessions) < 0.8:
                st.warning(f"⚠️ Lower success rate ({processing_quality['successful_embeddings']/(valid_sessions+skipped_sessions):.1%}). Check data quality.")
            
            return True
        else:
            st.error(f"❌ No valid text data found in {filename}")
            return False
        
    except Exception as e:
        st.error(f"❌ Failed to process session data: {str(e)}")
        # Force cleanup on error
        cleanup_memory()
        return False


def render_upload_screen():
    """Render the unified upload interface."""
    if st.session_state.get("upload_success") and len(st.session_state.get("memory", [])) > 0:
        st.success("✅ Dataset loaded. Continue to analysis.")
        if st.button("➡️ Go to analysis", type="primary"):
            st.session_state.pop("upload_success", None)
            st.rerun()

    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1>🌐 Semantic Tensor Analysis</h1>
        <h3 style="color: #666;">Analyze how meaning evolves across conversations and sessions</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Unified Upload Section  
        st.markdown("### 📁 **Upload Your Data**")
        st.markdown("Upload **any** text-based data for semantic analysis - we'll detect the format automatically!")
        
        # Enhanced file uploader that accepts multiple formats
        uploaded_file = st.file_uploader(
            "Choose your file",
            type=['csv', 'json', 'txt'],
            help="Supports: CSV files, ChatGPT/AI conversation exports (JSON/TXT), or any text data"
        )
        
        if uploaded_file is not None:
            if handle_unified_upload(uploaded_file):
                st.rerun()

        st.markdown("---")
        
        # Format examples
        with st.expander("📋 **Supported Formats**"):
            st.markdown("""
            ### 🤖 **AI Conversations**
            - **ChatGPT JSON exports** (conversations.json)
            - **Claude/AI text conversations** (copy-pasted chats)
            - **Any conversation format** with User:/Assistant: patterns
            
            ### 📊 **Traditional Data**  
            - **CSV files** with a 'text' column
            - **Journal entries, documents, surveys**
            - **Any structured text data**
            
            ### 🧠 **Auto-Detection**
            Our system automatically detects:
            - File format (CSV, JSON, TXT)
            - Content type (AI conversation vs. traditional sessions)
            - Optimal processing method for your specific data
            """)
        
        # Example datasets
        st.markdown("### 🎯 Try Example Datasets")
        st.caption("Loads the bundled `ultimate_demo_dataset.csv` sample journey.")

        if st.button("📚 Load Demo Dataset", type="primary"):
            # Check if demo dataset exists
            if DEMO_DATA_PATH.exists():
                import io
                content = DEMO_DATA_PATH.read_bytes()
                mock_file = io.BytesIO(content)
                mock_file.name = DEMO_DATA_PATH.name  # type: ignore[attr-defined]
                if handle_unified_upload(mock_file):
                    st.rerun()
            else:
                st.error(
                    f"Demo dataset not found at {DEMO_DATA_PATH}. Please upload your own file."
                )
        
        st.markdown("""
        <div style="background: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem;">
        <strong>💡 What can you analyze?</strong>
        <ul>
        <li>🤖 <strong>AI Conversations</strong>: ChatGPT, Claude, or any AI chat history</li>
        <li>📝 <strong>Journal entries</strong> over time</li>
        <li>📚 <strong>Document evolution</strong> in a project</li>
        <li>💬 <strong>Chat conversations</strong> or interviews</li>
        <li>📊 <strong>Survey responses</strong> across periods</li>
        <li>🎓 <strong>Learning journey</strong> documentation</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)


def render_simple_sidebar():
    """Render a clean, minimal sidebar."""
    # Import sidebar chat module
    try:
        from semantic_tensor_analysis.app.sidebar_chat import render_llm_config_sidebar, render_sidebar_chat
        SIDEBAR_CHAT_AVAILABLE = True
    except ImportError:
        SIDEBAR_CHAT_AVAILABLE = False

    with st.sidebar:
        st.image(str(LOGO_PATH), width=200)

        # Dataset status (compact)
        dataset_info = st.session_state.get('dataset_info', {})
        if dataset_info.get('session_count', 0) > 0:
            filename = dataset_info.get('filename', 'Unknown')
            session_count = dataset_info.get('session_count', 0)
            st.markdown(f"**📁 {filename}**")
            st.markdown(f"🔢 {session_count} sessions")

            # Quick actions
            if st.button("🔄 New Dataset"):
                st.session_state.memory = []
                st.session_state.meta = []
                st.session_state.dataset_info = {'session_count': 0}
                st.rerun()
        else:
            st.markdown("**No dataset loaded**")

        st.markdown("---")

        # Show preferred dimensionality method
        preferred_method = st.session_state.get('preferred_method')
        if preferred_method:
            st.markdown("**📐 Preferred Method:**")
            st.markdown(f"🎯 {preferred_method.upper()}")
            st.caption("Set in Dimensionality tab")

    # Add LLM config and chat to sidebar
    if SIDEBAR_CHAT_AVAILABLE:
        render_llm_config_sidebar()

        # Pass context about current tab
        context = {
            'current_tab': st.session_state.get('active_tab', 'Overview'),
            'data_summary': f"{dataset_info.get('session_count', 0)} sessions loaded",
            'has_data': dataset_info.get('session_count', 0) > 0
        }
        render_sidebar_chat(context)


def render_overview_dashboard():
    """Render an enhanced overview dashboard with meaningful concept-level insights and performance metrics."""
    dataset_info = st.session_state.get('dataset_info', {})
    filename = dataset_info.get('filename', 'Unknown Dataset')
    session_count = dataset_info.get('session_count', 0)
    source_type = dataset_info.get('source', 'unknown')
    
    # Header with better styling and source indicator
    icon = "🤖" if source_type == 'ai_conversation' else "📊"
    source_label = "AI Conversation Analysis" if source_type == 'ai_conversation' else "Semantic Analysis Overview"
    
    st.markdown(f"""
    <div style="text-align: center; padding: 1rem 0;">
        <h1>{icon} {filename}</h1>
        <h3 style="color: #666;">{source_label}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if multi-resolution is available
    has_multi_res = 'temporal_manager' in st.session_state
    
    if has_multi_res:
        render_temporal_zoom_interface()
        render_multi_resolution_summary()
    
    # Performance Metrics Dashboard (if available)
    if 'performance_metrics' in dataset_info:
        performance = dataset_info['performance_metrics']
        processing_time = dataset_info.get('processing_time', 0)
        memory_usage = dataset_info.get('memory_usage', 0)
        processing_strategy = dataset_info.get('processing_strategy', 'full_processing')
        complexity_score = dataset_info.get('complexity_score', 0)
        
        # Performance Summary Card
        with st.expander("📊 **Processing Performance Summary**", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("⚡ Processing Speed", f"{performance['sessions_per_second']:.1f}/s",
                         help="Sessions processed per second")
            
            with col2:
                st.metric("🧠 Memory Efficiency", f"{performance['memory_efficiency']:.1f}",
                         help="Sessions per MB of memory used")
            
            with col3:
                st.metric("✅ Success Rate", f"{performance['success_rate']:.1%}",
                         help="Percentage of sessions successfully processed")
            
            with col4:
                st.metric("🎯 Quality Score", f"{performance['estimated_quality']:.1%}",
                         help="Overall processing quality estimate")
            
            # Processing details
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"**⏱️ Total Time**: {processing_time:.1f}s")
                st.markdown(f"**🔧 Strategy**: {processing_strategy.replace('_', ' ').title()}")
            
            with col2:
                st.markdown(f"**💾 Memory Usage**: {memory_usage:.0f}MB")
                st.markdown(f"**📊 Dataset Complexity**: {complexity_score:.2f}/1.0")
            
            with col3:
                # Quality assessment
                if performance['estimated_quality'] > 0.8:
                    quality_status = "🟢 Excellent"
                elif performance['estimated_quality'] > 0.6:
                    quality_status = "🟡 Good"
                else:
                    quality_status = "🔴 Needs Review"
                st.markdown(f"**Quality Status**: {quality_status}")
                
                # Strategy recommendation
                if processing_strategy == "progressive_sampling":
                    st.markdown("**🎯 Used Smart Sampling** for large dataset")
                elif processing_strategy == "smart_batching":
                    st.markdown("**🔄 Used Adaptive Batching** for efficiency")
                else:
                    st.markdown("**✅ Full Processing** completed")
    
    # Show special indicator for AI conversations
    if source_type == 'ai_conversation':
        st.info("🤖 **AI Conversation Detected**: Analyzing your message evolution across AI interactions")
    
    if session_count < 2:
        st.info("📊 Upload at least 2 sessions to see meaningful semantic analysis.")
        return
    
    # Quick Auto-Analysis Section
    st.markdown("### 🚀 Quick Semantic Insights")
    
        # Try to generate quick concept analysis
    if st.button("🔍 Generate Quick Analysis", type="primary", key="quick_analysis_btn"):
        with st.spinner("🧠 Analyzing your semantic patterns..."):
            try:
                # Quick concept analysis with fewer clusters for overview
                from semantic_tensor_analysis.analytics.concept.concept_analysis import ConceptAnalyzer
                from semantic_tensor_analysis.memory.universal_core import UniversalMemoryStore
                
                # PERFORMANCE FIX: Use cached models for quick analysis
                if 'quick_analysis_store' not in st.session_state:
                    st.session_state.quick_analysis_store = get_cached_universal_store()
                    st.session_state.quick_analysis_embedder = get_cached_text_embedder()
                else:
                    # Clear previous data but reuse cached models
                    st.session_state.quick_analysis_store = get_cached_universal_store()
                
                universal_store = st.session_state.quick_analysis_store
                text_embedder = st.session_state.quick_analysis_embedder
                
                # Process sample of sessions for quick analysis
                sample_size = min(20, len(st.session_state.memory))
                memory_sample = st.session_state.memory[-sample_size:]  # Recent sessions
                
                for i, session_text in enumerate(memory_sample):
                    # Get original text from meta if available
                    if i < len(st.session_state.meta):
                        meta_text = st.session_state.meta[i].get('text', str(session_text))
                    else:
                        meta_text = str(session_text)
                    
                    embedding = text_embedder.process_raw_data(
                        meta_text, 
                        session_id=f"overview_session_{i}"
                    )
                    universal_store.add_session(embedding)
                
                # Quick analysis
                analyzer = ConceptAnalyzer(universal_store)
                concept_evolution = analyzer.analyze_complete_concept_evolution(n_clusters=3)
                
                # Store for reuse
                st.session_state.overview_concept_analysis = concept_evolution
                
            except Exception as e:
                st.error(f"Quick analysis failed: {str(e)}")
                concept_evolution = None
    
    # Display analysis if available
    if hasattr(st.session_state, 'overview_concept_analysis'):
        concept_evolution = st.session_state.overview_concept_analysis
        
        # Key Insights Cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Sessions Analyzed", concept_evolution.total_sessions)
        
        with col2:
            st.metric("🎯 Main Concepts", len(concept_evolution.concept_clusters))
        
        with col3:
            major_shifts = len(concept_evolution.major_shifts)
            st.metric("🚀 Major Shifts", major_shifts)
        
        with col4:
            if concept_evolution.drift_patterns:
                avg_drift = np.mean([p.drift_magnitude for p in concept_evolution.drift_patterns])
                st.metric("📈 Avg Drift", f"{avg_drift:.3f}")
            else:
                st.metric("📈 Avg Drift", "N/A")
        
        # Main concept insights
        st.markdown("### 💡 Your Main Concepts")
        
        if concept_evolution.concept_clusters:
            # Show top 3 concepts
            for i, cluster in enumerate(concept_evolution.concept_clusters[:3]):
                with st.expander(f"🎯 Concept {i+1}: {', '.join(cluster.theme_keywords[:3])}", expanded=i==0):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write(f"**Sessions:** {len(cluster.session_indices)}")
                        st.write(f"**Main themes:** {', '.join(cluster.theme_keywords[:5])}")
                        st.write(f"**Representative text:** _{cluster.representative_text[:200]}..._")
                    with col2:
                        st.metric("Coherence", f"{cluster.coherence_score:.3f}")
                        if concept_evolution.concept_persistence:
                            main_theme = cluster.theme_keywords[0] if cluster.theme_keywords else f"cluster_{cluster.cluster_id}"
                            persistence = concept_evolution.concept_persistence.get(main_theme, 0)
                            st.metric("Persistence", f"{persistence:.1%}")
        
        # Quick visualization
        st.markdown("### 📊 Concept Evolution Timeline")
        try:
            fig = visualize_concept_evolution(concept_evolution, "timeline")
            st.plotly_chart(fig, use_container_width=True, key="overview_concept_timeline")
        except Exception as e:
            st.error(f"Could not generate timeline: {str(e)}")
        
        # Actionable next steps
        st.markdown("### 🎯 Recommended Next Steps")
        
        insights = []
        if major_shifts > 2:
            insights.append("🚀 **High conceptual evolution** - Explore the 'Concepts' tab to understand major shifts")
        if concept_evolution.concept_clusters and len(concept_evolution.concept_clusters[0].session_indices) > len(concept_evolution.concept_clusters) * 0.6:
            insights.append("🎯 **Dominant theme detected** - Check 'Patterns' tab for deeper thematic analysis")
        if concept_evolution.drift_patterns and any(p.drift_magnitude > 0.5 for p in concept_evolution.drift_patterns):
            insights.append("🌊 **Strong semantic shifts** - Use 'Evolution' tab to track temporal changes")
        
        if insights:
            for insight in insights:
                st.markdown(f"- {insight}")
        else:
            st.markdown("- 📈 **Stable semantic patterns** - Your content shows consistent themes over time")
            st.markdown("- 🔍 **Explore deeper** - Use the 'Concepts' tab for detailed cluster analysis")
        
        # Call to action
        st.markdown("---")
        st.markdown("### 🧭 Explore Your Data")
        
        exploration_col1, exploration_col2, exploration_col3 = st.columns(3)
        
        with exploration_col1:
            if st.button("🧠 Deep Concept Analysis", key="goto_concepts"):
                st.info("👉 Navigate to the **'🧠 Concepts'** tab for comprehensive concept analysis!")
        
        with exploration_col2:
            if st.button("🌊 Evolution Patterns", key="goto_evolution"):
                st.info("👉 Check the **'🌊 Evolution'** tab to track semantic changes over time!")
        
        with exploration_col3:
            if st.button("🔍 Pattern Discovery", key="goto_patterns"):
                st.info("👉 Visit the **'🔍 Patterns'** tab for advanced pattern analysis!")
    
    else:
        # No analysis yet - show what's available
        st.markdown("""
        ### 🎯 What You Can Discover
        
        **Click 'Generate Quick Analysis' above to get:**
        - 🧠 **Main concepts** in your data
        - 📈 **Semantic evolution** patterns  
        - 🚀 **Major conceptual shifts**
        - 💡 **Personalized insights** and recommendations
        
        ### 📊 Your Dataset
        """)
        
        # Basic dataset info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📊 Sessions", session_count)
        
        with col2:
            total_tokens = dataset_info.get('total_tokens', 0)
            st.metric("🔤 Total Tokens", f"{total_tokens:,}")
        
        with col3:
            avg_tokens = total_tokens / max(session_count, 1)
            st.metric("📏 Avg Length", f"{avg_tokens:.0f} tokens")
        
        st.markdown("""
        ### 🚀 Ready to Analyze?
        
        Your semantic tensor memory system is loaded and ready! Here's what each tab offers:
        
        - **🧠 Concepts**: Advanced concept clustering and evolution analysis
        - **🌊 Evolution**: Track semantic changes over time  
        - **🔍 Patterns**: Discover hidden patterns and relationships
        - **📐 Dimensionality**: Compare different analysis methods
        - **🤖 AI Insights**: Get AI-powered analysis and interpretations
        
        **Start with the quick analysis above, then dive deeper into any tab!**
        """)

    # Mini similarity heatmap (sampled)
    if session_count >= 2 and 'memory' in st.session_state and st.session_state.memory:
        with st.expander("🔥 Mini Similarity Heatmap (sampled)", expanded=False):
            max_n = min(30, session_count)
            n = st.slider("Sessions (first N)", 2, max_n, min(10, max_n))
            try:
                fig = plot_heatmap_plotly(st.session_state.memory[:n])
                st.plotly_chart(fig, use_container_width=True, key="overview_mini_heatmap")
                st.caption("Go to 🔍 Patterns → Similarity Heatmap for full view and token alignment.")
            except Exception as e:
                st.error(f"Mini heatmap failed: {e}")


def render_semantic_evolution_tab():
    """Combined semantic evolution analysis (drift + trajectory)."""
    st.header("🌊 Semantic Evolution")
    
    if len(st.session_state.memory) > 1:
        # Evolution over time
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Drift Analysis")
            drifts, counts = drift_series(st.session_state.memory)
            fig = plot_drift_plotly(drifts, counts)
            st.plotly_chart(fig, use_container_width=True, key="evolution_drift_plot")
        
        with col2:
            st.subheader("🎯 3D Trajectory")
            trajectory_data = calculate_semantic_trajectory_data(
                st.session_state.memory, 
                st.session_state.meta
            )
            if trajectory_data:
                fig = create_3d_trajectory_plot(trajectory_data)
                st.plotly_chart(fig, use_container_width=True, key="evolution_3d_trajectory")
        
        # Detailed analysis table
        st.subheader("📊 Session-by-Session Analysis")
        if trajectory_data:
            table_data = display_trajectory_analysis_table(trajectory_data)
            st.dataframe(table_data, use_container_width=True)

        # Token alignment for consecutive sessions
        with st.expander("🔎 Token Drift Alignment (consecutive sessions)"):
            if len(st.session_state.memory) >= 2:
                max_pair = len(st.session_state.memory) - 1
                pair_idx = st.slider("Step (i → i+1)", 1, max_pair, 1)
                if st.button("Show alignment", key="evolution_token_alignment"):
                    try:
                        fig_align = token_alignment_heatmap(st.session_state.memory, pair_idx-1, pair_idx)
                        if fig_align is not None:
                            st.pyplot(fig_align, use_container_width=True)
                    except Exception as e:
                        st.error(f"Token alignment failed: {e}")

        # Token importance drift and coherence trends
        from semantic_tensor_analysis.memory.sequence_drift import (
            token_importance_drift,
            semantic_coherence_score,
        )
        with st.expander("📌 Token Importance Drift & Coherence"):
            try:
                top_k = st.slider("Top drifting tokens (K)", 5, 20, 10)
                top = token_importance_drift(st.session_state.memory, top_k=top_k)
                if top:
                    df_top = pd.DataFrame(top, columns=["SessionIndex", "DriftScore"]) 
                    df_top["SessionIndex"] = df_top["SessionIndex"].astype(int) + 1
                    st.dataframe(df_top, use_container_width=True)
                coherences = [semantic_coherence_score(t) for t in st.session_state.memory]
                st.line_chart(pd.DataFrame({"Coherence": coherences}))
            except Exception as e:
                st.caption(f"Drift/coherence summary unavailable: {e}")
    
    else:
        st.warning("Need ≥2 sessions for evolution analysis.")


def render_pattern_analysis_tab():
    """Pattern discovery with multiple visualization techniques."""
    st.header("🔍 Pattern Analysis")
    
    if len(st.session_state.memory) > 1:
        # Analysis type selector
        analysis_type = st.radio(
            "Choose analysis type:",
            [
                "🌐 Holistic Semantic Analysis",
                "🌊 Semantic Drift River (3D)",
                "📊 Ridgeline (Feature Evolution)",
                "🔥 Similarity Heatmap",
                "🎬 Animated Patterns",
            ],
            horizontal=True,
        )
        
        if analysis_type.startswith("🌐"):
            # Holistic Semantic Analysis
            render_holistic_semantic_analysis(st.session_state.memory, st.session_state.meta)
        
        elif analysis_type.startswith("🌊"):
            # 3D Semantic Drift River
            render_semantic_drift_river_analysis(st.session_state.memory, st.session_state.meta)
        
        elif analysis_type.startswith("📊"):
            # Enhanced ridgeline
            result = plot_enhanced_ridgeline_altair(
                st.session_state.memory, 
                st.session_state.meta,
                show_trends=True,
                highlight_changes=True
            )
            
            if result and len(result) == 2:
                fig, scaling_info = result
                if fig:
                    st.altair_chart(fig, use_container_width=True)
                    with st.expander("📈 Scaling Details"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Features", scaling_info.get('features', 'N/A'))
                        with col2:
                            st.metric("Value Range", scaling_info.get('value_range', 'N/A'))
                        with col3:
                            st.metric("Chart Height", f"{scaling_info.get('adaptive_height', 'N/A')}px")
        
        elif analysis_type.startswith("🔥"):
            # Heatmap
            fig = plot_heatmap_plotly(st.session_state.memory)
            st.plotly_chart(fig, use_container_width=True, key="pattern_heatmap")

            # Optional token alignment drilldown
            with st.expander("🔎 Token Alignment Heatmap (pairwise)"):
                if len(st.session_state.memory) >= 2:
                    col_i, col_j = st.columns(2)
                    with col_i:
                        i = st.number_input("Session i", min_value=1, max_value=len(st.session_state.memory), value=1)
                    with col_j:
                        j = st.number_input("Session j", min_value=1, max_value=len(st.session_state.memory), value=2)
                    if st.button("Show token alignment", key="show_token_alignment"):
                        try:
                            fig_align = token_alignment_heatmap(st.session_state.memory, int(i)-1, int(j)-1)
                            if fig_align is not None:
                                st.pyplot(fig_align, use_container_width=True)
                        except Exception as e:
                            st.error(f"Token alignment heatmap failed: {e}")
        
        elif analysis_type.startswith("🎬"):
            # Animated analysis
            st.subheader("🎬 Animated Semantic Evolution")
            
            # Animation type selector
            animation_type = st.radio(
                "Choose animation type:",
                ["🎯 Trajectory Evolution", "📈 PCA Over Time", "📊 Variance Build-up", "🌊 Liminal Tunnel", "🌌 4D Semantic Space", "🔥 Temporal Similarity (sliding window)"],
                horizontal=True
            )
            
            # Quick animation settings
            col1, col2 = st.columns(2)
            with col1:
                speed = st.selectbox("Speed", [300, 500, 800], index=1)
            with col2:
                include_3d = st.checkbox("3D View", value=True)
            
            # Enhanced PCA for animation
            results = robust_pca_pipeline(
                st.session_state.memory, 
                st.session_state.meta, 
                n_components=3 if include_3d else 2,
                method='auto'
            )
            
            # Check if user has a preferred method and can use it
            preferred_method = st.session_state.get('preferred_method')
            if preferred_method:
                st.info(f"🎯 Note: Your preferred method is {preferred_method.upper()}, but animations currently use PCA for temporal consistency.")
            
            if results:
                if animation_type.startswith("🎯"):
                    # Animated trajectory
                    trajectory_fig = create_animated_pca_trajectory(results, st.session_state.meta, speed)
                    if trajectory_fig:
                        st.plotly_chart(trajectory_fig, use_container_width=True, key="pattern_animated_trajectory")
                        # Axis explainer (LLM)
                        with st.expander("🧠 Axis Explainer (LLM)"):
                            try:
                                from semantic_tensor_analysis.visualization.viz.semantic_analysis import (
                                    analyze_pca_patterns,
                                )
                                # Build texts/scores for PC1 extremes
                                reduced = results['reduced']
                                session_ids = np.array(results['session_ids'])
                                pca1 = reduced[:,0]
                                idxs = np.argsort(pca1)
                                sample_idxs = np.concatenate([idxs[:3], idxs[-3:]]) if len(idxs) >= 6 else idxs
                                texts = [st.session_state.meta[session_ids[i]].get('text','') for i in sample_idxs]
                                scores = [float(pca1[i]) for i in sample_idxs]
                                if st.button("Explain axes", key="explain_axes_traj"):
                                    summary = analyze_pca_patterns(texts, scores)
                                    st.write(summary)
                            except Exception as e:
                                st.caption(f"Axis explainer unavailable: {e}")
                
                elif animation_type.startswith("📈"):
                    # PCA over time animation
                    st.info("🎥 Watch how the PCA space evolves as sessions are progressively added")
                    timeline_fig = create_pca_timeline_animation(
                        st.session_state.memory,
                        st.session_state.meta,
                        animation_speed=speed,
                        is_3d=include_3d
                    )
                    if timeline_fig:
                        st.plotly_chart(timeline_fig, use_container_width=True, key="pattern_pca_timeline")
                        
                        # Quick tips
                        st.markdown("""
                        **💡 Animation Tips:**
                        - 🔴 **Larger dots** = newest session in each frame
                        - 🌈 **Color** = temporal progression (blue → red)
                        - 📊 **Space changes** = how axes reorient with new data
                        - ⏭️ **Use Final** button to jump to complete dataset
                        """)
                
                elif animation_type.startswith("📊"):
                    # Variance evolution
                    variance_fig = create_variance_evolution_animation(results)
                    if variance_fig:
                        st.plotly_chart(variance_fig, use_container_width=True, key="pattern_variance_evolution")
                
                elif animation_type.startswith("🌊"):
                    # Liminal Tunnel visualization - PCA + t-SNE hybrid through temporal space
                    st.info("🌊 Liminal Tunnel: PCA + t-SNE hybrid flowing through 3D temporal space")
                    tunnel_fig = create_liminal_tunnel_visualization(
                        st.session_state.memory,
                        st.session_state.meta
                    )
                    if tunnel_fig:
                        st.plotly_chart(tunnel_fig, use_container_width=True, key="pattern_liminal_tunnel")
                        
                        # Liminal tunnel interpretation
                        st.markdown("""
                        **🌊 Liminal Tunnel Guide:**
                        - **🚇 Tunnel spine** = Smooth temporal path through hybrid PCA-t-SNE space
                        - **💎 Diamond anchors** = Actual session positions in hybrid space
                        - **⚪ Flow particles** = Temporal progression indicators
                        - **🌌 Liminal aesthetics** = Dark space with ethereal colors
                        - **📐 Hybrid dimensions** = Global PCA structure + local t-SNE patterns
                        """)
                    else:
                        st.error("Could not generate liminal tunnel visualization")
                
                elif animation_type.startswith("🌌"):
                    # 4D Semantic Space visualization
                    st.info("🌌 4D Semantic Space: Pure PCA with 4th dimension controlling visual properties")
                    tunnel_fig = create_4d_semantic_space_visualization(
                        st.session_state.memory,
                        st.session_state.meta
                    )
                    if tunnel_fig:
                        st.plotly_chart(tunnel_fig, use_container_width=True, key="pattern_4d_semantic_space")
                        
                        # 4D semantic space interpretation
                        st.markdown("""
                        **🌌 4D Semantic Space Guide:**
                        - **🎯 Rotate & zoom** to explore 4D space from different angles
                        - **🌈 Colors** represent the 4th semantic dimension (PC4)
                        - **📏 Sizes** vary with 4th dimension intensity
                        - **🔗 Connections** show semantic tunnels between sessions
                        - **➡️ Arrows** show temporal flow direction
                        """)
                    else:
                        st.error("Could not generate 4D semantic space visualization")

                elif animation_type.startswith("🔥"):
                    # Temporal similarity heatmap
                    window = st.slider("Window size", min_value=3, max_value=min(12, len(st.session_state.memory)), value=5)
                    temp_fig = create_temporal_heatmap(results, st.session_state.meta, window_size=window)
                    if temp_fig:
                        st.plotly_chart(temp_fig, use_container_width=True, key="pattern_temporal_similarity")
    
    else:
        st.warning("Need ≥2 sessions for pattern analysis.")


def render_dimensionality_tab():
    """Dimensionality reduction analysis and comparison."""
    st.header("📐 Dimensionality Analysis")
    
    if len(st.session_state.memory) > 1:
        # Analysis options
        analysis_mode = st.radio(
            "Analysis mode:",
            ["🎯 Enhanced PCA", "🔬 Method Comparison", "🌊 Liminal Tunnel Visualization"],
            horizontal=True
        )
        
        if analysis_mode.startswith("🎯"):
            # Enhanced PCA with timeline
            st.subheader("Enhanced PCA Analysis")
            
            # Show current method preference
            preferred_method = st.session_state.get('preferred_method')
            if preferred_method:
                st.info(f"🎯 Your preferred method: **{preferred_method.upper()}** (from method comparison)")
                
                # Option to use preferred method if available
                if preferred_method == 'umap' and 'method_results' in st.session_state:
                    use_preferred = st.checkbox("Use UMAP instead of PCA", value=False, 
                                              help="Use your preferred UMAP method for this analysis")
                else:
                    use_preferred = False
                    if preferred_method != 'pca':
                        st.warning(f"⚠️ {preferred_method.upper()} not available for interactive timeline analysis. Using PCA.")
            else:
                use_preferred = False
            
            col1, col2 = st.columns([3, 1])
            with col1:
                max_session = len(st.session_state.memory)
                timeline_idx = st.slider(
                    "Sessions to include:",
                    2, max_session, max_session,
                    help="Adjust to see how patterns develop over time"
                )
            with col2:
                is_3d = st.checkbox("3D View", value=False)
            
            # Add animation controls
            st.markdown("### 🎬 Animation Options")
            col1, col2, col3 = st.columns(3)
            with col1:
                show_timeline_animation = st.checkbox("📈 PCA Over Time Animation", 
                                                    value=False,
                                                    help="See how PCA space evolves as sessions are added")
            with col2:
                if show_timeline_animation:
                    animation_speed = st.selectbox("Animation Speed", [500, 800, 1200], index=1,
                                                  help="Milliseconds per frame")
                else:
                    animation_speed = 800
            with col3:
                if show_timeline_animation:
                    use_3d_animation = st.checkbox("3D Animation", value=is_3d,
                                                  help="Use 3D view for timeline animation")
                else:
                    use_3d_animation = is_3d
            
            # Point granularity toggle
            st.markdown("### 🔬 Point Granularity")
            granularity = st.radio(
                "Show points as:", ["Session means", "Tokens (sampled)"] , index=0, horizontal=True
            )

            # Run analysis with preferred method if available and selected
            if use_preferred and 'method_results' in st.session_state:
                # Use cached UMAP results
                method_results = st.session_state['method_results']
                if method_results.get('umap_results'):
                    fig = create_alternative_visualization(
                        method_results['umap_results'], 
                        st.session_state.meta, 
                        "UMAP"
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key="dimensionality_enhanced_umap")
                        st.caption("🎯 Using UMAP (your preferred method)")
                        
                        # Show UMAP metrics
                        umap_results = method_results['umap_results']
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Method", "UMAP")
                        with col2:
                            trust_score = method_results.get('results', {}).get('UMAP', {}).get('trust_score', 'N/A')
                            st.metric("Trust Score", f"{trust_score:.3f}" if isinstance(trust_score, (int, float)) else trust_score)
                        with col3:
                            st.metric("Samples", f"{umap_results.get('n_samples', 'N/A'):,}")
                else:
                    st.error("UMAP results not available. Falling back to PCA.")
                    use_preferred = False
            
            if not use_preferred:
                # Standard PCA analysis
                memory_slice = st.session_state.memory[:timeline_idx]
                meta_slice = st.session_state.meta[:timeline_idx]
                
                if show_timeline_animation:
                    # Show PCA over time animation
                    st.subheader("🎬 PCA Space Evolution Over Time")
                    st.info("🎥 Watch how the semantic space develops as sessions are progressively added")
                    
                    timeline_fig = create_pca_timeline_animation(
                        st.session_state.memory[:timeline_idx],
                        st.session_state.meta[:timeline_idx],
                        animation_speed=animation_speed,
                        is_3d=use_3d_animation
                    )
                    
                    if timeline_fig:
                        st.plotly_chart(timeline_fig, use_container_width=True, key="dimensionality_pca_timeline")
                        
                        # Animation interpretation guide
                        with st.expander("🎭 How to Interpret the Animation"):
                            st.markdown("""
                            **🎬 Animation Guide:**
                            - **▶️ Play Timeline**: Watch sessions being added chronologically
                            - **Larger dots**: Newest session in each frame
                            - **Color progression**: Blue → Red shows temporal order
                            - **Space shifts**: How PCA axes reorient as data grows
                            - **Quality changes**: Explained variance evolving over time
                            
                            **🔍 What to Look For:**
                            - **Stable patterns**: Consistent positioning across frames
                            - **Sudden shifts**: New sessions dramatically changing the space
                            - **Convergence**: Space stabilizing as more data is added
                            - **Outliers**: Sessions that reshape the entire analysis
                            """)
                else:
                    # Standard static analysis
                    results = robust_pca_pipeline(
                        memory_slice, meta_slice, 
                        n_components=3 if is_3d else 2,
                        method='auto'
                    )
                    
                    if results:
                        # Visualization
                        if granularity.startswith("Tokens"):
                            # Rebuild a token-level scatter directly from results (sampled)
                            try:
                                reduced = results['reduced']
                                session_ids = np.array(results['session_ids'])
                                max_points = 2000
                                if reduced.shape[0] > max_points:
                                    idx = np.linspace(0, reduced.shape[0]-1, max_points, dtype=int)
                                    reduced = reduced[idx]
                                    session_ids = session_ids[idx]
                                df = pd.DataFrame({
                                    'PC1': reduced[:,0],
                                    'PC2': reduced[:,1],
                                    'SessionIdx': session_ids
                                })
                                if is_3d and reduced.shape[1] > 2:
                                    df['PC3'] = reduced[:,2]
                                    fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3', color='SessionIdx', color_continuous_scale='RdYlBu')
                                else:
                                    fig = px.scatter(df, x='PC1', y='PC2', color='SessionIdx', color_continuous_scale='RdYlBu')
                            except Exception as e:
                                st.error(f"Token-level view failed, falling back to session means: {e}")
                                fig = create_pca_visualization(results, meta_slice, is_3d=is_3d)
                        else:
                            fig = create_pca_visualization(results, meta_slice, is_3d=is_3d)
                        st.plotly_chart(fig, use_container_width=True, key="dimensionality_enhanced_pca")
                        
                        # Quality metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Explained Variance", f"{results['cumulative_variance'][-1]:.1%}")
                        with col2:
                            st.metric("Quality", results['quality_assessment'].title())
                        with col3:
                            st.metric("Method", results['method_used'])
                        with col4:
                            st.metric("Features", f"{results['n_features']:,}")
        
        elif analysis_mode.startswith("🔬"):
            # Method comparison
            st.subheader("Method Comparison")
            
            if st.button("🚀 Compare Methods", type="primary"):
                with st.spinner("Comparing PCA, UMAP, and t-SNE..."):
                    comparison_results = compare_dimensionality_methods(
                        st.session_state.memory, 
                        st.session_state.meta
                    )
                
                if comparison_results:
                    st.session_state['comparison_results'] = comparison_results
                    
                    # Show best method
                    best_method = comparison_results['best_method']
                    st.success(f"🏆 Best method: **{best_method}**")
                    
                    # Visualization of best method
                    if best_method == "UMAP" and comparison_results.get('umap_results'):
                        fig = create_alternative_visualization(
                            comparison_results['umap_results'], 
                            st.session_state.meta, 
                            "UMAP"
                        )
                        if fig:
                            st.plotly_chart(fig, use_container_width=True, key="dimensionality_method_comparison")
                    
                    # Add actionable buttons
                    st.markdown("### 🎯 Apply Recommendation")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button(f"✅ Use {best_method} as Default", type="primary"):
                            st.session_state['preferred_method'] = best_method.lower()
                            st.session_state['method_results'] = comparison_results
                            st.success(f"✅ {best_method} is now your default dimensionality reduction method!")
                            st.info("💡 Other visualizations in the app will now use this method when possible.")
                            st.rerun()
                    
                    with col2:
                        if st.button("🔄 Reset to Auto"):
                            if 'preferred_method' in st.session_state:
                                del st.session_state['preferred_method']
                            if 'method_results' in st.session_state:
                                del st.session_state['method_results']
                            st.success("🔄 Reset to automatic method selection.")
                            st.rerun()
                    
                    # Results summary
                    with st.expander("📊 Detailed Results"):
                        results = comparison_results.get('results', {})
                        for method, data in results.items():
                            st.markdown(f"**{method}:**")
                            if isinstance(data, dict):
                                for key, value in data.items():
                                    if isinstance(value, (int, float)):
                                        st.write(f"  - {key}: {value:.3f}")
            
            # Show current method preference
            if 'preferred_method' in st.session_state:
                preferred = st.session_state['preferred_method'].upper()
                st.info(f"🎯 **Current default method**: {preferred}")
                
                # Show when this method is being used
                st.markdown("**This method is automatically used in:**")
                st.markdown("- 🏠 Overview dashboard visualizations")
                st.markdown("- 🌊 Evolution tab PCA plots")
                st.markdown("- 🔍 Pattern analysis animations")
            
            # Show stored results
            if 'comparison_results' in st.session_state:
                st.markdown("### Previous Comparison Results")
                results = st.session_state['comparison_results'].get('results', {})
                df_results = pd.DataFrame(results).T
                st.dataframe(df_results, use_container_width=True)
        
        elif analysis_mode.startswith("🌊"):
            # Liminal Tunnel Visualization
            st.subheader("🌊 Liminal Tunnel Visualization")
            st.info("🚇 Journey through hybrid PCA-t-SNE space with temporal tunneling effects")
            
            # Tunnel type selector
            tunnel_type = st.radio(
                "Choose tunnel type:",
                ["🌊 Liminal Tunnel (PCA + t-SNE)", "🌌 4D Semantic Space (Pure PCA)"],
                horizontal=True
            )
            
            # Tunnel visualization controls
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Generate Tunnel", type="primary"):
                    if tunnel_type.startswith("🌊"):
                        with st.spinner("Creating liminal tunnel through PCA-t-SNE hybrid space..."):
                            tunnel_fig = create_liminal_tunnel_visualization(
                                st.session_state.memory,
                                st.session_state.meta
                            )
                            tunnel_key = 'liminal_tunnel_visualization'
                    else:
                        with st.spinner("Creating 4D semantic space visualization..."):
                            tunnel_fig = create_4d_semantic_space_visualization(
                                st.session_state.memory,
                                st.session_state.meta
                            )
                            tunnel_key = '4d_semantic_space_visualization'
                    
                    if tunnel_fig:
                        st.session_state[tunnel_key] = tunnel_fig
                        st.success(f"✅ {tunnel_type.split(' ')[1]} visualization created!")
                    else:
                        st.error(f"❌ Could not create {tunnel_type.lower()} visualization")
            
            with col2:
                # Clear button for active visualizations
                active_visualizations = []
                if 'liminal_tunnel_visualization' in st.session_state:
                    active_visualizations.append('Liminal Tunnel')
                if '4d_semantic_space_visualization' in st.session_state:
                    active_visualizations.append('4D Semantic Space')
                
                if active_visualizations:
                    clear_option = st.selectbox("Clear visualization:", ["None"] + active_visualizations)
                    if st.button("🔄 Clear Selected"):
                        if clear_option == "Liminal Tunnel":
                            del st.session_state['liminal_tunnel_visualization']
                        elif clear_option == "4D Semantic Space":
                            del st.session_state['4d_semantic_space_visualization']
                        if clear_option != "None":
                            st.success(f"🗑️ {clear_option} visualization cleared")
                            st.rerun()
            
            # Show stored visualizations
            if 'liminal_tunnel_visualization' in st.session_state:
                st.markdown("### 🌊 Liminal Tunnel")
                st.plotly_chart(
                    st.session_state['liminal_tunnel_visualization'], 
                    use_container_width=True, 
                    key="dimensionality_liminal_tunnel"
                )
                
                # Liminal tunnel interpretation guide
                with st.expander("🌊 How to Interpret the Liminal Tunnel"):
                    st.markdown("""
                    **🌊 Liminal Tunnel Features:**
                    - **🚇 Tunnel Spine**: Smooth spline path through hybrid PCA-t-SNE space
                    - **💎 Session Anchors**: Diamond markers at actual session positions
                    - **⚪ Flow Particles**: White particles showing temporal progression
                    - **🌌 Tunnel Surface**: Semi-transparent surface creating tunnel effect
                    - **🎨 Liminal Aesthetics**: Dark ethereal space with plasma colors
                    
                    **🔍 What to Look For:**
                    - **Tunnel curvature**: How PCA global structure + t-SNE local patterns combine
                    - **Session positioning**: Where sessions anchor in the hybrid space
                    - **Color progression**: Temporal flow from purple to yellow
                    - **Tunnel width**: Varies based on session variance and characteristics
                    - **Smooth transitions**: Spline interpolation creating fluid movement
                    
                    **🎯 Interactive Features:**
                    - **Rotate & Zoom**: Explore the liminal space from different angles
                    - **Hover**: Get session details and coordinates
                    - **Flow visualization**: Follow particles along temporal path
                    """)
            
            if '4d_semantic_space_visualization' in st.session_state:
                st.markdown("### 🌌 4D Semantic Space")
                st.plotly_chart(
                    st.session_state['4d_semantic_space_visualization'], 
                    use_container_width=True, 
                    key="dimensionality_4d_semantic_space"
                )
                
                # 4D semantic space interpretation guide
                with st.expander("🌌 How to Interpret the 4D Semantic Space"):
                    st.markdown("""
                    **🌌 4D Semantic Space Features:**
                    - **📍 Session Centers**: Large spheres representing session centroids
                    - **🔗 Semantic Tunnels**: Connect consecutive sessions through 4D space
                    - **🌈 4th Dimension Colors**: PC4 values control visual properties
                    - **📏 Variable Sizes**: Marker and tunnel sizes based on PC4 intensity
                    - **➡️ Flow Arrows**: Cone arrows showing temporal direction
                    
                    **🔍 What to Look For:**
                    - **Color intensity**: High PC4 values = complex semantic patterns
                    - **Size variations**: Large elements = high 4th dimension activity
                    - **Tunnel paths**: How sessions connect through 4D semantic space
                    - **Arrow flow**: Temporal progression through the space
                    - **Clustering**: Periods of similar 4D semantic positioning
                    
                    **🎯 Interactive Features:**
                    - **4D Exploration**: Rotate to see different 4D perspectives
                    - **Hover Details**: Get complete 4D coordinates and session info
                    - **Legend Control**: Toggle elements on/off
                    """)
            
            # Show information if no visualizations are active
            if not ('liminal_tunnel_visualization' in st.session_state or '4d_semantic_space_visualization' in st.session_state):
                st.markdown("### 💡 About Tunnel Visualizations")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    **🌊 Liminal Tunnel**
                    
                    The **Liminal Tunnel** creates a hybrid visualization combining:
                    - **PCA global structure** (70% weight)
                    - **t-SNE local relationships** (30% weight)
                    - **Smooth spline interpolation** for temporal flow
                    - **Ethereal liminal aesthetics** for immersive experience
                    
                    **Perfect for:**
                    - Exploring temporal semantic evolution
                    - Seeing both global and local patterns
                    - Understanding transitional phases
                    - Immersive data exploration
                    """)
                
                with col2:
                    st.markdown("""
                    **🌌 4D Semantic Space**
                    
                    The **4D Semantic Space** provides:
                    - **Pure 4D PCA analysis** with all components
                    - **4th dimension visual mapping** to properties
                    - **Semantic tunnels** connecting sessions
                    - **Temporal flow indicators** with arrows
                    
                    **Perfect for:**
                    - Understanding high-dimensional relationships
                    - Exploring 4th semantic dimension effects
                    - Analyzing complex semantic patterns
                    - Mathematical precision in visualization
                    """)
                
                st.info("🚀 Choose a tunnel type and click **'Generate Tunnel'** to begin your journey!")
    
        else:
            st.warning("Need ≥2 sessions for dimensionality analysis.")


def render_enhanced_concept_analysis_tab():
    """Render enhanced concept analysis using existing S-BERT embeddings."""
    st.header("🧠 Enhanced Concept Analysis")
    st.markdown("**Leveraging existing S-BERT sequence embeddings for concept-level analysis**")
    
    if not st.session_state.get('memory') or len(st.session_state.memory) < 2:
        st.info("📁 Need at least 2 sessions for concept analysis")
        st.markdown("""
        **What you'll get with Enhanced Concept Analysis:**
        - **🎯 Concept Clustering**: Group sessions by semantic similarity using S-BERT embeddings
        - **📈 Drift Timeline**: Track how concepts evolve over time  
        - **⚡ Velocity Analysis**: Measure rate of concept change
        - **🌐 Network Visualization**: See relationships between concept clusters
        - **🥧 Persistence Analysis**: Identify long-lasting vs transient concepts
        - **📊 Comprehensive Dashboard**: All analyses in one view
        
        Upload your data to begin concept-level analysis!
        """)
        return
    
    # Analysis controls
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        n_clusters = st.slider("Number of concept clusters", 2, min(8, len(st.session_state.memory)), 5)
        
    with col2:
        analysis_scope = st.selectbox(
            "Analysis Scope",
            ["All Sessions", "Recent 50", "Recent 100", "First 50"]
        )
    
    with col3:
        st.write("")  # Spacing
        if st.button("🔍 Analyze Concepts", type="primary", key="concept_analyze_btn"):
            st.session_state.run_concept_analysis = True
    
    # Run analysis if requested
    if st.session_state.get('run_concept_analysis', False):
        with st.spinner("🧠 Analyzing concept evolution using S-BERT embeddings..."):
            try:
                # Import enhanced concept analysis helpers
                from semantic_tensor_analysis.analytics.concept.concept_analysis import ConceptAnalyzer
                
                # Create Universal STM store from existing memory
                from semantic_tensor_analysis.memory.universal_core import UniversalMemoryStore
                
                # Convert existing memory to Universal STM format using shared embedder
                universal_store = UniversalMemoryStore()
                text_embedder = get_cached_text_embedder()
                
                # Filter sessions based on scope
                memory_data = st.session_state.memory
                if analysis_scope == "Recent 50":
                    memory_data = memory_data[-50:] if len(memory_data) > 50 else memory_data
                elif analysis_scope == "Recent 100":
                    memory_data = memory_data[-100:] if len(memory_data) > 100 else memory_data
                elif analysis_scope == "First 50":
                    memory_data = memory_data[:50]
                
                # Process existing sessions
                progress_bar = st.progress(0)
                for i, session_text in enumerate(memory_data):
                    embedding = text_embedder.process_raw_data(
                        session_text, 
                        session_id=f"session_{i}"
                    )
                    universal_store.add_session(embedding)
                    progress_bar.progress((i + 1) / len(memory_data))
                
                progress_bar.empty()
                
                # Run concept analysis
                analyzer = ConceptAnalyzer(universal_store)
                concept_evolution = analyzer.analyze_complete_concept_evolution(n_clusters)
                
                # Store results
                st.session_state.concept_evolution = concept_evolution
                st.session_state.concept_store = universal_store
                st.session_state.run_concept_analysis = False
                
                st.success(f"✅ Analyzed {concept_evolution.total_sessions} sessions with {len(concept_evolution.concept_clusters)} concept clusters")
                
            except Exception as e:
                st.error(f"Error in concept analysis: {str(e)}")
                st.exception(e)
                st.session_state.run_concept_analysis = False
                return
    
    # Display results if available
    if hasattr(st.session_state, 'concept_evolution'):
        concept_evolution = st.session_state.concept_evolution
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Sessions", concept_evolution.total_sessions)
        with col2:
            st.metric("🎯 Clusters", len(concept_evolution.concept_clusters))
        with col3:
            major_shifts = len(concept_evolution.major_shifts)
            st.metric("🚀 Major Shifts", major_shifts)
        with col4:
            if concept_evolution.drift_patterns:
                avg_drift = np.mean([p.drift_magnitude for p in concept_evolution.drift_patterns])
                st.metric("📈 Avg Drift", f"{avg_drift:.3f}")
            else:
                st.metric("📈 Avg Drift", "N/A")
        
        # Visualization controls
        st.subheader("📊 Concept Evolution Visualizations")
        
        viz_choice = st.selectbox(
            "Choose Visualization",
            ["📈 Dashboard", "🔥 Cluster Heatmap", "📊 Drift Timeline", "⚡ Velocity Chart", "🌐 Network Graph", "🥧 Persistence"],
            key="concept_viz_choice"
        )
        
        # Generate visualization
        try:
            chart_type_map = {
                "📈 Dashboard": "dashboard",
                "🔥 Cluster Heatmap": "heatmap", 
                "📊 Drift Timeline": "timeline",
                "⚡ Velocity Chart": "velocity",
                "🌐 Network Graph": "network",
                "🥧 Persistence": "persistence"
            }
            
            chart_type = chart_type_map[viz_choice]
            fig = visualize_concept_evolution(concept_evolution, chart_type)
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional context based on visualization type
            if viz_choice == "🔥 Cluster Heatmap":
                st.subheader("🎯 Cluster Details")
                for cluster in concept_evolution.concept_clusters[:5]:  # Show top 5
                    with st.expander(f"Cluster {cluster.cluster_id}: {', '.join(cluster.theme_keywords[:3])}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Sessions:** {len(cluster.session_indices)}")
                            st.write(f"**Coherence:** {cluster.coherence_score:.3f}")
                        with col2:
                            st.write(f"**Keywords:** {', '.join(cluster.theme_keywords)}")
                            st.write(f"**Sample:** {cluster.representative_text[:150]}...")

                        # Token alignment for two representative sessions in this cluster
                        with st.expander("🔎 Token Alignment (top exemplars)"):
                            if len(cluster.session_indices) >= 2:
                                a = cluster.session_indices[0]
                                b = cluster.session_indices[1]
                                st.caption(f"Aligning Session {a+1} and Session {b+1}")
                                if st.button(f"Show alignment for Cluster {cluster.cluster_id}", key=f"cluster_align_{cluster.cluster_id}"):
                                    try:
                                        token_alignment_heatmap(st.session_state.memory, a, b)
                                        st.caption("Close the Matplotlib window to continue.")
                                    except Exception as e:
                                        st.error(f"Alignment failed: {e}")
            
            elif viz_choice == "📊 Drift Timeline":
                if concept_evolution.drift_patterns:
                    st.subheader("📋 Drift Pattern Summary")
                    drift_df = pd.DataFrame([
                        {
                            "From": p.session_from + 1,
                            "To": p.session_to + 1,
                            "Drift": f"{p.drift_magnitude:.3f}",
                            "Direction": p.drift_direction,
                            "New Concepts": ", ".join(p.concept_shift_keywords[:2])
                        }
                        for p in concept_evolution.drift_patterns[:10]
                    ])
                    st.dataframe(drift_df, use_container_width=True)
            
            elif viz_choice == "⚡ Velocity Chart":
                if concept_evolution.concept_velocity:
                    max_velocity_idx = np.argmax(concept_evolution.concept_velocity)
                    st.info(f"🚀 **Peak velocity** at transition {max_velocity_idx + 1}: {concept_evolution.concept_velocity[max_velocity_idx]:.3f}")
            
            elif viz_choice == "🥧 Persistence":
                if concept_evolution.concept_persistence:
                    st.subheader("🏆 Top Persistent Concepts")
                    persistence_df = pd.DataFrame([
                        {"Concept": concept, "Persistence": f"{persistence:.1%}"}
                        for concept, persistence in sorted(
                            concept_evolution.concept_persistence.items(),
                            key=lambda x: x[1], reverse=True
                        )[:5]
                    ])
                    st.dataframe(persistence_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Visualization error: {str(e)}")
    
    else:
        # Show informational content when no analysis is run yet
        st.subheader("💡 About Enhanced Concept Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🎯 Concept Clustering**
            - Groups sessions by semantic similarity
            - Uses existing S-BERT sequence embeddings  
            - Identifies coherent themes and topics
            - Shows which sessions belong together
            
            **📈 Temporal Drift Analysis**
            - Tracks concept evolution over time
            - Measures drift magnitude and direction
            - Identifies major conceptual shifts
            - Shows stability vs. change patterns
            """)
        
        with col2:
            st.markdown("""
            **⚡ Velocity & Network Analysis**
            - Measures rate of concept change
            - Maps relationships between clusters
            - Shows concept persistence over time
            - Provides comprehensive dashboards
            
            **🌟 Key Advantages**
            - Leverages existing S-BERT embeddings
            - No misleading PCA projections
            - Interpretable concept-focused views
            - Real semantic understanding
            """)
        
        st.info("🚀 **Ready to analyze?** Choose your settings above and click **'Analyze Concepts'** to begin!")


def render_explainability_dashboard():
    """Render the explainability dashboard with clear guidance and analysis explanations."""
    st.header("💡 Understanding Your Analysis Results")
    
    # Check if we have analysis results to explain
    dataset_info = st.session_state.get('dataset_info', {})
    has_performance_data = 'performance_metrics' in dataset_info
    has_concept_analysis = hasattr(st.session_state, 'concept_evolution')
    has_memory_data = len(st.session_state.get('memory', [])) > 0
    
    if not has_memory_data:
        st.info("📊 Upload data to see explanations and quality assessments.")
        return
    
    # Initialize explainability engine
    try:
        explainability_engine = ExplainabilityEngine()
        
        # Performance Quality Assessment
        st.subheader("🎯 Processing Quality Assessment")
        
        if has_performance_data:
            performance = dataset_info['performance_metrics']
            derived = False
        else:
            session_count = dataset_info.get('session_count', len(st.session_state.get('memory', [])))
            processing_time = dataset_info.get('processing_time', 0) or 0
            sessions_per_second = (session_count / processing_time) if processing_time else 0.0
            performance = {
                "sessions_per_second": sessions_per_second,
                "memory_efficiency": float(session_count) if session_count else 0.0,
                "success_rate": 1.0 if session_count else 0.0,
                "estimated_quality": 0.5 if session_count else 0.0,
            }
            derived = True
        
        if performance['sessions_per_second'] or performance['success_rate']:
            
            # Create quality explanation
            quality_explanation = explainability_engine.explain_processing_quality(
                success_rate=performance['success_rate'],
                memory_efficiency=performance['memory_efficiency'],
                processing_speed=performance['sessions_per_second'],
                estimated_quality=performance['estimated_quality']
            )
            
            # Display quality assessment
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### 📊 What These Numbers Mean")
                st.markdown(f"**Quality Score: {performance['estimated_quality']:.1%}**")
                st.markdown(quality_explanation.what_it_means)
                
                st.markdown("### 🔍 Why These Results?")
                st.markdown(quality_explanation.why_these_results)
                
                if quality_explanation.what_to_do_next:
                    st.markdown("### 💡 What You Can Do")
                    for rec in quality_explanation.what_to_do_next:
                        st.markdown(f"- {rec}")
            
            with col2:
                # Visual quality indicator
                quality_score = performance['estimated_quality']
                if quality_score > 0.8:
                    st.success("🟢 **Excellent Quality**")
                    st.markdown("Your data processed beautifully!")
                elif quality_score > 0.6:
                    st.warning("🟡 **Good Quality**")
                    st.markdown("Solid results with room for improvement.")
                else:
                    st.error("🔴 **Needs Attention**")
                    st.markdown("Several issues detected.")
                
                # Quick metrics
                st.metric("Success Rate", f"{performance['success_rate']:.1%}")
                st.metric("Memory Efficiency", f"{performance['memory_efficiency']:.1f}")
                st.metric("Processing Speed", f"{performance['sessions_per_second']:.1f}/s")
            if derived:
                st.info("ℹ️ Showing inferred quality (no recorded processing metrics). Re-run processing for full assessment.")
        
        else:
            st.info("🔄 Process data to see quality assessment")
        
        st.markdown("---")
        
        # Data Complexity Analysis
        st.subheader("📊 Dataset Complexity Analysis")
        
        if has_performance_data:
            complexity_score = dataset_info.get('complexity_score', 0)
            processing_strategy = dataset_info.get('processing_strategy', 'full_processing')
            
            complexity_explanation = explainability_engine.explain_complexity_score(
                complexity_score, processing_strategy
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Complexity Score", f"{complexity_score:.2f}/1.0")
                if complexity_score > 0.7:
                    st.markdown("🔴 **High Complexity**")
                    st.markdown("Large, diverse dataset")
                elif complexity_score > 0.4:
                    st.markdown("🟡 **Medium Complexity**") 
                    st.markdown("Moderate size and diversity")
                else:
                    st.markdown("🟢 **Low Complexity**")
                    st.markdown("Small, focused dataset")
            
            with col2:
                st.markdown("**Processing Strategy**")
                strategy_display = processing_strategy.replace('_', ' ').title()
                st.markdown(f"🎯 {strategy_display}")
                
                if processing_strategy == "progressive_sampling":
                    st.markdown("Applied intelligent sampling")
                elif processing_strategy == "smart_batching":
                    st.markdown("Used optimized batching")
                else:
                    st.markdown("Full processing applied")
            
            with col3:
                st.markdown("**Recommendation**")
                st.markdown(complexity_explanation.what_it_means)
        
        st.markdown("---")
        
        # Analysis Results Explanation
        st.subheader("🧠 Analysis Results Explainer")
        
        # Concept Analysis Explanation
        if has_concept_analysis:
            concept_evolution = st.session_state.concept_evolution
            
            st.markdown("### 🎯 Concept Analysis Results")
            
            # Explain cluster quality
            n_clusters = len(concept_evolution.concept_clusters)
            n_sessions = concept_evolution.total_sessions
            
            cluster_explanation = explainability_engine.explain_clustering_results(
                n_clusters=n_clusters,
                n_sessions=n_sessions,
                cluster_quality_score=0.7  # You could calculate this from silhouette scores
            )
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📊 Clustering Quality")
                st.markdown(cluster_explanation.what_it_means)
                st.markdown(cluster_explanation.why_these_results)
            
            with col2:
                st.markdown("#### 💡 What This Means")
                for rec in cluster_explanation.what_to_do_next[:3]:
                    st.markdown(f"• {rec}")
            
            # Explain drift patterns
            if concept_evolution.drift_patterns:
                major_drifts = len([d for d in concept_evolution.drift_patterns if d.drift_magnitude > 0.5])
                
                st.markdown("### 📈 Concept Drift Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Major Concept Shifts", major_drifts)
                    if major_drifts > 5:
                        st.markdown("🔄 **High Dynamism** - Your concepts evolve rapidly")
                    elif major_drifts > 2:
                        st.markdown("📊 **Moderate Evolution** - Steady concept development")
                    else:
                        st.markdown("📍 **Stable Concepts** - Consistent themes")
                
                with col2:
                    avg_drift = np.mean([p.drift_magnitude for p in concept_evolution.drift_patterns])
                    st.metric("Average Drift", f"{avg_drift:.3f}")
                    
                    if avg_drift > 0.5:
                        st.markdown("🌊 High conceptual fluidity")
                    elif avg_drift > 0.3:
                        st.markdown("🏃 Moderate concept evolution")
                    else:
                        st.markdown("🏠 Stable conceptual foundation")
        
        else:
            st.info("🧠 Run **Concept Analysis** to see detailed explanations of your semantic patterns")
        
        st.markdown("---")

        # Ragged tensors & masking explainer
        st.subheader("🧵 Ragged Tensors, Padding, and Masking")
        try:
            tokens_per_session = [t.shape[0] for t in st.session_state.memory]
            avg_tokens = np.mean(tokens_per_session) if tokens_per_session else 0
            max_tokens = np.max(tokens_per_session) if tokens_per_session else 0
            padded_ratio = 0.0
            if len(tokens_per_session) > 0 and max_tokens > 0:
                total_slots = len(tokens_per_session) * max_tokens
                total_real = sum(tokens_per_session)
                padded_ratio = 1 - (total_real / total_slots)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg tokens/session", f"{avg_tokens:.0f}")
            with col2:
                st.metric("Max tokens", f"{max_tokens}")
            with col3:
                st.metric("Padding ratio", f"{padded_ratio:.1%}")
            st.caption("We use padding + masks to compute stable session means and token-level PCA without biasing distances.")
        except Exception:
            pass
        
        # Interactive Help Section
        st.subheader("❓ Frequently Asked Questions")
        
        with st.expander("🤔 **Why is my success rate low?**"):
            st.markdown("""
            **Common Reasons:**
            - **Noisy Data**: Text contains many non-readable characters or formatting
            - **Empty Sessions**: Some rows have no meaningful text content
            - **Encoding Issues**: File encoding problems (try UTF-8)
            - **Large File Size**: Memory constraints causing processing failures
            
            **Solutions:**
            - Clean your data before upload
            - Use intelligent sampling for large datasets
            - Check file encoding and format
            """)
        
        with st.expander("📊 **How do I improve clustering quality?**"):
            st.markdown("""
            **For Better Clusters:**
            - **More Data**: At least 20-50 sessions for meaningful clusters
            - **Consistent Topics**: Data should have some thematic coherence
            - **Sufficient Text**: Each session should have meaningful content
            - **Optimal Cluster Number**: Try different cluster numbers (3-15 typically work well)
            
            **Red Flags:**
            - Too many tiny clusters → Reduce cluster count
            - One giant cluster → Increase cluster count or check data diversity
            """)
        
        with st.expander("🧠 **What does high concept drift mean?**"):
            st.markdown("""
            **High Drift Indicates:**
            - **Learning Journey**: You're exploring new topics over time
            - **Project Evolution**: Your focus is shifting as you progress
            - **Conversation Dynamics**: AI conversations evolving in complexity
            
            **This is Often GOOD:**
            - Shows intellectual growth
            - Indicates active exploration
            - Demonstrates learning progression
            
            **Only Concerning If:**
            - You expected consistency but see chaos
            - The drift doesn't match your memory of the conversations
            """)
        
        with st.expander("⚡ **My processing is slow - what can I do?**"):
            st.markdown("""
            **Speed Optimization:**
            - **Use Sampling**: For datasets >1000 sessions, try intelligent sampling
            - **Smaller Batches**: Process in smaller chunks
            - **Close Other Apps**: Free up system memory
            - **Check File Size**: Very large files may need preprocessing
            
            **Memory Optimization:**
            - **Progressive Analysis**: Build complexity gradually
            - **Quality Over Quantity**: Sometimes less data gives better insights
            - **System Resources**: 16GB+ RAM recommended for large datasets
            """)
        
        # Performance Recommendations
        st.subheader("🚀 Performance Recommendations")
        
        if has_performance_data:
            performance = dataset_info['performance_metrics']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🎯 For Your Next Upload")
                
                if performance['success_rate'] < 0.9:
                    st.markdown("- 🧹 **Clean your data** - Remove empty rows and formatting issues")
                
                if performance['memory_efficiency'] < 2.0:
                    st.markdown("- 📊 **Use sampling** - Try intelligent sampling for large datasets")
                
                if performance['sessions_per_second'] < 1.0:
                    st.markdown("- ⚡ **Optimize processing** - Close other applications to free memory")
                
                st.markdown("- 📈 **Gradual complexity** - Start with small datasets, build up")
            
            with col2:
                st.markdown("### 💡 Analysis Strategy")
                
                dataset_size = dataset_info.get('session_count', 0)
                
                if dataset_size > 500:
                    st.markdown("- 🎯 **Use progressive sampling** for initial exploration")
                    st.markdown("- 🔄 **Smart batching** for full analysis")
                elif dataset_size > 100:
                    st.markdown("- 📊 **Full processing** with quality monitoring")
                    st.markdown("- 🧠 **Focus on concept evolution** analysis")
                else:
                    st.markdown("- ✅ **Perfect size** for comprehensive analysis")
                    st.markdown("- 🌟 **Try all analysis methods** to compare")
    
    except Exception as e:
        st.error(f"Error initializing explainability engine: {str(e)}")
        st.info("💡 The explainability features require the analysis modules to be properly loaded.")


def main():
    """Main application with clean, intuitive interface."""
    if _missing_torch_error is not None:
        st.error("PyTorch is required to run Semantic Tensor Analysis. Install with `pip install torch` (CPU is fine).")
        return

    # Clean up any rerun flags
    if st.session_state.get("csv_imported", False):
        st.session_state["csv_imported"] = False
    
    # Check if we have data
    has_data = len(st.session_state.get('memory', [])) > 0
    
    if not has_data:
        # Ensure sidebar is allowed to show expanded on fresh/no-data state
        st.session_state['sidebar_minimized_after_load'] = False
        # Show upload screen for new users
        render_upload_screen()
    else:
        # Show main app interface
        collapse_sidebar_once_after_load()
        render_simple_sidebar()
        
        # Main tabs (now with Enhanced Concept Analysis and Explainability)
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "🏠 Overview", 
            "🌊 Evolution", 
            "🔍 Patterns", 
            "📐 Dimensionality",
            "⏱️ Temporal",
            "🧠 Concepts",
            "💡 Explain",
            "🤖 AI Insights"
        ])
        
        with tab1:
            render_overview_dashboard()
        
        with tab2:
            render_semantic_evolution_tab()
        
        with tab3:
            render_pattern_analysis_tab()
        
        with tab4:
            render_dimensionality_tab()
        
        with tab5:
            # Temporal multi-resolution analysis (fallback message if no chat data)
            try:
                from semantic_tensor_analysis.app.tabs.temporal import render_temporal_analysis_tab
                render_temporal_analysis_tab()
            except Exception as exc:
                st.info(f"Temporal analysis unavailable: {exc}")
        
        with tab6:
            render_enhanced_concept_analysis_tab()
        
        with tab7:
            render_explainability_dashboard()
        
        with tab8:
            st.header("🤖 AI-Powered Analysis")
            render_comprehensive_chat_analysis()


def render_temporal_zoom_interface():
    """Render the temporal zoom interface for multi-resolution analysis."""
    st.subheader("🔍 Temporal Resolution Control")
    
    if 'temporal_manager' not in st.session_state:
        st.warning("Multi-resolution data not available")
        return
    
    temporal_manager = st.session_state.temporal_manager
    from temporal_resolution_manager import TemporalResolution
    
    # Get navigation options
    nav_options = temporal_manager.get_temporal_navigation_options()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Resolution selector
        resolution_options = {
            "🔬 Turn Level": TemporalResolution.TURN,
            "💬 Conversation Level": TemporalResolution.CONVERSATION, 
            "📅 Day Level": TemporalResolution.DAY
        }
        
        selected_resolution_name = st.selectbox(
            "Analysis Resolution",
            options=list(resolution_options.keys()),
            index=0,  # Default to turn level
            help="Choose the temporal granularity for analysis"
        )
        selected_resolution = resolution_options[selected_resolution_name]
    
    with col2:
        # Date filter (if applicable)
        focus_date = None
        if nav_options['available_dates']:
            focus_date = st.selectbox(
                "Focus Date (Optional)",
                options=[None] + nav_options['available_dates'],
                format_func=lambda x: "All Dates" if x is None else str(x),
                help="Filter to specific date"
            )
    
    with col3:
        # Conversation filter (if applicable and turn level selected)
        focus_conversation = None
        if selected_resolution == TemporalResolution.TURN and nav_options['available_conversations']:
            conv_options = ["All Conversations"] + nav_options['available_conversations']
            selected_conv = st.selectbox(
                "Focus Conversation (Optional)",
                options=conv_options,
                help="Filter to specific conversation"
            )
            if selected_conv != "All Conversations":
                focus_conversation = selected_conv
    
    # Apply zoom if resolution changed
    current_res = st.session_state.get('current_temporal_resolution')
    current_date = st.session_state.get('current_focus_date')
    current_conv = st.session_state.get('current_focus_conversation')
    
    if (current_res != selected_resolution or 
        current_date != focus_date or 
        current_conv != focus_conversation):
        
        # Apply zoom
        resolution_data = temporal_manager.zoom_to_resolution(
            selected_resolution, 
            focus_date=focus_date,
            focus_conversation=focus_conversation
        )
        
        # Update session state
        st.session_state.current_temporal_resolution = selected_resolution
        st.session_state.current_focus_date = focus_date
        st.session_state.current_focus_conversation = focus_conversation
        st.session_state.current_resolution_data = resolution_data
        
        # Update the main memory and meta for compatibility with existing visualizations
        if resolution_data['embeddings']:
            # Convert embeddings to legacy format
            legacy_embeddings = []
            new_meta = []
            
            for i, (embedding, meta) in enumerate(zip(resolution_data['embeddings'], resolution_data['metadata'])):
                # Convert tensor to expected shape [tokens, dim]
                if len(embedding.shape) == 1:
                    legacy_emb = embedding.unsqueeze(0)  # [1, dim]
                else:
                    legacy_emb = embedding
                
                legacy_embeddings.append(legacy_emb)
                
                # Create metadata
                new_meta.append({
                    'text': meta.get('user_text', meta.get('text', '')),
                    'session_id': meta.get('turn_id', meta.get('session_id', f'item_{i}')),
                    'timestamp': str(meta.get('timestamp', meta.get('date', ''))),
                    'resolution': selected_resolution.value,
                    'tokens': legacy_emb.shape[0],
                    **meta  # Include original metadata
                })
            
            # Update session state
            st.session_state.memory = legacy_embeddings
            st.session_state.meta = new_meta
            
            st.success(f"🎯 Zoomed to {selected_resolution.value} level: {len(legacy_embeddings)} items")
            
            if focus_date:
                st.info(f"📅 Filtered to date: {focus_date}")
            if focus_conversation:
                st.info(f"💬 Filtered to conversation: {focus_conversation}")
        
        st.rerun()


def render_multi_resolution_summary():
    """Render summary of multi-resolution processing results."""
    if 'multi_res_summary' not in st.session_state:
        return
    
    summary = st.session_state.multi_res_summary
    
    with st.expander("🎯 Multi-Resolution Analysis Summary", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Conversation Turns", summary['total_turns'])
        with col2:
            st.metric("Conversations", summary['total_conversations'])
        with col3:
            st.metric("Days Covered", summary['total_days'])
        with col4:
            st.metric("Temporal Span", f"{summary['temporal_span_days']} days")
        
        st.info(f"🔍 Current view: **{summary['current_resolution'].title()} Level** - Use controls above to zoom in/out")


if __name__ == "__main__":
    main() 
