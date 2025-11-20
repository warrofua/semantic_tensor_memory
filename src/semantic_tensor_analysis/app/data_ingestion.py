"""Data ingestion utilities for the Streamlit app."""

from __future__ import annotations

import csv
import io
import os
import time
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from semantic_tensor_analysis.optimization.performance import AdaptiveDataProcessor
from semantic_tensor_analysis.memory.universal_core import UniversalEmbedding

from .models import (
    cleanup_memory,
    get_cached_text_embedder,
    get_cached_universal_store,
    get_memory_usage,
)
from .services import parse_chat_history


__all__ = [
    "detect_file_type_and_content",
    "convert_ai_conversation_to_sessions",
    "handle_unified_upload",
    "process_unified_sessions",
    "render_upload_screen",
]

AI_CONVERSATION_MAX_USER_MESSAGES = 1200


def _limit_ai_conversation_messages(messages, max_user_messages: int):
    """Cap AI conversation size to avoid memory blow-ups."""
    user_messages = [m for m in messages if getattr(m, "role", "").lower() == "user"]
    capped = len(user_messages) > max_user_messages
    if not capped:
        return messages, {
            "was_capped": False,
            "original_user_messages": len(user_messages),
            "kept_user_messages": len(user_messages),
            "kept_total_messages": len(messages),
            "stride": 1,
        }

    stride = max(1, len(user_messages) // max_user_messages)
    sampled_users = user_messages[::stride][:max_user_messages]
    sampled_ids = {id(msg) for msg in sampled_users}

    limited_messages = []
    attach_assistant = False
    kept_assistants = 0

    for msg in messages:
        role = getattr(msg, "role", "").lower()
        if role == "user" and id(msg) in sampled_ids:
            limited_messages.append(msg)
            attach_assistant = True  # keep the immediate assistant reply for context
        elif attach_assistant and role == "assistant":
            limited_messages.append(msg)
            kept_assistants += 1
            attach_assistant = False
        else:
            if role == "assistant":
                attach_assistant = False

    return limited_messages, {
        "was_capped": True,
        "original_user_messages": len(user_messages),
        "kept_user_messages": len(sampled_users),
        "kept_total_messages": len(limited_messages),
        "stride": stride,
        "kept_assistants": kept_assistants,
    }


def detect_file_type_and_content(uploaded_file) -> Tuple[str, str]:
    """Intelligently detect file type and content format."""
    file_name = uploaded_file.name.lower()
    content = uploaded_file.read()
    uploaded_file.seek(0)

    try:
        content_str = content.decode("utf-8")
    except Exception:
        return "unknown", "unknown"

    if file_name.endswith(".csv"):
        try:
            csv_reader = csv.DictReader(io.StringIO(content_str))
            first_row = next(csv_reader, {})
            if "text" in first_row:
                return "csv", "csv_sessions"
            return "csv", "unknown"
        except Exception:
            return "csv", "unknown"

    if file_name.endswith(".json"):
        try:
            import json

            data = json.loads(content_str)
            if isinstance(data, list) or (isinstance(data, dict) and "mapping" in data):
                return "json", "ai_conversation"
            return "json", "unknown"
        except Exception:
            return "json", "unknown"

    if file_name.endswith(".txt"):
        lines = content_str.split("\n")
        conversation_indicators = 0
        for line in lines[:20]:
            line_lower = line.strip().lower()
            if any(
                pattern in line_lower
                for pattern in [
                    "you:",
                    "user:",
                    "human:",
                    "assistant:",
                    "ai:",
                    "chatgpt:",
                    "claude:",
                    "**you**:",
                    "**assistant**:",
                    "> ",
                ]
            ):
                conversation_indicators += 1
        if conversation_indicators >= 2:
            return "txt", "ai_conversation"
        return "txt", "unknown"

    return "unknown", "unknown"


def convert_ai_conversation_to_sessions(
    messages, apply_limit: bool = True, max_user_messages: int = AI_CONVERSATION_MAX_USER_MESSAGES
) -> List[Dict[str, Any]]:
    """Convert chat messages to CSV-like session dictionaries."""
    if apply_limit:
        messages, _ = _limit_ai_conversation_messages(messages, max_user_messages)

    sessions: List[Dict[str, Any]] = []
    user_messages = [msg for msg in messages if getattr(msg, "role", "") == "user"]

    for i, msg in enumerate(user_messages):
        raw_timestamp = getattr(msg, "timestamp", None)
        if raw_timestamp is not None:
            iso_formatter = getattr(raw_timestamp, "isoformat", None)
            timestamp = iso_formatter() if callable(iso_formatter) else raw_timestamp
        else:
            timestamp = None

        sessions.append(
            {
                "text": msg.content,
                "session_id": i,
                "timestamp": timestamp,
                "conversation_id": getattr(msg, "conversation_id", "unknown"),
                "message_id": getattr(msg, "message_id", f"msg_{i}"),
                "source_type": "ai_conversation",
                "role": msg.role,
            }
        )

    return sessions


def handle_unified_upload(uploaded_file) -> bool:
    """Unified handler that processes both AI conversations and CSV data."""
    try:
        file_type, content_type = detect_file_type_and_content(uploaded_file)
        st.info(f"🔍 Detected: {file_type.upper()} file with {content_type.replace('_', ' ')}")

        if content_type == "ai_conversation":
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            try:
                with st.spinner("🤖 Processing AI conversation data..."):
                    status_text.info("📥 Reading file...")
                    file_content = uploaded_file.read().decode("utf-8")
                    progress_bar.progress(0.2)

                    status_text.info("🧩 Parsing conversation...")
                    messages = parse_chat_history(file_content)
                    progress_bar.progress(0.4)

                    if not messages:
                        st.error("❌ No conversation messages found in the uploaded file")
                        return False

                    status_text.info("🎯 Applying safety limits...")
                    messages, cap_info = _limit_ai_conversation_messages(
                        messages, AI_CONVERSATION_MAX_USER_MESSAGES
                    )
                    progress_bar.progress(0.6)
                    if cap_info["was_capped"]:
                        st.warning(
                            f"⚡ Large conversation detected ({cap_info['original_user_messages']:,} user messages). "
                            f"Sampled to {cap_info['kept_user_messages']:,} messages "
                            f"(kept every ~{cap_info['stride']}th message plus immediate replies) "
                            "to prevent memory issues."
                        )

                    status_text.info("📊 Converting to sessions...")
                    st.session_state["uploaded_chat_messages"] = messages
                    session_data = convert_ai_conversation_to_sessions(
                        messages, apply_limit=False
                    )
                    progress_bar.progress(0.85)
                    if len(session_data) < 2:
                        st.error("❌ Need at least 2 user messages for analysis")
                        return False
            finally:
                progress_bar.progress(1.0)
                progress_bar.empty()
                status_text.empty()
        elif content_type == "csv_sessions":
            with st.spinner("📊 Processing CSV session data..."):
                csv_data = uploaded_file.read().decode("utf-8")
                reader = csv.DictReader(io.StringIO(csv_data))
                session_data = list(reader)
                if not session_data:
                    st.error("❌ The CSV file appears to be empty")
                    return False
                text_found = any("text" in row and row["text"].strip() for row in session_data)
                if not text_found:
                    st.error("❌ No 'text' column with content found in CSV")
                    return False
        else:
            st.error("❌ Unsupported file format. Please upload:")
            st.markdown(
                """
                - **CSV files** with a 'text' column
                - **JSON files** from ChatGPT exports
                - **TXT files** with conversation format
                """
            )
            return False

        return process_unified_sessions(session_data, uploaded_file.name, content_type)
    except Exception as exc:  # pragma: no cover - UI feedback path
        st.error(f"❌ Failed to process {uploaded_file.name}: {str(exc)}")
        return False


def process_unified_sessions(session_data, filename: str, content_type: str) -> bool:
    """Process unified session data with adaptive performance optimization."""
    try:
        adaptive_processor = AdaptiveDataProcessor()
        profile = adaptive_processor.profile_dataset(session_data)

        st.info(
            f"""
        📊 **Dataset Analysis**:
        - **{profile.total_sessions:,} sessions** (~{profile.avg_tokens_per_session} tokens each)
        - **Complexity Score**: {profile.complexity_score:.2f}/1.0
        - **Strategy**: {profile.processing_strategy.replace('_', ' ').title()}
        - **Estimated Memory**: {profile.estimated_memory_mb:.0f}MB
        """
        )

        if profile.processing_strategy == "progressive_sampling":
            st.warning("🎯 **Large Dataset Detected**: Using intelligent sampling for optimal performance")

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(
                    f"🔬 Smart Sample ({profile.recommended_batch_size} sessions)", type="primary"
                ):
                    session_data, selected_indices = adaptive_processor.apply_intelligent_sampling(
                        session_data, profile.recommended_batch_size
                    )
                    st.success(
                        f"✅ Applied intelligent sampling: {len(session_data)} sessions selected"
                    )
                else:
                    return False

            with col2:
                if st.button(f"📊 Process First {profile.recommended_batch_size}"):
                    session_data = session_data[: profile.recommended_batch_size]
                    selected_indices = list(range(profile.recommended_batch_size))
                    st.info(f"✅ Processing first {len(session_data)} sessions")
                else:
                    return False

            with col3:
                if st.button("🔄 Cancel & Try Smaller File"):
                    st.info("Consider splitting your data or using the sampling option")
                    return False

            return False

        if profile.processing_strategy == "smart_batching":
            st.info(
                f"🔄 **Smart Batching**: Processing in optimized batches of {profile.recommended_batch_size}"
            )
        else:
            st.success("✅ **Full Processing**: Dataset size is optimal for direct processing")

        start_time = time.time()
        start_memory = get_memory_usage()

        memory: List = []
        meta: List[Dict[str, Any]] = []

        progress_bar = st.progress(0)
        status_text = st.empty()
        performance_metrics = st.empty()

        text_embedder = get_cached_text_embedder()
        universal_store = get_cached_universal_store()
        universal_store.clear()

        valid_sessions = 0
        skipped_sessions = 0
        batch_size = min(profile.recommended_batch_size, 50)
        memory_threshold = min(
            2000, int(adaptive_processor.available_memory_gb * 1024 * 0.5)
        )

        processing_quality = {
            "successful_embeddings": 0,
            "failed_embeddings": 0,
            "skipped_short": 0,
        }

        for batch_start in range(0, len(session_data), batch_size):
            batch_end = min(batch_start + batch_size, len(session_data))
            batch = session_data[batch_start:batch_end]

            for session in batch:
                text = session.get("text", "").strip()
                if not text:
                    skipped_sessions += 1
                    processing_quality["skipped_short"] += 1
                    continue

                try:
                    embedding = text_embedder.process_raw_data(
                        text,
                        session_id=session.get("session_id", str(valid_sessions)),
                    )
                    universal_store.add_session(embedding)
                    st.session_state.active_modalities.add("text")

                    # Store legacy tensor for compatibility (avoid pickling UniversalEmbedding)
                    legacy_tensor = embedding.event_embeddings
                    session["tokens"] = legacy_tensor.shape[0] if legacy_tensor is not None else 0
                    memory.append(legacy_tensor)
                    meta.append(session)
                    valid_sessions += 1
                    processing_quality["successful_embeddings"] += 1
                except Exception:
                    skipped_sessions += 1
                    processing_quality["failed_embeddings"] += 1

                current_memory = get_memory_usage()
                if current_memory > memory_threshold:
                    st.warning(
                        f"🧠 Memory usage high ({current_memory:.0f}MB). Consider sampling for larger datasets."
                    )
                    cleanup_memory()

            progress_pct = batch_end / len(session_data)
            progress_bar.progress(progress_pct)

            current_memory = get_memory_usage()
            elapsed_time = time.time() - start_time
            estimated_total_time = elapsed_time / progress_pct if progress_pct > 0 else 0
            eta = estimated_total_time - elapsed_time if estimated_total_time > elapsed_time else 0
            batch_efficiency = processing_quality["successful_embeddings"] / max(valid_sessions, 1)

            status_text.markdown(
                f"""
            **Processing Batch {batch_start // batch_size + 1}**: {batch_start + 1}-{batch_end} of {len(session_data)}
            - ✅ **Sessions**: {valid_sessions} processed, {skipped_sessions} skipped
            - 🧠 **Memory**: {current_memory:.0f}MB ({(current_memory/start_memory-1)*100:+.0f}%)
            - ⏱️ **Time**: {elapsed_time:.1f}s elapsed, ~{eta:.0f}s remaining
            - 📊 **Batch Efficiency**: {batch_efficiency:.1%}
            """
            )

            with performance_metrics.container():
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Progress", f"{progress_pct:.1%}")
                with col2:
                    st.metric("Memory Usage", f"{current_memory:.0f}MB", f"{current_memory-start_memory:+.0f}MB")
                with col3:
                    st.metric("Processing Rate", f"{valid_sessions/elapsed_time:.1f}/s")
                with col4:
                    success_rate = processing_quality["successful_embeddings"] / max(
                        1,
                        processing_quality["successful_embeddings"]
                        + processing_quality["failed_embeddings"],
                    )
                    st.metric("Success Rate", f"{success_rate:.1%}")

            time.sleep(0.01)

        progress_bar.empty()
        status_text.empty()
        performance_metrics.empty()

        if not memory:
            st.error(f"❌ No valid text data found in {filename}")
            return False

        total_time = time.time() - start_time
        final_memory = get_memory_usage()
        memory_efficiency = (
            len(memory) / (final_memory - start_memory)
            if final_memory > start_memory
            else 1
        )

        st.session_state.memory = memory
        st.session_state.meta = meta
        st.session_state.universal_store = universal_store
        from semantic_tensor_analysis.memory.store import save
        def _token_count(session):
            if isinstance(session, UniversalEmbedding):
                if session.event_embeddings is not None:
                    return session.event_embeddings.shape[0]
                return 0
            # Fallback for legacy tensors
            try:
                return session.shape[0]
            except Exception:
                return 0

        st.session_state.dataset_info = {
            "source": content_type,
            "filename": filename,
            "upload_timestamp": time.time(),
            "session_count": len(memory),
            "total_tokens": sum(_token_count(m) for m in memory),
            "universal_sessions": len(universal_store.embeddings),
            "active_modalities": list(st.session_state.active_modalities),
            "memory_usage": final_memory,
            "processing_time": total_time,
            "processing_strategy": profile.processing_strategy,
            "complexity_score": profile.complexity_score,
            "performance_metrics": {
                "sessions_per_second": len(memory) / total_time,
                "memory_efficiency": memory_efficiency,
                "success_rate": processing_quality["successful_embeddings"]
                / max(1, valid_sessions + skipped_sessions),
                "estimated_quality": min(
                    1.0,
                    (processing_quality["successful_embeddings"] / max(1, valid_sessions + skipped_sessions))
                    * memory_efficiency,
                ),
            },
        }

        # Only save legacy tensor format when applicable; UniversalEmbedding is already persisted via UniversalMemoryStore.
        if memory and not isinstance(memory[0], UniversalEmbedding):
            save(memory, meta)
        else:
            st.info("💾 Sessions stored via Universal Memory; legacy tensor save skipped.")

        st.success(
            f"""
        🎉 **Processing Complete!**
        - **Processed**: {valid_sessions:,} sessions in {total_time:.1f}s
        - **Performance**: {len(memory)/total_time:.1f} sessions/second
        - **Memory**: {final_memory:.0f}MB final ({final_memory-start_memory:+.0f}MB change)
        - **Quality**: {processing_quality['successful_embeddings']/(valid_sessions+skipped_sessions):.1%} success rate
        """
        )
        st.session_state["upload_success"] = True
        if st.button("➡️ Go to analysis", type="primary"):
            st.rerun()

        if profile.processing_strategy != "full_processing":
            st.info(
                f"🎯 **Strategy: {profile.processing_strategy.replace('_', ' ').title()}** - Optimized for your system capabilities"
            )

        memory_increase = final_memory - start_memory
        if memory_increase > 1000:
            st.warning(
                f"📊 High memory usage: +{memory_increase:.0f}MB. Consider using sampling for larger datasets."
            )
        elif (
            processing_quality["successful_embeddings"]
            / max(1, valid_sessions + skipped_sessions)
            < 0.8
        ):
            st.warning(
                f"⚠️ Lower success rate ({processing_quality['successful_embeddings']/(valid_sessions+skipped_sessions):.1%}). Check data quality."
            )

        return True
    except Exception as exc:
        st.error(f"❌ Failed to process session data: {str(exc)}")
        cleanup_memory()
        return False


def render_upload_screen() -> None:
    """Render the unified upload interface."""
    # If a dataset was just loaded, offer a quick way back to the app surface
    if st.session_state.get("upload_success") and len(st.session_state.get("memory", [])) > 0:
        st.success("✅ Dataset loaded. Continue to analysis.")
        if st.button("➡️ Go to analysis", type="primary"):
            st.session_state.pop("upload_success", None)
            st.rerun()

    st.markdown(
        """
    <div style="text-align: center; padding: 2rem 0;">
        <h1>🌐 Semantic Tensor Analysis</h1>
        <h3 style="color: #666;">Analyze how meaning evolves across conversations and sessions</h3>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("### 📁 **Upload Your Data**")
        st.markdown(
            "Upload **any** text-based data for semantic analysis - we'll detect the format automatically!"
        )

        uploaded_file = st.file_uploader(
            "Choose your file",
            type=["csv", "json", "txt"],
            help="Supports: CSV files, ChatGPT/AI conversation exports (JSON/TXT), or any text data",
        )

        if uploaded_file is not None and handle_unified_upload(uploaded_file):
            st.rerun()

        st.markdown("---")

        with st.expander("📋 **Supported Formats**"):
            st.markdown(
                """
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
            """
            )

        st.markdown("### 🎯 Try Example Datasets")
        st.caption("Loads the bundled `ultimate_demo_dataset.csv` sample journey.")
        if st.button("📚 Load Demo Dataset", type="primary"):
            demo_path = os.path.join("data", "ultimate_demo_dataset.csv")
            if os.path.exists(demo_path):
                with open(demo_path, "rb") as file_handle:
                    content = file_handle.read()
                import io
                mock_file = io.BytesIO(content)
                mock_file.name = "ultimate_demo_dataset.csv"  # type: ignore[attr-defined]
                if handle_unified_upload(mock_file):
                    st.rerun()
            else:
                st.error(
                    f"Demo dataset not found at {demo_path}. Please upload your own file."
                )

        st.markdown(
            """
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
        """,
            unsafe_allow_html=True,
        )
