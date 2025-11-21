"""Persistent sidebar chat panel with LLM configuration.

This module provides a context-aware chat interface that appears in the sidebar
across all tabs, allowing users to ask questions about any visualization or analysis.
"""

import os
import streamlit as st
from pathlib import Path
from typing import Optional, Dict, Any, Iterator, List
import streamlit.components.v1 as components
import base64
import plotly.io as pio

# Try to import file browser components
try:
    from streamlit_file_browser import st_file_browser
    FILE_BROWSER_AVAILABLE = True
except ImportError:
    FILE_BROWSER_AVAILABLE = False

# Try to import tkinter for native file dialog
# Note: Disabled due to threading issues with Streamlit on macOS
# try:
#     import tkinter as tk
#     from tkinter import filedialog
#     TKINTER_AVAILABLE = True
# except ImportError:
#     TKINTER_AVAILABLE = False
TKINTER_AVAILABLE = False  # Disabled - causes "Notifier not initialized" error on macOS

# Import LLM components
try:
    from semantic_tensor_analysis.chat.llama_cpp_analyzer import (
        is_llama_cpp_available,
        get_recommended_models
    )
    from semantic_tensor_analysis.chat.unified_analyzer import UnifiedLLMAnalyzer
    from semantic_tensor_analysis.chat.analysis import (
        create_semantic_insights_prompt,
        create_targeted_insights_prompt,
        stream_unified_response,
    )
    from semantic_tensor_analysis.streamlit.utils import (
        collect_comprehensive_analysis_data,
    )
    CHAT_AVAILABLE = True
except ImportError:
    CHAT_AVAILABLE = False
    # Fallback stubs to avoid runtime NameErrors when chat stack is unavailable
    def is_llama_cpp_available() -> bool:  # type: ignore
        return False

    def get_recommended_models() -> list:  # type: ignore
        return []

    create_semantic_insights_prompt = lambda *args, **kwargs: ""  # type: ignore
    create_targeted_insights_prompt = lambda *args, **kwargs: []  # type: ignore
    stream_unified_response = None  # type: ignore
    def collect_comprehensive_analysis_data():  # type: ignore
        return {}


def _build_sidebar_prompt(base_context: Dict[str, Any], user_question: Optional[str], deep: bool) -> str:
    """Construct a sidebar prompt aligned with the AI Insights tab."""
    base_prompt = create_semantic_insights_prompt(base_context)
    if deep:
        targeted = create_targeted_insights_prompt(base_context)
        base_prompt = base_prompt + "\n\n" + "\n".join(targeted)
    if user_question:
        base_prompt += f"\n\nUSER QUESTION:\n{user_question}\n\nGround your answer in the analysis above."
    return base_prompt


def _get_latest_plotly_snapshot() -> Optional[str]:
    """Return a base64 PNG of the most recent Plotly figure, if available."""
    figs = st.session_state.get("recent_plotly_figs") or []
    if not figs:
        return None
    fig = figs[-1]
    try:
        img_bytes = pio.to_image(fig, format="png", width=1200, height=800, scale=1)
        return base64.b64encode(img_bytes).decode("utf-8")
    except Exception:
        return None


def _stream_sidebar_response(prompt: str, include_snapshot: bool) -> None:
    """Send prompt to LLM with safety checks; renders once to reduce websocket churn."""
    backend = st.session_state.get('llm_backend', 'none')
    backend_config = st.session_state.get('llm_backend_config', {})

    if include_snapshot:
        snapshot = st.session_state.get("page_snapshot_b64")
        if snapshot:
            backend_config = {**backend_config, "image_base64": snapshot}
        else:
            st.info("📸 Snapshot not available; sending text-only this time.")
            include_snapshot = False

    if include_snapshot and backend != "llama_server":
        st.warning("📸 Page snapshots require the llama.cpp server backend (vision-capable model).")
        return

    if backend == 'none':
        response = "AI assistance is disabled. Please select an LLM backend above to enable chat."
        st.session_state['sidebar_chat_history'].append(("assistant", response))
        return
    if not CHAT_AVAILABLE or stream_unified_response is None:
        response = "Chat components are unavailable in this environment."
        st.session_state['sidebar_chat_history'].append(("assistant", response))
        return
    if backend == "llama_server" and not backend_config.get("server_url"):
        response = "Please provide the llama-server URL (e.g., http://localhost:8080)."
        st.session_state['sidebar_chat_history'].append(("assistant", response))
        return
    if backend == "llama_cpp":
        model_path = backend_config.get("model_path") or st.session_state.get("llama_cpp_model_path")
        if not model_path or not Path(model_path).exists():
            st.warning("Please specify a valid path to a GGUF model file in the configuration above.")
            return

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            # Collect once to avoid websocket flooding/freezes
            response = "".join(stream_unified_response(prompt, backend=backend, **backend_config))
            placeholder.markdown(response)
            st.session_state['sidebar_chat_history'].append(("assistant", response))
        except Exception as e:
            # Fallback: if vision not supported, retry without image
            msg = str(e)
            if include_snapshot and "image input is not supported" in msg.lower():
                st.warning("📸 Backend does not support images (mmproj missing). Sending text-only.")
                try:
                    response = "".join(stream_unified_response(prompt, backend=backend))
                    placeholder.markdown(response)
                    st.session_state['sidebar_chat_history'].append(("assistant", response))
                    return
                except Exception as e2:
                    msg = str(e2)
            error_msg = f"Error: {e}"
            placeholder.error(error_msg)
            st.session_state['sidebar_chat_history'].append(("assistant", error_msg))


def _find_local_models(search_roots: Optional[List[Path]] = None, limit: int = 80) -> List[str]:
    """Return a list of GGUF model paths by scanning common locations."""
    extra_roots: List[Path] = []
    env_dirs = os.environ.get("LLM_MODEL_DIRS")
    if env_dirs:
        extra_roots.extend(Path(p).expanduser() for p in env_dirs.split(os.pathsep) if p)

    roots = search_roots or [
        Path.home() / "models",
        Path.home() / "Downloads",
    ]
    roots.extend(extra_roots)

    candidates: List[str] = []
    for root in roots:
        try:
            if not root.exists() or not root.is_dir():
                continue

            # Check root and immediate subdirectories to avoid slow deep scans
            paths_to_scan = [root]
            for child in root.iterdir():
                if child.is_dir():
                    paths_to_scan.append(child)

            for path in paths_to_scan:
                for gguf in path.glob("*.gguf"):
                    candidates.append(str(gguf))
                    if len(candidates) >= limit:
                        return sorted(set(candidates))
        except Exception:
            # Fail silently; this is a best-effort helper
            continue

    return sorted(set(candidates))


def select_file_with_tkinter(initial_dir: Optional[str] = None, file_types: Optional[list] = None) -> Optional[str]:
    """Open native OS file dialog using tkinter.

    Args:
        initial_dir: Initial directory to open
        file_types: List of file type tuples, e.g., [("GGUF files", "*.gguf")]

    Returns:
        Selected file path or None if cancelled
    """
    if not TKINTER_AVAILABLE:
        return None

    try:
        # Create root window and hide it immediately
        root = tk.Tk()
        root.withdraw()

        # Make sure the dialog appears on top
        root.lift()
        root.attributes('-topmost', True)
        root.after_idle(root.attributes, '-topmost', False)

        if file_types is None:
            file_types = [("GGUF Model Files", "*.gguf"), ("All files", "*.*")]

        file_path = filedialog.askopenfilename(
            parent=root,
            initialdir=initial_dir or str(Path.home()),
            title="Select GGUF Model File",
            filetypes=file_types
        )

        # Properly destroy the root window
        root.update()
        root.destroy()

        return file_path if file_path else None
    except Exception as e:
        st.error(f"Error opening file dialog: {e}")
        return None


def select_folder_with_tkinter(initial_dir: Optional[str] = None) -> Optional[str]:
    """Open native OS folder dialog using tkinter.

    Args:
        initial_dir: Initial directory to open

    Returns:
        Selected folder path or None if cancelled
    """
    if not TKINTER_AVAILABLE:
        return None

    try:
        # Create root window and hide it immediately
        root = tk.Tk()
        root.withdraw()

        # Make sure the dialog appears on top
        root.lift()
        root.attributes('-topmost', True)
        root.after_idle(root.attributes, '-topmost', False)

        folder_path = filedialog.askdirectory(
            parent=root,
            initialdir=initial_dir or str(Path.home()),
            title="Select Model Directory"
        )

        # Properly destroy the root window
        root.update()
        root.destroy()

        return folder_path if folder_path else None
    except Exception as e:
        st.error(f"Error opening folder dialog: {e}")
        return None


def render_llm_config_sidebar():
    """Render minimal, auto-configured LLM sidebar (llama.cpp server)."""

    with st.sidebar:
        st.markdown("---")
        st.subheader("🤖 AI Assistant")

        # Fixed llama.cpp server configuration (no user controls)
        st.session_state["llm_backend"] = "llama_server"
        st.session_state["llama_server_url"] = "http://localhost:8080"
        st.session_state["llama_server_model"] = "local"
        st.session_state["llm_backend_config"] = {
            "server_url": st.session_state["llama_server_url"],
            "server_model": st.session_state["llama_server_model"],
        }
        st.caption("Using llama.cpp server at http://localhost:8080 (model 'local').")


def render_sidebar_chat(context: Optional[Dict[str, Any]] = None):
    """Render persistent chat panel in sidebar.

    Args:
        context: Optional context about current page/visualization
    """
    # Ensure snapshot key exists and allow deferred resets before widgets instantiate.
    if "page_snapshot_b64" not in st.session_state:
        st.session_state["page_snapshot_b64"] = ""
    if st.session_state.pop("reset_page_snapshot", False):
        st.session_state["page_snapshot_b64"] = ""

    with st.sidebar:
        st.markdown("---")
        st.subheader("💬 Ask AI")

        # Initialize chat history
        if 'sidebar_chat_history' not in st.session_state:
            st.session_state['sidebar_chat_history'] = []

        # Chat container with fixed height
        chat_container = st.container()

        with chat_container:
            # Display chat history
            for role, message in st.session_state['sidebar_chat_history'][-5:]:  # Show last 5 messages
                with st.chat_message(role):
                    st.markdown(message)

        # Gather analysis context for prompts
        analysis_context = collect_comprehensive_analysis_data() if CHAT_AVAILABLE else {}

        # Quick actions aligned with AI Insights tab
        col_a, col_b = st.columns(2)
        with col_a:
            analyze_clicked = st.button(
                "🧠 Analyze Journey",
                key="sidebar_analyze_journey",
                help="Run the same semantic journey analysis used in AI Insights",
            )
        with col_b:
            deep_clicked = st.button(
                "💡 Deep Insights",
                key="sidebar_deep_insights",
                help="Use AI Insights deep-dive angles for targeted guidance",
            )

        # Vision-powered explain button (uses llama.cpp server + page snapshot)
        vision_clicked = st.button(
            "🖼️ Explain with Vision-Model",
            key="sidebar_vision_explain",
            help="Send the current page snapshot to a vision-capable llama.cpp server model (e.g., Qwen3-VL)",
        )

        # Chat input
        user_question = st.chat_input(
            "Ask about this analysis...",
            key="sidebar_chat_input"
        )

        if user_question:
            st.session_state['sidebar_chat_history'].append(("user", user_question))
            prompt = _build_sidebar_prompt(
                base_context=analysis_context,
                user_question=user_question,
                deep=False,
            )
            _stream_sidebar_response(prompt, include_snapshot=False)
            st.rerun()

        if analyze_clicked:
            prompt = _build_sidebar_prompt(base_context=analysis_context, user_question=None, deep=False)
            _stream_sidebar_response(prompt, include_snapshot=False)
            st.rerun()

        if deep_clicked:
            prompt = _build_sidebar_prompt(base_context=analysis_context, user_question=None, deep=True)
            _stream_sidebar_response(prompt, include_snapshot=False)
            st.rerun()

        if vision_clicked:
            prompt = _build_sidebar_prompt(
                base_context=analysis_context,
                user_question="Explain this page and its key insights using the attached snapshot of the current chart/figure.",
                deep=True,
            )

            snapshot_b64 = _get_latest_plotly_snapshot()
            if snapshot_b64:
                st.session_state["page_snapshot_b64"] = snapshot_b64
                _stream_sidebar_response(prompt, include_snapshot=True)
            else:
                st.warning("📸 No chart snapshot available (no recent Plotly figure or kaleido missing); sending text-only.")
                st.session_state["page_snapshot_b64"] = ""
                _stream_sidebar_response(prompt, include_snapshot=False)
            st.rerun()

        # Clear chat button
        if st.button("🗑️ Clear Chat", key="sidebar_clear_chat"):
            st.session_state['sidebar_chat_history'] = []
            st.rerun()


def render_sidebar_with_chat():
    """Render complete sidebar with LLM config and chat."""

    # LLM Configuration
    render_llm_config_sidebar()

    # Chat Interface
    context = {
        'current_tab': st.session_state.get('current_tab', 'Unknown'),
        'data_summary': f"{st.session_state.get('total_sessions', 0)} sessions loaded"
    }
    render_sidebar_chat(context)
