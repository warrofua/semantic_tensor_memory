"""Persistent sidebar chat panel with LLM configuration.

This module provides a context-aware chat interface that appears in the sidebar
across all tabs, allowing users to ask questions about any visualization or analysis.
"""

import os
import streamlit as st
from pathlib import Path
from typing import Optional, Dict, Any, Iterator, List
import streamlit.components.v1 as components

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
        stream_unified_response
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
    stream_unified_response = None  # type: ignore


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
    """Render LLM configuration in sidebar."""

    with st.sidebar:
        st.markdown("---")
        st.subheader("🤖 AI Assistant")

        # Backend selection
        backend_choices = [("ollama", "Ollama"), ("none", "None")]
        # Local llama.cpp option only if importable
        if CHAT_AVAILABLE and is_llama_cpp_available():
            backend_choices.insert(0, ("llama_cpp", "llama.cpp (local)"))
        # HTTP llama.cpp server option (does not require local lib)
        backend_choices.insert(1, ("llama_server", "llama.cpp server (HTTP)"))

        labels = {value: label for value, label in backend_choices}
        llm_backend = st.selectbox(
            "LLM Backend",
            options=[value for value, _ in backend_choices],
            format_func=lambda v: labels.get(v, v),
            key="sidebar_llm_backend",
            help="Select AI backend for chat assistance"
        )

        # Backend-specific configuration
        if llm_backend == "llama_cpp":
            st.markdown("**Model Selection:**")

            # Initialize model path in session state
            if 'llama_cpp_model_path' not in st.session_state:
                st.session_state['llama_cpp_model_path'] = ''

            # Show recent models if any
            if 'recent_llama_models' in st.session_state and st.session_state['recent_llama_models']:
                # Find current selection index
                current_model = st.session_state.get('llama_cpp_model_path', '')
                recent_options = ["[Select a recent model]"] + st.session_state['recent_llama_models']
                default_index = 0
                if current_model in st.session_state['recent_llama_models']:
                    default_index = recent_options.index(current_model)

                recent_model = st.selectbox(
                    "Recent Models",
                    recent_options,
                    index=default_index,
                    key="sidebar_recent_models_select"
                )
                if recent_model != "[Select a recent model]":
                    st.session_state['llama_cpp_model_path'] = recent_model

            # File browser interface - Native OS Dialog (Primary Method)
            if TKINTER_AVAILABLE:
                st.markdown("**Select Model File:**")

                col1, col2 = st.columns([2, 1])
                with col1:
                    if st.button("🗂️ Browse for GGUF Model...", key="sidebar_browse_file", use_container_width=True):
                        # Get initial directory from current path or use home
                        initial_dir = None
                        if st.session_state['llama_cpp_model_path']:
                            current_path = Path(st.session_state['llama_cpp_model_path'])
                            if current_path.exists():
                                initial_dir = str(current_path.parent)

                        selected_file = select_file_with_tkinter(initial_dir=initial_dir)
                        if selected_file:
                            st.session_state['llama_cpp_model_path'] = selected_file
                            # Add to recent models
                            if 'recent_llama_models' not in st.session_state:
                                st.session_state['recent_llama_models'] = []
                            if selected_file not in st.session_state['recent_llama_models']:
                                st.session_state['recent_llama_models'].insert(0, selected_file)
                                st.session_state['recent_llama_models'] = st.session_state['recent_llama_models'][:5]
                            st.rerun()

                with col2:
                    if st.button("📁 Browse Folder", key="sidebar_browse_folder", use_container_width=True):
                        selected_folder = select_folder_with_tkinter()
                        if selected_folder:
                            # Find GGUF files in selected folder
                            gguf_files = list(Path(selected_folder).glob("*.gguf"))
                            if gguf_files:
                                st.session_state['_browse_dir'] = selected_folder
                                st.session_state['_browse_files'] = [str(f) for f in sorted(gguf_files)]
                            else:
                                st.warning(f"No GGUF files found in {selected_folder}")

                # Show browsed files from folder selection
                if '_browse_files' in st.session_state and st.session_state['_browse_files']:
                    st.caption(f"Found {len(st.session_state['_browse_files'])} models in {Path(st.session_state['_browse_dir']).name}:")

                    selected_file = st.selectbox(
                        "Select model:",
                        ["[Choose a model]"] + st.session_state['_browse_files'],
                        key="sidebar_folder_file_select"
                    )

                    if selected_file != "[Choose a model]":
                        st.session_state['llama_cpp_model_path'] = selected_file
                        if 'recent_llama_models' not in st.session_state:
                            st.session_state['recent_llama_models'] = []
                        if selected_file not in st.session_state['recent_llama_models']:
                            st.session_state['recent_llama_models'].insert(0, selected_file)
                            st.session_state['recent_llama_models'] = st.session_state['recent_llama_models'][:5]

            # Fallback: File browser is intentionally disabled to avoid exposing full directory trees.
            elif FILE_BROWSER_AVAILABLE:
                st.info(
                    "📦 File browser disabled to avoid exposing your filesystem. "
                    "Use the quick scanner below or enter a path manually."
                )

            # Fallback: Manual path entry only
            else:
                st.warning("⚠️ No file browser available. Install tkinter for native dialogs.")

            # Quick scan for GGUF files when native picker is unavailable or disabled
            if not TKINTER_AVAILABLE:
                st.markdown("**Quick Model Picker (common locations):**")
                st.caption("Scans ~/models, ~/Downloads, and any paths in $LLM_MODEL_DIRS for .gguf files.")

                if "quick_llama_models" not in st.session_state:
                    st.session_state["quick_llama_models"] = []

                if st.button("🔍 Scan for models", key="sidebar_scan_models", use_container_width=True):
                    st.session_state["quick_llama_models"] = _find_local_models()
                    if not st.session_state["quick_llama_models"]:
                        st.info("No GGUF models found in common locations. Enter a path manually.")

                if st.session_state.get("quick_llama_models"):
                    discovered = st.selectbox(
                        "Select discovered model",
                        ["[Choose a model]"] + st.session_state["quick_llama_models"],
                        key="sidebar_quick_model_select",
                    )
                    if discovered != "[Choose a model]":
                        st.session_state["llama_cpp_model_path"] = discovered

            # Manual path input (always available)
            with st.expander("✏️ Or Enter Path Manually"):
                model_path = st.text_input(
                    "Full model path:",
                    value=st.session_state['llama_cpp_model_path'],
                    key="sidebar_llama_model_path_input",
                    help="Full path to your GGUF model file",
                    placeholder="e.g., /Users/you/models/model.gguf"
                )

                # Update state from text input
                if model_path != st.session_state['llama_cpp_model_path']:
                    st.session_state['llama_cpp_model_path'] = model_path

            # Save valid paths to recent models
            final_model_path = st.session_state['llama_cpp_model_path']
            if final_model_path and Path(final_model_path).exists():
                if 'recent_llama_models' not in st.session_state:
                    st.session_state['recent_llama_models'] = []
                if final_model_path not in st.session_state['recent_llama_models']:
                    st.session_state['recent_llama_models'].insert(0, final_model_path)
                    st.session_state['recent_llama_models'] = st.session_state['recent_llama_models'][:5]

            # Performance settings
            with st.expander("⚙️ Performance Settings"):
                n_threads = st.number_input("CPU Threads", min_value=1, max_value=16, value=4, key="sidebar_n_threads")
                n_gpu_layers = st.number_input("GPU Layers", min_value=0, max_value=100, value=0, key="sidebar_n_gpu_layers")

            st.session_state['llm_backend'] = 'llama_cpp'
            st.session_state['llm_backend_config'] = {
                'model_path': final_model_path,
                'n_threads': n_threads,
                'n_gpu_layers': n_gpu_layers
            }

            # Status indicator
            if final_model_path and Path(final_model_path).exists():
                file_size_gb = Path(final_model_path).stat().st_size / (1024**3)
                st.success(f"✅ {Path(final_model_path).name[:30]} ({file_size_gb:.1f}GB)")
            elif final_model_path:
                st.error(f"⚠️ Model not found: {final_model_path}")
            else:
                st.warning("⚠️ No model selected")

        elif llm_backend == "llama_server":
            st.markdown("**Remote llama.cpp server:**")
            server_url = st.text_input(
                "Server URL",
                value=st.session_state.get("llama_server_url", "http://localhost:8080"),
                key="sidebar_llama_server_url",
                help="URL where llama-server is running (e.g., http://localhost:8080)",
            )
            server_model = st.text_input(
                "Server Model Name",
                value=st.session_state.get("llama_server_model", "local"),
                key="sidebar_llama_server_model",
                help="Model name configured in the server (default: local)",
            )

            st.session_state["llama_server_url"] = server_url
            st.session_state["llama_server_model"] = server_model

            st.session_state['llm_backend'] = 'llama_server'
            st.session_state['llm_backend_config'] = {
                'server_url': server_url.rstrip("/"),
                'server_model': server_model,
            }

            st.info(f"Using llama-server at {server_url} with model '{server_model}'")

        elif llm_backend == "ollama":
            model_options = {
                "Qwen3": "qwen3:latest",
                "Mistral": "mistral:latest",
                "Llama 3": "llama3:latest"
            }
            selected_model_label = st.selectbox(
                "Ollama Model",
                list(model_options.keys()),
                key="sidebar_ollama_model"
            )
            selected_model = model_options[selected_model_label]

            st.session_state['llm_backend'] = 'ollama'
            st.session_state['llm_backend_config'] = {
                'model_name': selected_model
            }
            st.success(f"✅ {selected_model}")

        else:  # None
            st.info("AI assistance disabled. Core analysis still works!")
            st.session_state['llm_backend'] = 'none'


def render_sidebar_chat(context: Optional[Dict[str, Any]] = None):
    """Render persistent chat panel in sidebar.

    Args:
        context: Optional context about current page/visualization
    """

    with st.sidebar:
        st.markdown("---")
        st.subheader("💬 Ask AI")

        # Optional: capture a page snapshot for VLMs
        include_snapshot = st.checkbox("📸 Include page snapshot", value=False, key="sidebar_include_snapshot")
        if include_snapshot:
            # Hidden field to hold base64 snapshot
            snapshot_holder = st.text_input(
                "page_snapshot_b64",
                key="page_snapshot_b64",
                label_visibility="collapsed",
                help="Automatically filled with a page snapshot",
            )
            # JS-based capture using html2canvas (best-effort; requires network access to CDN)
            components.html(
                """
                <script src="https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js"></script>
                <script>
                (function() {
                  const doCapture = () => {
                    const input = window.parent.document.querySelector('input[aria-label="page_snapshot_b64"]');
                    if (!input || typeof html2canvas === 'undefined') return;
                    html2canvas(window.parent.document.body, {scale: 0.5}).then(canvas => {
                      const dataUrl = canvas.toDataURL('image/png').replace(/^data:image\\/png;base64,/, '');
                      input.value = dataUrl;
                      const event = new Event('input', { bubbles: true });
                      input.dispatchEvent(event);
                    }).catch(() => {});
                  };
                  // Throttle capture to avoid excessive calls
                  if (!window.__sta_snapshot_cooldown) {
                    window.__sta_snapshot_cooldown = true;
                    doCapture();
                    setTimeout(() => { window.__sta_snapshot_cooldown = false; }, 3000);
                  }
                })();
                </script>
                """,
                height=0,
            )

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

        # Chat input
        user_question = st.chat_input(
            "Ask about this analysis...",
            key="sidebar_chat_input"
        )

        if user_question:
            # Add user message
            st.session_state['sidebar_chat_history'].append(("user", user_question))

            # Get LLM response
            backend = st.session_state.get('llm_backend', 'none')
            backend_config = st.session_state.get('llm_backend_config', {})
            if include_snapshot:
                snapshot = st.session_state.get("page_snapshot_b64")
                if snapshot:
                    backend_config = {**backend_config, "image_base64": snapshot}
                else:
                    st.info("📸 Snapshot not available yet; sending text-only this time.")

            if backend == 'none':
                response = "AI assistance is disabled. Please select an LLM backend above to enable chat."
                st.session_state['sidebar_chat_history'].append(("assistant", response))
            elif not CHAT_AVAILABLE or stream_unified_response is None:
                response = "Chat components are unavailable in this environment."
                st.session_state['sidebar_chat_history'].append(("assistant", response))
            elif backend == "llama_server" and not backend_config.get("server_url"):
                response = "Please provide the llama-server URL (e.g., http://localhost:8080)."
                st.session_state['sidebar_chat_history'].append(("assistant", response))
            else:
                # Create context-aware prompt
                context_info = []
                if context and context.get('has_data'):
                    context_info.append(f"Dataset: {context.get('data_summary', 'No data loaded')}")

                if context_info:
                    prompt = f"Context: {' | '.join(context_info)}\n\nUser Question: {user_question}\n\nProvide a helpful answer based on semantic tensor analysis principles."
                else:
                    prompt = f"User Question: {user_question}\n\nNote: No dataset is currently loaded. Provide general guidance about semantic tensor analysis."

                # Stream response
                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    response = ""

                    try:
                        for token in stream_unified_response(prompt, backend=backend, **backend_config):
                            response += token
                            placeholder.markdown(response)

                        st.session_state['sidebar_chat_history'].append(("assistant", response))
                    except Exception as e:
                        error_msg = f"Error: {e}"
                        placeholder.error(error_msg)
                        st.session_state['sidebar_chat_history'].append(("assistant", error_msg))

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
