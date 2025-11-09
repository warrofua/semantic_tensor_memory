"""Chat and LLM analysis functionality for Semantic Tensor Memory.

This module contains chat interface and LLM-powered analysis functions
for providing behavioral insights and interactive explanations focused on
semantic content rather than technical metrics.
"""

import streamlit as st
import requests
import json
from semantic_tensor_memory.streamlit.utils import (
    add_chat_message,
    is_ollama_model_available,
    collect_comprehensive_analysis_data,
)


def create_semantic_insights_prompt(analysis_data):
    """Create a content-focused, domain-adaptive prompt for semantic analysis."""

    # Extract key content insights
    session_texts = analysis_data.get('session_texts', [])
    drift_analysis = analysis_data.get('drift_analysis', {})
    trajectory = analysis_data.get('semantic_trajectory', {})
    
    prompt_parts = [
        "You are an expert analyst who adapts to the dataset domain.",
        "First infer the domain from the session texts and metadata (e.g., ABA therapy/clinical notes, learning journals, software/project logs, research notes, general conversations).",
        "Then adopt the most suitable perspective (e.g., ABA clinician, therapist, learning coach, project analyst, research mentor) and use domain-appropriate language.",
        "Focus on meaning and actionable insights rather than technical model metrics.",
        "",
        "SEMANTIC JOURNEY ANALYSIS:",
        f"- Total sessions: {analysis_data.get('total_sessions', 0)}",
    ]
    
    # Add session content analysis
    if session_texts and len(session_texts) > 0:
        prompt_parts.extend([
            "",
            "SESSION CONTENT EVOLUTION:",
            f"First session themes: {session_texts[0][:200]}..." if len(session_texts[0]) > 200 else f"First session: {session_texts[0]}",
        ])
        
        if len(session_texts) > 1:
            mid_point = len(session_texts) // 2
            prompt_parts.append(f"Mid-journey themes: {session_texts[mid_point][:200]}..." if len(session_texts[mid_point]) > 200 else f"Mid-journey: {session_texts[mid_point]}")
            
            prompt_parts.append(f"Recent themes: {session_texts[-1][:200]}..." if len(session_texts[-1]) > 200 else f"Recent session: {session_texts[-1]}")
    
    # Add semantic shift analysis
    if 'significant_shifts' in trajectory and trajectory['significant_shifts']:
        prompt_parts.extend([
            "",
            "SIGNIFICANT SEMANTIC SHIFTS detected at sessions:",
            f"Sessions with major changes: {trajectory['significant_shifts']}",
        ])
        
        # Add context for significant shifts
        for shift_session in trajectory['significant_shifts'][:3]:  # Limit to first 3
            if shift_session <= len(session_texts):
                shift_text = session_texts[shift_session - 1]
                prompt_parts.append(f"Session {shift_session} content: {shift_text[:150]}...")
    
    # Add drift pattern analysis
    if 'drift_trend' in drift_analysis:
        trend = drift_analysis['drift_trend']
        avg_drift = drift_analysis.get('avg_drift', 0)
        prompt_parts.extend([
            "",
            "SEMANTIC EVOLUTION PATTERNS:",
            f"- Change pattern: {trend} (avg rate: {avg_drift:.3f})",
            f"- Interpretation (domain-aware): if decreasing → consolidation/stability; if increasing → exploration/change",
        ])
    
    # Add trajectory insights
    if 'velocity_trend' in trajectory:
        velocity_trend = trajectory['velocity_trend']
        prompt_parts.extend([
            f"- Growth velocity: {velocity_trend}",
            f"- Interpretation: {'settling into expertise/mastery phase' if velocity_trend == 'decreasing' else 'accelerating learning and exploration'}",
        ])
    
    # Adaptive time-scale guidance
    # Infer time granularity from metadata if available
    time_hint = ""
    try:
        dates = analysis_data.get('dates', []) or []
        if dates and len(dates) >= 2:
            # assume ISO or YYYY-MM-DD; basic spread heuristic
            import datetime as _dt
            def _parse(d):
                try:
                    return _dt.datetime.fromisoformat(d)
                except Exception:
                    return None
            parsed = [p for p in (_parse(d) for d in dates) if p]
            if len(parsed) >= 2:
                span_days = (max(parsed) - min(parsed)).days
                if span_days >= 365 * 2:
                    time_hint = "Use quarters as the primary time scale; consider years for summaries."
                elif span_days >= 365:
                    time_hint = "Use months as the primary time scale; summarize by quarters where helpful."
                elif span_days >= 120:
                    time_hint = "Use weeks as the primary time scale; summarize by months."
                elif span_days >= 30:
                    time_hint = "Use weeks as the primary time scale."
                else:
                    time_hint = "Use days as the primary time scale."
    except Exception:
        time_hint = ""

    prompt_parts.extend([
        "",
        "ANALYSIS FOCUS (adapt to inferred domain):",
        "Please provide insights on:",
        "1. OVERALL NARRATIVE: What story do these sessions tell about goals, themes, and change?",
        "2. KEY SHIFTS: What major transition points occurred and what likely drove them?",
        "3. CURRENT STATE: Based on recent patterns, what is the present phase/status?",
        "4. TIME-SCALE: Infer an appropriate time-scale for summarizing patterns based on the date span (e.g., if months of data, use weeks; if years, use months or quarters). " + (f"Hint: {time_hint}" if time_hint else ""),
        "5. ACTIONABLE NEXT STEPS: Domain-appropriate recommendations (e.g., ABA interventions, study plan tweaks, project milestones, reflective practices).",
        "5. RISKS & WATCHPOINTS: Any emerging issues or regressions to monitor.",
        "6. DATA CAVEATS: Any limitations in the dataset that affect confidence.",
        "",
        "Ground every point in the session content. Keep language suitable for the inferred domain."
    ])
    
    return "\n".join(prompt_parts)


def create_targeted_insights_prompt(analysis_data):
    """Create adaptive, domain-agnostic deep-dive prompts, relying on the LLM to infer domain."""

    session_texts = analysis_data.get('session_texts', [])
    drift_analysis = analysis_data.get('drift_analysis', {})
    trajectory = analysis_data.get('semantic_trajectory', {})

    base = [
        "DEEP-DIVE ANGLES (choose those most appropriate for the inferred domain):",
        "- Thematic structure: dominant themes and how they evolve.",
        "- Phase mapping: identify distinct phases/stages and transition drivers.",
        "- Intervention/strategy review (if applicable): what worked, what didn’t, and why.",
        "- Risks/regressions: early warning signs and mitigation ideas.",
        "- Next experiments/milestones: concrete, domain-appropriate steps.",
        "- Outcome metrics/KPIs: propose realistic indicators to track, tailored to the domain.",
    ]

    # Add a concise context anchor to aid domain inference without forcing it
    if session_texts:
        base.append("")
        base.append("Context anchor excerpts (for domain inference):")
        base.append(f"- Start: {session_texts[0][:140]}...")
        if len(session_texts) > 2:
            mid_point = len(session_texts) // 2
            base.append(f"- Mid: {session_texts[mid_point][:140]}...")
        base.append(f"- Recent: {session_texts[-1][:140]}...")

    return base


def stream_ollama_response(prompt_text, model_name):
    """Stream response from Ollama API."""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt_text,
        "stream": True
    }
    try:
        with requests.post(url, json=payload, stream=True, timeout=180) as r:
            for line in r.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                        yield data.get("response", "")
                    except Exception:
                        continue
    except Exception as e:
        yield f"Error connecting to Ollama: {e}"


def render_chat_analysis_panel(context=None, tab_id=None):
    """Render chat analysis panel with content-focused LLM interaction."""
    # Model selection and buttons section
    col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1])
    
    with col1:
        # AI Model selection dropdown
        model_options = {
            "Qwen3": "qwen3:latest",
            "Mistral": "mistral:latest"
        }
        selected_model_label = st.selectbox(
            "AI Model:",
            list(model_options.keys()),
            key=f"model_selection_{tab_id}" if tab_id else "model_selection",
            help="Choose the AI model for analysis"
        )
        selected_model = model_options[selected_model_label]
        st.session_state["selected_model"] = selected_model
    
    chat_key = f"chat_history_{tab_id}" if tab_id else "chat_history"
    
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []
    
    chat_history = st.session_state[chat_key]
    
    # Check for streaming state
    streaming_key = f'streaming_{tab_id}' if tab_id else 'streaming'
    is_streaming = st.session_state.get(streaming_key, False)
    
    # Display existing chat history first (before new streaming)
    for role, msg in chat_history:
        with st.chat_message(role):
            st.markdown(msg)
    
    # Analyze buttons
    with col2:
        explain_clicked = st.button("🧠 Analyze Journey", key=f"explain_btn_{tab_id}", disabled=is_streaming)
    
    with col3:
        insights_clicked = st.button("💡 Get Insights", key=f"insights_btn_{tab_id}", disabled=is_streaming)
    
    with col4:
        if is_streaming:
            pause_clicked = st.button("⏸️ Pause", key=f"pause_btn_{tab_id}")
            if pause_clicked:
                st.session_state[streaming_key] = False
                st.rerun()
    
    # Handle button clicks
    prompt = None
    
    if explain_clicked:
        prompt = create_semantic_insights_prompt(context or {})
        st.session_state[streaming_key] = True
    
    elif insights_clicked:
        targeted_prompts = create_targeted_insights_prompt(context or {})
        base_prompt = create_semantic_insights_prompt(context or {})
        prompt = base_prompt + "\n\n" + "\n".join(targeted_prompts)
        st.session_state[streaming_key] = True
    
    # Process the prompt if one was generated
    if prompt and st.session_state.get(streaming_key, False):
        if not is_ollama_model_available(selected_model):
            st.warning(
                f"The model '{selected_model}' is not available in your local Ollama. "
                f"To download it, run this command in your terminal:\n\n"
                f"```sh\nollama pull {selected_model}\n```"
            )
            st.session_state[streaming_key] = False
            return
        
        with st.chat_message("assistant"):
            placeholder = st.empty()
            streamed = ""
            for token in stream_ollama_response(prompt, selected_model):
                if not st.session_state.get(streaming_key, False):
                    break
                streamed += token
                placeholder.markdown(streamed)
            
            # Finalize the response
            st.session_state[chat_key].append(("assistant", streamed))
            add_chat_message("assistant", streamed)
            st.session_state[streaming_key] = False
    
    # User input
    user_input = st.chat_input("Ask about your semantic journey, growth patterns, or next steps...", key=f"chat_input_{chat_key}")
    if user_input and not is_streaming:
        st.session_state[chat_key].append(("user", user_input))
        add_chat_message("user", user_input)
        st.rerun()


def render_comprehensive_chat_analysis():
    """
    Render the comprehensive chat analysis panel with all collected data.
    """
    st.header("🧠 Comprehensive Behavioral Analysis & Chat")
    
    # Collect all analysis data
    analysis_data = collect_comprehensive_analysis_data()
    
    # Display summary metrics with more semantic context
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Sessions", analysis_data['total_sessions'])
    with col2:
        if 'avg_drift' in analysis_data['drift_analysis']:
            drift_val = analysis_data['drift_analysis']['avg_drift']
            drift_interpretation = "High Change" if drift_val > 0.2 else "Moderate" if drift_val > 0.1 else "Stable"
            st.metric("Semantic Change", f"{drift_val:.3f}", drift_interpretation)
        else:
            st.metric("Semantic Change", "N/A")
    with col3:
        if 'cumulative_variance' in analysis_data['pca_2d_analysis']:
            var_val = analysis_data['pca_2d_analysis']['cumulative_variance']
            st.metric("Pattern Clarity", f"{var_val:.1%}")
        else:
            st.metric("Pattern Clarity", "N/A")
    with col4:
        if 'total_significant_shifts' in analysis_data['semantic_trajectory']:
            shifts = analysis_data['semantic_trajectory']['total_significant_shifts']
            st.metric("Major Transitions", shifts)
        else:
            st.metric("Major Transitions", "N/A")
    
    # Add semantic journey summary
    if analysis_data.get('session_texts'):
        st.subheader("📖 Journey Overview")
        total_sessions = len(analysis_data['session_texts'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Journey Start:**")
            start_preview = analysis_data['session_texts'][0][:200] + "..." if len(analysis_data['session_texts'][0]) > 200 else analysis_data['session_texts'][0]
            st.info(start_preview)
        
        with col2:
            st.markdown("**Current Phase:**")
            end_preview = analysis_data['session_texts'][-1][:200] + "..." if len(analysis_data['session_texts'][-1]) > 200 else analysis_data['session_texts'][-1]
            st.info(end_preview)
        
        # Show significant shifts if any
        if 'significant_shifts' in analysis_data.get('semantic_trajectory', {}):
            shifts = analysis_data['semantic_trajectory']['significant_shifts']
            if shifts:
                st.markdown("**🎯 Key Transition Points:**")
                shift_info = []
                for shift_session in shifts[:3]:  # Show up to 3 shifts
                    if shift_session <= len(analysis_data['session_texts']):
                        shift_text = analysis_data['session_texts'][shift_session - 1][:100] + "..."
                        shift_info.append(f"**Session {shift_session}:** {shift_text}")
                st.markdown("\n".join(shift_info))
    
    # Show detailed analysis in expander (keep technical details hidden)
    with st.expander("🔍 Technical Analysis Data", expanded=False):
        st.json(analysis_data)
    
    # Render chat panel with comprehensive context
    render_chat_analysis_panel(context=analysis_data, tab_id="comprehensive") 
