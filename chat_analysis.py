"""Chat and LLM analysis functionality for Semantic Tensor Memory.

This module contains chat interface and LLM-powered analysis functions
for providing behavioral insights and interactive explanations focused on
semantic content rather than technical metrics.
"""

import streamlit as st
import requests
import json
from streamlit_utils import add_chat_message, is_ollama_model_available, collect_comprehensive_analysis_data


def create_semantic_insights_prompt(analysis_data):
    """Create a content-focused prompt for semantic analysis."""
    
    # Extract key content insights
    session_texts = analysis_data.get('session_texts', [])
    drift_analysis = analysis_data.get('drift_analysis', {})
    trajectory = analysis_data.get('semantic_trajectory', {})
    
    prompt_parts = [
        "You are an expert in analyzing personal and professional development patterns through text analysis.",
        "Focus on the semantic meaning, life journey, and personal growth insights rather than technical metrics.",
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
            f"- This suggests: {'semantic stabilization and focus development' if trend == 'decreasing' else 'increasing exploration and change'}",
        ])
    
    # Add trajectory insights
    if 'velocity_trend' in trajectory:
        velocity_trend = trajectory['velocity_trend']
        prompt_parts.extend([
            f"- Growth velocity: {velocity_trend}",
            f"- Interpretation: {'settling into expertise/mastery phase' if velocity_trend == 'decreasing' else 'accelerating learning and exploration'}",
        ])
    
    prompt_parts.extend([
        "",
        "ANALYSIS FOCUS:",
        "Please provide insights on:",
        "1. PERSONAL/PROFESSIONAL DEVELOPMENT JOURNEY: What story do these sessions tell about growth, learning, and change?",
        "2. KEY BREAKTHROUGH MOMENTS: What do the significant shifts reveal about important life/career transitions?",
        "3. CURRENT PHASE: Based on recent patterns, what phase of development is this person in?",
        "4. ACTIONABLE INSIGHTS: What specific next steps would support continued growth based on the semantic patterns?",
        "5. PATTERN INTERPRETATION: What do the evolution trends suggest about learning style, career progression, or personal development?",
        "",
        "Focus on human insights, growth patterns, and meaningful recommendations rather than technical analysis.",
        "Be specific about what the content reveals about this person's journey and potential next steps."
    ])
    
    return "\n".join(prompt_parts)


def create_targeted_insights_prompt(analysis_data):
    """Create focused prompts based on the specific patterns detected."""
    
    session_texts = analysis_data.get('session_texts', [])
    trajectory = analysis_data.get('semantic_trajectory', {})
    drift_analysis = analysis_data.get('drift_analysis', {})
    
    # Detect the type of journey based on content patterns
    if session_texts:
        combined_text = " ".join(session_texts).lower()
        
        # Career transition detection
        career_keywords = ['job', 'career', 'work', 'manager', 'team', 'company', 'interview', 'promotion', 'skills']
        learning_keywords = ['learn', 'study', 'course', 'bootcamp', 'python', 'programming', 'data', 'algorithm']
        research_keywords = ['research', 'paper', 'study', 'analysis', 'experiment', 'hypothesis', 'publication']
        personal_keywords = ['feel', 'think', 'believe', 'growth', 'change', 'journey', 'reflection']
        
        career_score = sum(1 for keyword in career_keywords if keyword in combined_text)
        learning_score = sum(1 for keyword in learning_keywords if keyword in combined_text)
        research_score = sum(1 for keyword in research_keywords if keyword in combined_text)
        personal_score = sum(1 for keyword in personal_keywords if keyword in combined_text)
        
        # Determine primary journey type
        scores = {
            'career_transition': career_score,
            'learning_journey': learning_score,
            'research_development': research_score,
            'personal_growth': personal_score
        }
        
        primary_journey = max(scores, key=scores.get)
        
        journey_prompts = {
            'career_transition': [
                "This appears to be a CAREER TRANSITION journey. Analyze:",
                "- What career shift is happening and what's driving it?",
                "- What skills and competencies are being developed?",
                "- What challenges and breakthroughs are evident?",
                "- What's the next logical step in this career progression?"
            ],
            'learning_journey': [
                "This appears to be a LEARNING AND SKILL DEVELOPMENT journey. Analyze:",
                "- What subjects/skills are being mastered and in what sequence?",
                "- What learning patterns and preferences are evident?",
                "- Where are the knowledge gaps or struggle points?",
                "- What advanced topics should be tackled next?"
            ],
            'research_development': [
                "This appears to be a RESEARCH AND ACADEMIC journey. Analyze:",
                "- What research interests and methodologies are developing?",
                "- How is academic thinking and expertise evolving?",
                "- What collaboration and publication patterns emerge?",
                "- What research directions show the most promise?"
            ],
            'personal_growth': [
                "This appears to be a PERSONAL DEVELOPMENT journey. Analyze:",
                "- What personal insights and self-awareness patterns emerge?",
                "- How are values, beliefs, and perspectives evolving?",
                "- What life challenges and growth opportunities are present?",
                "- What personal development focus would be most beneficial?"
            ]
        }
        
        return journey_prompts.get(primary_journey, journey_prompts['personal_growth'])
    
    return ["Analyze this semantic journey focusing on personal and professional development patterns."]


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