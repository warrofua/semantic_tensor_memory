#!/usr/bin/env python3
"""
Demo Dataset Loader for Semantic Tensor Memory Analysis

This script demonstrates the power of semantic tensor memory analysis using
a carefully crafted dataset showing a person's journey from career dissatisfaction
to becoming a researcher in AI and computational neuroscience.

The dataset showcases:
- Clear semantic evolution over time
- Vocabulary development (accounting → Python → ML → neuroscience)
- Emotional progression (lost → excited → confident → visionary)
- Conceptual complexity growth
- Natural phase transitions with semantic shifts
"""

from pathlib import Path

import pandas as pd
import streamlit as st
from datetime import datetime

DATA_ROOT = Path(__file__).resolve().parents[3]
DEMO_DATA_PATH = DATA_ROOT / "data" / "ultimate_demo_dataset.csv"

def load_demo_dataset():
    """Load the demonstration dataset and return analysis insights."""
    
    # Load the dataset
    data_path = DEMO_DATA_PATH
    if not data_path.exists():
        raise FileNotFoundError(
            "ultimate_demo_dataset.csv not found. Ensure the dataset is located in the project's data directory."
        )

    df = pd.read_csv(data_path)
    
    # Convert timestamps to readable dates
    df['date'] = pd.to_datetime(df['timestamp'], unit='s')
    df['formatted_date'] = df['date'].dt.strftime('%Y-%m-%d')
    
    # Add analysis metadata
    analysis_insights = {
        'total_sessions': len(df),
        'time_span': f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}",
        'phases': df['phase'].unique().tolist(),
        'session_types': df['session_type'].unique().tolist(),
        'expected_patterns': {
            'vocabulary_evolution': [
                'Early: accounting, meaningless, lost, job',
                'Middle: Python, algorithms, machine learning, neural networks', 
                'Late: neuroscience, research, AI ethics, brain-computer interfaces'
            ],
            'emotional_progression': [
                'Seeking (uncertain, lost, questioning)',
                'Discovery (excited, curious, alive)',
                'Development (learning, challenging, growing)',
                'Challenge (struggling, doubt, perseverance)',
                'Mastery (confident, purposeful, leading)',
                'Integration (teaching, mentoring, serving)',
                'Evolution (researching, visionary, pioneering)'
            ],
            'semantic_shifts': [
                'Session 7: Career transition (accounting → data science)',
                'Session 12-14: Major challenge and recovery',
                'Session 16: Professional breakthrough',
                'Session 20-23: Shift to leadership and teaching',
                'Session 24+: Evolution toward research and AI ethics'
            ]
        }
    }
    
    return df, analysis_insights

def print_dataset_overview():
    """Print an overview of the demonstration dataset."""
    df, insights = load_demo_dataset()
    
    print("🎯 SEMANTIC TENSOR MEMORY DEMONSTRATION DATASET")
    print("=" * 60)
    print(f"📊 Total Sessions: {insights['total_sessions']}")
    print(f"📅 Time Span: {insights['time_span']}")
    print(f"🔄 Phases: {', '.join(insights['phases'])}")
    print(f"📝 Session Types: {', '.join(insights['session_types'])}")
    
    print("\n🧠 EXPECTED SEMANTIC PATTERNS:")
    print("-" * 40)
    
    print("\n📚 Vocabulary Evolution:")
    for i, stage in enumerate(insights['expected_patterns']['vocabulary_evolution'], 1):
        print(f"  {i}. {stage}")
    
    print("\n💭 Emotional Progression:")
    for phase in insights['expected_patterns']['emotional_progression']:
        print(f"  • {phase}")
    
    print("\n🚀 Key Semantic Shifts:")
    for shift in insights['expected_patterns']['semantic_shifts']:
        print(f"  • {shift}")
    
    print("\n🎯 ANALYSIS OPPORTUNITIES:")
    print("-" * 40)
    print("✅ Clear drift patterns across career transition")
    print("✅ Vocabulary sophistication progression") 
    print("✅ Emotional tone evolution")
    print("✅ Conceptual complexity growth")
    print("✅ Phase transition detection")
    print("✅ Semantic clustering by life stages")
    print("✅ Trajectory analysis showing development arc")

def get_streamlit_import_instructions():
    """Return instructions for importing into Streamlit app."""
    return """
    🚀 TO IMPORT INTO STREAMLIT APP:
    
    1. Open your Semantic Tensor Memory app at http://localhost:8501
    2. In the sidebar, use "Import sessions from CSV"
    3. Upload the file: ultimate_demo_dataset.csv
    4. The app will load 30 sessions showing a complete learning journey
    
    🔬 RECOMMENDED ANALYSIS WORKFLOW:
    
    1. Start with "Drift Analysis" to see overall change patterns
    2. Try "Enhanced PCA Map" to visualize semantic evolution
    3. Use "Method Comparison" to find optimal dimensionality reduction
    4. Explore "Semantic Trajectory" for development path analysis
    5. Check "Ridgeline Plot" for feature evolution over time
    6. Use "AI Insights & Chat" for AI insights
    
    🎯 WHAT TO LOOK FOR:
    
    • Clear semantic clusters corresponding to life phases
    • Vocabulary evolution from basic to technical terms
    • Emotional progression from uncertainty to confidence
    • Semantic "jumps" at major life transitions
    • Different dimensionality reduction methods revealing different patterns
    """

if __name__ == "__main__":
    print_dataset_overview()
    print(get_streamlit_import_instructions()) 
