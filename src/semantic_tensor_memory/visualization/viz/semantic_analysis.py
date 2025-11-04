import json
import subprocess
from typing import List, Dict, Tuple
from rich.console import Console
from rich.panel import Panel
import numpy as np

console = Console()

def analyze_with_ollama(texts: List[str], prompt: str) -> str:
    """Run analysis using Ollama with the specified prompt."""
    # Prepare the input for Ollama
    input_data = {
        "model": "qwen3:latest",
        "prompt": prompt,
        "stream": False
    }
    
    try:
        # Call Ollama
        result = subprocess.run(
            ["ollama", "run", "qwen3:latest", prompt],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error calling Ollama:[/red] {e}")
        return ""

def analyze_pca_patterns(texts: List[str], scores: List[float]) -> str:
    """Use Ollama to analyze patterns in PCA results."""
    # Create a prompt for pattern analysis
    prompt = f"""Analyze these clinical session notes and their PCA scores to identify key behavioral patterns.
Focus on identifying meaningful clinical patterns, not just surface-level differences.

Texts and scores:
{chr(10).join(f'Score: {score:.2f} - {text}' for text, score in zip(texts, scores))}

Please provide:
1. A brief summary of the key behavioral patterns
2. Clinical significance of these patterns
3. Any notable transitions or changes

Keep the analysis concise and clinically relevant."""

    return analyze_with_ollama(texts, prompt)

def generate_clinical_summary(reduced: np.ndarray, session_ids: np.ndarray,
                            token_ids: np.ndarray, meta: List[Dict]) -> str:
    """Generate a clinical summary of the PCA analysis using Ollama."""
    # Get extreme points for each axis
    pca1_scores = reduced[:, 0]
    pca2_scores = reduced[:, 1]
    
    # Find most extreme points
    pca1_pos = np.argmax(pca1_scores)
    pca1_neg = np.argmin(pca1_scores)
    pca2_pos = np.argmax(pca2_scores)
    pca2_neg = np.argmin(pca2_scores)
    
    # Get corresponding texts
    pca1_pos_text = meta[session_ids[pca1_pos]]['text']
    pca1_neg_text = meta[session_ids[pca1_neg]]['text']
    pca2_pos_text = meta[session_ids[pca2_pos]]['text']
    pca2_neg_text = meta[session_ids[pca2_neg]]['text']
    
    # Create prompt for clinical analysis
    prompt = f"""Analyze these clinical session notes to identify key behavioral patterns and their clinical significance.
Focus on understanding the underlying behavioral and emotional patterns.

Primary Axis (PCA-1) Examples:
Positive: {pca1_pos_text}
Negative: {pca1_neg_text}

Secondary Axis (PCA-2) Examples:
Positive: {pca2_pos_text}
Negative: {pca2_neg_text}

Please provide:
1. A clinical interpretation of each axis
2. Key behavioral patterns identified
3. Clinical significance of these patterns
4. Any notable transitions or changes in behavior

Keep the analysis concise and clinically relevant."""

    return analyze_with_ollama([pca1_pos_text, pca1_neg_text, pca2_pos_text, pca2_neg_text], prompt)

def print_clinical_analysis(analysis: str) -> None:
    """Print the clinical analysis in a nicely formatted panel."""
    console.print(Panel(analysis, title="Clinical Analysis", border_style="blue")) 