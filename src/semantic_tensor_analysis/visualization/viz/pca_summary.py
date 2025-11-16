import numpy as np
from rich.console import Console
from rich.table import Table
from typing import List, Dict, Tuple
from collections import Counter
import re

console = Console()

def extract_keywords(text: str) -> List[str]:
    """Extract meaningful keywords from text, filtering out common words."""
    # Convert to lowercase and split
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Filter out common words and short terms
    stop_words = {'the', 'and', 'a', 'to', 'of', 'in', 'is', 'that', 'with', 'was', 'for', 'as', 'on', 'at', 'by', 'this', 'but', 'not', 'from', 'or', 'an', 'be', 'are', 'it', 'have', 'has', 'had', 'if', 'they', 'their', 'there', 'what', 'when', 'where', 'which', 'who', 'why', 'how'}
    return [w for w in words if len(w) > 2 and w not in stop_words]

def analyze_axis_patterns(texts: List[str], scores: List[float]) -> Tuple[str, List[str]]:
    """Analyze patterns in texts along a PCA axis to generate a summary."""
    # Get keywords from each text, weighted by score
    all_keywords = []
    for text, score in zip(texts, scores):
        keywords = extract_keywords(text)
        # Weight keywords by absolute score
        all_keywords.extend(keywords * int(abs(score) * 10))
    
    # Get most common keywords
    keyword_counts = Counter(all_keywords)
    top_keywords = [word for word, _ in keyword_counts.most_common(5)]
    
    # Generate simple summary
    if len(texts) >= 2:
        pos_text = texts[0]
        neg_text = texts[-1]
        summary = f"Distinguishes between patterns like '{pos_text[:50]}...' and '{neg_text[:50]}...'"
    else:
        summary = "Insufficient data for pattern analysis"
    
    return summary, top_keywords

def explain_pca_axes(reduced: np.ndarray, session_ids: np.ndarray, 
                    token_ids: np.ndarray, meta: List[Dict], n: int = 2) -> None:
    """
    Print detailed analysis of PCA axes, including extreme points and patterns.
    
    Args:
        reduced: PCA output (shape: [num_tokens, 2])
        session_ids: Array mapping each point to its session
        token_ids: Array mapping each point to its token position
        meta: List of session metadata dictionaries
        n: Number of top examples to print per axis/direction
    """
    dim_names = ["PCA-1 (Horizontal)", "PCA-2 (Vertical)"]
    
    for dim in range(reduced.shape[1]):
        scores = reduced[:, dim]
        
        # Get extreme points
        top_idx = np.argsort(scores)[-n:][::-1]
        bot_idx = np.argsort(scores)[:n]
        
        # Collect texts for pattern analysis
        top_texts = [meta[session_ids[idx]]['text'] for idx in top_idx]
        bot_texts = [meta[session_ids[idx]]['text'] for idx in bot_idx]
        top_scores = [scores[idx] for idx in top_idx]
        bot_scores = [scores[idx] for idx in bot_idx]
        
        # Analyze patterns
        pos_summary, pos_keywords = analyze_axis_patterns(top_texts, top_scores)
        neg_summary, neg_keywords = analyze_axis_patterns(bot_texts, bot_scores)
        
        # Create rich table for axis summary
        table = Table(title=f"{dim_names[dim]} Analysis")
        table.add_column("Direction")
        table.add_column("Summary")
        table.add_column("Key Terms")
        
        table.add_row(
            "Positive",
            pos_summary,
            ", ".join(pos_keywords)
        )
        table.add_row(
            "Negative",
            neg_summary,
            ", ".join(neg_keywords)
        )
        
        console.print("\n")
        console.print(table)
        
        # Print detailed examples
        console.rule(f"{dim_names[dim]} Most Positive Examples")
        for idx in top_idx:
            session_idx = session_ids[idx]
            token_idx = token_ids[idx]
            text = meta[session_idx]['text']
            score = scores[idx]
            
            console.print(f"[green]Session {session_idx + 1}[/green] (Token {token_idx})")
            console.print(f"Score: {score:.3f}")
            console.print(f"Text: {text}\n")
        
        console.rule(f"{dim_names[dim]} Most Negative Examples")
        for idx in bot_idx:
            session_idx = session_ids[idx]
            token_idx = token_ids[idx]
            text = meta[session_idx]['text']
            score = scores[idx]
            
            console.print(f"[red]Session {session_idx + 1}[/red] (Token {token_idx})")
            console.print(f"Score: {score:.3f}")
            console.print(f"Text: {text}\n")

def generate_narrative_summary(reduced: np.ndarray, session_ids: np.ndarray,
                             token_ids: np.ndarray, meta: List[Dict]) -> str:
    """
    Generate a narrative summary of the PCA analysis.
    
    Returns:
        A string containing a human-readable summary of the PCA patterns.
    """
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
    
    # Generate summary
    summary = []
    summary.append("PCA Analysis Summary:")
    summary.append("\n1. Primary Axis (PCA-1):")
    summary.append(f"   • Positive end: {pca1_pos_text[:100]}...")
    summary.append(f"   • Negative end: {pca1_neg_text[:100]}...")
    
    summary.append("\n2. Secondary Axis (PCA-2):")
    summary.append(f"   • Positive end: {pca2_pos_text[:100]}...")
    summary.append(f"   • Negative end: {pca2_neg_text[:100]}...")
    
    return "\n".join(summary) 
