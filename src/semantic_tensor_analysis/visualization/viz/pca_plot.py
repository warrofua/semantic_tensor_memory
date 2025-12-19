# import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch, numpy as np
from typing import List, Dict, Tuple
from rich import print
from rich.table import Table
from rich.console import Console
from .pca_summary import explain_pca_axes, generate_narrative_summary
from .semantic_analysis import generate_pca_interpretation, print_pca_interpretation
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def check_tensor_health(tensor: torch.Tensor, name: str = "tensor") -> bool:
    """Check tensor for numerical issues and print diagnostics.
    
    Args:
        tensor: Input tensor to check
        name: Optional name for the tensor in diagnostic messages
    
    Returns:
        bool: True if tensor is healthy, False if issues were found
    
    Checks for:
        - NaN values
        - Inf values
        - Zero rows (potential padding)
        - Constant rows (no variance)
    """
    is_healthy = True
    
    # Check for NaN/Inf
    if torch.isnan(tensor).any():
        print(f"[red]Warning:[/red] NaN values found in {name}")
        is_healthy = False
    if torch.isinf(tensor).any():
        print(f"[red]Warning:[/red] Inf values found in {name}")
        is_healthy = False
        
    # Check for zero rows
    zero_rows = (tensor.abs().sum(dim=1) < 1e-6).sum().item()
    if zero_rows > 0:
        print(f"[yellow]Note:[/yellow] Found {zero_rows} zero rows in {name}")
        
    # Check for constant rows
    row_vars = tensor.var(dim=1)
    const_rows = (row_vars < 1e-6).sum().item()
    if const_rows > 0:
        print(f"[yellow]Note:[/yellow] Found {const_rows} constant rows in {name}")
        is_healthy = False
        
    return is_healthy

def prepare_for_pca(tensors: List[torch.Tensor]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare tensor data for PCA with proper masking and checks.
    
    Args:
        tensors: List of session embeddings, where each tensor has shape [tokens, embed_dim]
    
    Returns:
        Tuple containing:
            - flat: Flattened and normalized numpy array of shape [total_tokens, embed_dim]
            - session_ids: Array mapping each point to its session
            - token_ids: Array mapping each point to its token position
    
    This function handles:
        - Flattening ragged tensors
        - Removing problematic rows (NaN, Inf, zero, constant)
        - Normalizing the data
        - Maintaining session and token indices
    """
    # Flatten and track indices
    flat = torch.cat(tensors)
    session_ids = np.repeat(np.arange(len(tensors)), [t.shape[0] for t in tensors])
    token_ids = np.concatenate([np.arange(t.shape[0]) for t in tensors])
    
    # 1. Check for empty sessions
    empty_sessions = [i for i, t in enumerate(tensors) if t.shape[0] == 0]
    if empty_sessions:
        print(f"[yellow]Warning:[/yellow] Found {len(empty_sessions)} empty sessions: {empty_sessions}")
    
    # 2. Check for NaN/Inf values
    nan_mask = torch.isnan(flat).any(dim=1)
    inf_mask = torch.isinf(flat).any(dim=1)
    if nan_mask.any() or inf_mask.any():
        print(f"[yellow]Warning:[/yellow] Found {nan_mask.sum().item()} NaN and {inf_mask.sum().item()} Inf values")
        # Replace NaN/Inf with zeros
        flat = torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 3. Check for zero rows (padding)
    zero_mask = (flat.abs().sum(dim=1) < 1e-6)
    if zero_mask.any():
        print(f"[yellow]Warning:[/yellow] Found {zero_mask.sum().item()} zero rows (likely padding)")
    
    # 4. Check for constant rows (no variance)
    row_vars = flat.var(dim=1)
    const_mask = (row_vars < 1e-6)
    if const_mask.any():
        print(f"[yellow]Warning:[/yellow] Found {const_mask.sum().item()} constant rows")
    
    # 5. Combine all problematic rows
    bad_rows = zero_mask | const_mask | nan_mask | inf_mask
    
    if bad_rows.any():
        print(f"[yellow]Removing[/yellow] {bad_rows.sum().item()} problematic rows")
        # Keep track of which rows we're removing
        kept_indices = ~bad_rows
        flat = flat[kept_indices]
        session_ids = session_ids[kept_indices]
        token_ids = token_ids[kept_indices]
    
    # 6. Normalize the data
    # First, center the data
    mean = flat.mean(dim=0)
    flat = flat - mean
    
    # Then, scale by standard deviation (with epsilon to avoid division by zero)
    std = flat.std(dim=0)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)  # Replace near-zero std with 1
    flat = flat / std
    
    # 7. Final check for any remaining issues
    if torch.isnan(flat).any() or torch.isinf(flat).any():
        print("[red]Error:[/red] Still found NaN/Inf values after normalization")
        # Last resort: replace with zeros
        flat = torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    
    return flat.numpy(), session_ids, token_ids

def interpret_pca(reduced: np.ndarray, session_ids: np.ndarray, token_ids: np.ndarray, 
                 tensors: List[torch.Tensor], meta: List[Dict]) -> None:
    """Analyze and print interpretation of PCA dimensions."""
    # Find extreme points along each PCA axis
    pca1_scores = reduced[:, 0]
    pca2_scores = reduced[:, 1]
    
    extremes = {
        'PCA1+': np.argmax(pca1_scores),
        'PCA1-': np.argmin(pca1_scores),
        'PCA2+': np.argmax(pca2_scores),
        'PCA2-': np.argmin(pca2_scores)
    }
    
    # Create rich table for interpretation
    table = Table(title="PCA Dimension Interpretation")
    table.add_column("Dimension")
    table.add_column("Direction")
    table.add_column("Session")
    table.add_column("Text")
    table.add_column("Score")
    
    for dim, idx in extremes.items():
        pca_axis, direction = dim[:3], dim[3]
        session_idx = session_ids[idx]
        token_idx = token_ids[idx]
        score = pca1_scores[idx] if pca_axis == 'PCA1' else pca2_scores[idx]
        
        # Get the full text for context
        text = meta[session_idx]['text']
        
        table.add_row(
            pca_axis,
            direction,
            f"Session {session_idx + 1}",
            text,
            f"{score:.3f}"
        )
    
    console = Console()
    console.print("\n[bold]PCA Dimension Analysis:[/bold]")
    console.print(table)
    
    # Print semantic patterns
    print("\n[bold]Observed Patterns:[/bold]")
    pca1_pos_text = meta[session_ids[extremes['PCA1+']]]['text']
    pca1_neg_text = meta[session_ids[extremes['PCA1-']]]['text']
    
    print(f"\nPCA-1 (Horizontal) appears to capture:")
    print(f"→ Positive end: {pca1_pos_text}")
    print(f"→ Negative end: {pca1_neg_text}")
    
    pca2_pos_text = meta[session_ids[extremes['PCA2+']]]['text']
    pca2_neg_text = meta[session_ids[extremes['PCA2-']]]['text']
    
    print(f"\nPCA-2 (Vertical) appears to capture:")
    print(f"→ Positive end: {pca2_pos_text}")
    print(f"→ Negative end: {pca2_neg_text}")

def plot(tensors: List[torch.Tensor], meta: List[Dict], title: str = "Semantic Drift Map"):
    """Create 2D PCA visualization of token embeddings from ragged tensors.
    
    Args:
        tensors: List of session embeddings, where each tensor has shape [tokens, embed_dim]
        meta: List of session metadata dictionaries
        title: Optional title for the plot
    
    The plot shows:
        - Token embeddings projected onto first two PCA components
        - Color-coded by session
        - Session boundaries marked with vertical lines
        - Narrative summaries of the patterns
    """
    # Prepare data for PCA
    flat, session_ids, token_ids = prepare_for_pca(tensors)
    
    # PCA reduction with numerical stability
    try:
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(flat)
        print(f"\n[green]PCA explained variance:[/green] {pca.explained_variance_ratio_}")
    except Exception as e:
        print(f"[red]Error in PCA:[/red] {str(e)}")
        print("Falling back to random projection...")
        # Fallback: random projection
        np.random.seed(42)
        proj = np.random.randn(flat.shape[1], 2)
        reduced = flat @ proj
    
    # Generate and print narrative summary
    print("\n" + generate_narrative_summary(reduced, session_ids, token_ids, meta))
    
    # Generate and print PCA interpretation using Ollama
    print("\n[bold]Generating PCA interpretation...[/bold]")
    interpretation = generate_pca_interpretation(reduced, session_ids, token_ids, meta)
    print_pca_interpretation(interpretation)
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'PCA1': reduced[:, 0],
        'PCA2': reduced[:, 1],
        'Session': [f"Session {i+1}" for i in session_ids],
        'Text': [meta[i]['text'] for i in session_ids]
    })
    
    # Create scatter plot
    fig = px.scatter(
        df,
        x='PCA1',
        y='PCA2',
        color='Session',
        hover_data=['Text'],
        title=title
    )
    
    # Update layout
    fig.update_layout(
        hovermode="closest",
        showlegend=True
    )
    
    return fig

def plot_drift(drifts: List[float], token_counts: List[int], title: str = "Session Drift"):
    """Plot drift scores with token count context.
    
    Args:
        drifts: List of drift scores between consecutive sessions
        token_counts: List of token counts per session
        title: Optional title for the plot
    
    The plot shows:
        - Drift scores as a line plot
        - Token counts as a bar plot
        - Grid lines for reference
    """
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Session': list(range(len(drifts))),
        'Drift Score': drifts,
        'Token Count': token_counts
    })
    
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # Add drift scores
    fig.add_trace(go.Scatter(
        x=df['Session'],
        y=df['Drift Score'],
        name="Drift Score",
        line=dict(color='blue')
    ))
    
    # Add token counts
    fig.add_trace(go.Bar(
        x=df['Session'],
        y=df['Token Count'],
        name="Token Count",
        opacity=0.3,
        marker_color='gray'
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Session",
        yaxis_title="Drift Score",
        hovermode="x unified",
        showlegend=True
    )
    
    return fig

# plt.show() 
