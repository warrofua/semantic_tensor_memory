"""semantic_tensor_analysis.visualization.plots
=============================================

Shared PCA visualisation utilities used across demos and the Streamlit app.
The module unifies the functionality that previously lived in ``viz.pca_plot``,
``viz.pca_summary`` and ``viz.semantic_analysis``.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.decomposition import PCA


def check_tensor_health(tensor: torch.Tensor, name: str = "tensor") -> bool:
    """Return ``True`` when ``tensor`` is free from obvious numerical issues."""

    is_healthy = True
    if torch.isnan(tensor).any():
        print(f"[red]Warning:[/red] NaN values found in {name}")
        is_healthy = False
    if torch.isinf(tensor).any():
        print(f"[red]Warning:[/red] Inf values found in {name}")
        is_healthy = False

    zero_rows = (tensor.abs().sum(dim=1) < 1e-6).sum().item()
    if zero_rows > 0:
        print(f"[yellow]Note:[/yellow] Found {zero_rows} zero rows in {name}")

    row_vars = tensor.var(dim=1)
    const_rows = (row_vars < 1e-6).sum().item()
    if const_rows > 0:
        print(f"[yellow]Note:[/yellow] Found {const_rows} constant rows in {name}")
        is_healthy = False

    return is_healthy


def prepare_for_pca(tensors: Sequence[torch.Tensor]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalise ragged tensors and return arrays suitable for PCA."""

    if not tensors:
        raise ValueError("No tensors provided to prepare_for_pca")

    flat = torch.cat(tensors)
    session_ids = np.repeat(np.arange(len(tensors)), [t.shape[0] for t in tensors])
    token_ids = np.concatenate([np.arange(t.shape[0]) for t in tensors])

    nan_mask = torch.isnan(flat).any(dim=1)
    inf_mask = torch.isinf(flat).any(dim=1)
    if nan_mask.any() or inf_mask.any():
        print(
            f"[yellow]Warning:[/yellow] Found {nan_mask.sum().item()} NaN and {inf_mask.sum().item()} Inf values"
        )
        flat = torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)

    zero_mask = flat.abs().sum(dim=1) < 1e-6
    const_mask = flat.var(dim=1) < 1e-6
    bad_rows = zero_mask | const_mask | nan_mask | inf_mask
    if bad_rows.any():
        print(f"[yellow]Removing[/yellow] {bad_rows.sum().item()} problematic rows")
        kept_indices = ~bad_rows
        flat = flat[kept_indices]
        session_ids = session_ids[kept_indices]
        token_ids = token_ids[kept_indices]

    mean = flat.mean(dim=0)
    flat = flat - mean
    std = flat.std(dim=0)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    flat = flat / std

    if torch.isnan(flat).any() or torch.isinf(flat).any():
        print("[red]Error:[/red] Still found NaN/Inf values after normalization")
        flat = torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)

    return (
        flat.detach().cpu().numpy(),
        session_ids.detach().cpu().numpy(),
        token_ids.detach().cpu().numpy(),
    )


def interpret_pca(
    reduced: np.ndarray,
    session_ids: np.ndarray,
    token_ids: np.ndarray,
    tensors: Sequence[torch.Tensor],
    meta: Sequence[Dict],
) -> None:
    """Print an interpretation of PCA axes using ``rich`` tables."""

    pca1_scores = reduced[:, 0]
    pca2_scores = reduced[:, 1]
    extremes = {
        "PCA1+": np.argmax(pca1_scores),
        "PCA1-": np.argmin(pca1_scores),
        "PCA2+": np.argmax(pca2_scores),
        "PCA2-": np.argmin(pca2_scores),
    }

    table = Table(title="PCA Dimension Interpretation")
    table.add_column("Dimension")
    table.add_column("Direction")
    table.add_column("Session")
    table.add_column("Text")
    table.add_column("Score")

    for dim, idx in extremes.items():
        pca_axis, direction = dim[:4], dim[4]
        session_idx = session_ids[idx]
        score = pca1_scores[idx] if pca_axis == "PCA1" else pca2_scores[idx]
        text = meta[session_idx]["text"]

        table.add_row(
            pca_axis,
            direction,
            f"Session {session_idx + 1}",
            text,
            f"{score:.3f}",
        )

    console = Console()
    console.print("\n[bold]PCA Dimension Analysis:[/bold]")
    console.print(table)

    console.print("\n[bold]Observed Patterns:[/bold]")
    console.print(f"\nPCA-1 (Horizontal) appears to capture:\n→ Positive end: {meta[session_ids[extremes['PCA1+']]]['text']}\n→ Negative end: {meta[session_ids[extremes['PCA1-']]]['text']}")
    console.print(f"\nPCA-2 (Vertical) appears to capture:\n→ Positive end: {meta[session_ids[extremes['PCA2+']]]['text']}\n→ Negative end: {meta[session_ids[extremes['PCA2-']]]['text']}")


def plot(tensors: Sequence[torch.Tensor], meta: Sequence[Dict], title: str = "Semantic Drift Map") -> go.Figure:
    """Return a Plotly scatter visualisation of PCA-reduced token embeddings."""

    flat, session_ids, token_ids = prepare_for_pca(tensors)

    try:
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(flat)
        print(f"\n[green]PCA explained variance:[/green] {pca.explained_variance_ratio_}")
    except Exception as exc:  # pragma: no cover - defensive fallback
        print(f"[red]Error in PCA:[/red] {exc}")
        print("Falling back to random projection...")
        np.random.seed(42)
        proj = np.random.randn(flat.shape[1], 2)
        reduced = flat @ proj

    narrative = generate_narrative_summary(reduced, session_ids, token_ids, meta)
    print("\n" + narrative)

    print("\n[bold]Generating clinical analysis...[/bold]")
    clinical_analysis = generate_clinical_summary(reduced, session_ids, token_ids, meta)
    print_clinical_analysis(clinical_analysis)

    df = pd.DataFrame(
        {
            "PCA1": reduced[:, 0],
            "PCA2": reduced[:, 1],
            "Session": [f"Session {i + 1}" for i in session_ids],
            "Text": [meta[i]["text"] for i in session_ids],
        }
    )

    fig = px.scatter(df, x="PCA1", y="PCA2", color="Session", hover_data=["Text"], title=title)
    fig.update_layout(hovermode="closest", showlegend=True)
    return fig


def plot_drift(drifts: Sequence[float], token_counts: Sequence[int], title: str = "Session Drift") -> go.Figure:
    """Render a combined drift/token count Plotly figure."""

    df = pd.DataFrame(
        {
            "Session": list(range(len(drifts))),
            "Drift Score": list(drifts),
            "Token Count": list(token_counts),
        }
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Session"], y=df["Drift Score"], name="Drift Score", line=dict(color="blue")))
    fig.add_trace(
        go.Bar(
            x=df["Session"],
            y=df["Token Count"],
            name="Token Count",
            opacity=0.3,
            marker_color="gray",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Session",
        yaxis_title="Drift Score",
        hovermode="x unified",
        showlegend=True,
    )
    return fig


def extract_keywords(text: str) -> List[str]:
    words = re.findall(r"\b\w+\b", text.lower())
    stop_words = {
        "the",
        "and",
        "a",
        "to",
        "of",
        "in",
        "is",
        "that",
        "with",
        "was",
        "for",
        "as",
        "on",
        "at",
        "by",
        "this",
        "but",
        "not",
        "from",
        "or",
        "an",
        "be",
        "are",
        "it",
        "have",
        "has",
        "had",
        "if",
        "they",
        "their",
        "there",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "how",
    }
    return [word for word in words if len(word) > 2 and word not in stop_words]


def analyze_axis_patterns(texts: Sequence[str], scores: Sequence[float]) -> Tuple[str, List[str]]:
    all_keywords: List[str] = []
    for text, score in zip(texts, scores):
        keywords = extract_keywords(text)
        all_keywords.extend(keywords * max(int(abs(score) * 10), 1))

    keyword_counts = Counter(all_keywords)
    top_keywords = [word for word, _ in keyword_counts.most_common(5)]

    if len(texts) >= 2:
        pos_text = texts[0]
        neg_text = texts[-1]
        summary = (
            "Distinguishes between patterns like "
            f"'{pos_text[:50]}...' and '{neg_text[:50]}...'"
        )
    else:
        summary = "Insufficient data for pattern analysis"

    return summary, top_keywords


def explain_pca_axes(
    reduced: np.ndarray,
    session_ids: np.ndarray,
    token_ids: np.ndarray,
    meta: Sequence[Dict],
    n: int = 2,
) -> None:
    """Print rich console tables explaining each PCA dimension."""

    dim_names = ["PCA-1 (Horizontal)", "PCA-2 (Vertical)"]
    for dim in range(reduced.shape[1]):
        scores = reduced[:, dim]
        top_idx = np.argsort(scores)[-n:][::-1]
        bot_idx = np.argsort(scores)[:n]

        top_texts = [meta[session_ids[idx]]["text"] for idx in top_idx]
        bot_texts = [meta[session_ids[idx]]["text"] for idx in bot_idx]
        top_scores = [scores[idx] for idx in top_idx]
        bot_scores = [scores[idx] for idx in bot_idx]

        pos_summary, pos_keywords = analyze_axis_patterns(top_texts, top_scores)
        neg_summary, neg_keywords = analyze_axis_patterns(bot_texts, bot_scores)

        table = Table(title=f"{dim_names[dim]} Analysis")
        table.add_column("Direction")
        table.add_column("Summary")
        table.add_column("Key Terms")
        table.add_row("Positive", pos_summary, ", ".join(pos_keywords))
        table.add_row("Negative", neg_summary, ", ".join(neg_keywords))

        console = Console()
        console.print("\n")
        console.print(table)

        console.rule(f"{dim_names[dim]} Most Positive Examples")
        for idx in top_idx:
            session_idx = session_ids[idx]
            text = meta[session_idx]["text"]
            score = scores[idx]
            console.print(f"[green]Session {session_idx + 1}[/green] (Token {token_ids[idx]})")
            console.print(f"Score: {score:.3f}")
            console.print(f"Text: {text}\n")

        console.rule(f"{dim_names[dim]} Most Negative Examples")
        for idx in bot_idx:
            session_idx = session_ids[idx]
            text = meta[session_idx]["text"]
            score = scores[idx]
            console.print(f"[red]Session {session_idx + 1}[/red] (Token {token_ids[idx]})")
            console.print(f"Score: {score:.3f}")
            console.print(f"Text: {text}\n")


def generate_narrative_summary(
    reduced: np.ndarray,
    session_ids: np.ndarray,
    token_ids: np.ndarray,
    meta: Sequence[Dict],
) -> str:
    """Return a string summarising PCA axis extremes in plain language."""

    pca1_scores = reduced[:, 0]
    pca2_scores = reduced[:, 1]

    pca1_pos = np.argmax(pca1_scores)
    pca1_neg = np.argmin(pca1_scores)
    pca2_pos = np.argmax(pca2_scores)
    pca2_neg = np.argmin(pca2_scores)

    summary = ["PCA Analysis Summary:"]
    summary.append("\n1. Primary Axis (PCA-1):")
    summary.append(f"   • Positive end: {meta[session_ids[pca1_pos]]['text'][:100]}...")
    summary.append(f"   • Negative end: {meta[session_ids[pca1_neg]]['text'][:100]}...")
    summary.append("\n2. Secondary Axis (PCA-2):")
    summary.append(f"   • Positive end: {meta[session_ids[pca2_pos]]['text'][:100]}...")
    summary.append(f"   • Negative end: {meta[session_ids[pca2_neg]]['text'][:100]}...")
    return "\n".join(summary)


def analyze_with_ollama(texts: Sequence[str], prompt: str) -> str:
    """Run the Ollama CLI with ``prompt`` and return the resulting output."""

    try:
        result = subprocess.run(
            ["ollama", "run", "qwen3:latest", prompt],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as exc:  # pragma: no cover - CLI fallback
        console = Console()
        console.print(f"[red]Error calling Ollama:[/red] {exc}")
        return ""


def analyze_pca_patterns(texts: Sequence[str], scores: Sequence[float]) -> str:
    """Invoke :func:`analyze_with_ollama` with a PCA-oriented prompt."""

    prompt = """Analyze these clinical session notes and their PCA scores to identify key behavioral patterns.
Focus on identifying meaningful clinical patterns, not just surface-level differences.

Texts and scores:
{entries}

Please provide:
1. A brief summary of the key behavioral patterns
2. Clinical significance of these patterns
3. Any notable transitions or changes

Keep the analysis concise and clinically relevant.""".format(
        entries="\n".join(
            f"Score: {score:.2f} - {text}" for text, score in zip(texts, scores)
        )
    )
    return analyze_with_ollama(texts, prompt)


def generate_clinical_summary(
    reduced: np.ndarray,
    session_ids: np.ndarray,
    token_ids: np.ndarray,
    meta: Sequence[Dict],
) -> str:
    """Create a clinical interpretation prompt and call Ollama."""

    pca1_scores = reduced[:, 0]
    pca2_scores = reduced[:, 1]
    pca1_pos = np.argmax(pca1_scores)
    pca1_neg = np.argmin(pca1_scores)
    pca2_pos = np.argmax(pca2_scores)
    pca2_neg = np.argmin(pca2_scores)

    pca1_pos_text = meta[session_ids[pca1_pos]]["text"]
    pca1_neg_text = meta[session_ids[pca1_neg]]["text"]
    pca2_pos_text = meta[session_ids[pca2_pos]]["text"]
    pca2_neg_text = meta[session_ids[pca2_neg]]["text"]

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

    return analyze_with_ollama(
        [pca1_pos_text, pca1_neg_text, pca2_pos_text, pca2_neg_text], prompt
    )


def print_clinical_analysis(analysis: str) -> None:
    """Render the Ollama response in a :class:`rich.panel.Panel`."""

    console = Console()
    console.print(Panel(analysis, title="Clinical Analysis", border_style="blue"))


from collections import Counter  # noqa: E402  (import after function definitions)
import re  # noqa: E402
import subprocess  # noqa: E402
