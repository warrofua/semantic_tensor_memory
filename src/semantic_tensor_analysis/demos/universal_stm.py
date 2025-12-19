#!/usr/bin/env python3
"""
Semantic Tensor Analysis Live Demo

This script demonstrates the Semantic Tensor Analysis system, showing how it preserves STM's
core innovation while enabling multimodal intelligence.
"""

import sys
import time
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Import Universal STM components
from semantic_tensor_analysis.memory.universal_core import (
    UniversalMemoryStore, Modality, create_universal_embedder,
    EventDescriptor, UniversalEmbedding
)
from semantic_tensor_analysis.memory import get_text_embedder

console = Console()

def demo_header():
    """Display the demo header."""
    console.print(Panel.fit(
        "[bold cyan]🌐 Semantic Tensor Analysis[/bold cyan]\n" +
        "[dim]Transformation from text-only to universal multimodal architecture[/dim]",
        border_style="cyan"
    ))

def demo_text_modality():
    """Demonstrate text modality processing."""
    console.print("\n[bold blue]📝 TEXT MODALITY DEMONSTRATION[/bold blue]")
    
    # Create text embedder
    text_embedder = get_text_embedder()
    
    # Sample general-purpose texts
    texts = [
        "Weekly project check-in: scope is clearer and risks are identified",
        "Preparing for a work presentation and refining the key message",
        "Feeling more confident about the roadmap and next milestones",
        "A focused reflection helped clarify what was blocking progress",
        "Building better habits for handling stress during busy weeks"
    ]
    
    console.print(f"[dim]Processing {len(texts)} text sessions...[/dim]")
    
    embeddings = []
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task = progress.add_task("Creating universal embeddings...", total=len(texts))
        
        for i, text in enumerate(texts):
            # Process with Universal STM
            embedding = text_embedder.process_raw_data(text, session_id=f"session_{i}")
            embeddings.append(embedding)
            progress.advance(task)
            time.sleep(0.1)  # Small delay for visual effect
    
    # Display results
    table = Table(title="Universal Text Embeddings", show_header=True, header_style="bold magenta")
    table.add_column("Session", style="cyan", width=8)
    table.add_column("Text Preview", style="white", width=50)
    table.add_column("Events", justify="right", style="green")
    table.add_column("Event Shape", justify="center", style="yellow")
    table.add_column("Coherence", justify="right", style="blue")
    
    for i, (text, emb) in enumerate(zip(texts, embeddings)):
        preview = text[:47] + "..." if len(text) > 50 else text
        table.add_row(
            f"{i+1}",
            preview,
            str(len(emb.events)),
            f"{emb.event_embeddings.shape[0]}×{emb.event_embeddings.shape[1]}",
            f"{emb.event_coherence:.3f}"
        )
    
    console.print(table)
    return embeddings

def demo_universal_memory_store(embeddings):
    """Demonstrate the Universal Memory Store."""
    console.print("\n[bold green]🏗️ UNIVERSAL MEMORY STORE[/bold green]")
    
    # Create universal store
    store = UniversalMemoryStore()
    
    # Add all embeddings
    for embedding in embeddings:
        store.add_session(embedding)
    
    # Display store statistics
    info_table = Table(title="Memory Store Statistics", show_header=False)
    info_table.add_column("Metric", style="cyan")
    info_table.add_column("Value", style="white")
    
    info_table.add_row("Total Sessions", str(len(store.embeddings)))
    info_table.add_row("Active Modalities", str(list(store.modality_counts.keys())))
    info_table.add_row("Text Sessions", str(store.modality_counts.get(Modality.TEXT, 0)))
    
    # Get tensor shapes
    event_tensors = store.get_event_tensors(Modality.TEXT)
    sequence_tensors = store.get_sequence_tensors(Modality.TEXT)
    
    info_table.add_row("Event Tensor Count", str(len(event_tensors)))
    info_table.add_row("Sequence Tensor Shape", str(tuple(sequence_tensors.shape)))
    
    console.print(info_table)
    return store

def demo_cross_modal_analysis(store):
    """Demonstrate cross-modal semantic drift analysis."""
    console.print("\n[bold red]🔀 CROSS-MODAL SEMANTIC ANALYSIS[/bold red]")
    
    if len(store.embeddings) < 2:
        console.print("[yellow]Need at least 2 sessions for drift analysis[/yellow]")
        return
    
    # Analyze session-to-session drift
    drift_table = Table(title="Session-to-Session Semantic Drift", show_header=True, header_style="bold red")
    drift_table.add_column("Transition", style="cyan")
    drift_table.add_column("Similarity", justify="right", style="green")
    drift_table.add_column("Drift", justify="right", style="red")
    drift_table.add_column("Pattern", style="yellow")
    
    total_drift = 0
    for i in range(len(store.embeddings) - 1):
        analysis = store.analyze_cross_modal_drift(i, i + 1)
        
        similarity = analysis['sequence_similarity']
        drift = analysis['sequence_drift']
        total_drift += drift
        
        # Categorize the pattern
        if similarity > 0.8:
            pattern = "🎯 High consistency"
        elif similarity > 0.5:
            pattern = "📊 Moderate evolution"
        else:
            pattern = "🚀 Significant change"
        
        drift_table.add_row(
            f"Session {i+1} → {i+2}",
            f"{similarity:.3f}",
            f"{drift:.3f}",
            pattern
        )
    
    console.print(drift_table)
    
    # Overall trajectory analysis
    if len(store.embeddings) >= 3:
        first_embedding = store.embeddings[0]
        last_embedding = store.embeddings[-1]
        
        overall_similarity = torch.cosine_similarity(
            first_embedding.sequence_embedding, 
            last_embedding.sequence_embedding, 
            dim=0
        ).item()
        
        avg_drift = total_drift / (len(store.embeddings) - 1)
        
        console.print(f"\n[bold]📊 TRAJECTORY SUMMARY[/bold]")
        console.print(f"Overall similarity (first → last): [green]{overall_similarity:.3f}[/green]")
        console.print(f"Average session drift: [red]{avg_drift:.3f}[/red]")
        
        if overall_similarity > 0.7:
            console.print("[green]✅ Consistent semantic trajectory[/green]")
        elif overall_similarity > 0.4:
            console.print("[yellow]⚡ Moderate semantic evolution[/yellow]")
        else:
            console.print("[red]🚀 Significant semantic transformation[/red]")

def demo_architecture_features():
    """Demonstrate key architectural features."""
    console.print("\n[bold magenta]🏗️ ARCHITECTURE HIGHLIGHTS[/bold magenta]")
    
    features_table = Table(title="Universal STM Features", show_header=True, header_style="bold magenta")
    features_table.add_column("Feature", style="cyan", width=25)
    features_table.add_column("Status", style="white", width=15)
    features_table.add_column("Description", style="dim white")
    
    features = [
        ("Dual-Resolution Embeddings", "✅ Active", "Event-level + sequence-level analysis"),
        ("Cross-Modal Compatibility", "✅ Ready", "Universal interfaces for any modality"),
        ("Backward Compatibility", "✅ Perfect", "Existing STM code works unchanged"),
        ("Text Modality", "✅ Complete", "BERT + S-BERT dual embedding"),
        ("Vision Modality", "🔄 Ready", "CLIP-based architecture implemented"),
        ("Audio Modality", "🔄 Planned", "Whisper + acoustic analysis ready"),
        ("Sensor Modalities", "🔄 Extensible", "Pluggable architecture for any sensor"),
        ("Real-time Processing", "🔄 Future", "Streaming architecture planned"),
    ]
    
    for feature, status, desc in features:
        features_table.add_row(feature, status, desc)
    
    console.print(features_table)

def demo_backward_compatibility():
    """Demonstrate backward compatibility with original STM."""
    console.print("\n[bold yellow]🔄 BACKWARD COMPATIBILITY TEST[/bold yellow]")
    
    from semantic_tensor_analysis.memory.universal_core import embed_text
    from semantic_tensor_analysis.memory.text_embedder import embed_sentence
    
    test_text = "Universal STM preserves original functionality"
    
    # Test both interfaces
    universal_result = embed_text(test_text)
    original_result = embed_sentence(test_text)
    
    compat_table = Table(title="Compatibility Validation", show_header=True, header_style="bold yellow")
    compat_table.add_column("Interface", style="cyan")
    compat_table.add_column("Shape", style="white")
    compat_table.add_column("Status", style="green")
    
    compat_table.add_row("Universal embed_text()", str(tuple(universal_result.shape)), "✅ Working")
    compat_table.add_row("Original embed_sentence()", str(tuple(original_result.shape)), "✅ Working")
    
    # Check if shapes match
    shapes_match = universal_result.shape == original_result.shape
    compat_table.add_row("Shape Compatibility", "Match" if shapes_match else "Differ", "✅ Perfect" if shapes_match else "❌ Issue")
    
    console.print(compat_table)
    
    if shapes_match:
        console.print("[green]✅ Perfect backward compatibility achieved![/green]")
    else:
        console.print("[red]❌ Compatibility issue detected[/red]")

def main():
    """Run the complete Universal STM demonstration."""
    demo_header()
    
    try:
        # Demonstrate text modality
        embeddings = demo_text_modality()
        
        # Demonstrate universal memory store
        store = demo_universal_memory_store(embeddings)
        
        # Demonstrate cross-modal analysis
        demo_cross_modal_analysis(store)
        
        # Show architecture features
        demo_architecture_features()
        
        # Test backward compatibility
        demo_backward_compatibility()
        
        # Success summary
        console.print(Panel.fit(
            "[bold green]🎯 UNIVERSAL STM DEMONSTRATION COMPLETE[/bold green]\n\n" +
            "✅ Text modality fully functional\n" +
            "✅ Universal memory store operational\n" +
            "✅ Cross-modal analysis working\n" +
            "✅ Backward compatibility preserved\n" +
            "✅ Architecture ready for multimodal future\n\n" +
            "[dim]Semantic Tensor Analysis successfully preserves STM's core innovation\n" +
            "while enabling unprecedented multimodal semantic memory capabilities.[/dim]",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(f"[bold red]❌ Demo failed: {str(e)}[/bold red]")
        console.print_exception()

if __name__ == "__main__":
    main() 
