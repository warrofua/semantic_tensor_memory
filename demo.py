import time, torch
import sys
from pathlib import Path
import os

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from memory.embedder import embed_sentence
from memory.store import load, append
from memory.drift import drift_series, token_drift
from semantic_tensor_memory.visualization import (
    heatmap,
    plot,
    plot_drift,
    token_heatmap,
)

# For CSV import
import csv
import tkinter as tk
from tkinter import filedialog

def import_csv_to_memory():
    """Open a file dialog to select a CSV and import its contents as memory sessions."""
    # Open file dialog
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select CSV file",
        filetypes=[("CSV Files", "*.csv")]
    )
    if not file_path:
        print("No file selected.")
        return None, None
    print(f"Selected file: {file_path}")
    
    memory = []
    meta = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            text = row.get('text', '').strip()
            if not text:
                continue
            emb = embed_sentence(text)
            # Store all columns as metadata
            meta_row = dict(row)
            meta_row['tokens'] = emb.shape[0]
            memory.append(emb)
            meta.append(meta_row)
    print(f"Imported {len(memory)} sessions from CSV.")
    return memory, meta

def main():
    memory, meta = load()
    
    print("\nSemantic Tensor Memory Demo")
    print("Commands:")
    print("  plot     - Show drift visualizations")
    print("  drift    - Show drift metrics")
    print("  tokens   - Show token-level drift")
    print("  import   - Import sessions from CSV")
    print("  exit     - Quit\n")
    
    while True:
        text = input("\U0001F464> ").strip()
        if text.lower() == "exit":
            break
        
        if text.lower() == "import":
            memory, meta = import_csv_to_memory()
            if memory is not None and meta is not None:
                # Save to disk using append logic (overwrite old memory)
                from memory.store import save
                save(memory, meta)
                print(f"Imported and saved {len(memory)} sessions.")
            continue
        
        if text.lower() == "plot":
            if len(memory) > 1:
                plot(memory, meta)
                heatmap(memory)
                if len(memory) >= 3:
                    token_heatmap(memory)
            else:
                print("Need ≥2 sessions to plot drift.")
            continue
        
        if text.lower() == "drift":
            if len(memory) > 1:
                drifts, counts = drift_series(memory)
                plot_drift(drifts, counts)
                print("\nDrift scores between sessions:")
                for i, (d, c) in enumerate(zip(drifts, counts)):
                    print(f"Session {i+1} → {i+2}: {d:.3f} (tokens: {c})")
            else:
                print("Need ≥2 sessions for drift analysis.")
            continue
        
        if text.lower() == "tokens":
            if len(memory) >= 3:
                drifts = token_drift(memory)
                if drifts:
                    print("\nTokens showing significant drift:")
                    for idx, score in drifts[:5]:  # Show top 5
                        print(f"Token {idx}: drift = {score:.3f}")
                else:
                    print("No significant token drift detected.")
            else:
                print("Need ≥3 sessions for token drift analysis.")
            continue
        
        # Process new input
        emb = embed_sentence(text)
        meta_row = {
            "ts": time.time(),
            "text": text,
            "tokens": emb.shape[0]
        }
        memory, meta = append(memory, emb, meta, meta_row)
        print(f"Stored session {len(meta)} with {emb.shape[0]} tokens.")

if __name__ == "__main__":
    main() 