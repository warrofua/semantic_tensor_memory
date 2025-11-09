"""
Embedding configuration system for STM.

Allows switching between different embedding approaches:
- 'bert_average': Original token averaging (problematic)
- 'bert_cls': CLS token approach (good semantics, loses granularity)  
- 'dual': BERT tokens + S-BERT sentences (best of both worlds)
- 'hybrid': Configurable granularity with efficiency options
"""

import os
from enum import Enum
from typing import Tuple, Dict, Any, Optional
import torch

class EmbeddingMode(Enum):
    BERT_AVERAGE = "bert_average"  # Original approach (deprecated)
    BERT_CLS = "bert_cls"         # CLS token only  
    DUAL = "dual"                 # BERT tokens + S-BERT sentences
    HYBRID = "hybrid"             # Configurable approach

# Global configuration
CURRENT_MODE = EmbeddingMode.DUAL  # Default to best approach
HYBRID_CONFIG = {
    'store_tokens': True,         # Preserve STM tensor nature
    'store_sentence': True,       # Add semantic intelligence  
    'use_cls_summary': True,      # Efficient session summaries
    'token_model': 'bert-base-uncased',
    'sentence_model': 'sentence-transformers/all-mpnet-base-v2'
}

def set_embedding_mode(mode: str, config: Optional[Dict] = None):
    """Set the global embedding mode."""
    global CURRENT_MODE, HYBRID_CONFIG
    
    CURRENT_MODE = EmbeddingMode(mode)
    
    if config and mode == 'hybrid':
        HYBRID_CONFIG.update(config)
    
    print(f"🔧 Embedding mode set to: {mode}")
    if mode == 'hybrid':
        print(f"   Config: {HYBRID_CONFIG}")

def get_embedder():
    """Get the appropriate embedder based on current mode."""
    
    if CURRENT_MODE == EmbeddingMode.BERT_AVERAGE:
        from .embedder import embed_sentence, get_token_count
        print("⚠️  Using deprecated BERT averaging (destroys semantic relationships)")
        return embed_sentence, get_token_count
        
    elif CURRENT_MODE == EmbeddingMode.BERT_CLS:
        from .embedder import embed_sentence, get_token_count  # Assumes CLS version
        print("🎯 Using BERT CLS approach (good semantics, loses granularity)")
        return embed_sentence, get_token_count
        
    elif CURRENT_MODE == EmbeddingMode.DUAL:
        from .dual_embedder import embed_sentence, get_token_count
        print("🚀 Using dual embedding system (BERT tokens + S-BERT sentences)")
        return embed_sentence, get_token_count
        
    elif CURRENT_MODE == EmbeddingMode.HYBRID:
        return get_hybrid_embedder()
        
    else:
        raise ValueError(f"Unknown embedding mode: {CURRENT_MODE}")

def get_hybrid_embedder():
    """Get hybrid embedder with current configuration."""
    from .embedder_hybrid import embed_sentence_with_summary, get_token_count
    
    print(f"🎪 Using hybrid approach: {HYBRID_CONFIG}")
    
    def hybrid_embed(text: str) -> torch.Tensor:
        """Wrapper that returns appropriate representation based on config."""
        if HYBRID_CONFIG['store_tokens'] and HYBRID_CONFIG['store_sentence']:
            # Return tokens (STM core) - sentence available via separate call
            token_embs, sentence_emb = embed_sentence_with_summary(text)
            return token_embs
        elif HYBRID_CONFIG['store_sentence']:
            # Return sentence embedding only
            _, sentence_emb = embed_sentence_with_summary(text)
            return sentence_emb.unsqueeze(0)  # Make it 2D for consistency
        else:
            # Return tokens only
            token_embs, _ = embed_sentence_with_summary(text)
            return token_embs
    
    return hybrid_embed, get_token_count

def get_memory_store():
    """Get the appropriate memory store for current embedding mode."""
    
    if CURRENT_MODE == EmbeddingMode.DUAL:
        from .dual_embedder import DualMemoryStore
        return DualMemoryStore()
    else:
        # Use standard memory store for other modes
        from .memory import Memory
        return Memory()

def analyze_embedding_quality(test_texts: list) -> Dict[str, Any]:
    """Compare different embedding approaches on test texts."""
    
    results = {}
    
    # Test each mode
    for mode in EmbeddingMode:
        try:
            print(f"\n🧪 Testing {mode.value}...")
            
            # Temporarily switch to this mode
            original_mode = CURRENT_MODE
            set_embedding_mode(mode.value)
            
            embed_fn, count_fn = get_embedder()
            
            # Test semantic preservation
            if len(test_texts) >= 2:
                emb1 = embed_fn(test_texts[0])
                emb2 = embed_fn(test_texts[1])
                
                # Compare mean embeddings (crude semantic similarity)
                if emb1.dim() > 1 and emb2.dim() > 1:
                    sim = torch.cosine_similarity(emb1.mean(0), emb2.mean(0), dim=0).item()
                else:
                    sim = torch.cosine_similarity(emb1, emb2, dim=0).item()
                
                results[mode.value] = {
                    'semantic_similarity': sim,
                    'emb1_shape': emb1.shape,
                    'emb2_shape': emb2.shape,
                    'token_counts': [count_fn(t) for t in test_texts[:2]]
                }
            
            # Restore original mode
            set_embedding_mode(original_mode.value)
            
        except Exception as e:
            results[mode.value] = {'error': str(e)}
    
    return results

def print_embedding_comparison():
    """Print a comparison of all embedding approaches."""
    
    print("\n📊 EMBEDDING APPROACH COMPARISON")
    print("=" * 60)
    
    approaches = {
        'BERT Average': {
            'strengths': ['Simple', 'Fast'],
            'weaknesses': ['Destroys semantic relationships', 'Poor quality'],
            'use_case': 'Deprecated - avoid using'
        },
        'BERT CLS': {
            'strengths': ['Good semantic quality', 'Efficient', 'No averaging issues'],
            'weaknesses': ['Loses token-level granularity', 'Not true "tensor memory"'],
            'use_case': 'When you need semantic quality over granularity'
        },
        'Dual (BERT + S-BERT)': {
            'strengths': ['Best semantic quality', 'Preserves token granularity', 'Multi-resolution analysis'],
            'weaknesses': ['Higher memory usage', 'Two model loads'],
            'use_case': 'Production systems needing both quality and granularity'
        },
        'Hybrid': {
            'strengths': ['Configurable', 'Preserves STM nature', 'Flexible efficiency'],
            'weaknesses': ['More complex', 'Config-dependent quality'],
            'use_case': 'Research and development, customizable deployments'
        }
    }
    
    for name, info in approaches.items():
        print(f"\n🔧 {name}:")
        print(f"  ✅ Strengths: {', '.join(info['strengths'])}")
        print(f"  ❌ Weaknesses: {', '.join(info['weaknesses'])}")
        print(f"  🎯 Use case: {info['use_case']}")

# Initialize with default mode
print(f"🏁 STM Embedding system initialized in {CURRENT_MODE.value} mode") 
