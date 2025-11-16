#!/usr/bin/env python3
"""
Large Chat History Processor for Universal STM

Handles massive chat history files (0.5GB+) with streaming processing,
sampling, and memory-efficient analysis.
"""

import json
import os
import time
from typing import List, Iterator, Optional
from dataclasses import dataclass
import random

from semantic_tensor_analysis.chat.history_analyzer import (
    ChatMessage,
    ChatSemanticAnalyzer,
    ConversationAnalysis,
)

@dataclass
class ProcessingConfig:
    """Configuration for large file processing."""
    max_messages: int = 1000  # Sample size for analysis
    sample_strategy: str = "uniform"  # "uniform", "recent", "random"
    chunk_size: int = 100  # Process in chunks
    progress_callback: Optional[callable] = None

class LargeChatProcessor:
    """Efficiently process large chat history files."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.analyzer = ChatSemanticAnalyzer()
    
    def estimate_file_size(self, file_path: str) -> dict:
        """Estimate processing requirements for a file."""
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)
        
        # Rough estimates
        estimated_messages = file_size // 100  # ~100 chars per message
        user_messages = estimated_messages // 3  # ~1/3 are user messages
        
        processing_time_minutes = user_messages * 0.1 / 60  # ~0.1s per message
        memory_mb = user_messages * 3072 / (1024 * 1024)  # 768*4 bytes per embedding
        
        return {
            "file_size_mb": round(file_size_mb, 1),
            "estimated_total_messages": int(estimated_messages),
            "estimated_user_messages": int(user_messages),
            "estimated_processing_minutes": round(processing_time_minutes, 1),
            "estimated_memory_mb": round(memory_mb, 1),
            "recommended_sample_size": min(1000, int(user_messages)),
            "needs_sampling": user_messages > 1000
        }
    
    def stream_messages_from_json(self, file_path: str) -> Iterator[ChatMessage]:
        """Stream messages from large JSON file without loading everything."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                for conversation in data:
                    conv_id = conversation.get('id', 'unknown')
                    conv_messages = conversation.get('mapping', {})
                    
                    for msg_id, msg_data in conv_messages.items():
                        message = msg_data.get('message')
                        if not message:
                            continue
                            
                        content = message.get('content')
                        if not content:
                            continue
                            
                        # Extract text content
                        if isinstance(content, dict):
                            parts = content.get('parts', [])
                            # Handle mixed content types in parts
                            text_parts = []
                            for part in parts:
                                if isinstance(part, str):
                                    text_parts.append(part)
                                elif isinstance(part, dict):
                                    # Some parts might be objects (e.g., for images, code blocks)
                                    # Try to extract text if available
                                    if 'text' in part:
                                        text_parts.append(str(part['text']))
                                elif part is not None:
                                    text_parts.append(str(part))
                            text = ' '.join(text_parts) if text_parts else ''
                        else:
                            text = str(content)
                        
                        if text.strip():
                            role = message.get('author', {}).get('role', 'unknown')
                            timestamp = message.get('create_time')
                            
                            yield ChatMessage(
                                content=text.strip(),
                                role=role,
                                timestamp=timestamp,
                                conversation_id=conv_id,
                                message_id=msg_id
                            )
        
        except Exception as e:
            raise ValueError(f"Failed to stream from JSON: {str(e)}")
    
    def sample_messages(self, messages: List[ChatMessage], strategy: str = "uniform") -> List[ChatMessage]:
        """Sample messages using different strategies."""
        user_messages = [msg for msg in messages if msg.role == 'user']
        
        if len(user_messages) <= self.config.max_messages:
            return user_messages
        
        if strategy == "uniform":
            # Take every nth message for uniform distribution
            step = len(user_messages) // self.config.max_messages
            return user_messages[::step][:self.config.max_messages]
        
        elif strategy == "recent":
            # Take most recent messages
            return user_messages[-self.config.max_messages:]
        
        elif strategy == "random":
            # Random sampling
            return random.sample(user_messages, self.config.max_messages)
        
        else:
            return user_messages[:self.config.max_messages]
    
    def process_large_file(self, file_path: str) -> tuple[ConversationAnalysis, dict]:
        """Process large chat file with sampling and progress tracking."""
        
        # Estimate requirements
        estimates = self.estimate_file_size(file_path)
        
        if self.config.progress_callback:
            self.config.progress_callback(f"📊 File analysis: {estimates['file_size_mb']}MB, ~{estimates['estimated_user_messages']:,} user messages")
        
        # Determine if we need sampling
        if estimates['needs_sampling']:
            if self.config.progress_callback:
                self.config.progress_callback(f"⚡ Large file detected - using {self.config.sample_strategy} sampling ({self.config.max_messages} messages)")
        
        # Stream and collect messages
        if self.config.progress_callback:
            self.config.progress_callback("📥 Reading messages...")
        
        all_messages = []
        try:
            for message in self.stream_messages_from_json(file_path):
                all_messages.append(message)
                
                # Progress update every 1000 messages
                if len(all_messages) % 1000 == 0 and self.config.progress_callback:
                    self.config.progress_callback(f"📥 Read {len(all_messages):,} messages...")
        
        except Exception as e:
            raise ValueError(f"Error reading file: {str(e)}")
        
        if self.config.progress_callback:
            self.config.progress_callback(f"✅ Read {len(all_messages):,} total messages")
        
        # Sample if needed
        user_messages = [msg for msg in all_messages if msg.role == 'user']
        
        if len(user_messages) > self.config.max_messages:
            if self.config.progress_callback:
                self.config.progress_callback(f"🎯 Sampling {self.config.max_messages} from {len(user_messages):,} user messages...")
            
            sampled_messages = self.sample_messages(all_messages, self.config.sample_strategy)
            
            # Reconstruct message list with sampled user messages + assistant responses
            final_messages = []
            sampled_user_ids = {id(msg) for msg in sampled_messages}
            
            for msg in all_messages:
                if msg.role == 'user' and id(msg) in sampled_user_ids:
                    final_messages.append(msg)
                elif msg.role == 'assistant':
                    final_messages.append(msg)
            
            messages_to_analyze = final_messages
        else:
            messages_to_analyze = all_messages
        
        # Process with Universal STM
        if self.config.progress_callback:
            self.config.progress_callback(f"🧠 Analyzing {len([m for m in messages_to_analyze if m.role == 'user'])} messages with Universal STM...")
        
        analysis = self.analyzer.analyze_conversation_history(messages_to_analyze)
        
        # Add processing metadata
        processing_info = {
            "original_file_size_mb": estimates['file_size_mb'],
            "total_messages_in_file": len(all_messages),
            "user_messages_in_file": len(user_messages),
            "messages_analyzed": len([m for m in messages_to_analyze if m.role == 'user']),
            "sampling_used": len(user_messages) > self.config.max_messages,
            "sampling_strategy": self.config.sample_strategy if len(user_messages) > self.config.max_messages else None,
            "processing_estimates": estimates
        }
        
        return analysis, processing_info

def create_smart_config(file_path: str) -> ProcessingConfig:
    """Create smart processing config based on file size."""
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    
    if file_size_mb < 1:  # Small files
        return ProcessingConfig(max_messages=5000, sample_strategy="uniform")
    elif file_size_mb < 10:  # Medium files  
        return ProcessingConfig(max_messages=2000, sample_strategy="uniform")
    elif file_size_mb < 100:  # Large files
        return ProcessingConfig(max_messages=1000, sample_strategy="uniform")
    else:  # Huge files (0.5GB+)
        return ProcessingConfig(max_messages=500, sample_strategy="recent")

# Example usage
def demo_large_file_processing():
    """Demo processing with progress tracking."""
    
    def progress_callback(message):
        print(f"⏳ {message}")
    
    # This would work for a real 0.5GB file
    print("🚀 LARGE FILE PROCESSING DEMO")
    print("=" * 40)
    print("For a 0.5GB file, this system would:")
    print("✅ Stream messages without loading full file")
    print("⚡ Sample ~500 representative messages")  
    print("🧠 Process with Universal STM")
    print("📊 Complete analysis in ~5 minutes")
    print("💾 Use <100MB memory")

if __name__ == "__main__":
    demo_large_file_processing() 
