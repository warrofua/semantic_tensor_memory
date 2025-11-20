#!/usr/bin/env python3
"""
Chat History Analyzer for Semantic Tensor Analysis

Analyzes conversation histories from ChatGPT, Claude, and other AI services
to reveal semantic evolution patterns in your thinking and communication style.
"""

import json
import re
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from semantic_tensor_analysis.memory import get_text_embedder
from semantic_tensor_analysis.memory.universal_core import UniversalMemoryStore, Modality

@dataclass
class ChatMessage:
    """Represents a single chat message."""
    content: str
    role: str  # 'user' or 'assistant'
    timestamp: Optional[datetime] = None
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None

@dataclass
class ConversationAnalysis:
    """Results of analyzing a conversation history."""
    total_messages: int
    user_messages: int
    conversations: int
    semantic_drift: float
    topic_evolution: List[str]
    communication_patterns: Dict[str, Any]
    time_span: Optional[str] = None

class ChatHistoryParser:
    """Parses various chat history formats."""
    
    @staticmethod
    def parse_chatgpt_json(file_content: str) -> List[ChatMessage]:
        """Parse ChatGPT JSON export format (handles both old and new formats)."""
        try:
            data = json.loads(file_content)
            messages = []
            
            # Handle new format (array of conversations)
            if isinstance(data, list):
                conversations = data
            # Handle old format (single conversation or dict with conversations)
            elif isinstance(data, dict):
                if 'mapping' in data:
                    # Single conversation
                    conversations = [data]
                else:
                    # Assume it's a dict of conversations
                    conversations = list(data.values())
            else:
                raise ValueError("Unexpected JSON structure")
            
            for conversation in conversations:
                conv_id = conversation.get('id', conversation.get('conversation_id', 'unknown'))
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
                                    text_parts.append(part['text'])
                                elif 'content' in part:
                                    text_parts.append(str(part['content']))
                            elif part is not None:
                                text_parts.append(str(part))
                        text = ' '.join(text_parts) if text_parts else ''
                    else:
                        text = str(content) if content is not None else ''
                    
                    if text.strip():
                        role = message.get('author', {}).get('role', 'unknown')
                        timestamp = message.get('create_time')
                        if timestamp:
                            timestamp = datetime.fromtimestamp(timestamp)
                        
                        messages.append(ChatMessage(
                            content=text.strip(),
                            role=role,
                            timestamp=timestamp,
                            conversation_id=conv_id,
                            message_id=msg_id
                        ))
            
            return messages
            
        except Exception as e:
            raise ValueError(f"Failed to parse ChatGPT JSON: {str(e)}")
    
    @staticmethod
    def parse_plain_text(file_content: str) -> List[ChatMessage]:
        """Parse plain text chat format."""
        messages = []
        lines = file_content.split('\n')
        current_role = None
        current_content = []
        
        # Common patterns for identifying user vs assistant messages
        user_patterns = [
            r'^(You|User|Human):\s*',
            r'^\*\*You\*\*:\s*',
            r'^>\s*',
        ]
        
        assistant_patterns = [
            r'^(Assistant|AI|ChatGPT|Claude):\s*',
            r'^\*\*Assistant\*\*:\s*',
            r'^AI:\s*',
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line starts a new message
            is_user = any(re.match(pattern, line, re.IGNORECASE) for pattern in user_patterns)
            is_assistant = any(re.match(pattern, line, re.IGNORECASE) for pattern in assistant_patterns)
            
            if is_user or is_assistant:
                # Save previous message
                if current_role and current_content:
                    content = ' '.join(current_content).strip()
                    if content:
                        messages.append(ChatMessage(
                            content=content,
                            role=current_role
                        ))
                
                # Start new message
                current_role = 'user' if is_user else 'assistant'
                # Remove the role prefix
                for pattern in (user_patterns if is_user else assistant_patterns):
                    line = re.sub(pattern, '', line, flags=re.IGNORECASE)
                current_content = [line] if line.strip() else []
            else:
                # Continue current message
                if line.strip():
                    current_content.append(line)
        
        # Save final message
        if current_role and current_content:
            content = ' '.join(current_content).strip()
            if content:
                messages.append(ChatMessage(
                    content=content,
                    role=current_role
                ))
        
        return messages
    
    @staticmethod
    def auto_detect_format(file_content: str) -> List[ChatMessage]:
        """Auto-detect and parse chat history format."""
        # Try JSON first (ChatGPT export)
        try:
            if file_content.strip().startswith('{') or file_content.strip().startswith('['):
                return ChatHistoryParser.parse_chatgpt_json(file_content)
        except:
            pass
        
        # Fall back to plain text
        return ChatHistoryParser.parse_plain_text(file_content)

class ChatSemanticAnalyzer:
    """Analyzes semantic evolution in chat histories."""
    
    def __init__(self, max_user_messages: int = 1200):
        self.text_embedder = get_text_embedder()
        self.store = UniversalMemoryStore()
        self.max_user_messages = max_user_messages
    
    def analyze_conversation_history(self, messages: List[ChatMessage]) -> ConversationAnalysis:
        """Analyze semantic evolution in conversation history."""
        
        # Filter to user messages only (focus on user's semantic evolution)
        user_messages = [msg for msg in messages if msg.role == 'user']
        user_messages = self._limit_user_messages(user_messages)
        
        if len(user_messages) < 2:
            raise ValueError("Need at least 2 user messages for analysis")
        
        # Process messages with Universal STM
        embeddings = []
        for i, msg in enumerate(user_messages):
            embedding = self.text_embedder.process_raw_data(
                msg.content, 
                session_id=f"message_{i}"
            )
            embeddings.append(embedding)
            self.store.add_session(embedding)
        
        # Calculate overall semantic drift
        first_embedding = embeddings[0]
        last_embedding = embeddings[-1]
        
        import torch
        overall_similarity = torch.cosine_similarity(
            first_embedding.sequence_embedding,
            last_embedding.sequence_embedding,
            dim=0
        ).item()
        
        semantic_drift = 1 - overall_similarity
        
        # Analyze topic evolution (simple keyword extraction)
        topic_evolution = self._extract_topic_evolution(user_messages)
        
        # Analyze communication patterns
        communication_patterns = self._analyze_communication_patterns(user_messages)
        
        # Calculate time span
        time_span = None
        timestamped_messages = [msg for msg in user_messages if msg.timestamp]
        if len(timestamped_messages) >= 2:
            first_time = min(msg.timestamp for msg in timestamped_messages)
            last_time = max(msg.timestamp for msg in timestamped_messages)
            
            # Handle both Unix timestamps (float) and datetime objects
            try:
                if isinstance(first_time, (int, float)) and isinstance(last_time, (int, float)):
                    # Convert Unix timestamps to datetime objects
                    from datetime import datetime
                    first_dt = datetime.fromtimestamp(first_time)
                    last_dt = datetime.fromtimestamp(last_time)
                    time_span = f"{(last_dt - first_dt).days} days"
                else:
                    # Assume they're already datetime objects
                    time_span = f"{(last_time - first_time).days} days"
            except Exception:
                # If timestamp conversion fails, calculate rough days from raw timestamps
                if isinstance(first_time, (int, float)) and isinstance(last_time, (int, float)):
                    days_diff = int((last_time - first_time) / 86400)  # 86400 seconds per day
                    time_span = f"{days_diff} days"
        
        return ConversationAnalysis(
            total_messages=len(messages),
            user_messages=len(user_messages),
            conversations=len(set(msg.conversation_id for msg in messages if msg.conversation_id)),
            semantic_drift=semantic_drift,
            topic_evolution=topic_evolution,
            communication_patterns=communication_patterns,
            time_span=time_span
        )
    
    def _extract_topic_evolution(self, messages: List[ChatMessage]) -> List[str]:
        """Extract evolving topics from messages."""
        # Simple keyword extraction - could be enhanced with more sophisticated NLP
        topics = []
        
        # Split messages into thirds to show evolution
        chunk_size = len(messages) // 3
        if chunk_size < 1:
            chunk_size = 1
        
        for i in range(0, len(messages), chunk_size):
            chunk = messages[i:i+chunk_size]
            text = ' '.join(msg.content for msg in chunk)
            
            # Extract key terms (simple approach)
            words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
            word_freq = {}
            for word in words:
                if word not in ['that', 'this', 'with', 'from', 'they', 'have', 'been', 'will', 'would', 'could', 'should']:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get top 3 words for this period
            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:3]
            period_topics = [word for word, count in top_words]
            topics.extend(period_topics)
        
        return topics[:10]  # Return top 10 topics
    
    def _analyze_communication_patterns(self, messages: List[ChatMessage]) -> Dict[str, Any]:
        """Analyze patterns in communication style."""
        if not messages:
            return {}
        
        total_length = sum(len(msg.content) for msg in messages)
        avg_length = total_length / len(messages)
        
        # Count questions
        question_count = sum(1 for msg in messages if '?' in msg.content)
        question_rate = question_count / len(messages)
        
        # Count exclamations (enthusiasm)
        exclamation_count = sum(1 for msg in messages if '!' in msg.content)
        enthusiasm_rate = exclamation_count / len(messages)
        
        # Analyze complexity (rough measure)
        complex_words = 0
        total_words = 0
        for msg in messages:
            words = msg.content.split()
            total_words += len(words)
            complex_words += sum(1 for word in words if len(word) > 6)
        
        complexity_rate = complex_words / total_words if total_words > 0 else 0
        
        return {
            'avg_message_length': round(avg_length, 1),
            'question_rate': round(question_rate, 3),
            'enthusiasm_rate': round(enthusiasm_rate, 3),
            'complexity_rate': round(complexity_rate, 3),
            'total_words': total_words
        }
    
    def get_session_drift_analysis(self) -> List[Dict[str, Any]]:
        """Get detailed session-to-session drift analysis."""
        if len(self.store.embeddings) < 2:
            return []
        
        analyses = []
        for i in range(len(self.store.embeddings) - 1):
            analysis = self.store.analyze_cross_modal_drift(i, i + 1)
            analyses.append({
                'from_session': i + 1,
                'to_session': i + 2,
                'similarity': round(analysis['sequence_similarity'], 3),
                'drift': round(analysis['sequence_drift'], 3)
            })
        
        return analyses
    
    def _limit_user_messages(self, user_messages: List[ChatMessage]) -> List[ChatMessage]:
        """Sample user messages to avoid runaway memory usage."""
        if len(user_messages) <= self.max_user_messages:
            return user_messages
        
        step = max(1, len(user_messages) // self.max_user_messages)
        return user_messages[::step][:self.max_user_messages]

def create_chat_analysis_summary(analysis: ConversationAnalysis) -> str:
    """Create a human-readable summary of chat analysis."""
    
    # Semantic drift interpretation
    if analysis.semantic_drift < 0.2:
        drift_desc = "very consistent thinking patterns"
    elif analysis.semantic_drift < 0.4:
        drift_desc = "moderate evolution in communication style"
    elif analysis.semantic_drift < 0.6:
        drift_desc = "significant changes in topics and thinking"
    else:
        drift_desc = "dramatic transformation in communication patterns"
    
    # Communication style insights
    patterns = analysis.communication_patterns
    avg_length = patterns.get('avg_message_length', 0)
    question_rate = patterns.get('question_rate', 0)
    
    style_insights = []
    if avg_length > 100:
        style_insights.append("tends to write detailed messages")
    elif avg_length < 30:
        style_insights.append("prefers concise communication")
    
    if question_rate > 0.3:
        style_insights.append("asks many questions (curious/exploratory style)")
    elif question_rate < 0.1:
        style_insights.append("more declarative communication style")
    
    # Time evolution
    time_info = f" over {analysis.time_span}" if analysis.time_span else ""
    
    summary = f"""
📊 **Your Conversation Analysis**

**Messages Analyzed**: {analysis.user_messages} of your messages{time_info}
**Semantic Evolution**: {drift_desc} (drift: {analysis.semantic_drift:.1%})

**Communication Style**: You {', '.join(style_insights) if style_insights else 'have a balanced communication approach'}.

**Topic Evolution**: Your conversations have touched on: {', '.join(analysis.topic_evolution[:5])}

**Key Metrics**:
- Average message length: {avg_length:.0f} characters
- Question frequency: {question_rate:.1%} of messages
- Vocabulary complexity: {patterns.get('complexity_rate', 0):.1%} complex words
"""
    
    return summary

# Example usage functions
def demo_chat_analysis():
    """Demo function for testing chat analysis."""
    
    # Sample conversation data
    sample_messages = [
        ChatMessage("Hi, I'm just getting started with Python programming", "user"),
        ChatMessage("Can you help me understand how lists work?", "user"), 
        ChatMessage("I'm working on a data analysis project now", "user"),
        ChatMessage("How do I optimize this machine learning model?", "user"),
        ChatMessage("I'm building a neural network for image classification", "user"),
    ]
    
    analyzer = ChatSemanticAnalyzer()
    analysis = analyzer.analyze_conversation_history(sample_messages)
    
    print(create_chat_analysis_summary(analysis))
    return analysis

if __name__ == "__main__":
    demo_chat_analysis() 
