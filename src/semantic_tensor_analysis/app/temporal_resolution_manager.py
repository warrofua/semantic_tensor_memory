#!/usr/bin/env python3
"""
Multi-Resolution Temporal Embedding System

Provides hierarchical temporal analysis:
- Turn-level: Individual conversation turns (user → assistant)
- Conversation-level: Complete conversation sessions 
- Day-level: Daily semantic aggregations
- Week/Month-level: Long-term temporal patterns

This addresses the core limitation where we were losing granular 
conversational dynamics by only having daily aggregations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import torch
from datetime import datetime, date, timedelta
import numpy as np
from collections import defaultdict

from semantic_tensor_analysis.memory import get_text_embedder
from semantic_tensor_analysis.memory.universal_core import UniversalEmbedding, UniversalMemoryStore
from semantic_tensor_analysis.chat.history_analyzer import ChatMessage


class TemporalResolution(Enum):
    """Different temporal resolutions for analysis."""
    TURN = "turn"              # Individual conversation turns
    CONVERSATION = "conversation"  # Complete conversation sessions
    DAY = "day"               # Daily aggregations
    WEEK = "week"             # Weekly patterns
    MONTH = "month"           # Monthly trends


@dataclass
class ConversationTurn:
    """A single conversation turn (user message + optional assistant response)."""
    turn_id: str
    user_message: ChatMessage
    assistant_message: Optional[ChatMessage] = None
    conversation_id: str = ""
    timestamp: Optional[datetime] = None
    turn_index: int = 0
    
    # Embeddings at different levels
    user_embedding: Optional[UniversalEmbedding] = None
    assistant_embedding: Optional[UniversalEmbedding] = None
    turn_summary_embedding: Optional[UniversalEmbedding] = None


@dataclass
class ConversationSession:
    """A complete conversation session containing multiple turns."""
    session_id: str
    conversation_id: str
    turns: List[ConversationTurn] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    date: Optional[date] = None
    
    # Session-level embeddings
    session_summary_embedding: Optional[UniversalEmbedding] = None
    semantic_trajectory: Optional[torch.Tensor] = None  # Turn-by-turn evolution
    
    # Metadata
    total_turns: int = 0
    user_message_count: int = 0
    assistant_message_count: int = 0
    session_duration_minutes: float = 0.0


@dataclass
class DayAggregation:
    """Daily aggregation of all conversations."""
    date: date
    conversations: List[ConversationSession] = field(default_factory=list)
    
    # Day-level embeddings
    day_summary_embedding: Optional[UniversalEmbedding] = None
    daily_semantic_trajectory: Optional[torch.Tensor] = None  # Conversation-by-conversation evolution
    
    # Daily statistics
    total_conversations: int = 0
    total_turns: int = 0
    total_messages: int = 0
    semantic_diversity_score: float = 0.0
    
    # Peak semantic activity periods (hours of day)
    peak_activity_hours: List[int] = field(default_factory=list)


class TemporalResolutionManager:
    """
    Manages multi-resolution temporal analysis of conversation data.
    
    Provides seamless zoom-in/zoom-out capability:
    - Zoom IN: Day → Conversations → Turns → Messages
    - Zoom OUT: Messages → Turns → Conversations → Days → Weeks
    """
    
    def __init__(self):
        # Reuse shared embedder to avoid repeated large model loads.
        self.text_embedder = get_text_embedder()
        self.universal_store = UniversalMemoryStore()
        
        # Multi-resolution storage
        self.turns: List[ConversationTurn] = []
        self.conversations: List[ConversationSession] = []
        self.daily_aggregations: Dict[date, DayAggregation] = {}
        
        # Index mappings for fast lookup
        self.turn_index: Dict[str, ConversationTurn] = {}
        self.conversation_index: Dict[str, ConversationSession] = {}
        
        # Temporal navigation
        self.current_resolution = TemporalResolution.DAY
        self.current_focus_date: Optional[date] = None
        self.current_focus_conversation: Optional[str] = None
    
    def process_conversation_messages(self, messages: List[ChatMessage]) -> Dict[str, Any]:
        """
        Process raw conversation messages into multi-resolution temporal structure.
        
        Returns:
            Dict with processing results and temporal structure
        """
        processing_results = {
            'total_messages': len(messages),
            'turns_created': 0,
            'conversations_created': 0,
            'days_covered': 0,
            'temporal_span_days': 0
        }
        
        # Group messages into conversation turns
        turns = self._group_messages_into_turns(messages)
        processing_results['turns_created'] = len(turns)
        
        # Group turns into conversation sessions
        conversations = self._group_turns_into_conversations(turns)
        processing_results['conversations_created'] = len(conversations)
        
        # Group conversations into daily aggregations
        daily_aggs = self._group_conversations_into_days(conversations)
        processing_results['days_covered'] = len(daily_aggs)
        
        # Calculate temporal span
        if daily_aggs:
            dates = list(daily_aggs.keys())
            processing_results['temporal_span_days'] = (max(dates) - min(dates)).days + 1
        
        # Generate embeddings at all resolutions
        self._generate_multi_resolution_embeddings(turns, conversations, daily_aggs)
        
        # Store in indices
        self._update_indices(turns, conversations, daily_aggs)
        
        return processing_results
    
    def _group_messages_into_turns(self, messages: List[ChatMessage]) -> List[ConversationTurn]:
        """Group messages into conversation turns (user → assistant pairs)."""
        turns = []
        current_user_message = None
        turn_counter = 0
        
        for msg in messages:
            if msg.role == 'user':
                # Start new turn
                if current_user_message is not None:
                    # Finish previous turn without assistant response
                    turn = ConversationTurn(
                        turn_id=f"turn_{turn_counter}",
                        user_message=current_user_message,
                        conversation_id=current_user_message.conversation_id or "unknown",
                        timestamp=current_user_message.timestamp,
                        turn_index=turn_counter
                    )
                    turns.append(turn)
                    turn_counter += 1
                
                current_user_message = msg
                
            elif msg.role == 'assistant' and current_user_message is not None:
                # Complete the turn
                turn = ConversationTurn(
                    turn_id=f"turn_{turn_counter}",
                    user_message=current_user_message,
                    assistant_message=msg,
                    conversation_id=current_user_message.conversation_id or "unknown",
                    timestamp=current_user_message.timestamp,
                    turn_index=turn_counter
                )
                turns.append(turn)
                turn_counter += 1
                current_user_message = None
        
        # Handle final user message without response
        if current_user_message is not None:
            turn = ConversationTurn(
                turn_id=f"turn_{turn_counter}",
                user_message=current_user_message,
                conversation_id=current_user_message.conversation_id or "unknown",
                timestamp=current_user_message.timestamp,
                turn_index=turn_counter
            )
            turns.append(turn)
        
        return turns
    
    def _group_turns_into_conversations(self, turns: List[ConversationTurn]) -> List[ConversationSession]:
        """Group turns into conversation sessions."""
        conversations_dict = defaultdict(list)
        
        # Group by conversation_id
        for turn in turns:
            conversations_dict[turn.conversation_id].append(turn)
        
        conversations = []
        for conv_id, conv_turns in conversations_dict.items():
            # Sort turns by timestamp or turn_index
            conv_turns.sort(key=lambda t: t.timestamp or datetime.min)
            
            # Extract session metadata
            start_time = conv_turns[0].timestamp if conv_turns[0].timestamp else None
            end_time = conv_turns[-1].timestamp if conv_turns[-1].timestamp else None
            session_date = start_time.date() if start_time else None
            
            duration = 0.0
            if start_time and end_time:
                duration = (end_time - start_time).total_seconds() / 60.0
            
            # Count messages
            user_count = sum(1 for t in conv_turns if t.user_message)
            assistant_count = sum(1 for t in conv_turns if t.assistant_message)
            
            session = ConversationSession(
                session_id=f"session_{conv_id}",
                conversation_id=conv_id,
                turns=conv_turns,
                start_time=start_time,
                end_time=end_time,
                date=session_date,
                total_turns=len(conv_turns),
                user_message_count=user_count,
                assistant_message_count=assistant_count,
                session_duration_minutes=duration
            )
            
            conversations.append(session)
        
        return conversations
    
    def _group_conversations_into_days(self, conversations: List[ConversationSession]) -> Dict[date, DayAggregation]:
        """Group conversations into daily aggregations."""
        daily_dict = defaultdict(list)
        
        for conv in conversations:
            if conv.date:
                daily_dict[conv.date].append(conv)
        
        daily_aggregations = {}
        for day_date, day_conversations in daily_dict.items():
            # Calculate daily statistics
            total_convs = len(day_conversations)
            total_turns = sum(c.total_turns for c in day_conversations)
            total_msgs = sum(c.user_message_count + c.assistant_message_count for c in day_conversations)
            
            # Calculate peak activity hours
            activity_hours = []
            for conv in day_conversations:
                if conv.start_time:
                    activity_hours.append(conv.start_time.hour)
            
            # Find peak hours (most active hours)
            if activity_hours:
                hour_counts = defaultdict(int)
                for hour in activity_hours:
                    hour_counts[hour] += 1
                peak_hours = [hour for hour, count in hour_counts.items() 
                             if count == max(hour_counts.values())]
            else:
                peak_hours = []
            
            daily_agg = DayAggregation(
                date=day_date,
                conversations=day_conversations,
                total_conversations=total_convs,
                total_turns=total_turns,
                total_messages=total_msgs,
                peak_activity_hours=peak_hours
            )
            
            daily_aggregations[day_date] = daily_agg
        
        return daily_aggregations
    
    def _generate_multi_resolution_embeddings(self, 
                                            turns: List[ConversationTurn],
                                            conversations: List[ConversationSession],
                                            daily_aggs: Dict[date, DayAggregation]):
        """Generate embeddings at all temporal resolutions."""
        
        # 1. Turn-level embeddings
        for turn in turns:
            # User message embedding
            if turn.user_message:
                turn.user_embedding = self.text_embedder.process_raw_data(
                    turn.user_message.content,
                    session_id=f"{turn.turn_id}_user"
                )
                self.universal_store.add_session(turn.user_embedding)
            
            # Assistant message embedding
            if turn.assistant_message:
                turn.assistant_embedding = self.text_embedder.process_raw_data(
                    turn.assistant_message.content,
                    session_id=f"{turn.turn_id}_assistant"
                )
                self.universal_store.add_session(turn.assistant_embedding)
            
            # Turn summary embedding (combine user + assistant)
            turn_text = turn.user_message.content
            if turn.assistant_message:
                turn_text += "\n\n" + turn.assistant_message.content
            
            turn.turn_summary_embedding = self.text_embedder.process_raw_data(
                turn_text,
                session_id=f"{turn.turn_id}_summary"
            )
            self.universal_store.add_session(turn.turn_summary_embedding)
        
        # 2. Conversation-level embeddings
        for conversation in conversations:
            # Collect all text from conversation
            conversation_text = ""
            turn_embeddings = []
            
            for turn in conversation.turns:
                if turn.user_message:
                    conversation_text += f"User: {turn.user_message.content}\n"
                if turn.assistant_message:
                    conversation_text += f"Assistant: {turn.assistant_message.content}\n"
                
                if turn.turn_summary_embedding:
                    turn_embeddings.append(turn.turn_summary_embedding.sequence_embedding)
            
            # Session summary embedding
            conversation.session_summary_embedding = self.text_embedder.process_raw_data(
                conversation_text,
                session_id=conversation.session_id
            )
            self.universal_store.add_session(conversation.session_summary_embedding)
            
            # Semantic trajectory (turn-by-turn evolution)
            if turn_embeddings:
                conversation.semantic_trajectory = torch.stack(turn_embeddings)
        
        # 3. Day-level embeddings
        for day_date, daily_agg in daily_aggs.items():
            # Collect all text from the day
            day_text = ""
            conversation_embeddings = []
            
            for conversation in daily_agg.conversations:
                for turn in conversation.turns:
                    if turn.user_message:
                        day_text += f"{turn.user_message.content}\n"
                
                if conversation.session_summary_embedding:
                    conversation_embeddings.append(conversation.session_summary_embedding.sequence_embedding)
            
            # Day summary embedding
            daily_agg.day_summary_embedding = self.text_embedder.process_raw_data(
                day_text,
                session_id=f"day_{day_date.isoformat()}"
            )
            self.universal_store.add_session(daily_agg.day_summary_embedding)
            
            # Daily semantic trajectory (conversation-by-conversation evolution)
            if conversation_embeddings:
                daily_agg.daily_semantic_trajectory = torch.stack(conversation_embeddings)
                
                # Calculate semantic diversity score
                if len(conversation_embeddings) > 1:
                    # Calculate pairwise cosine similarities
                    similarities = []
                    for i in range(len(conversation_embeddings)):
                        for j in range(i+1, len(conversation_embeddings)):
                            sim = torch.cosine_similarity(
                                conversation_embeddings[i], 
                                conversation_embeddings[j], 
                                dim=0
                            ).item()
                            similarities.append(sim)
                    
                    # Diversity is inverse of average similarity
                    daily_agg.semantic_diversity_score = 1.0 - np.mean(similarities)
                else:
                    daily_agg.semantic_diversity_score = 0.0
    
    def _update_indices(self, 
                       turns: List[ConversationTurn],
                       conversations: List[ConversationSession],
                       daily_aggs: Dict[date, DayAggregation]):
        """Update internal indices for fast lookup."""
        self.turns.extend(turns)
        self.conversations.extend(conversations)
        self.daily_aggregations.update(daily_aggs)
        
        # Update lookup indices
        for turn in turns:
            self.turn_index[turn.turn_id] = turn
        
        for conversation in conversations:
            self.conversation_index[conversation.session_id] = conversation
    
    def get_embeddings_at_resolution(self, resolution: TemporalResolution) -> List[torch.Tensor]:
        """Get embeddings at specified temporal resolution."""
        embeddings = []
        
        if resolution == TemporalResolution.TURN:
            for turn in self.turns:
                if turn.turn_summary_embedding:
                    embeddings.append(turn.turn_summary_embedding.sequence_embedding)
        
        elif resolution == TemporalResolution.CONVERSATION:
            for conversation in self.conversations:
                if conversation.session_summary_embedding:
                    embeddings.append(conversation.session_summary_embedding.sequence_embedding)
        
        elif resolution == TemporalResolution.DAY:
            for daily_agg in self.daily_aggregations.values():
                if daily_agg.day_summary_embedding:
                    embeddings.append(daily_agg.day_summary_embedding.sequence_embedding)
        
        return embeddings
    
    def get_metadata_at_resolution(self, resolution: TemporalResolution) -> List[Dict[str, Any]]:
        """Get metadata at specified temporal resolution."""
        metadata = []
        
        if resolution == TemporalResolution.TURN:
            for turn in self.turns:
                metadata.append({
                    'turn_id': turn.turn_id,
                    'conversation_id': turn.conversation_id,
                    'timestamp': turn.timestamp,
                    'turn_index': turn.turn_index,
                    'has_user_message': turn.user_message is not None,
                    'has_assistant_message': turn.assistant_message is not None,
                    'user_text': turn.user_message.content if turn.user_message else "",
                    'assistant_text': turn.assistant_message.content if turn.assistant_message else ""
                })
        
        elif resolution == TemporalResolution.CONVERSATION:
            for conversation in self.conversations:
                metadata.append({
                    'session_id': conversation.session_id,
                    'conversation_id': conversation.conversation_id,
                    'date': conversation.date,
                    'start_time': conversation.start_time,
                    'end_time': conversation.end_time,
                    'total_turns': conversation.total_turns,
                    'user_message_count': conversation.user_message_count,
                    'assistant_message_count': conversation.assistant_message_count,
                    'duration_minutes': conversation.session_duration_minutes
                })
        
        elif resolution == TemporalResolution.DAY:
            for daily_agg in self.daily_aggregations.values():
                metadata.append({
                    'date': daily_agg.date,
                    'total_conversations': daily_agg.total_conversations,
                    'total_turns': daily_agg.total_turns,
                    'total_messages': daily_agg.total_messages,
                    'semantic_diversity_score': daily_agg.semantic_diversity_score,
                    'peak_activity_hours': daily_agg.peak_activity_hours
                })
        
        return metadata
    
    def zoom_to_resolution(self, resolution: TemporalResolution, 
                          focus_date: Optional[date] = None,
                          focus_conversation: Optional[str] = None) -> Dict[str, Any]:
        """
        Change temporal resolution and return relevant data.
        
        This enables the "zoom in/out" functionality.
        """
        self.current_resolution = resolution
        self.current_focus_date = focus_date
        self.current_focus_conversation = focus_conversation
        
        embeddings = self.get_embeddings_at_resolution(resolution)
        metadata = self.get_metadata_at_resolution(resolution)
        
        # Filter by focus if specified
        if focus_date and resolution in [TemporalResolution.CONVERSATION, TemporalResolution.TURN]:
            filtered_embeddings = []
            filtered_metadata = []
            
            for i, meta in enumerate(metadata):
                if resolution == TemporalResolution.CONVERSATION:
                    if meta.get('date') == focus_date:
                        filtered_embeddings.append(embeddings[i])
                        filtered_metadata.append(meta)
                elif resolution == TemporalResolution.TURN:
                    # Find turn's conversation and check date
                    turn = self.turn_index.get(meta['turn_id'])
                    if turn and turn.conversation_id in self.conversation_index:
                        conv = self.conversation_index[turn.conversation_id]
                        if conv.date == focus_date:
                            filtered_embeddings.append(embeddings[i])
                            filtered_metadata.append(meta)
            
            embeddings = filtered_embeddings
            metadata = filtered_metadata
        
        if focus_conversation and resolution == TemporalResolution.TURN:
            filtered_embeddings = []
            filtered_metadata = []
            
            for i, meta in enumerate(metadata):
                if meta.get('conversation_id') == focus_conversation:
                    filtered_embeddings.append(embeddings[i])
                    filtered_metadata.append(meta)
            
            embeddings = filtered_embeddings
            metadata = filtered_metadata
        
        return {
            'embeddings': embeddings,
            'metadata': metadata,
            'resolution': resolution,
            'focus_date': focus_date,
            'focus_conversation': focus_conversation,
            'total_items': len(embeddings)
        }
    
    def get_temporal_navigation_options(self) -> Dict[str, Any]:
        """Get available navigation options based on current data."""
        options = {
            'available_dates': sorted(list(self.daily_aggregations.keys())),
            'available_conversations': sorted(list(self.conversation_index.keys())),
            'total_turns': len(self.turns),
            'total_conversations': len(self.conversations),
            'total_days': len(self.daily_aggregations),
            'date_range': None
        }
        
        if self.daily_aggregations:
            dates = list(self.daily_aggregations.keys())
            options['date_range'] = {
                'start': min(dates),
                'end': max(dates),
                'span_days': (max(dates) - min(dates)).days + 1
            }
        
        return options
