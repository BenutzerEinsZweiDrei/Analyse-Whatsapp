"""
Type definitions for WhatsApp Analyzer.

Defines dataclasses and types used throughout the application for analysis results.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Message:
    """
    Represents a single WhatsApp message.
    
    Note: The canonical Message class is in app.core.parser.
    This is a re-export for convenience.
    """

    datetime: Optional[str]
    date: str
    time: str
    user: Optional[str]
    message: str
    file_origin: Optional[str] = None


@dataclass
class AnalysisResult:
    """
    Complete analysis result for a WhatsApp conversation.
    
    Contains all metrics, topics, sentiment, and personality data.
    """

    # Core data
    matrix: Dict[str, Any]
    summary: Dict[str, Any]
    conversation_messages: List[Dict[str, Any]]
    
    # Metadata
    username: str
    total_messages: int = 0
    analysis_time: float = 0.0
    
    # Optional components
    topics: List[str] = field(default_factory=list)
    positive_topics: List[str] = field(default_factory=list)
    negative_topics: List[str] = field(default_factory=list)
    sentiment_score: float = 0.0
    emotion_variability: float = 0.0


@dataclass
class PersonalityProfile:
    """
    Personality profile result from local or AI analysis.
    """

    # Profile content
    profile_text: str
    results: Dict[str, Any]
    
    # Metadata
    generation_method: str  # "local" or "ai"
    confidence: float = 0.0
    
    # Personality traits (if available)
    big_five: Optional[Dict[str, float]] = None
    mbti_type: Optional[str] = None
    
    # Export data
    exports: Dict[str, str] = field(default_factory=dict)


@dataclass
class MergedProfile:
    """
    Result from merging multiple personality profiles.
    """

    profile_text: str
    results: Dict[str, Any]
    merged_from: List[str]  # List of source file names
    num_profiles_merged: int = 0
    exports: Dict[str, str] = field(default_factory=dict)


# Type aliases for common data structures
ConversationMatrix = Dict[str, Any]
SummaryData = Dict[str, Any]
MessageList = List[Dict[str, Any]]
