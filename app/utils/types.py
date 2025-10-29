"""
Type definitions for WhatsApp Analyzer.

Defines dataclasses and types used throughout the application for analysis results.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    """
    Represents a single WhatsApp message.

    Note: The canonical Message class is in app.core.parser.
    This is a re-export for convenience.
    """

    datetime: str | None
    date: str
    time: str
    user: str | None
    message: str
    file_origin: str | None = None


@dataclass
class AnalysisResult:
    """
    Complete analysis result for a WhatsApp conversation.

    Contains all metrics, topics, sentiment, and personality data.
    """

    # Core data
    matrix: dict[str, Any]
    summary: dict[str, Any]
    conversation_messages: list[dict[str, Any]]

    # Metadata
    username: str
    total_messages: int = 0
    analysis_time: float = 0.0

    # Optional components
    topics: list[str] = field(default_factory=list)
    positive_topics: list[str] = field(default_factory=list)
    negative_topics: list[str] = field(default_factory=list)
    sentiment_score: float = 0.0
    emotion_variability: float = 0.0


@dataclass
class PersonalityProfile:
    """
    Personality profile result from local or AI analysis.
    """

    # Profile content
    profile_text: str
    results: dict[str, Any]

    # Metadata
    generation_method: str  # "local" or "ai"
    confidence: float = 0.0

    # Personality traits (if available)
    big_five: dict[str, float] | None = None
    mbti_type: str | None = None

    # Export data
    exports: dict[str, str] = field(default_factory=dict)


@dataclass
class MergedProfile:
    """
    Result from merging multiple personality profiles.
    """

    profile_text: str
    results: dict[str, Any]
    merged_from: list[str]  # List of source file names
    num_profiles_merged: int = 0
    exports: dict[str, str] = field(default_factory=dict)


# Type aliases for common data structures
ConversationMatrix = dict[str, Any]
SummaryData = dict[str, Any]
MessageList = list[dict[str, Any]]
