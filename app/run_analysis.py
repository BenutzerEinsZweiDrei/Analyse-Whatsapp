"""
Main analysis orchestration for WhatsApp Analyzer.

Coordinates all analysis modules to process conversations and generate results.
"""

import logging
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Union

from app.cache import cached_data, cached_resource
from app.config import get_settings
from app.core.emojis import evaluate_emoji_string, extract_emojis
from app.core.keywords import get_keywords
from app.core.metrics import (
    calculate_emotional_reciprocity,
    calculate_response_times,
    calculate_topic_response_time,
)
from app.core.nouns import extract_nouns
from app.core.parser import merge_and_deduplicate_messages, parse_conversations_from_text
from app.core.personality import (
    calculate_big_five_scores,
    calculate_emotion_analysis,
    map_big_five_to_mbti,
)
from app.core.sentiment import analyze_sentiment
from app.services.jina_client import classify_texts

logger = logging.getLogger("whatsapp_analyzer")


@cached_resource
def get_empath_lexicon():
    """Load Empath lexicon once and cache it."""
    from empath import Empath

    logger.debug("Initializing Empath lexicon (cached)")
    return Empath()


@cached_resource
def get_emot_object():
    """Load emot object once and cache it."""
    import emot

    logger.debug("Initializing emot object (cached)")
    return emot.core.emot()


@cached_data(show_spinner=False)
def cached_run_analysis(
    file_content: Union[str, List[str]],
    username: str,
    file_metadata: List[dict] = None,
):
    """
    Cached wrapper for run_analysis to avoid recomputing identical analysis.

    Supports both single file (backward compatible) and multiple files.

    Args:
        file_content: WhatsApp chat export text (single string or list of strings)
        username: Username to analyze
        file_metadata: Optional list of dicts with filename, size, decode info

    Returns:
        Tuple of (matrix, conversation_messages)
    """
    logger.info(f"cached_run_analysis called for username={username} (may use cache)")

    # Handle both single file (backward compatible) and multiple files
    if isinstance(file_content, str):
        # Single file - legacy behavior
        return run_analysis(file_content, username)
    else:
        # Multiple files - merge and analyze
        return run_analysis_multiple_files(file_content, username, file_metadata or [])


def run_analysis_multiple_files(
    file_contents: List[str],
    username: str,
    file_metadata: List[dict],
) -> Tuple[Dict, Dict]:
    """
    Run analysis on multiple merged files.

    Args:
        file_contents: List of raw WhatsApp chat export texts
        username: Username to analyze
        file_metadata: List of dicts with filename, size, decode info

    Returns:
        Tuple of (matrix, conversation_messages)
    """
    logger.info(f"run_analysis_multiple_files: processing {len(file_contents)} files")

    # Parse each file separately with file_origin metadata
    all_conversations = []
    for idx, content in enumerate(file_contents):
        file_origin = (
            file_metadata[idx].get("filename", f"file_{idx+1}")
            if idx < len(file_metadata)
            else f"file_{idx+1}"
        )
        logger.debug(f"Parsing file {idx+1}/{len(file_contents)}: {file_origin}")

        conversations = parse_conversations_from_text(content, file_origin=file_origin)
        all_conversations.append(conversations)

        # Log parse stats
        total_messages = sum(len(conv) for conv in conversations)
        logger.info(
            f"File '{file_origin}': {len(conversations)} conversations, {total_messages} messages"
        )

    # Merge and deduplicate
    merged_conversations = merge_and_deduplicate_messages(all_conversations)

    # Continue with standard analysis on merged data
    return run_analysis_from_conversations(merged_conversations, username)


def run_analysis_from_conversations(
    conversations: List[List[dict]],
    username: str,
) -> Tuple[Dict, Dict]:
    """
    Run analysis on pre-parsed conversations.

    Args:
        conversations: Pre-parsed conversation list
        username: Username to analyze

    Returns:
        Tuple of (matrix, conversation_messages)
    """
    logger.info(f"run_analysis_from_conversations: {len(conversations)} conversations")
    start_time = time.time()

    # Get settings (API keys, etc.)
    settings = get_settings()

    # Load cached heavy resources
    empath_lex = get_empath_lexicon()
    emot_obj = get_emot_object()

    matrix = {}
    mergetext = {}
    conversation_messages = {}

    for idx, conv_msgs in enumerate(conversations, 1):
        t0 = time.time()
        matrix[idx] = {"idx": idx}

        # Store conversation messages for metrics
        conversation_messages[idx] = conv_msgs

        # Track file origins for this conversation
        file_origins = set(msg.get("file_origin") for msg in conv_msgs if msg.get("file_origin"))
        if file_origins:
            matrix[idx]["file_origins"] = list(file_origins)

        # Extract user messages
        user_msgs = [msg["message"] for msg in conv_msgs if msg.get("user") == username]
        mergetext[idx] = " ".join(user_msgs)
        text = mergetext[idx]

        logger.debug(
            f"Conversation #{idx}: user_messages={len(user_msgs)}, merged_text_len={len(text or '')}"
        )

        if not text.strip():
            logger.debug(f"Conversation #{idx} skipped (no user messages)")
            continue


def run_analysis(file_content: str, username: str) -> Tuple[Dict, Dict]:
    """
    Run complete conversation analysis pipeline.

    Processes WhatsApp chat export to extract:
    - Sentiment and emotions
    - Personality traits (Big Five, MBTI)
    - Topics and keywords
    - Response times and reciprocity
    - Emoji analysis

    Args:
        file_content: Raw WhatsApp chat export text
        username: Username to analyze

    Returns:
        Tuple of (matrix, conversation_messages) where:
        - matrix: Dict mapping conversation idx to analysis results
        - conversation_messages: Dict mapping conversation idx to message list
    """
    logger.info(
        f"run_analysis started for username={username}, content_length={len(file_content or '')}"
    )

    # Parse conversations
    conversations = parse_conversations_from_text(file_content)
    logger.debug(f"run_analysis: parsed {len(conversations)} conversations")

    # Use the shared processing logic
    return run_analysis_from_conversations(conversations, username)
