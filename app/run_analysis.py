"""
Main analysis orchestration for WhatsApp Analyzer.

Coordinates all analysis modules to process conversations and generate results.
"""

import logging
import time
from typing import Dict, List, Tuple
from collections import defaultdict

from app.config import get_settings
from app.cache import cached_resource, cached_data
from app.core.parser import parse_conversations_from_text
from app.core.emojis import extract_emojis, evaluate_emoji_string
from app.core.keywords import get_keywords
from app.core.nouns import extract_nouns
from app.core.sentiment import analyze_sentiment
from app.core.personality import (
    calculate_big_five_scores,
    map_big_five_to_mbti,
    calculate_emotion_analysis
)
from app.core.metrics import (
    calculate_response_times,
    calculate_topic_response_time,
    calculate_emotional_reciprocity
)
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
def cached_run_analysis(file_content: str, username: str):
    """
    Cached wrapper for run_analysis to avoid recomputing identical analysis.
    
    Args:
        file_content: WhatsApp chat export text
        username: Username to analyze
        
    Returns:
        Tuple of (matrix, conversation_messages)
    """
    logger.info(f"cached_run_analysis called for username={username} (may use cache)")
    return run_analysis(file_content, username)


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
    logger.info(f"run_analysis started for username={username}, content_length={len(file_content or '')}")
    start_time = time.time()
    
    # Get settings (API keys, etc.)
    settings = get_settings()
    
    # Load cached heavy resources
    empath_lex = get_empath_lexicon()
    emot_obj = get_emot_object()
    
    # Parse conversations
    conversations = parse_conversations_from_text(file_content)
    logger.debug(f"run_analysis: parsed {len(conversations)} conversations")
    
    matrix = {}
    mergetext = {}
    conversation_messages = {}
    
    for idx, conv_msgs in enumerate(conversations, 1):
        t0 = time.time()
        matrix[idx] = {"idx": idx}
        
        # Store conversation messages for metrics
        conversation_messages[idx] = conv_msgs
        
        # Extract user messages
        user_msgs = [msg["message"] for msg in conv_msgs if msg.get("user") == username]
        mergetext[idx] = " ".join(user_msgs)
        text = mergetext[idx]
        
        logger.debug(f"Conversation #{idx}: user_messages={len(user_msgs)}, merged_text_len={len(text or '')}")
        
        if not text.strip():
            logger.debug(f"Conversation #{idx} skipped (no user messages)")
            continue
        
        # Extract emojis
        try:
            emoji_result = emot_obj.emoji(text)
            emojis = emoji_result.get("value", [])
            matrix[idx]["emojies"] = emojis
            matrix[idx]["emo_bew"] = [evaluate_emoji_string(emojis)]
            logger.debug(f"Conversation #{idx}: emojis={len(emojis)}, emo_bew={matrix[idx]['emo_bew']}")
        except Exception as e:
            logger.exception(f"Error extracting emojis for conversation {idx}: {e}")
            matrix[idx]["emojies"] = []
            matrix[idx]["emo_bew"] = ["error"]
        
        # Empath lexical analysis
        try:
            lex_analysis = empath_lex.analyze(text, normalize=True) if text else {}
            filtered_lex = {k: v for k, v in lex_analysis.items() if v > 0}
            matrix[idx]["lex"] = filtered_lex
            logger.debug(f"Conversation #{idx}: empath_categories={len(filtered_lex)}")
        except Exception as e:
            logger.exception(f"Error running Empath analysis for conversation {idx}: {e}")
            matrix[idx]["lex"] = {}
        
        # Sentiment analysis
        if matrix[idx].get("lex"):
            try:
                t_call = time.time()
                sentiment_result = analyze_sentiment(text)
                
                matrix[idx]["sentiment"] = [sentiment_result["label"]]
                matrix[idx]["sent_rating"] = [sentiment_result["scaled_rating"]]
                matrix[idx]["vader_scores"] = sentiment_result["vader"]
                matrix[idx]["sentiment_compound"] = sentiment_result["compound"]
                
                logger.debug(
                    f"Conversation #{idx}: sentiment={sentiment_result['label']}, "
                    f"rating={sentiment_result['scaled_rating']}, "
                    f"compound={sentiment_result['compound']:.3f}, "
                    f"time={time.time() - t_call:.2f}s"
                )
            except Exception as e:
                logger.exception(f"Error analyzing sentiment for conversation {idx}: {e}")
                matrix[idx]["sentiment"] = ["error"]
                matrix[idx]["sent_rating"] = []
                matrix[idx]["sentiment_compound"] = 0.0
        
        # Big Five personality traits
        try:
            emojis = matrix[idx].get("emojies", [])
            big_five = calculate_big_five_scores(text, emojis)
            matrix[idx]["big_five"] = big_five
            logger.debug(f"Conversation #{idx}: Big Five calculated")
        except Exception as e:
            logger.exception(f"Error calculating Big Five for conversation {idx}: {e}")
            matrix[idx]["big_five"] = {}
        
        # MBTI mapping
        try:
            if matrix[idx].get("big_five"):
                mbti = map_big_five_to_mbti(matrix[idx]["big_five"])
                matrix[idx]["mbti"] = mbti
                logger.debug(f"Conversation #{idx}: MBTI={mbti}")
        except Exception as e:
            logger.exception(f"Error mapping MBTI for conversation {idx}: {e}")
            matrix[idx]["mbti"] = "XXXX"
        
        # Enhanced emotion analysis
        try:
            emojis = matrix[idx].get("emojies", [])
            compound = matrix[idx].get("sentiment_compound", 0.0)
            emotion_analysis = calculate_emotion_analysis(emojis, compound)
            matrix[idx]["emotion_analysis"] = emotion_analysis
            logger.debug(f"Conversation #{idx}: emotion={emotion_analysis.get('dominant_emotion')}")
        except Exception as e:
            logger.exception(f"Error in emotion analysis for conversation {idx}: {e}")
            matrix[idx]["emotion_analysis"] = {}
        
        # Extract keywords and nouns
        matrix[idx]["keywords"] = get_keywords(text) if text else []
        matrix[idx]["nouns"] = extract_nouns(text) if text else []
        
        # Combine and deduplicate words
        combined_words = []
        seen_lower = set()
        
        # Add keywords first (weighted by topic relevance)
        for keyword in matrix[idx]["keywords"]:
            keyword_lower = keyword.lower()
            if keyword_lower not in seen_lower and len(keyword) > 2:
                combined_words.append(keyword)
                seen_lower.add(keyword_lower)
        
        # Add nouns that aren't already present
        for noun in matrix[idx]["nouns"]:
            noun_lower = noun.lower()
            if noun_lower not in seen_lower and len(noun) > 2:
                combined_words.append(noun)
                seen_lower.add(noun_lower)
        
        # Limit total categories
        matrix[idx]["words"] = combined_words[:20] or ["no topic"]
        
        # Topic classification
        categories = matrix[idx]["words"]
        try:
            if categories and settings.jina_api_key:
                result = classify_texts(text, categories, api_key=settings.jina_api_key)
                topic_pred = result["data"][0]["prediction"]
                matrix[idx]["topic"] = [topic_pred]
                logger.debug(f"Conversation #{idx}: topic={topic_pred}")
            else:
                matrix[idx]["topic"] = ["no topic"]
                if not settings.jina_api_key:
                    logger.debug(f"Conversation #{idx}: no API key for topic classification")
        except Exception as e:
            logger.exception(f"Error classifying topic for conversation {idx}: {e}")
            matrix[idx]["topic"] = ["error"]
        
        # Response times
        try:
            response_times = calculate_response_times(conv_msgs)
            topic_avg_response = calculate_topic_response_time(conv_msgs)
            matrix[idx]["response_times"] = {
                "per_user": response_times,
                "topic_average": topic_avg_response
            }
            logger.debug(f"Conversation #{idx}: response times calculated")
        except Exception as e:
            logger.exception(f"Error calculating response times for conversation {idx}: {e}")
            matrix[idx]["response_times"] = {"per_user": {}, "topic_average": 0.0}
        
        # Emotional reciprocity
        try:
            # Enrich messages with sentiment for reciprocity calculation
            enriched_msgs = []
            for msg in conv_msgs:
                enriched_msg = msg.copy()
                enriched_msg["sentiment_compound"] = matrix[idx].get("sentiment_compound", 0.0)
                # Extract emojis for each message
                try:
                    msg_emoji_result = emot_obj.emoji(msg.get("message", ""))
                    enriched_msg["emojis"] = msg_emoji_result.get("value", [])
                except:
                    enriched_msg["emojis"] = []
                enriched_msgs.append(enriched_msg)
            
            reciprocity = calculate_emotional_reciprocity(enriched_msgs)
            matrix[idx]["emotional_reciprocity"] = reciprocity
            logger.debug(f"Conversation #{idx}: emotional_reciprocity={reciprocity:.3f}")
        except Exception as e:
            logger.exception(f"Error calculating emotional reciprocity for conversation {idx}: {e}")
            matrix[idx]["emotional_reciprocity"] = 0.5
        
        logger.debug(f"Conversation #{idx} processing time: {time.time() - t0:.2f}s")
    
    logger.info(f"run_analysis finished in {time.time() - start_time:.2f}s")
    return matrix, conversation_messages
