"""
Conversation metrics module for WhatsApp analysis.
Provides response time tracking and emotional reciprocity scoring.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger("whatsapp_analyzer")


def calculate_response_times(messages: List[Dict]) -> Dict[str, float]:
    """
    Calculate average response times per user based on message timestamps.
    
    Response time is defined as the time between a message and the previous
    message from a different user.
    
    Args:
        messages: List of message dictionaries with 'datetime', 'user', and 'message' keys
        
    Returns:
        Dictionary mapping username to average response time in minutes
    """
    logger.debug(f"Calculating response times for {len(messages)} messages")
    
    response_times = {}  # user -> list of response times in minutes
    
    for i in range(1, len(messages)):
        current_msg = messages[i]
        previous_msg = messages[i - 1]
        
        # Only calculate if different users and both have valid datetimes
        if (current_msg.get("user") != previous_msg.get("user") and 
            current_msg.get("datetime") and previous_msg.get("datetime")):
            
            try:
                current_time = datetime.fromisoformat(current_msg["datetime"])
                previous_time = datetime.fromisoformat(previous_msg["datetime"])
                
                # Calculate time difference in minutes
                time_diff = (current_time - previous_time).total_seconds() / 60.0
                
                # Only count reasonable response times (< 24 hours)
                if 0 < time_diff < 1440:  # 1440 minutes = 24 hours
                    user = current_msg["user"]
                    if user not in response_times:
                        response_times[user] = []
                    response_times[user].append(time_diff)
                    
            except (ValueError, TypeError) as e:
                logger.debug(f"Error parsing datetime for response time: {e}")
                continue
    
    # Calculate averages
    avg_response_times = {}
    for user, times in response_times.items():
        if times:
            avg_response_times[user] = round(sum(times) / len(times), 2)
    
    logger.debug(f"Average response times: {avg_response_times}")
    return avg_response_times


def calculate_topic_response_time(topic_messages: List[Dict]) -> float:
    """
    Calculate average response time for a specific topic/conversation.
    
    Args:
        topic_messages: List of messages belonging to one topic/conversation
        
    Returns:
        Average response time in minutes for this topic
    """
    response_times = calculate_response_times(topic_messages)
    
    if not response_times:
        return 0.0
    
    # Calculate overall average for the topic
    all_times = list(response_times.values())
    avg = sum(all_times) / len(all_times) if all_times else 0.0
    
    return round(avg, 2)


def calculate_emoji_valence(emojis: List[str], emoji_dict: Optional[Dict] = None) -> float:
    """
    Calculate the average emotional valence of emojis.
    
    Args:
        emojis: List of emojis
        emoji_dict: Optional dictionary mapping emoji to meaning (positiv/traurig/neutral)
        
    Returns:
        Average valence score: 1.0 (positive), 0.0 (neutral), -1.0 (negative)
    """
    if not emojis:
        return 0.0
    
    if emoji_dict is None:
        try:
            import json
            with open("data/emos.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                emoji_dict = {entry["emoji"]: entry["meaning"] for entry in data["emojis"]}
        except Exception as e:
            logger.debug(f"Could not load emoji dict: {e}")
            emoji_dict = {}
    
    valence_sum = 0.0
    count = 0
    
    for emoji in emojis:
        meaning = emoji_dict.get(emoji, "neutral")
        if meaning == "positiv":
            valence_sum += 1.0
        elif meaning == "traurig":
            valence_sum -= 1.0
        # neutral adds 0
        count += 1
    
    return valence_sum / count if count > 0 else 0.0


def calculate_emotional_reciprocity(messages: List[Dict], emoji_dict: Optional[Dict] = None) -> float:
    """
    Calculate emotional reciprocity as similarity between consecutive participants'
    sentiment or emoji valence within the same conversation/topic.
    
    Reciprocity is measured as the correlation/similarity of emotional expression
    between consecutive messages from different users.
    
    Args:
        messages: List of message dictionaries for one topic/conversation
        emoji_dict: Optional emoji-to-meaning mapping
        
    Returns:
        Reciprocity score (0-1, where 1 = perfect reciprocity)
    """
    logger.debug(f"Calculating emotional reciprocity for {len(messages)} messages")
    
    if len(messages) < 2:
        return 0.5  # Neutral reciprocity for single message
    
    reciprocity_scores = []
    
    for i in range(1, len(messages)):
        current_msg = messages[i]
        previous_msg = messages[i - 1]
        
        # Only compare consecutive messages from different users
        if current_msg.get("user") == previous_msg.get("user"):
            continue
        
        # Get sentiment scores if available
        current_sentiment = current_msg.get("sentiment_compound", 0)
        previous_sentiment = previous_msg.get("sentiment_compound", 0)
        
        # Get emoji valences
        current_emojis = current_msg.get("emojis", [])
        previous_emojis = previous_msg.get("emojis", [])
        
        current_emoji_valence = calculate_emoji_valence(current_emojis, emoji_dict)
        previous_emoji_valence = calculate_emoji_valence(previous_emojis, emoji_dict)
        
        # Combine sentiment and emoji for overall emotional tone
        current_emotion = (current_sentiment + current_emoji_valence) / 2.0 if current_emojis else current_sentiment
        previous_emotion = (previous_sentiment + previous_emoji_valence) / 2.0 if previous_emojis else previous_sentiment
        
        # Calculate similarity (reciprocity)
        # Using inverse of absolute difference, normalized to 0-1
        # If both emotions are in same direction (both positive or both negative), reciprocity is higher
        emotion_diff = abs(current_emotion - previous_emotion)
        similarity = 1.0 - min(emotion_diff / 2.0, 1.0)  # Normalize to 0-1
        
        # Bonus for matching emotional direction (both positive or both negative)
        if (current_emotion * previous_emotion) > 0:  # Same sign
            similarity = min(similarity + 0.1, 1.0)
        
        reciprocity_scores.append(similarity)
    
    if not reciprocity_scores:
        return 0.5  # Neutral if no valid comparisons
    
    # Average reciprocity across all consecutive pairs
    avg_reciprocity = sum(reciprocity_scores) / len(reciprocity_scores)
    
    logger.debug(f"Emotional reciprocity: {avg_reciprocity:.3f}")
    return round(avg_reciprocity, 3)


def aggregate_topic_metrics(
    topic: str,
    messages: List[Dict],
    big_five: Dict[str, float],
    mbti: str,
    emotion_analysis: Dict,
    emoji_dict: Optional[Dict] = None
) -> Dict:
    """
    Aggregate all computed metrics for a single topic.
    
    Args:
        topic: Topic label
        messages: Messages belonging to this topic
        big_five: Big Five personality scores
        mbti: MBTI type
        emotion_analysis: Emotion analysis results
        emoji_dict: Optional emoji dictionary
        
    Returns:
        Dictionary with all aggregated metrics for the topic
    """
    logger.debug(f"Aggregating metrics for topic: {topic}")
    
    # Calculate response times
    response_times = calculate_response_times(messages)
    avg_topic_response_time = calculate_topic_response_time(messages)
    
    # Calculate emotional reciprocity
    reciprocity_score = calculate_emotional_reciprocity(messages, emoji_dict)
    
    aggregated = {
        "topic": topic,
        "personality": {
            "big_five": big_five,
            "mbti": mbti
        },
        "emotions": emotion_analysis,
        "response_times": {
            "per_user": response_times,
            "topic_average": avg_topic_response_time
        },
        "emotional_reciprocity": reciprocity_score,
        "message_count": len(messages)
    }
    
    logger.debug(f"Topic metrics aggregated: {topic}")
    return aggregated
