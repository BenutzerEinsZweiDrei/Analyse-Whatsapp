"""
Emoji extraction and evaluation for WhatsApp Analyzer.

Provides functions to extract emojis from text and evaluate their sentiment.
"""

import logging
import re
from typing import List, Dict, Optional

from app.data.loaders import load_emoji_mappings

logger = logging.getLogger("whatsapp_analyzer")


def extract_emojis(text: str) -> List[str]:
    """
    Extract emojis from text.
    
    Tries to use emot library if available, otherwise falls back to regex.
    
    Args:
        text: Input text containing emojis
        
    Returns:
        List of extracted emojis
        
    Example:
        >>> extract_emojis("Hello 😊 World 👍")
        ['😊', '👍']
    """
    logger.debug(f"Extracting emojis from text (length={len(text)})")
    
    if not text:
        return []
    
    # Try using emot library
    try:
        import emot
        emot_obj = emot.core.emot()
        emoji_result = emot_obj.emoji(text)
        emojis = emoji_result.get("value", [])
        logger.debug(f"Extracted {len(emojis)} emojis using emot")
        return emojis
    except ImportError:
        logger.debug("emot library not available, using regex fallback")
    except Exception as e:
        logger.warning(f"emot extraction failed: {e}, using regex fallback")
    
    # Fallback: regex-based emoji extraction
    # Unicode emoji ranges
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA00-\U0001FA6F"  # chess symbols
        "]+",
        flags=re.UNICODE
    )
    
    emojis = emoji_pattern.findall(text)
    logger.debug(f"Extracted {len(emojis)} emojis using regex")
    return emojis


def evaluate_emoji_string(emojis: List[str], emoji_dict: Optional[Dict[str, str]] = None) -> str:
    """
    Evaluate the sentiment of a list of emojis.
    
    Returns one of: "sehr positiv", "eher positiv", "neutral", "eher traurig", "sehr traurig"
    
    Args:
        emojis: List of emojis to evaluate
        emoji_dict: Optional emoji-to-meaning mapping. If None, loads from file.
        
    Returns:
        Sentiment label based on emoji meanings
        
    Example:
        >>> evaluate_emoji_string(['😊', '😄'])
        'sehr positiv'
        >>> evaluate_emoji_string(['😢', '😭'])
        'sehr traurig'
    """
    logger.debug(f"evaluate_emoji_string called with {len(emojis) if emojis else 0} emojis")
    
    if not emojis:
        return "neutral"
    
    # Load emoji mappings if not provided
    if emoji_dict is None:
        try:
            emoji_dict = load_emoji_mappings("emos.json")
            logger.debug(f"Loaded emoji mapping ({len(emoji_dict)} entries)")
        except Exception as e:
            logger.warning(f"Could not load emoji mappings: {e}")
            emoji_dict = {}
    
    if not emoji_dict:
        return "neutral"
    
    # Calculate sentiment score
    score = 0
    num = len(emojis)
    
    for emoji in emojis:
        meaning = emoji_dict.get(emoji, "neutral")
        if meaning == "positiv":
            score += 1
        elif meaning == "traurig":
            score -= 1
        # neutral adds 0
    
    # Return neutral if score is 0
    if score == 0:
        return "neutral"
    
    # Calculate ratio: -1 to +1 normalized to 0 to 1
    anteil = (score / num + 1) / 2
    
    # Classify based on ratio
    if anteil >= 2 / 3:
        return "sehr positiv"
    elif anteil >= 0.5:
        return "eher positiv"
    elif anteil >= 0.3:
        return "eher traurig"
    else:
        return "sehr traurig"


def bewerte_emoji_string(emojis_im_text: List[str], emoji_dict: Optional[Dict[str, str]] = None) -> str:
    """
    Legacy function name for backward compatibility.
    
    Wrapper around evaluate_emoji_string.
    
    Args:
        emojis_im_text: List of emojis to evaluate
        emoji_dict: Optional emoji-to-meaning mapping
        
    Returns:
        Sentiment label
    """
    return evaluate_emoji_string(emojis_im_text, emoji_dict)
