"""
Noun extraction for WhatsApp Analyzer.

Extracts and filters meaningful nouns from text using POS tagging.
"""

import logging
from collections import Counter
from typing import List
from nltk.tag import pos_tag

from app.core.preprocessing import preprocess_text, init_nltk

logger = logging.getLogger("whatsapp_analyzer")


def extract_nouns(text: str, max_nouns: int = 20) -> List[str]:
    """
    Extract meaningful nouns from text with improved filtering.
    
    Prioritizes proper nouns and common nouns while filtering out low-quality terms.
    
    Args:
        text: Input text to analyze
        max_nouns: Maximum number of nouns to return
        
    Returns:
        List of filtered nouns
        
    Example:
        >>> extract_nouns("Alice and Bob went to the park")
        ['Alice', 'Bob', 'park']
    """
    logger.debug("extract_nouns called")
    
    if not text or not text.strip():
        return []
    
    # Ensure NLTK resources are available
    init_nltk()
    
    # Preprocess text
    tokens = preprocess_text(text)
    
    if not tokens:
        return []
    
    # POS tag tokens
    try:
        tagged_tokens = pos_tag(tokens)
    except Exception as e:
        logger.warning(f"POS tagging failed: {e}")
        return []
    
    # Tags to filter out (verbs, adjectives, adverbs, particles)
    filter_tags = {
        "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",  # Verbs
        "JJ", "JJR", "JJS",                        # Adjectives
        "RB", "RBR", "RBS",                        # Adverbs
        "RP"                                        # Particles
    }
    
    # Extract nouns (NN, NNS, NNP, NNPS)
    nouns = [word for word, pos in tagged_tokens 
             if pos in ("NN", "NNS", "NNP", "NNPS") and pos not in filter_tags]
    
    # Count noun frequencies to prioritize important terms
    noun_freq = Counter(nouns)
    
    # Identify proper nouns (capitalized in original text, high POS confidence)
    proper_nouns = {word for word, pos in tagged_tokens if pos in ("NNP", "NNPS")}
    
    # Filter nouns: keep those that appear more than once OR are proper nouns
    filtered_nouns = []
    seen = set()
    
    for noun in nouns:
        if noun not in seen:
            # Keep if: appears multiple times, is a proper noun, or is sufficiently long
            if noun_freq[noun] > 1 or noun in proper_nouns or len(noun) > 5:
                filtered_nouns.append(noun)
                seen.add(noun)
    
    logger.debug(f"extract_nouns found {len(nouns)} nouns (filtered to {len(filtered_nouns)})")
    
    # Limit to max_nouns
    return filtered_nouns[:max_nouns]
