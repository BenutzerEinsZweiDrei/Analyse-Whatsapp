# Enhanced Analysis Features

This document describes the enhanced analysis capabilities added to the WhatsApp Conversation Analyzer.

## Overview

The analyzer has been significantly enhanced with:
- **Robust feature extraction** with comprehensive message metadata
- **Ensemble sentiment analysis** combining multiple methods
- **Multi-method emotion detection** with confidence scores
- **Advanced topic extraction** with coherence measurement
- **Enriched analysis outputs** with evidence and confidence

## New Modules

### 1. Feature Extraction (`app/core/feature_extraction.py`)

Extracts comprehensive per-message metadata including:

**Temporal Features:**
- Timestamp parsing and normalization to UTC
- Time-of-day and day-of-week analysis
- Inter-message intervals (for conversation dynamics)

**Text Statistics:**
- Token count, word count, sentence count
- Average word length
- Lexical diversity (type-token ratio)
- Stopword ratio
- Uppercase ratio
- Punctuation density

**Special Elements:**
- Emoji extraction and normalization (with descriptive names)
- Emoticon detection (text-based smileys)
- URL detection and extraction
- @mention detection
- Hashtag detection
- Question/exclamation mark counting

**Language Detection:**
- Per-message language detection using langdetect
- Aggregate language statistics
- Dominant language identification

**Usage:**
```python
from app.core.feature_extraction import extract_message_features
from datetime import datetime

features = extract_message_features(
    message_id=1,
    author="Alice",
    text="Hey @Bob, check out this link: https://example.com 😊",
    timestamp=datetime.now(),
)

print(f"Emoji count: {features.emoji_count}")
print(f"Mention count: {features.mention_count}")
print(f"URL count: {features.url_count}")
print(f"Detected language: {features.detected_language}")
```

### 2. Enhanced Sentiment Analysis (`app/core/sentiment_enhanced.py`)

Provides ensemble sentiment analysis combining:

**Methods:**
- VADER lexicon-based sentiment (baseline, always available)
- Transformer models (optional, when transformers library installed)
  - Default: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- Emoji sentiment mapping
- Weighted ensemble combination

**Outputs:**
- Polarity: positive/negative/neutral
- Compound score: -1 to 1
- Scaled rating: 0 to 10
- Confidence score: 0 to 1
- Method used: indicates which approach was used

**Usage:**
```python
from app.core.sentiment_enhanced import analyze_sentiment_ensemble

result = analyze_sentiment_ensemble(
    text="I love this new feature! 🎉",
    emojis=["🎉"],
    use_transformer=True  # Set False to use only VADER
)

print(f"Sentiment: {result.polarity}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Method: {result.method_used}")
```

**Configuration:**
- Works with or without transformers library
- Automatically falls back to lexicon-based if transformers unavailable
- Emoji sentiment provides additional signal

### 3. Emotion Detection (`app/core/emotion_detection.py`)

Multi-method emotion classification:

**Emotion Categories:**
- Joy, Sadness, Anger, Fear, Disgust, Surprise, Neutral

**Methods:**
- Keyword-based detection (multilingual: English + German)
- Emoji-to-emotion mapping
- Transformer models (optional)
  - Default: `j-hartmann/emotion-english-distilroberta-base`
- Ensemble combination

**Outputs:**
- Primary emotion with score
- Scores for all 7 emotion categories
- Confidence score
- Method indicator

**Usage:**
```python
from app.core.emotion_detection import detect_emotion

result = detect_emotion(
    text="I'm so happy about this!",
    emojis=["😊"],
    use_transformer=True
)

print(f"Primary: {result.primary_emotion}")
print(f"Score: {result.primary_score:.2f}")
print(f"All emotions: {result.emotion_scores}")
```

### 4. Topic Extraction (`app/core/topic_extraction.py`)

Advanced keyword and topic extraction:

**Methods:**
- TF-IDF (simple, fast baseline)
- YAKE (statistical keyword extraction)
- KeyBERT (embedding-based, semantically aware)
- Automatic method selection with fallbacks

**Outputs:**
- Top keywords with scores and support
- Topic coherence score
- Representative messages (evidence)
- Method used

**Usage:**
```python
from app.core.topic_extraction import extract_topics

texts = ["Python programming is fun", "Learning Python today", "Code review"]
message_ids = [1, 2, 3]

result = extract_topics(texts, message_ids, top_n=5)

print(f"Keywords: {result.keywords}")
print(f"Coherence: {result.coherence_score:.2f}")
print(f"Representative messages: {result.representative_messages}")
```

## Enhanced Outputs

### Enriched Summary (from `summarize_matrix`)

The summary now includes additional fields:

**New Fields:**
```python
{
    # Existing fields (preserved for backward compatibility)
    "positive_topics": [...],
    "negative_topics": [...],
    "emotion_variability": 1.23,
    "matrix": {...},
    "analysis": {...},
    
    # NEW enrichment fields
    "top_keywords": [
        {"keyword": "python", "score": 0.8, "support": 10},
        ...
    ],
    "topic_coherence_scores": {"overall": 0.75},
    "per_author_stats": {
        "Alice": {
            "message_count": 50,
            "avg_sentiment": 6.2,
            "avg_length": 42.5,
            "emoji_ratio": 0.15,
            "question_ratio": 0.08
        }
    },
    "feature_summary": {
        "total_conversations": 10,
        "avg_sentiment": 6.5,
        "sentiment_variability": 1.2,
        "keyword_diversity": 45,
        "topic_coverage": 0.8
    },
    "summary_confidence": 0.85,
    "analysis_text": "Brief human-readable summary..."
}
```

### Enhanced AI Profile Generation

The g4f_client now uses structured prompts:

**Features:**
- Strict JSON schema enforcement
- Big Five personality traits with:
  - Score (0-1)
  - Label (low/medium/high)
  - Confidence score
  - Evidence snippets
- Fallback to local profile on failure
- JSON validation and parsing

**Schema:**
```json
{
  "schema_version": "1.0",
  "traits": {
    "openness": {
      "score": 0.75,
      "label": "high",
      "confidence": 0.8,
      "evidence": [...]
    },
    ...
  },
  "summary_text": "...",
  "warnings": [...]
}
```

## Optional Dependencies

For best results, install optional dependencies:

```bash
# Basic enhancements (recommended)
pip install emoji langdetect yake

# Advanced analysis (optional, large downloads)
pip install transformers torch sentence-transformers keybert

# If using transformers on CPU
pip install transformers torch --index-url https://download.pytorch.org/whl/cpu
```

**Dependency Tiers:**

**Tier 1 - Core (always required):**
- streamlit, nltk, empath, vaderSentiment

**Tier 2 - Basic Enhanced (small, fast):**
- emoji (emoji normalization)
- langdetect (language detection)
- yake (keyword extraction)
- scikit-learn (clustering, coherence)

**Tier 3 - Advanced (large models):**
- transformers (neural sentiment/emotion)
- torch (transformer backend)
- keybert (semantic keywords)
- sentence-transformers (embeddings)

**Graceful Degradation:**
- All enhanced features work without Tier 3 dependencies
- System automatically falls back to simpler methods
- Clear logging indicates which methods are available
- No errors if advanced libraries missing

## Performance Considerations

**With Only Core Dependencies:**
- Analysis time: ~1-2 seconds per file
- Memory: ~200MB
- Accuracy: Good (lexicon-based)

**With All Dependencies:**
- Analysis time: ~3-5 seconds per file (first run)
- Memory: ~1-2GB (models cached)
- Accuracy: Excellent (ensemble methods)

**Recommendations:**
- For personal use: Install all dependencies
- For server deployment: Use Tier 1+2 only (lighter)
- For high accuracy: Install transformers (Tier 3)

## Testing

All new modules include comprehensive unit tests:

```bash
# Run all tests
pytest tests/

# Run specific module tests
pytest tests/test_feature_extraction.py
pytest tests/test_sentiment_emotion.py
pytest tests/test_topic_extraction.py

# Check test coverage
pytest --cov=app tests/
```

**Test Coverage:**
- Feature extraction: 28 tests
- Sentiment & emotion: 31 tests
- Topic extraction: 16 tests
- Integration tests: 9 tests
- Total: 195 tests, all passing

## Migration Guide

**No breaking changes!** All existing code continues to work.

**To use new features:**

1. Install optional dependencies (see above)
2. Enhanced analysis runs automatically
3. Access new fields in summary:
   ```python
   summary = summarize_matrix(matrix)
   
   # Existing fields still work
   positive = summary["positive_topics"]
   
   # New fields available
   keywords = summary["top_keywords"]
   confidence = summary["summary_confidence"]
   ```

4. All public function signatures unchanged
5. New fields are always present (with sensible defaults)

## Future Enhancements

Planned improvements:
- [ ] BERTopic for dynamic topic modeling
- [ ] spaCy NER integration
- [ ] Personality trait prediction using ML
- [ ] Multi-language support expansion
- [ ] Real-time analysis streaming
- [ ] Conversation clustering
- [ ] Anomaly detection

## Troubleshooting

**Issue: "transformers not available" warnings**
- Expected if transformers not installed
- System will use lexicon-based methods (still accurate)
- To enable: `pip install transformers torch`

**Issue: Slow first analysis**
- Transformer models download on first use (~500MB)
- Subsequent analyses use cached models (fast)
- Models cached in `~/.cache/huggingface/`

**Issue: Out of memory**
- Reduce batch sizes
- Use CPU instead of GPU: `export CUDA_VISIBLE_DEVICES=""`
- Install lighter models or use Tier 1+2 only

**Issue: Language detection errors**
- Requires text length > 10 characters
- Returns None for very short messages (expected)
- Install langdetect: `pip install langdetect`

## Documentation

- Main README: `README.md`
- Architecture: `IMPLEMENTATION_SUMMARY.md`
- Testing guide: `TESTING.md`
- This doc: `ENHANCED_ANALYSIS.md`
