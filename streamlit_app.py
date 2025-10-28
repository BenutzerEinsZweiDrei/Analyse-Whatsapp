import streamlit as st
import re
import regex
import json
import requests
from datetime import datetime
from collections import defaultdict, Counter
import time
import logging
import io
import traceback
import os
import platform
import importlib
import importlib.metadata

import g4f
import emot
from empath import Empath
import textrazor
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from gensim import corpora
from gensim.models import LdaModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Import new analysis modules
from personality_analyzer import (
    calculate_big_five_scores,
    map_big_five_to_mbti,
    calculate_emotion_analysis
)
from conversation_metrics import (
    calculate_response_times,
    calculate_topic_response_time,
    calculate_emotional_reciprocity,
    aggregate_topic_metrics
)

# ---------------------------
# Constants
# ---------------------------
# Maximum weight for term frequency in keyword scoring
MAX_FREQ_WEIGHT = 0.5

# ---------------------------
# Debug / Logging Setup
# ---------------------------
# Memory buffer to show logs in the Streamlit UI
log_stream = io.StringIO()
stream_handler = logging.StreamHandler(log_stream)
stream_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)

console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)

logger = logging.getLogger("whatsapp_analyzer")
logger.setLevel(logging.INFO)
# Attach handlers if not attached yet (avoid duplicate handlers on rerun)
if not logger.handlers:
    logger.addHandler(stream_handler)
    logger.addHandler(console_handler)


def set_debug_mode(enabled: bool):
    logger.setLevel(logging.DEBUG if enabled else logging.INFO)
    # clear log buffer on mode change
    log_stream.truncate(0)
    log_stream.seek(0)
    logger.debug(f"Debug mode set to {enabled}")


def get_logs() -> str:
    logger.debug("Collecting logs from buffer")
    stream_handler.flush()
    return log_stream.getvalue()


def mask_key(key: str) -> str:
    if not key:
        return "<empty>"
    if len(key) <= 8:
        return key[:2] + "..." + key[-2:]
    return key[:4] + "..." + key[-4:]


# ---------------------------
# Environment and API keys
# Prefer st.secrets or environment variables; fallback to hard-coded (kept for compatibility)
# ---------------------------
jina_key = os.environ.get("JINA_API_KEY") or st.secrets.get("JINA_API_KEY", None) if hasattr(st, "secrets") else None
if not jina_key:
    jina_key = "jina_7010ba5005d74ef7bf3d3d767638ad97BnKkR5OSxO1hxE9qSpR4I943z-2K"  # fallback (consider removing)
    logger.warning("Using fallback hard-coded jina_key. Consider setting JINA_API_KEY in env or Streamlit secrets.")

textrazor_key = os.environ.get("TEXTRAZOR_API_KEY") or (st.secrets.get("TEXTRAZOR_API_KEY", None) if hasattr(st, "secrets") else None)
if not textrazor_key:
    textrazor_key = "2decf4a27aec43292ef8f925ff7b230db1c2589f94a52acbf147a9a7"  # fallback
    logger.warning("Using fallback hard-coded textrazor key. Consider setting TEXTRAZOR_API_KEY in env or Streamlit secrets.")

textrazor.api_key = textrazor_key
client = textrazor.TextRazor(extractors=["entities"])
client.set_language_override("ger")

# ---------------------------
# NLTK downloads (only the common ones required)
# ---------------------------
# Download the minimum set and log. Avoid 'all' which is heavy.
nltk_needed = ["all", "punkt", "punkt_tab", "averaged_perceptron_tagger", "wordnet", "stopwords", "omw-1.4"]
for res in nltk_needed:
    try:
        nltk.data.find(res)
        logger.debug(f"NLTK resource already present: {res}")
    except LookupError:
        logger.info(f"NLTK resource {res} not found. Downloading...")
        try:
            nltk.download(res, quiet=True)
            logger.info(f"Downloaded NLTK resource: {res}")
        except Exception as e:
            logger.exception(f"Failed to download NLTK resource {res}: {e}")

# ---------------------------
# Helpers
# ---------------------------
def classify_texts(inputs, labels, model="jina-embeddings-v3", api_key=None):
    start = time.time()
    logger.debug("Calling classify_texts: model=%s labels_count=%d", model, len(labels) if labels else 0)
    if api_key is None:
        raise ValueError("API Key must be provided.")
    url = "https://api.jina.ai/v1/classify"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    data = {
        "model": model,
        "input": inputs,
        "labels": labels,
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        elapsed = time.time() - start
        logger.debug("classify_texts response status: %s time: %.2fs", response.status_code, elapsed)
        # Avoid logging full content - just size and truncated body for debug
        text_len = len(response.text or "")
        logger.debug("classify_texts response length: %d", text_len)
        truncated = (response.text[:1000] + "...") if text_len > 1000 else response.text
        logger.debug("classify_texts truncated response: %s", truncated)
        return response.json()
    except Exception as e:
        logger.exception("Error in classify_texts: %s", e)
        raise


def parse_conversations_from_text(text):
    logger.debug("parse_conversations_from_text: input length=%d", len(text or ""))
    conversations = [conv.strip() for conv in re.split(r"\n\s*\n", text) if conv.strip()]
    logger.debug("Found %d conversation blocks", len(conversations))
    pattern = re.compile(r"^(\d{2}\.\d{2}\.\d{2}), (\d{2}:\d{2}) - (.*)", re.MULTILINE)
    all_conversations = []
    for ci, conv in enumerate(conversations, start=1):
        messages = []
        for match in pattern.finditer(conv):
            date_str, time_str, rest = match.groups()
            if ": " in rest:
                user, msg = rest.split(": ", 1)
            else:
                user, msg = None, rest
            try:
                dt = datetime.strptime(f"{date_str} {time_str}", "%y.%m.%d %H:%M")
            except Exception:
                # Fallback for unexpected date formats
                try:
                    dt = datetime.strptime(f"{date_str} {time_str}", "%d.%m.%y %H:%M")
                except Exception:
                    dt = None
                    logger.debug("Failed to parse datetime for line: %s", match.group(0))
            messages.append({
                "datetime": dt.isoformat() if dt else None,
                "date": date_str,
                "time": time_str,
                "user": user,
                "message": msg.strip(),
            })
        messages.sort(key=lambda x: (x["datetime"] or "", x["user"] or "zzz"))
        logger.debug("Conversation #%d -> messages parsed: %d", ci, len(messages))
        all_conversations.append(messages)
    logger.info("Total parsed conversations: %d", len(all_conversations))
    return all_conversations


def get_keywords(text, num_topics=3, num_keywords=5):
    """
    Extract keywords from text using gensim LDA topic modeling with improved filtering.
    
    Args:
        text: Input text to analyze
        num_topics: Number of topics to extract (default: 3)
        num_keywords: Number of keywords per topic to return (default: 5)
    
    Returns:
        List of keywords extracted from the most relevant topics with frequency weighting
    """
    logger.debug("get_keywords called with gensim topic analysis")
    
    if not text or not text.strip():
        logger.debug("Empty text provided to get_keywords")
        return []
    
    # Preprocess the text to get tokens
    tokens = preprocess(text)
    
    if not tokens or len(tokens) < 3:
        logger.debug("Insufficient tokens after preprocessing: %d", len(tokens))
        return []
    
    # Calculate term frequencies for importance weighting
    token_freq = Counter(tokens)
    
    # Create a corpus: list of tokenized documents (we have one document)
    # For LDA, we need at least some variation, so we'll treat the text as if 
    # it contains multiple sub-documents by splitting it
    try:
        # Split tokens into smaller chunks to give LDA some structure to work with
        chunk_size = max(10, len(tokens) // 5)  # At least 10 tokens per chunk
        chunks = [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]
        
        # Filter out very small chunks
        chunks = [chunk for chunk in chunks if len(chunk) >= 3]
        
        if not chunks:
            logger.debug("No valid chunks created from tokens")
            return []
        
        # Create dictionary and corpus for gensim
        dictionary = corpora.Dictionary(chunks)
        
        # More aggressive filtering: keep terms that appear at least once but not everywhere
        # and prioritize terms with medium frequency (not too rare, not too common)
        dictionary.filter_extremes(no_below=1, no_above=0.7, keep_n=100)
        
        if len(dictionary) == 0:
            logger.debug("Dictionary is empty after filtering")
            return []
        
        # Create bag-of-words corpus
        corpus = [dictionary.doc2bow(chunk) for chunk in chunks]
        
        # Build LDA model with fewer topics for short texts
        actual_num_topics = min(num_topics, len(chunks), len(dictionary))
        if actual_num_topics < 1:
            logger.debug("Cannot create topics with current parameters")
            return []
        
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=actual_num_topics,
            random_state=42,
            passes=10,
            alpha='auto',
            per_word_topics=True
        )
        
        # Extract keywords from all topics with probability weighting
        keyword_scores = {}
        for topic_id in range(actual_num_topics):
            topic_words = lda_model.show_topic(topic_id, topn=num_keywords)
            # topic_words is a list of (word, probability) tuples
            for word, prob in topic_words:
                if len(word) > 2:  # Avoid very short words
                    # Weight by both LDA probability and term frequency
                    freq_weight = min(token_freq.get(word, 1) / len(tokens), MAX_FREQ_WEIGHT)
                    combined_score = prob * (1 + freq_weight)
                    if word not in keyword_scores or combined_score > keyword_scores[word]:
                        keyword_scores[word] = combined_score
        
        # Sort by combined score and return top keywords
        sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, score in sorted_keywords]
        
        logger.debug("Extracted %d keywords using gensim LDA with frequency weighting", len(keywords))
        return keywords[:15]  # Return top 15 keywords maximum
        
    except Exception as e:
        logger.exception("Error in gensim topic analysis: %s", e)
        return []


def get_relevance(text):
    logger.debug("get_relevance called (placeholder)")
    return ["limit reached"][:10]


def preprocess(text):
    logger.debug("Preprocess called: input length=%d", len(text or ""))
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in string.punctuation]
    stop_words = set()
    try:
        stop_words = set(stopwords.words('german'))
    except Exception:
        logger.exception("Could not load NLTK german stopwords; continuing without them.")
    # Try to load extra stopwords file
    try:
        with open('data/stwd.json', 'r', encoding='utf-8') as file:
            extra_stopwords = json.load(file)
            logger.debug("Loaded %d extra stopwords from data/stwd.json", len(extra_stopwords) if extra_stopwords else 0)
            stop_words.update(extra_stopwords)
    except FileNotFoundError:
        logger.warning("data/stwd.json not found; proceeding without extra stopwords.")
    except Exception:
        logger.exception("Error reading data/stwd.json; proceeding without extra stopwords.")
    tokens = [t for t in tokens if t not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    logger.debug("Preprocess result tokens=%d", len(tokens))
    return tokens


def extract_nouns(text):
    """
    Extract meaningful nouns from text with improved filtering.
    Prioritizes proper nouns and common nouns while filtering out low-quality terms.
    """
    logger.debug("extract_nouns called")
    tokens = preprocess(text)
    if not tokens:
        return []
    tagged_tokens = pos_tag(tokens)
    filter_tags = {
        "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
        "JJ", "JJR", "JJS",
        "RB", "RBR", "RBS",
        "RP"
    }
    
    # Extract nouns with frequency counting
    nouns = [word for word, pos in tagged_tokens if pos in ("NN", "NNS", "NNP", "NNPS") and pos not in filter_tags]
    
    # Count noun frequencies to prioritize important terms
    noun_freq = Counter(nouns)
    
    # Filter nouns: keep those that appear more than once OR are proper nouns
    # This helps focus on conversation-relevant topics
    proper_nouns = {word for word, pos in tagged_tokens if pos in ("NNP", "NNPS")}
    filtered_nouns = []
    seen = set()
    
    for noun in nouns:
        if noun not in seen:
            # Keep if: appears multiple times, is a proper noun, or is sufficiently long
            if noun_freq[noun] > 1 or noun in proper_nouns or len(noun) > 5:
                filtered_nouns.append(noun)
                seen.add(noun)
    
    logger.debug("extract_nouns found %d nouns (filtered to %d)", len(nouns), len(filtered_nouns))
    return filtered_nouns[:20]  # Limit to top 20 most relevant nouns


def bewerte_emoji_string(emojis_im_text, emoji_dict=None):
    logger.debug("bewerte_emoji_string called with %d emojis", len(emojis_im_text) if emojis_im_text else 0)
    if not emojis_im_text:
        return ""
    if emoji_dict is None:
        try:
            with open("data/emos.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                emoji_dict = {entry["emoji"]: entry["meaning"] for entry in data["emojis"]}
                logger.debug("Loaded emojis mapping (%d entries) from data/emos.json", len(emoji_dict))
        except FileNotFoundError:
            logger.warning("data/emos.json not found; emoji evaluation will be neutral by default.")
            emoji_dict = {}
        except Exception:
            logger.exception("Error loading data/emos.json; emoji evaluation may be incomplete.")
            emoji_dict = {}
    score = 0
    num = len(emojis_im_text)
    if num == 0:
        return "neutral"
    for em in emojis_im_text:
        meaning = emoji_dict.get(em, "neutral")
        if meaning == "positiv":
            score += 1
        elif meaning == "traurig":
            score -= 1
    if score == 0:
        return "neutral"
    anteil = (score / num + 1) / 2
    if anteil >= 2 / 3:
        return "sehr positiv"
    elif anteil >= 0.5:
        return "eher positiv"
    elif anteil >= 0.3:
        return "eher traurig"
    else:
        return "sehr traurig"


def rate_text_from_file(word, sentiment_dict=None):
    logger.debug("rate_text_from_file called for word=%s", word)
    if sentiment_dict is None:
        try:
            with open("data/sent_rating.json", "r", encoding="utf-8") as f:
                sentiment_dict = json.load(f)
                logger.debug("Loaded sentiment ratings (%d entries)", len(sentiment_dict))
        except FileNotFoundError:
            logger.warning("data/sent_rating.json not found; sentiment rating will be empty.")
            sentiment_dict = {}
        except Exception:
            logger.exception("Error reading data/sent_rating.json; sentiment rating will be empty.")
            sentiment_dict = {}
    score = sentiment_dict.get(word.lower(), False)
    return [score] if score else []


def run_analysis(file_content, username):
    logger.info("run_analysis started for username=%s content_length=%d", username, len(file_content or ""))
    start_time = time.time()
    empath_lex = Empath()
    emot_obj = emot.core.emot()
    vader_analyzer = SentimentIntensityAnalyzer()  # Initialize once outside the loop
    conversations = parse_conversations_from_text(file_content)
    logger.debug("run_analysis: parsed %d conversations", len(conversations))
    matrix = {}
    mergetext = {}
    
    # Store original conversation messages for metrics calculation
    conversation_messages = {}

    for idx, conv_msgs in enumerate(conversations, 1):
        t0 = time.time()
        matrix[idx] = {"idx": idx}
        
        # Store conversation messages for metrics
        conversation_messages[idx] = conv_msgs
        
        user_msgs = [msg["message"] for msg in conv_msgs if msg.get("user") == username]
        mergetext[idx] = " ".join(user_msgs)
        text = mergetext[idx]
        logger.debug("Conversation #%d user messages=%d merged_text_len=%d", idx, len(user_msgs), len(text or ""))

        if not text.strip():
            logger.debug("Conversation #%d skipped (no user messages)", idx)
            continue

        try:
            emoji_result = emot_obj.emoji(text)
            matrix[idx]["emojies"] = emoji_result.get("value", [])
            matrix[idx]["emo_bew"] = [bewerte_emoji_string(matrix[idx]["emojies"])]
            logger.debug("Conversation #%d emojis=%d emo_bew=%s", idx, len(matrix[idx]["emojies"]), matrix[idx]["emo_bew"])
        except Exception:
            logger.exception("Error extracting emojis for conversation %d", idx)
            matrix[idx]["emojies"] = []
            matrix[idx]["emo_bew"] = ["error"]

        try:
            lex_analysis = empath_lex.analyze(text, normalize=True) if text else {}
            filtered_lex = {k: v for k, v in lex_analysis.items() if v > 0}
            matrix[idx]["lex"] = filtered_lex
            logger.debug("Conversation #%d empath categories=%d", idx, len(filtered_lex))
        except Exception:
            logger.exception("Error running Empath analysis for conversation %d", idx)
            matrix[idx]["lex"] = {}

        if matrix[idx].get("lex"):
            categories = list(matrix[idx]["lex"].keys())
            try:
                t_call = time.time()
                # Use vaderSentiment for sentiment analysis
                vader_scores = vader_analyzer.polarity_scores(text)
                compound_score = vader_scores['compound']
                
                # Convert compound score (-1 to 1) to 0-10 scale
                sent_rating_value = round((compound_score + 1) * 5, 1)
                
                # Determine sentiment label based on compound score
                if compound_score >= 0.05:
                    sentiment_label = "positive"
                elif compound_score <= -0.05:
                    sentiment_label = "negative"
                else:
                    sentiment_label = "neutral"
                
                logger.debug("Conversation #%d vader analysis took %.2fs", idx, time.time() - t_call)
                matrix[idx]["sentiment"] = [sentiment_label]
                matrix[idx]["sent_rating"] = [sent_rating_value]
                matrix[idx]["vader_scores"] = vader_scores  # Store detailed vader scores
                matrix[idx]["sentiment_compound"] = compound_score  # Store for reciprocity calculation
                logger.debug("Conversation #%d sentiment=%s sent_rating=%s compound=%s", 
                           idx, sentiment_label, sent_rating_value, compound_score)
            except Exception:
                logger.exception("Error analyzing sentiment for conversation %d", idx)
                matrix[idx]["sentiment"] = ["error"]
                matrix[idx]["sent_rating"] = []
                matrix[idx]["sentiment_compound"] = 0.0
        
        # Calculate Big Five personality traits
        try:
            emojis = matrix[idx].get("emojies", [])
            big_five = calculate_big_five_scores(text, emojis)
            matrix[idx]["big_five"] = big_five
            logger.debug("Conversation #%d Big Five calculated", idx)
        except Exception:
            logger.exception("Error calculating Big Five for conversation %d", idx)
            matrix[idx]["big_five"] = {}
        
        # Map Big Five to MBTI
        try:
            if matrix[idx].get("big_five"):
                mbti = map_big_five_to_mbti(matrix[idx]["big_five"])
                matrix[idx]["mbti"] = mbti
                logger.debug("Conversation #%d MBTI=%s", idx, mbti)
        except Exception:
            logger.exception("Error mapping MBTI for conversation %d", idx)
            matrix[idx]["mbti"] = "XXXX"
        
        # Enhanced emotion analysis
        try:
            emojis = matrix[idx].get("emojies", [])
            compound = matrix[idx].get("sentiment_compound", 0.0)
            emotion_analysis = calculate_emotion_analysis(emojis, compound)
            matrix[idx]["emotion_analysis"] = emotion_analysis
            logger.debug("Conversation #%d emotion analysis: dominant=%s", 
                        idx, emotion_analysis.get("dominant_emotion"))
        except Exception:
            logger.exception("Error in emotion analysis for conversation %d", idx)
            matrix[idx]["emotion_analysis"] = {}

        matrix[idx]["keywords"] = get_keywords(text) if text else []
        matrix[idx]["nouns"] = extract_nouns(text) if text else []

        # Combine and deduplicate words with improved filtering
        # Prioritize keywords (topic modeling results) over simple noun extraction
        combined_words = []
        seen_lower = set()
        
        # Add keywords first (they're weighted by topic relevance)
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
        
        # Limit total categories to avoid noise in classification
        matrix[idx]["words"] = combined_words[:20] or ["no topic"]

        categories = matrix[idx]["words"]
        try:
            if categories:
                result = classify_texts(text, categories, api_key=jina_key)
                topic_pred = result["data"][0]["prediction"]
                matrix[idx]["topic"] = [topic_pred]
                logger.debug("Conversation #%d topic=%s", idx, topic_pred)
            else:
                matrix[idx]["topic"] = ["no topic"]
                logger.debug("Conversation #%d no categories to classify topic", idx)
        except Exception:
            logger.exception("Error classifying topic for conversation %d", idx)
            matrix[idx]["topic"] = ["error"]
        
        # Calculate response times for this conversation
        try:
            response_times = calculate_response_times(conv_msgs)
            topic_avg_response = calculate_topic_response_time(conv_msgs)
            matrix[idx]["response_times"] = {
                "per_user": response_times,
                "topic_average": topic_avg_response
            }
            logger.debug("Conversation #%d response times calculated", idx)
        except Exception:
            logger.exception("Error calculating response times for conversation %d", idx)
            matrix[idx]["response_times"] = {"per_user": {}, "topic_average": 0.0}
        
        # Calculate emotional reciprocity
        try:
            # Enrich messages with sentiment for reciprocity calculation
            enriched_msgs = []
            for msg in conv_msgs:
                enriched_msg = msg.copy()
                enriched_msg["sentiment_compound"] = matrix[idx].get("sentiment_compound", 0.0)
                # Extract emojis for each message if needed
                try:
                    msg_emoji_result = emot_obj.emoji(msg.get("message", ""))
                    enriched_msg["emojis"] = msg_emoji_result.get("value", [])
                except:
                    enriched_msg["emojis"] = []
                enriched_msgs.append(enriched_msg)
            
            reciprocity = calculate_emotional_reciprocity(enriched_msgs)
            matrix[idx]["emotional_reciprocity"] = reciprocity
            logger.debug("Conversation #%d emotional reciprocity=%.3f", idx, reciprocity)
        except Exception:
            logger.exception("Error calculating emotional reciprocity for conversation %d", idx)
            matrix[idx]["emotional_reciprocity"] = 0.5

        logger.debug("Conversation #%d processing time: %.2fs", idx, time.time() - t0)

    logger.info("run_analysis finished in %.2fs", time.time() - start_time)
    return matrix, conversation_messages


def summarize_matrix(matrix):
    logger.debug("summarize_matrix called with matrix_size=%d", len(matrix))
    addtopic, negtopic, emo_vars = [], [], []
    analysis = {}
    for idx, entry in matrix.items():
        emo_bew = entry.get("emo_bew", [])
        topic = entry.get("topic", [])
        sentiment = entry.get("sentiment", [])
        sent_rating = entry.get("sent_rating", [])
        words = entry.get("words", [])
        lex = list(entry.get("lex", []))

        emo_vars.append(sent_rating)

        if topic and topic != ["no topic"]:
            if sent_rating:
                if sent_rating[0] >= 5.0:
                    addtopic.append(topic)
                else:
                    negtopic.append(topic)
            elif emo_bew:
                if emo_bew[0] in ["sehr positiv", "eher positiv"]:
                    addtopic.append(topic)
                else:
                    negtopic.append(topic)

        analysis[idx] = {
            "topic": topic,
            "emojies": emo_bew,
            "sentiment": sentiment,
            "wordcloud": words,
            # Add new metrics
            "big_five": entry.get("big_five", {}),
            "mbti": entry.get("mbti", ""),
            "emotion_analysis": entry.get("emotion_analysis", {}),
            "response_times": entry.get("response_times", {}),
            "emotional_reciprocity": entry.get("emotional_reciprocity", 0.5)
        }

    emo_vars = [x for x in emo_vars if x]
    flat_emo_vars = [x[0] for x in emo_vars]
    mittelwert = sum(flat_emo_vars) / len(flat_emo_vars) if flat_emo_vars else 0
    varianz = sum((x - mittelwert) ** 2 for x in flat_emo_vars) / len(flat_emo_vars) if flat_emo_vars else 0
    std_abweichung = varianz ** 0.5

    # Flatten topics
    flat_addtopic = [item for sublist in addtopic for item in sublist]
    flat_negtopic = [item for sublist in negtopic for item in sublist]

    addtopic_nouns = extract_nouns(" ".join(flat_addtopic))
    negtopic_nouns = extract_nouns(" ".join(flat_negtopic))

    # Filter URLs and symbols
    url_pattern = re.compile(r'\b(?:https?://|www\.)?\S+\.\S+\b', re.IGNORECASE)
    addtopic_clean = url_pattern.sub('', " ".join(addtopic_nouns))
    negtopic_clean = url_pattern.sub('', " ".join(negtopic_nouns))

    symbol_pattern = regex.compile(r'[\p{S}\p{P}\p{Emoji}]', regex.UNICODE)
    addtopic_clean = symbol_pattern.sub('', addtopic_clean)
    negtopic_clean = symbol_pattern.sub('', negtopic_clean)

    addtopic_set = set(addtopic_clean.split())
    negtopic_set = set(negtopic_clean.split())

    addtopic_final = extract_nouns(" ".join(addtopic_set))
    negtopic_final = extract_nouns(" ".join(negtopic_set))

    logger.debug("summarize_matrix positive_topics=%d negative_topics=%d", len(addtopic_final), len(negtopic_final))

    return {
        "positive_topics": addtopic_final,
        "negative_topics": negtopic_final,
        "emotion_variability": std_abweichung,
        "matrix": matrix,
        "analysis": analysis
    }


# ---------------------------
# Streamlit UI
# ---------------------------
st.title("WhatsApp Conversation Analyzer (with Debug Info)")

# Debug toggle in UI
debug_mode = st.checkbox("Enable debug mode (show logs and detailed info)", value=False)
set_debug_mode(debug_mode)

# Show some environment info in debug mode
if debug_mode:
    with st.expander("Environment & Dependency Info (debug)"):
        # Core environment
        env_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "cwd": os.getcwd(),
        }
        st.write(env_info)
        # Package versions for important libs
        key_packages = ["streamlit", "nltk", "textrazor", "empath", "emot", "g4f", "requests", "regex"]
        pkg_versions = {}
        for pkg in key_packages:
            try:
                pkg_versions[pkg] = importlib.metadata.version(pkg)
            except Exception:
                try:
                    pkg_versions[pkg] = importlib.import_module(pkg).__version__
                except Exception:
                    pkg_versions[pkg] = "not installed / unknown"
        st.write(pkg_versions)

        # Masked keys
        st.write({
            "jina_key": mask_key(jina_key),
            "textrazor_key": mask_key(textrazor_key),
        })
        st.write("Note: For production, configure API keys via Streamlit secrets or environment variables.")

uploaded_file = st.file_uploader("Upload your whatsapp.txt file", type=["txt"])

username = st.text_input("Enter the username to analyze")

if st.button("Start Analysis"):
    try:
        if not uploaded_file:
            st.error("Please upload a whatsapp.txt file.")
            logger.warning("Start Analysis pressed but no file uploaded.")
        elif not username.strip():
            st.error("Please enter a username.")
            logger.warning("Start Analysis pressed but username is empty.")
        else:
            file_bytes = uploaded_file.read()
            file_size = len(file_bytes)
            logger.info("File uploaded: filename=%s size=%d bytes", getattr(uploaded_file, "name", "<unknown>"), file_size)
            try:
                file_content = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    file_content = file_bytes.decode("latin-1")
                    logger.warning("File decoded with latin-1 fallback.")
                except Exception:
                    st.error("Could not decode uploaded file. Please provide a UTF-8 encoded text file.")
                    logger.exception("Failed to decode uploaded file.")
                    raise

            with st.spinner("Analyzing conversations... This may take a while."):
                total_start = time.time()
                matrix, conversation_messages = run_analysis(file_content, username.strip())
                summary = summarize_matrix(matrix)
                total_time = time.time() - total_start
                logger.info("Full analysis completed in %.2fs", total_time)

            st.success("Analysis completed!")

            st.subheader("Summary")
            st.write(f"**Positive Topics ({len(summary['positive_topics'])}):** {', '.join(summary['positive_topics'])}")
            st.write(f"**Negative Topics ({len(summary['negative_topics'])}):** {', '.join(summary['negative_topics'])}")
            st.write(f"**Emotional Variability:** {summary['emotion_variability']:.3f}")

            st.subheader("Analysis")
            message = "Erstelle ein kurzes psychologisches Profil anhand der folgenden Whatsapp Analyse \n\n"
            # pass analysis data to model
            message = f'{message}\n{json.dumps(summary["analysis"], ensure_ascii=False, indent=2)}'
            try:
                logger.debug("Sending prompt to g4f model (truncated)")
                response = g4f.ChatCompletion.create(
                    model=g4f.models.gpt_4,
                    messages=[{"role": "user", "content": message}],
                )
                logger.debug("g4f response type: %s", type(response))
            except Exception:
                logger.exception("Error while calling g4f ChatCompletion")
                response = "Error while generating profile."

            st.write(response)
            st.subheader("Detailed Conversation Matrix")
            st.json(summary['matrix'])

            # Provide JSON download
            json_data = json.dumps(summary['matrix'], ensure_ascii=False, indent=4)
            st.download_button(
                label="Download conv_matrix.json",
                data=json_data,
                file_name="conv_matrix.json",
                mime="application/json",
            )

            # If debug mode, show logs and offer download
            if debug_mode:
                st.subheader("Debug Logs")
                logs = get_logs()
                st.text_area("Logs", value=logs, height=300)
                st.download_button("Download logs", data=logs, file_name="analysis_logs.txt", mime="text/plain")
    except Exception as main_e:
        logger.exception("Unhandled exception during analysis: %s", main_e)
        st.error(f"An unexpected error occurred: {main_e}")
        if debug_mode:
            st.exception(main_e)
