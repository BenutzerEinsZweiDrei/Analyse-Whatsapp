import streamlit as st
import re
import regex
import json
import requests
from datetime import datetime
from collections import defaultdict

import emot
from empath import Empath
import textrazor
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# nltk download
nltk.download('all')

# Setup API keys (consider securing these)
jina_key = "jina_87f1a0df417c40b589cf8bf99d15deacGrG0M_BhwOAwxoYQTpkjIrLEtAS0"
textrazor.api_key = "2decf4a27aec43292ef8f925ff7b230db1c2589f94a52acbf147a9a7"
client = textrazor.TextRazor(extractors=["entities"])
client.set_language_override("ger")

# NLP helpers (reuse your functions here, but refactor to use in Streamlit)
def classify_texts(inputs, labels, model="jina-embeddings-v3", api_key=None):
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
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()

def parse_conversations_from_text(text):
    conversations = [
        conv.strip() for conv in re.split(r"\n\s*\n", text) if conv.strip()
    ]
    pattern = re.compile(r"^(\d{2}\.\d{2}\.\d{2}), (\d{2}:\d{2}) - (.*)", re.MULTILINE)
    all_conversations = []
    for conv in conversations:
        messages = []
        for match in pattern.finditer(conv):
            date_str, time_str, rest = match.groups()
            if ": " in rest:
                user, msg = rest.split(": ", 1)
            else:
                user, msg = None, rest
            dt = datetime.strptime(f"{date_str} {time_str}", "%y.%m.%d %H:%M")
            messages.append({
                "datetime": dt.isoformat(),
                "date": date_str,
                "time": time_str,
                "user": user,
                "message": msg.strip(),
            })
        messages.sort(key=lambda x: (x["datetime"], x["user"] or "zzz"))
        all_conversations.append(messages)
    return all_conversations

# Add your preprocess, extract_nouns, bewerte_emoji_string, rate_text_from_file here exactly as you wrote them,
# But modify preprocess to load stwd.json from a fixed path or a default list for stopwords,
# and bewerte_emoji_string to load emos.json from a fixed path or embed example emojis (or ask user to upload)

def get_keywords(text):
    # Placeholder for now
    return ["limit reached"]

def get_relevance(text):
    return ["limit reached"][:10]

def preprocess(text):
    # Basic preprocessing similar to your version without file loading to simplify
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in string.punctuation]
    with open('data/stwd.json', 'r', encoding='utf-8') as file:
        extra_stopwords = json.load(file)
    stop_words = set(stopwords.words('german'))
    stop_words.update(extra_stopwords)
    tokens = [t for t in tokens if t not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens

def extract_nouns(text):
    tokens = preprocess(text)
    tagged_tokens = pos_tag(tokens)
    filter_tags = {
        "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
        "JJ", "JJR", "JJS",
        "RB", "RBR", "RBS",
        "RP"
    }
    return [word for word, pos in tagged_tokens if pos in ("NN","NNS","NNP","NNPS") and pos not in filter_tags]

def bewerte_emoji_string(emojis_im_text, emoji_dict=None):
    if not emojis_im_text:
        return ""
    if emoji_dict is None:
        with open("data/emos.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            emoji_dict = {entry["emoji"]: entry["meaning"] for entry in data["emojis"]}
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
    if sentiment_dict is None:
        with open("data/sent_rating.json", "r", encoding="utf-8") as f:
            sentiment_dict = json.load(f)
    score = sentiment_dict.get(word.lower(), False)
    return [score] if score else []

def run_analysis(file_content, username):
    empath_lex = Empath()
    emot_obj = emot.core.emot()
    conversations = parse_conversations_from_text(file_content)
    matrix = {}
    mergetext = {}

    for idx, conv_msgs in enumerate(conversations, 1):
        matrix[idx] = {"idx": idx}
        user_msgs = [msg["message"] for msg in conv_msgs if msg.get("user") == username]
        mergetext[idx] = " ".join(user_msgs)
        text = mergetext[idx]
        if not text.strip():
            continue

        emoji_result = emot_obj.emoji(text)
        matrix[idx]["emojies"] = emoji_result.get("value", [])
        matrix[idx]["emo_bew"] = [bewerte_emoji_string(matrix[idx]["emojies"])]
        
        lex_analysis = empath_lex.analyze(text, normalize=True) if text else {}
        filtered_lex = {k: v for k, v in lex_analysis.items() if v > 0}
        matrix[idx]["lex"] = filtered_lex

        if filtered_lex:
            categories = list(filtered_lex.keys())
            try:
                result = classify_texts(text, categories, api_key=jina_key)
                sentiment_pred = result["data"][0]["prediction"]
                matrix[idx]["sentiment"] = [sentiment_pred]
                matrix[idx]["sent_rating"] = rate_text_from_file(sentiment_pred)
            except Exception as e:
                matrix[idx]["sentiment"] = ["error"]
                matrix[idx]["sent_rating"] = []

        matrix[idx]["keywords"] = get_keywords(text) if text else []
        matrix[idx]["nouns"] = extract_nouns(text) if text else []

        matrix[idx]["words"] = list(matrix[idx]["nouns"]) + list(matrix[idx]["keywords"]) or ["no topic"]

        categories = matrix[idx]["words"]
        try:
            result = classify_texts(text, categories, api_key=jina_key)
            topic_pred = result["data"][0]["prediction"]
            matrix[idx]["topic"] = [topic_pred]
        except Exception as e:
            matrix[idx]["topic"] = ["error"]

    return matrix

def summarize_matrix(matrix):
    addtopic, negtopic, emo_vars = [], [], []
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
                if sent_rating[0] > 4:
                    addtopic.append(topic)
                else:
                    negtopic.append(topic)
            elif emo_bew:
                if emo_bew[0] in ["sehr positiv", "eher positiv"]:
                    addtopic.append(topic)
                else:
                    negtopic.append(topic)

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

    return {
        "positive_topics": addtopic_final,
        "negative_topics": negtopic_final,
        "emotion_variability": std_abweichung,
        "matrix": matrix
    }


# Streamlit UI
st.title("WhatsApp Conversation Analyzer")

uploaded_file = st.file_uploader("Upload your whatsapp.txt file", type=["txt"])

username = st.text_input("Enter the username to analyze")

if st.button("Start Analysis"):
    if not uploaded_file:
        st.error("Please upload a whatsapp.txt file.")
    elif not username.strip():
        st.error("Please enter a username.")
    else:
        file_content = uploaded_file.read().decode("utf-8")
        with st.spinner("Analyzing conversations... This may take a while."):
            matrix = run_analysis(file_content, username.strip())
            summary = summarize_matrix(matrix)
        
        st.success("Analysis completed!")

        st.subheader("Summary")
        st.write(f"**Positive Topics ({len(summary['positive_topics'])}):** {', '.join(summary['positive_topics'])}")
        st.write(f"**Negative Topics ({len(summary['negative_topics'])}):** {', '.join(summary['negative_topics'])}")
        st.write(f"**Emotional Variability:** {summary['emotion_variability']:.3f}")

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
