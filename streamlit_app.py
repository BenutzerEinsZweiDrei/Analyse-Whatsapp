import streamlit as st
import re
import json
from datetime import datetime
from collections import defaultdict
import emot
from empath import Empath
import textrazor
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import io
import g4f

textrazor.api_key = "2decf4a27aec43292ef8f925ff7b230db1c2589f94a52acbf147a9a7"
client = textrazor.TextRazor(extractors=["topics"]) 


# Sicherstellen, dass die benötigten NLTK Daten vorhanden sind
for resource in ['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'taggers/{resource}')
    except LookupError:
        nltk.download(resource)


def parse_conversations_from_text(text):
    conversations = [conv.strip() for conv in re.split(r'\n\s*\n', text) if conv.strip()]
    pattern = re.compile(r'^(\d{2}\.\d{2}\.\d{2}), (\d{2}:\d{2}) - (.*)', re.MULTILINE)

    all_conversations = []
    for conv in conversations:
        messages = []
        for match in pattern.finditer(conv):
            date_str = match.group(1)
            time_str = match.group(2)
            rest = match.group(3)
            if ": " in rest:
                user, msg = rest.split(": ", 1)
            else:
                user = None
                msg = rest
            dt = datetime.strptime(date_str + " " + time_str, "%y.%m.%d %H:%M")
            messages.append({
                "datetime": dt.isoformat(),
                "date": date_str,
                "time": time_str,
                "user": user,
                "message": msg.strip()
            })
        messages.sort(key=lambda x: (x["datetime"], x["user"] if x["user"] else "zzz"))
        all_conversations.append(messages)
    return all_conversations

def get_keywords(text):
    response = client.analyze(text)
    labels = [topic.label for topic in response.topics() if topic.score > 0.4]
    return labels[:3]

def extract_nouns(text):
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    nouns = [word for word, pos in tagged_tokens if pos in ('NNP', 'NNPS')]
    return nouns


def bewerte_emoji_string(text, json_path='data/emos.json'):
    # JSON mit Emojis und Bedeutungen laden
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # emoji -> meaning Dictionary erstellen
    emoji_dict = {entry["emoji"]: entry["meaning"] for entry in data["emojis"]}

    # Emoji-Regex (für die meisten Emojis)
    emoji_pattern = re.compile("["
                               "\U0001F600-\U0001F64F"  # Emoticons
                               "\U0001F300-\U0001F5FF"  # Symbole & Piktogramme
                               "\U0001F680-\U0001F6FF"  # Transport & Karten
                               "\U0001F1E0-\U0001F1FF"  # Flaggen (regional)
                               "]+", flags=re.UNICODE)

    # Alle Emojis aus dem Text extrahieren
    emojis_im_text = emoji_pattern.findall(text)

    # Score berechnen
    score = 0
    for em in emojis_im_text:
        meaning = emoji_dict.get(em, "neutral")
        if meaning == "positiv":
            score += 1
        elif meaning == "traurig":
            score -= 1
        # neutral oder nicht vorhanden = 0 Punkte

    # Bewertung basierend auf Score zurückgeben
    if score >= 3:
        return "sehr positiv"
    elif 1 <= score <= 2:
        return "eher positiv"
    elif -2 <= score <= -1:
        return "eher traurig"
    elif score <= -3:
        return "sehr traurig"
    else:
        return "neutral"

def analyze_conversations(data, user_input):
    lexicon = Empath()
    emot_obj = emot.core.emot()

    matrix = {}
    mergetext = {}

    for idx, conv_msgs in enumerate(data, 1):
        matrix[idx] = {"idx": idx}
        mergetext[idx] = ""

        user_msgs = [msg["message"] for msg in conv_msgs if msg.get("user") == user_input]
        mergetext[idx] = "".join(user_msgs)
        text = mergetext[idx]

        emoji_result = emot_obj.emoji(text)
        matrix[idx]["emojies"] = emoji_result.get("value", [])

        matrix[idx]["keywords"] = []
        if text:
            try:
                keys = get_keywords(text)
            except Exception as e:
                keys = []
            matrix[idx]["keywords"] = keys

        matrix[idx]["nouns"] = []
        if text:
            nouns = extract_nouns(text)
            matrix[idx]["nouns"] = nouns

        lex_analysis = lexicon.analyze(text, normalize=True) if text else {}
        filtered_lex = {k: v for k, v in lex_analysis.items() if v > 0}
        matrix[idx]["lex"] = filtered_lex

    return matrix, mergetext

def group_by_emoji(matrix):
    emoji_groups = defaultdict(lambda: {"lex": set(), "keywords": set(), "nouns": set()})

    for entry in matrix.values():
        emojies = entry.get("emojies", [])
        lex = entry.get("lex", {})
        keywords = entry.get("keywords", [])
        nouns = entry.get("nouns", [])

        emoji_key = "".join(sorted(emojies)) if emojies else "no_emoji"

        emoji_groups[emoji_key]["lex"].update(lex.keys())
        emoji_groups[emoji_key]["keywords"].update(keywords)
        emoji_groups[emoji_key]["nouns"].update(nouns)

    # Merged string per emoji group
    merged = {}
    for emoji_key, group in emoji_groups.items():
        all_words = group["lex"] | group["keywords"] | group["nouns"]
        merged[emoji_key] = " ".join(sorted(all_words))
        

    return merged

# --- Streamlit UI ---
st.title("WhatsApp Chat Analyse")

uploaded_file = st.file_uploader("Lade whatsapp.txt hoch", type=["txt"])
user_input = st.text_input("Gib den Usernamen ein, dessen Nachrichten analysiert werden sollen")

if uploaded_file and user_input:
    if st.button("Analyse starten"):
        with st.spinner("Analyse lädt..."):
            text = uploaded_file.read().decode("utf-8")
            data = parse_conversations_from_text(text)
            matrix, mergetext = analyze_conversations(data, user_input)
            merged_by_emoji = group_by_emoji(matrix)

            fullmerge = ""
            st.header("Zusammengeführte Texte je Emoji-Gruppe")
            for emoji_key, merged_text in merged_by_emoji.items():
                sh = f"Emoji-Gruppe: {bewerte_emoji_string(emoji_key)}"
                st.subheader(sh)
                sw = merged_text if merged_text else "_Keine Wörter gefunden_"
                st.write(sw)
                fullmerge = fullmerge + sh + "\n" + sw + "\n"
            
            response = g4f.ChatCompletion.create(
                model=g4f.models.gpt_4,
                messages=[{"role": "user", "content": "Beschreibe mich kurz anhand der folgenden Informationen aus einer Whatsapp Chat Analyse: " + fullmerge}],
            )  
            st.write(response)
            
            # Download conv_matrix.json
            json_str = json.dumps(matrix, ensure_ascii=False, indent=4)
            st.download_button(
                label="conv_matrix.json herunterladen",
                data=json_str,
                file_name="conv_matrix.json",
                mime="application/json"
            )
else:
    st.info("Bitte lade eine whatsapp.txt Datei hoch und gib den Usernamen ein.")
