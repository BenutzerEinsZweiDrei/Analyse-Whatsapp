import re
import json
from datetime import datetime

import streamlit as st
import emot
from empath import Empath


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


def analyze_conversations(conversations, user_input):
    lexicon = Empath()
    emot_obj = emot.core.emot()
    matrix = {}
    mergetext = {}

    for idx, conv_msgs in enumerate(conversations, 1):
        # Filter messages from specified user
        user_msgs = [msg["message"] for msg in conv_msgs if msg.get("user") == user_input]
        text = "".join(user_msgs)
        if not text.strip():
            # No messages from user in this conversation, skip
            continue

        matrix[idx] = {"idx": idx}

        # Extract emojis
        emoji_result = emot_obj.emoji(text)
        matrix[idx]["emojies"] = emoji_result.get("value", [])

        # Empath lexicon analysis
        lex_analysis = lexicon.analyze(text, normalize=True) if text else {}
        filtered_lex = {k: v for k, v in lex_analysis.items() if v > 0}
        matrix[idx]["lex"] = filtered_lex

        mergetext[idx] = text

        # Only keep conversations with emojis or lexicon data
        if not matrix[idx]["emojies"] and not filtered_lex:
            del matrix[idx]

    return matrix, mergetext


def main():
    st.title("WhatsApp Conversation Analyzer")

    uploaded_file = st.file_uploader("Upload WhatsApp chat text file", type=["txt"])
    user_input = st.text_input("Enter user name to analyze (case sensitive)")

    if uploaded_file and user_input:
        if st.button("Start Analysis"):
            text = uploaded_file.read().decode("utf-8")
            st.subheader("Parsing conversations...")
            conversations = parse_conversations_from_text(text)
            st.success(f"Parsed {len(conversations)} conversations.")

            st.subheader(f"Analyzing conversations for user: {user_input}")
            matrix, mergetext = analyze_conversations(conversations, user_input)

            if not matrix:
                st.info("No conversations found with emojis or lexicon data for this user.")
                return

            for idx in sorted(matrix.keys()):
                st.markdown(f"### Conversation {idx}")
                emojis = matrix[idx].get("emojies", [])
                lex = matrix[idx].get("lex", {})
                st.write("Emojis:", " ".join(emojis) if emojis else "None")
                if lex:
                    st.write("Lexicon analysis:")
                    for key, value in lex.items():
                        st.write(f"- {key}: {value}")
                else:
                    st.write("No lexicon categories detected.")

            # Downloads for JSON files
            if st.button("Download parsed conversations JSON"):
                json_data = json.dumps(conversations, ensure_ascii=False, indent=4)
                st.download_button("Download JSON", data=json_data, file_name="konversationen.json", mime="application/json")

            if st.button("Download analysis matrix JSON"):
                json_data = json.dumps(matrix, ensure_ascii=False, indent=4)
                st.download_button("Download JSON", data=json_data, file_name="conv_matrix.json", mime="application/json")


if __name__ == "__main__":
    main()
