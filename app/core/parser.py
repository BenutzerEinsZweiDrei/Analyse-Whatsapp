"""
WhatsApp conversation parser.

Parses WhatsApp chat export text into structured message objects.
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger("whatsapp_analyzer")


@dataclass
class Message:
    """
    Represents a single WhatsApp message.

    Attributes:
        datetime: ISO format datetime string (or None if parsing failed)
        date: Date string from original export (e.g., "23.01.21")
        time: Time string from original export (e.g., "14:30")
        user: Username of sender (or None for system messages)
        message: Message text content
        file_origin: Optional filename indicating source file
    """

    datetime: Optional[str]
    date: str
    time: str
    user: Optional[str]
    message: str
    file_origin: Optional[str] = None


def parse_conversations(text: str) -> List[List[Message]]:
    """
    Parse WhatsApp chat export text into conversations.

    Handles common WhatsApp export formats:
    - DD.MM.YY, HH:MM - Username: Message
    - YY.MM.DD, HH:MM - Username: Message

    Conversations are separated by blank lines in the export.

    Args:
        text: Raw WhatsApp chat export text

    Returns:
        List of conversations, where each conversation is a list of Message objects

    Example:
        >>> text = "23.01.21, 14:30 - Alice: Hello\\n23.01.21, 14:31 - Bob: Hi!"
        >>> conversations = parse_conversations(text)
        >>> len(conversations)
        1
        >>> len(conversations[0])
        2
    """
    logger.debug(f"parse_conversations: input length={len(text or '')}")

    if not text or not text.strip():
        logger.warning("Empty text provided to parse_conversations")
        return []

    # Split by blank lines to separate conversations
    conversation_blocks = [conv.strip() for conv in re.split(r"\n\s*\n", text) if conv.strip()]
    logger.debug(f"Found {len(conversation_blocks)} conversation blocks")

    # Regex pattern for WhatsApp message format
    # Matches: DD.MM.YY, HH:MM - Username: Message
    # Or: YY.MM.DD, HH:MM - Username: Message
    pattern = re.compile(r"^(\d{2}\.\d{2}\.\d{2}), (\d{2}:\d{2}) - (.*)", re.MULTILINE)

    all_conversations = []

    for conv_idx, conv_block in enumerate(conversation_blocks, start=1):
        messages = []

        for match in pattern.finditer(conv_block):
            date_str, time_str, rest = match.groups()

            # Split username and message (format: "Username: Message")
            if ": " in rest:
                user, msg = rest.split(": ", 1)
            else:
                # System message or message without colon
                user, msg = None, rest

            # Parse datetime - try different formats
            dt = _parse_datetime(date_str, time_str)

            messages.append(
                Message(
                    datetime=dt.isoformat() if dt else None,
                    date=date_str,
                    time=time_str,
                    user=user,
                    message=msg.strip(),
                )
            )

        if messages:
            # Sort messages by datetime (with None values last)
            messages.sort(key=lambda x: (x.datetime or "9999", x.user or "zzz"))
            logger.debug(f"Conversation #{conv_idx} -> {len(messages)} messages parsed")
            all_conversations.append(messages)
        else:
            logger.debug(
                f"Conversation #{conv_idx} -> no messages parsed (possibly invalid format)"
            )

    logger.info(f"Total parsed conversations: {len(all_conversations)}")
    return all_conversations


def _parse_datetime(date_str: str, time_str: str) -> Optional[datetime]:
    """
    Parse date and time strings into datetime object.

    Tries multiple formats to handle different WhatsApp export formats.

    Args:
        date_str: Date string (e.g., "23.01.21" or "21.01.23")
        time_str: Time string (e.g., "14:30")

    Returns:
        datetime object or None if parsing fails
    """
    # Try YY.MM.DD format first (year.month.day)
    formats = [
        "%y.%m.%d %H:%M",  # YY.MM.DD HH:MM
        "%d.%m.%y %H:%M",  # DD.MM.YY HH:MM
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(f"{date_str} {time_str}", fmt)
            return dt
        except ValueError:
            continue

    # If all formats fail, log and return None
    logger.debug(f"Failed to parse datetime: {date_str} {time_str}")
    return None


def parse_conversations_from_text(text: str, file_origin: Optional[str] = None) -> List[List[dict]]:
    """
    Legacy function for backward compatibility.

    Parses conversations and returns them as list of dicts instead of Message objects.

    Args:
        text: Raw WhatsApp chat export text
        file_origin: Optional filename to track message source

    Returns:
        List of conversations, where each conversation is a list of message dicts
    """
    conversations = parse_conversations(text)

    # Convert Message objects to dicts
    result = []
    for conv in conversations:
        conv_dicts = []
        for msg in conv:
            conv_dicts.append(
                {
                    "datetime": msg.datetime,
                    "date": msg.date,
                    "time": msg.time,
                    "user": msg.user,
                    "message": msg.message,
                    "file_origin": file_origin or msg.file_origin,
                }
            )
        result.append(conv_dicts)

    return result


def merge_and_deduplicate_messages(
    all_conversations: List[List[List[dict]]],
) -> List[List[dict]]:
    """
    Merge multiple conversation lists and deduplicate messages.

    Takes conversations from multiple files, flattens them into a single message list,
    sorts by datetime, and removes duplicates based on (datetime, user, message).

    Args:
        all_conversations: List of conversation lists from different files

    Returns:
        List of conversations (merged and deduplicated)
    """
    logger.info(f"Merging {len(all_conversations)} file conversation lists")

    # Flatten all conversations into a single message list
    all_messages = []
    for file_convs in all_conversations:
        for conv in file_convs:
            all_messages.extend(conv)

    logger.debug(f"Total messages before deduplication: {len(all_messages)}")

    # Sort by datetime (None values last)
    all_messages.sort(key=lambda m: (m.get("datetime") or "9999", m.get("user") or "zzz"))

    # Deduplicate based on (datetime, user, message)
    seen = set()
    deduplicated_messages = []
    duplicate_count = 0

    for msg in all_messages:
        # Create hash key for deduplication
        key = (
            msg.get("datetime"),
            msg.get("user"),
            msg.get("message", "").strip(),
        )

        if key not in seen:
            deduplicated_messages.append(msg)
            seen.add(key)
        else:
            duplicate_count += 1

    logger.info(
        f"Deduplication complete: {len(deduplicated_messages)} unique messages, "
        f"{duplicate_count} duplicates removed"
    )

    # Re-split into conversations based on blank line logic
    # For now, treat all merged messages as one conversation
    # since blank lines were lost during merge
    if deduplicated_messages:
        return [deduplicated_messages]
    return []
