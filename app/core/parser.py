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
    """

    datetime: Optional[str]
    date: str
    time: str
    user: Optional[str]
    message: str


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


def parse_conversations_from_text(text: str) -> List[List[dict]]:
    """
    Legacy function for backward compatibility.

    Parses conversations and returns them as list of dicts instead of Message objects.

    Args:
        text: Raw WhatsApp chat export text

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
                }
            )
        result.append(conv_dicts)

    return result
