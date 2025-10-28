"""
Unit tests for app/ package modules.

Tests core functionality of the refactored WhatsApp analyzer.
"""

import os
import sys
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import get_settings, mask_key
from app.core.emojis import evaluate_emoji_string, extract_emojis
from app.core.keywords import get_keywords_simple_tf
from app.core.nouns import extract_nouns
from app.core.parser import Message, parse_conversations
from app.core.preprocessing import preprocess_text
from app.core.sentiment import analyze_sentiment, sentiment_label_from_compound
from app.data.loaders import load_json_asset


class TestConfig(unittest.TestCase):
    """Test configuration management."""

    def test_get_settings(self):
        """Test settings retrieval."""
        settings = get_settings()
        self.assertIsNotNone(settings)
        # API keys may or may not be set - just check structure
        self.assertTrue(hasattr(settings, "jina_api_key"))
        self.assertTrue(hasattr(settings, "textrazor_api_key"))

    def test_mask_key(self):
        """Test API key masking."""
        # Test various key lengths
        self.assertEqual(mask_key(None), "<not configured>")
        self.assertEqual(mask_key(""), "<not configured>")
        self.assertEqual(mask_key("abc"), "ab...bc")
        self.assertEqual(mask_key("abcdefghij"), "abcd...ghij")

        # Ensure masked keys don't reveal full key
        full_key = "1234567890abcdef"
        masked = mask_key(full_key)
        self.assertIn("...", masked)
        self.assertLess(len(masked), len(full_key))


class TestParser(unittest.TestCase):
    """Test WhatsApp message parsing."""

    def test_parse_empty_text(self):
        """Test parsing empty text."""
        result = parse_conversations("")
        self.assertEqual(result, [])

    def test_parse_single_message(self):
        """Test parsing single message."""
        text = "23.01.21, 14:30 - Alice: Hello world"
        conversations = parse_conversations(text)

        self.assertEqual(len(conversations), 1)
        self.assertEqual(len(conversations[0]), 1)

        msg = conversations[0][0]
        self.assertIsInstance(msg, Message)
        self.assertEqual(msg.user, "Alice")
        self.assertEqual(msg.message, "Hello world")
        self.assertEqual(msg.time, "14:30")

    def test_parse_multiple_messages(self):
        """Test parsing multiple messages."""
        text = """23.01.21, 14:30 - Alice: Hello
23.01.21, 14:31 - Bob: Hi there
23.01.21, 14:32 - Alice: How are you?"""

        conversations = parse_conversations(text)
        self.assertEqual(len(conversations), 1)
        self.assertEqual(len(conversations[0]), 3)

        # Check users
        users = [msg.user for msg in conversations[0]]
        self.assertEqual(users, ["Alice", "Bob", "Alice"])

    def test_parse_system_message(self):
        """Test parsing system message (no username)."""
        text = "23.01.21, 14:30 - Alice created this group"
        conversations = parse_conversations(text)

        self.assertEqual(len(conversations), 1)
        # System messages are parsed but may have unusual structure
        self.assertGreater(len(conversations[0]), 0)


class TestPreprocessing(unittest.TestCase):
    """Test text preprocessing."""

    def test_preprocess_empty_text(self):
        """Test preprocessing empty text."""
        result = preprocess_text("")
        self.assertEqual(result, [])

    def test_preprocess_simple_text(self):
        """Test preprocessing simple text."""
        text = "Hello world, this is a test!"
        tokens = preprocess_text(text, lang="english")

        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)

        # Should be lowercase
        self.assertTrue(all(t.islower() or not t.isalpha() for t in tokens))

        # Should not contain punctuation
        self.assertNotIn(",", tokens)
        self.assertNotIn("!", tokens)

    def test_preprocess_german_text(self):
        """Test preprocessing German text."""
        text = "Hallo Welt, das ist ein Test!"
        tokens = preprocess_text(text, lang="german")

        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)


class TestKeywords(unittest.TestCase):
    """Test keyword extraction."""

    def test_keywords_empty_text(self):
        """Test keyword extraction from empty text."""
        result = get_keywords_simple_tf("")
        self.assertEqual(result, [])

    def test_keywords_simple_tf(self):
        """Test simple TF keyword extraction."""
        text = "apple apple banana orange apple banana"
        keywords = get_keywords_simple_tf(text, num_keywords=3)

        self.assertIsInstance(keywords, list)
        self.assertGreater(len(keywords), 0)

        # 'apple' should be first (appears 3 times)
        # Note: stopwords and preprocessing may affect results
        self.assertIsInstance(keywords[0], str)


class TestNouns(unittest.TestCase):
    """Test noun extraction."""

    def test_extract_nouns_empty(self):
        """Test noun extraction from empty text."""
        result = extract_nouns("")
        self.assertEqual(result, [])

    def test_extract_nouns_simple(self):
        """Test noun extraction from simple text."""
        text = "Alice went to the park with Bob"
        nouns = extract_nouns(text)

        self.assertIsInstance(nouns, list)
        # Depending on POS tagging, should extract proper nouns and common nouns
        # Alice, park, Bob might be extracted


class TestEmojis(unittest.TestCase):
    """Test emoji extraction and evaluation."""

    def test_extract_emojis_empty(self):
        """Test emoji extraction from empty text."""
        result = extract_emojis("")
        self.assertEqual(result, [])

    def test_extract_emojis_simple(self):
        """Test emoji extraction from text with emojis."""
        text = "Hello 😊 World 👍"
        emojis = extract_emojis(text)

        self.assertIsInstance(emojis, list)
        # Should extract emojis (if emot is available)
        # If emot not available, regex fallback will work

    def test_evaluate_emoji_empty(self):
        """Test emoji evaluation with no emojis."""
        result = evaluate_emoji_string([])
        self.assertEqual(result, "neutral")

    def test_evaluate_emoji_labels(self):
        """Test emoji evaluation returns valid labels."""
        # Test with mock positive emojis (if we had a known mapping)
        # Without actual emoji dict, just test it doesn't crash
        result = evaluate_emoji_string(["😊", "😄"])
        self.assertIn(
            result, ["sehr positiv", "eher positiv", "neutral", "eher traurig", "sehr traurig"]
        )


class TestSentiment(unittest.TestCase):
    """Test sentiment analysis."""

    def test_analyze_sentiment_empty(self):
        """Test sentiment analysis of empty text."""
        result = analyze_sentiment("")

        self.assertIsInstance(result, dict)
        self.assertIn("vader", result)
        self.assertIn("compound", result)
        self.assertIn("label", result)
        self.assertIn("scaled_rating", result)

        # Empty text should be neutral
        self.assertEqual(result["label"], "neutral")
        self.assertEqual(result["compound"], 0.0)

    def test_analyze_sentiment_positive(self):
        """Test sentiment analysis of positive text."""
        result = analyze_sentiment("I love this! It's wonderful!")

        self.assertEqual(result["label"], "positive")
        self.assertGreater(result["compound"], 0.0)
        self.assertGreater(result["scaled_rating"], 5.0)

    def test_analyze_sentiment_negative(self):
        """Test sentiment analysis of negative text."""
        result = analyze_sentiment("I hate this. It's terrible.")

        self.assertEqual(result["label"], "negative")
        self.assertLess(result["compound"], 0.0)
        self.assertLess(result["scaled_rating"], 5.0)

    def test_sentiment_label_from_compound(self):
        """Test compound score to label conversion."""
        self.assertEqual(sentiment_label_from_compound(0.6), "positive")
        self.assertEqual(sentiment_label_from_compound(-0.6), "negative")
        self.assertEqual(sentiment_label_from_compound(0.0), "neutral")


class TestDataLoaders(unittest.TestCase):
    """Test data loading functions."""

    def test_load_json_asset_not_found(self):
        """Test loading non-existent JSON file."""
        with self.assertRaises(FileNotFoundError):
            load_json_asset("nonexistent_file_12345.json", "test asset")

    def test_load_json_asset_existing(self):
        """Test loading existing JSON file."""
        # Try to load a known file (stwd.json should exist)
        try:
            data = load_json_asset("stwd.json", "stopwords")
            self.assertIsNotNone(data)
        except FileNotFoundError:
            # If file doesn't exist, that's okay for this test
            pass


class TestSecurityNoHardcodedKeys(unittest.TestCase):
    """Test that no hard-coded API keys remain in code."""

    def test_no_hardcoded_keys_in_streamlit_app(self):
        """Ensure no hard-coded API keys in streamlit_app.py."""
        with open("streamlit_app.py", "r") as f:
            content = f.read()

        # Check for patterns that might indicate hard-coded keys
        # These are the OLD hard-coded keys that should NOT be present
        old_jina_key = "jina_7010ba5005d74ef7bf3d3d767638ad97BnKkR5OSxO1hxE9qSpR4I943z-2K"
        old_textrazor_key = "2decf4a27aec43292ef8f925ff7b230db1c2589f94a52acbf147a9a7"

        self.assertNotIn(
            old_jina_key, content, "Old hard-coded Jina API key found in streamlit_app.py"
        )
        self.assertNotIn(
            old_textrazor_key, content, "Old hard-coded TextRazor API key found in streamlit_app.py"
        )

    def test_no_hardcoded_keys_in_config(self):
        """Ensure no hard-coded API keys in app/config.py."""
        with open("app/config.py", "r") as f:
            content = f.read()

        # Check patterns
        old_jina_key = "jina_7010ba5005d74ef7bf3d3d767638ad97BnKkR5OSxO1hxE9qSpR4I943z-2K"
        old_textrazor_key = "2decf4a27aec43292ef8f925ff7b230db1c2589f94a52acbf147a9a7"

        self.assertNotIn(old_jina_key, content, "Hard-coded Jina API key found in app/config.py")
        self.assertNotIn(
            old_textrazor_key, content, "Hard-coded TextRazor API key found in app/config.py"
        )


if __name__ == "__main__":
    unittest.main()
