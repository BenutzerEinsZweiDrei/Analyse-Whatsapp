"""
Tests for multi-file WhatsApp chat merging and deduplication.
"""

import os
import unittest
from pathlib import Path

from app.core.parsing import (
    merge_and_deduplicate_messages,
    parse_conversations_from_text,
)


class TestMultiFileMerge(unittest.TestCase):
    """Test multi-file parsing and merging functionality."""

    @classmethod
    def setUpClass(cls):
        """Load test fixtures."""
        fixtures_dir = Path(__file__).parent / "fixtures"
        
        cls.chat_part1_path = fixtures_dir / "chat_part1.txt"
        cls.chat_part2_path = fixtures_dir / "chat_part2.txt"
        
        if cls.chat_part1_path.exists():
            with open(cls.chat_part1_path, "r", encoding="utf-8") as f:
                cls.chat_part1 = f.read()
        else:
            # Fallback to inline test data
            cls.chat_part1 = """23.01.21, 14:30 - Alice: Hello Bob!
23.01.21, 14:31 - Bob: Hi Alice, how are you?
23.01.21, 14:32 - Alice: I'm doing great, thanks!

23.01.21, 15:00 - Alice: Did you see the movie yesterday?
23.01.21, 15:01 - Bob: Yes! It was amazing!"""
        
        if cls.chat_part2_path.exists():
            with open(cls.chat_part2_path, "r", encoding="utf-8") as f:
                cls.chat_part2 = f.read()
        else:
            # Fallback with overlap
            cls.chat_part2 = """23.01.21, 15:01 - Bob: Yes! It was amazing!
23.01.21, 15:02 - Alice: I loved the ending!

23.01.21, 16:00 - Bob: Want to grab lunch tomorrow?
23.01.21, 16:01 - Alice: Sure, what time?"""

    def test_parse_single_file_with_origin(self):
        """Test parsing a single file with file_origin metadata."""
        conversations = parse_conversations_from_text(self.chat_part1, file_origin="test1.txt")
        
        # Should have conversations
        self.assertGreater(len(conversations), 0)
        
        # Check that file_origin is set
        for conv in conversations:
            for msg in conv:
                self.assertEqual(msg.get("file_origin"), "test1.txt")

    def test_merge_two_files_without_overlap(self):
        """Test merging two files with no overlapping messages."""
        # Create non-overlapping test data
        file1 = """23.01.21, 10:00 - Alice: Morning!
23.01.21, 10:01 - Bob: Good morning!"""
        
        file2 = """23.01.21, 11:00 - Alice: How's your day?
23.01.21, 11:01 - Bob: Going well!"""
        
        conv1 = parse_conversations_from_text(file1, file_origin="file1.txt")
        conv2 = parse_conversations_from_text(file2, file_origin="file2.txt")
        
        merged = merge_and_deduplicate_messages([conv1, conv2])
        
        # Should have merged all messages
        self.assertGreater(len(merged), 0)
        
        total_messages = sum(len(conv) for conv in merged)
        # Should have 4 messages (2 from each file)
        self.assertEqual(total_messages, 4)

    def test_merge_with_duplicates(self):
        """Test that duplicate messages are removed during merge."""
        # Both files contain the overlapping message
        conv1 = parse_conversations_from_text(self.chat_part1, file_origin="part1.txt")
        conv2 = parse_conversations_from_text(self.chat_part2, file_origin="part2.txt")
        
        # Count total messages before merge
        total_before = sum(len(conv) for file_convs in [conv1, conv2] for conv in file_convs)
        
        merged = merge_and_deduplicate_messages([conv1, conv2])
        
        # Count total messages after merge
        total_after = sum(len(conv) for conv in merged)
        
        # Should have deduplicated at least one message
        # (the "Yes! It was amazing!" message appears in both files)
        self.assertLess(total_after, total_before)
        
        # Check that messages are sorted chronologically
        if merged:
            prev_datetime = None
            for conv in merged:
                for msg in conv:
                    curr_datetime = msg.get("datetime")
                    if prev_datetime and curr_datetime:
                        # Allow None values but check chronological order when both exist
                        self.assertLessEqual(prev_datetime, curr_datetime)
                    if curr_datetime:
                        prev_datetime = curr_datetime

    def test_merge_preserves_file_origins(self):
        """Test that file_origin metadata is preserved after merge."""
        conv1 = parse_conversations_from_text(self.chat_part1, file_origin="source1.txt")
        conv2 = parse_conversations_from_text(self.chat_part2, file_origin="source2.txt")
        
        merged = merge_and_deduplicate_messages([conv1, conv2])
        
        # Check that file origins are present
        file_origins = set()
        for conv in merged:
            for msg in conv:
                origin = msg.get("file_origin")
                if origin:
                    file_origins.add(origin)
        
        # Should have messages from both files
        self.assertIn("source1.txt", file_origins)
        self.assertIn("source2.txt", file_origins)

    def test_merge_empty_lists(self):
        """Test merging with empty conversation lists."""
        merged = merge_and_deduplicate_messages([])
        self.assertEqual(len(merged), 0)
        
        # Test with empty conversations inside
        merged = merge_and_deduplicate_messages([[]])
        self.assertEqual(len(merged), 0)

    def test_chronological_ordering(self):
        """Test that merged messages are in chronological order."""
        # Create messages that would be out of order if not sorted
        file1 = """23.01.21, 14:00 - Alice: First
23.01.21, 16:00 - Alice: Third"""
        
        file2 = """23.01.21, 15:00 - Bob: Second
23.01.21, 17:00 - Bob: Fourth"""
        
        conv1 = parse_conversations_from_text(file1, file_origin="early.txt")
        conv2 = parse_conversations_from_text(file2, file_origin="late.txt")
        
        merged = merge_and_deduplicate_messages([conv1, conv2])
        
        # Extract all messages
        all_messages = []
        for conv in merged:
            all_messages.extend(conv)
        
        # Verify chronological order
        for i in range(len(all_messages) - 1):
            curr = all_messages[i].get("datetime")
            next_msg = all_messages[i + 1].get("datetime")
            if curr and next_msg:
                self.assertLessEqual(curr, next_msg)


class TestMultiFileIntegration(unittest.TestCase):
    """Integration tests for multi-file analysis pipeline."""

    def test_single_file_backward_compatibility(self):
        """Test that single file analysis still works (backward compatibility)."""
        test_content = """23.01.21, 14:30 - Alice: Hello!
23.01.21, 14:31 - Bob: Hi there!"""
        
        conversations = parse_conversations_from_text(test_content)
        
        # Should parse successfully
        self.assertGreater(len(conversations), 0)
        self.assertGreater(len(conversations[0]), 0)
        
        # Check message structure
        first_msg = conversations[0][0]
        self.assertIn("datetime", first_msg)
        self.assertIn("user", first_msg)
        self.assertIn("message", first_msg)


if __name__ == "__main__":
    unittest.main()
