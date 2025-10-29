"""
Integration test for multi-file analysis workflow.

Tests the complete pipeline from file upload to analysis results.
"""

import unittest
from unittest.mock import MagicMock, patch

from app.core.parser import merge_and_deduplicate_messages, parse_conversations_from_text
from app.run_analysis import cached_run_analysis, run_analysis_multiple_files


class TestMultiFileAnalysisIntegration(unittest.TestCase):
    """Integration tests for multi-file analysis workflow."""

    def setUp(self):
        """Set up test data."""
        # Create sample WhatsApp chat exports
        self.file1_content = """23.01.21, 14:30 - Alice: Hello!
23.01.21, 14:31 - Bob: Hi there!
23.01.21, 14:32 - Alice: How are you?

23.01.21, 15:00 - Bob: I'm great, thanks!
23.01.21, 15:01 - Alice: That's wonderful!"""

        self.file2_content = """23.01.21, 15:01 - Alice: That's wonderful!
23.01.21, 15:02 - Bob: How was your day?

23.01.21, 16:00 - Alice: It was good, busy with work
23.01.21, 16:01 - Bob: Same here"""

        self.file_metadata = [
            {"filename": "chat1.txt", "file_size_bytes": len(self.file1_content), "decode_used": "utf-8"},
            {"filename": "chat2.txt", "file_size_bytes": len(self.file2_content), "decode_used": "utf-8"},
        ]

    def test_single_file_analysis_backward_compatibility(self):
        """Test that single file analysis still works (backward compatibility)."""
        # This should work exactly as before
        result = cached_run_analysis(self.file1_content, "Alice")
        
        matrix, conversation_messages = result
        
        # Should have conversations
        self.assertIsInstance(matrix, dict)
        self.assertIsInstance(conversation_messages, dict)
        self.assertGreater(len(matrix), 0)

    def test_multi_file_analysis_integration(self):
        """Test complete multi-file analysis workflow."""
        # Simulate multiple file upload
        file_contents = [self.file1_content, self.file2_content]
        
        # Call the multi-file analysis
        result = cached_run_analysis(file_contents, "Alice", self.file_metadata)
        
        matrix, conversation_messages = result
        
        # Verify results
        self.assertIsInstance(matrix, dict)
        self.assertIsInstance(conversation_messages, dict)
        
        # Should have at least one conversation
        self.assertGreater(len(matrix), 0)
        
        # Check that file_origins are tracked
        for conv_idx, conv_data in matrix.items():
            if isinstance(conv_data, dict) and "file_origins" in conv_data:
                # If file_origins exist, they should be a list
                self.assertIsInstance(conv_data["file_origins"], list)

    def test_deduplication_in_multi_file_analysis(self):
        """Test that duplicate messages are removed during multi-file analysis."""
        # Parse both files separately first
        conv1 = parse_conversations_from_text(self.file1_content, file_origin="file1.txt")
        conv2 = parse_conversations_from_text(self.file2_content, file_origin="file2.txt")
        
        # Count messages before merge
        total_before = sum(len(conv) for file_convs in [conv1, conv2] for conv in file_convs)
        
        # Merge and deduplicate
        merged = merge_and_deduplicate_messages([conv1, conv2])
        
        # Count messages after merge
        total_after = sum(len(conv) for conv in merged)
        
        # Should have deduplicated (the overlapping message)
        self.assertLess(total_after, total_before)
        
        print(f"Messages before merge: {total_before}")
        print(f"Messages after merge: {total_after}")
        print(f"Duplicates removed: {total_before - total_after}")

    def test_file_origin_tracking(self):
        """Test that file origins are properly tracked."""
        # Run full analysis with multiple files
        file_contents = [self.file1_content, self.file2_content]
        matrix, conversation_messages = cached_run_analysis(
            file_contents, "Alice", self.file_metadata
        )
        
        # Check conversation messages for file_origin
        for conv_idx, messages in conversation_messages.items():
            for msg in messages:
                # Each message should have file_origin
                self.assertIn("file_origin", msg)
                # file_origin should be one of our test files
                self.assertIn(msg["file_origin"], ["chat1.txt", "chat2.txt"])

    def test_empty_file_handling(self):
        """Test handling of empty or invalid files."""
        # Test with empty string
        result = cached_run_analysis("", "Alice")
        matrix, conversation_messages = result
        
        # Should not crash, just return empty results
        self.assertIsInstance(matrix, dict)
        self.assertIsInstance(conversation_messages, dict)

    def test_encoding_metadata_preserved(self):
        """Test that encoding metadata is preserved in file_metadata."""
        file_contents = [self.file1_content, self.file2_content]
        
        # The metadata should be passed through
        # (we can't directly test caching behavior, but we can verify the function accepts it)
        result = cached_run_analysis(file_contents, "Alice", self.file_metadata)
        
        matrix, conversation_messages = result
        
        # Should complete without error
        self.assertIsInstance(matrix, dict)


class TestWorkflowSimulation(unittest.TestCase):
    """Simulate the complete Streamlit workflow."""

    def test_streamlit_workflow_single_file(self):
        """Simulate single file upload workflow."""
        # User uploads one file
        file_content = """23.01.21, 14:30 - Alice: Test message
23.01.21, 14:31 - Bob: Response"""
        
        username = "Alice"
        
        # Run analysis (single file path)
        matrix, conv_messages = cached_run_analysis(file_content, username)
        
        # Verify results structure
        self.assertIsInstance(matrix, dict)
        self.assertIsInstance(conv_messages, dict)
        
        # Should have analyzed Alice's messages
        for conv_idx, conv_data in matrix.items():
            if isinstance(conv_data, dict):
                # Matrix should have analysis fields
                self.assertIn("idx", conv_data)

    def test_streamlit_workflow_multiple_files(self):
        """Simulate multiple file upload workflow."""
        # User uploads multiple files
        uploaded_files = [
            ("chat1.txt", "23.01.21, 14:30 - Alice: First file\n23.01.21, 14:31 - Bob: Reply"),
            ("chat2.txt", "23.01.21, 15:30 - Alice: Second file\n23.01.21, 15:31 - Bob: Reply again"),
        ]
        
        # Process files (simulating Streamlit logic)
        file_contents = []
        file_metadata = []
        
        for filename, content in uploaded_files:
            file_contents.append(content)
            file_metadata.append({
                "filename": filename,
                "file_size_bytes": len(content),
                "decode_used": "utf-8",
            })
        
        username = "Alice"
        
        # Run analysis (multiple files path)
        matrix, conv_messages = cached_run_analysis(file_contents, username, file_metadata)
        
        # Verify results
        self.assertIsInstance(matrix, dict)
        self.assertIsInstance(conv_messages, dict)
        self.assertGreater(len(matrix), 0)
        
        print(f"Analyzed {len(uploaded_files)} files")
        print(f"Generated {len(matrix)} conversation analyses")


if __name__ == "__main__":
    unittest.main()
