"""
Unit tests for merge_local_profiles function.

Tests the merging of multiple per-file personality profile results.
"""

from app.core.local_profile import merge_local_profiles


def test_merge_empty_list():
    """Test merging with empty list returns error."""
    result, text = merge_local_profiles([])
    
    assert "error" in result
    assert "No profiles to merge" in result["error"]
    assert "No profiles available" in text


def test_merge_single_profile():
    """Test merging single profile returns it as-is with note."""
    mock_results = {
        "basic_metrics": {"per_conversation_count": 10},
        "big_five_aggregation": {
            "openness": {"mean": 7.0, "std": 0.5, "n": 10},
            "conscientiousness": {"mean": 6.0, "std": 0.3, "n": 10},
        },
        "emotion_insights": {
            "flagged_conversations": [],
            "most_common_emotion": "joy",
            "average_emotion_ratios": {"joy": 0.6, "sadness": 0.2},
        },
    }
    mock_text = "Test profile text"
    mock_filename = "test.txt"
    
    result, text = merge_local_profiles([(mock_results, mock_text, mock_filename)])
    
    assert "error" not in result
    assert "Single File" in text
    assert mock_filename in text


def test_merge_multiple_profiles():
    """Test merging multiple profiles aggregates correctly."""
    # Create two mock profiles
    profile1 = {
        "basic_metrics": {
            "per_conversation_count": 10,
            "average_emotional_reciprocity": {"mean": 0.7, "std": 0.1, "n": 10},
            "dominant_emotion_counts": {"joy": 5, "sadness": 3, "neutral": 2},
            "mbti_distribution": {"INTJ": 6, "INFJ": 4},
            "response_time_stats": {"mean": 30.0, "std": 5.0, "n": 10},
        },
        "big_five_aggregation": {
            "openness": {"mean": 7.0, "std": 0.5, "n": 10},
            "conscientiousness": {"mean": 6.0, "std": 0.3, "n": 10},
            "extraversion": {"mean": 5.0, "std": 0.4, "n": 10},
            "agreeableness": {"mean": 8.0, "std": 0.2, "n": 10},
            "neuroticism": {"mean": 4.0, "std": 0.6, "n": 10},
        },
        "emotion_insights": {
            "flagged_conversations": [{"conversation_id": "1", "reason": "low_reciprocity"}],
            "most_common_emotion": "joy",
            "average_emotion_ratios": {"joy": 0.6, "sadness": 0.2},
        },
        "topics_summary": {},
        "mbti_summary": {},
        "per_conversation_table": [{"conversation_id": "1"}],
    }
    
    profile2 = {
        "basic_metrics": {
            "per_conversation_count": 5,
            "average_emotional_reciprocity": {"mean": 0.5, "std": 0.15, "n": 5},
            "dominant_emotion_counts": {"joy": 2, "sadness": 2, "neutral": 1},
            "mbti_distribution": {"INTJ": 3, "INFJ": 2},
            "response_time_stats": {"mean": 60.0, "std": 10.0, "n": 5},
        },
        "big_five_aggregation": {
            "openness": {"mean": 6.0, "std": 0.4, "n": 5},
            "conscientiousness": {"mean": 7.0, "std": 0.2, "n": 5},
            "extraversion": {"mean": 4.0, "std": 0.5, "n": 5},
            "agreeableness": {"mean": 7.0, "std": 0.3, "n": 5},
            "neuroticism": {"mean": 5.0, "std": 0.4, "n": 5},
        },
        "emotion_insights": {
            "flagged_conversations": [{"conversation_id": "2", "reason": "high_sadness"}],
            "most_common_emotion": "sadness",
            "average_emotion_ratios": {"joy": 0.3, "sadness": 0.5},
        },
        "topics_summary": {},
        "mbti_summary": {},
        "per_conversation_table": [{"conversation_id": "2"}],
    }
    
    profiles_list = [
        (profile1, "Profile 1 text", "file1.txt"),
        (profile2, "Profile 2 text", "file2.txt"),
    ]
    
    result, text = merge_local_profiles(profiles_list)
    
    # Check merged results
    assert "error" not in result
    assert result["merged_file_count"] == 2
    assert "file1.txt" in result["merged_from_files"]
    assert "file2.txt" in result["merged_from_files"]
    
    # Check merged basic metrics
    assert result["basic_metrics"]["per_conversation_count"] == 15  # 10 + 5
    
    # Check weighted average reciprocity: (0.7*10 + 0.5*5) / 15 = 0.6333...
    recip_mean = result["basic_metrics"]["average_emotional_reciprocity"]["mean"]
    assert 0.63 <= recip_mean <= 0.64
    assert result["basic_metrics"]["average_emotional_reciprocity"]["n"] == 15
    
    # Check merged emotion counts
    emotion_counts = result["basic_metrics"]["dominant_emotion_counts"]
    assert emotion_counts["joy"] == 7  # 5 + 2
    assert emotion_counts["sadness"] == 5  # 3 + 2
    assert emotion_counts["neutral"] == 3  # 2 + 1
    
    # Check merged MBTI distribution
    mbti_dist = result["basic_metrics"]["mbti_distribution"]
    assert mbti_dist["INTJ"] == 9  # 6 + 3
    assert mbti_dist["INFJ"] == 6  # 4 + 2
    
    # Check weighted average response time: (30*10 + 60*5) / 15 = 40.0
    rt_mean = result["basic_metrics"]["response_time_stats"]["mean"]
    assert rt_mean == 40.0
    
    # Check merged Big Five (weighted averages)
    big_five = result["big_five_aggregation"]
    # Openness: (7.0*10 + 6.0*5) / 15 = 6.666...
    assert 6.6 <= big_five["openness"]["mean"] <= 6.7
    # Conscientiousness: (6.0*10 + 7.0*5) / 15 = 6.333...
    assert 6.3 <= big_five["conscientiousness"]["mean"] <= 6.4
    
    # Check merged emotion insights
    assert len(result["emotion_insights"]["flagged_conversations"]) == 2
    assert result["emotion_insights"]["flagged_conversations"][0]["source_file"] == "file1.txt"
    assert result["emotion_insights"]["flagged_conversations"][1]["source_file"] == "file2.txt"
    
    # Check merged per-conversation table
    assert len(result["per_conversation_table"]) == 2
    assert result["per_conversation_table"][0]["source_file"] == "file1.txt"
    assert result["per_conversation_table"][1]["source_file"] == "file2.txt"
    
    # Check exports exist
    assert "exports" in result
    assert "metrics_json" in result["exports"]
    assert "per_conversation_csv" in result["exports"]
    assert "flagged_json" in result["exports"]
    
    # Check merged text
    assert "Merged" in text
    assert "file1.txt" in text
    assert "file2.txt" in text


def test_merge_with_error_profiles():
    """Test merging skips profiles with errors."""
    profile_ok = {
        "basic_metrics": {"per_conversation_count": 5},
        "big_five_aggregation": {},
        "emotion_insights": {},
        "topics_summary": {},
        "mbti_summary": {},
        "per_conversation_table": [],
    }
    
    profile_error = {
        "error": "Analysis failed",
        "basic_metrics": {},
    }
    
    profiles_list = [
        (profile_ok, "Profile OK", "file1.txt"),
        (profile_error, "Profile Error", "file2.txt"),
    ]
    
    result, text = merge_local_profiles(profiles_list)
    
    # Should only merge the valid profile
    assert "error" not in result
    assert result.get("merged_file_count") == 1


def test_merge_all_error_profiles():
    """Test merging all error profiles returns error."""
    profile_error1 = {"error": "Error 1"}
    profile_error2 = {"error": "Error 2"}
    
    profiles_list = [
        (profile_error1, "", "file1.txt"),
        (profile_error2, "", "file2.txt"),
    ]
    
    result, text = merge_local_profiles(profiles_list)
    
    assert "error" in result
    assert "All profiles had errors" in result["error"]
    assert "All profiles had errors" in text


if __name__ == "__main__":
    # Run tests if pytest not available
    test_merge_empty_list()
    test_merge_single_profile()
    test_merge_multiple_profiles()
    test_merge_with_error_profiles()
    test_merge_all_error_profiles()
    print("✓ All tests passed!")
