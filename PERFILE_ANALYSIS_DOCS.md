# Per-File Analysis Feature Documentation

## Overview

The WhatsApp Conversation Analyzer now supports per-file analysis, per-file personality profile generation, and merging of multiple personality profiles. This allows users to:

1. Upload multiple WhatsApp export files (1-5 files)
2. Analyze each file individually
3. Generate personality profiles for each file
4. Merge multiple profiles into a combined analysis

## Features

### 1. Multi-File Upload

- Upload 1-5 WhatsApp export `.txt` files
- Each file is processed and tracked independently
- Files are displayed as individual cards in the UI

### 2. Per-File Analysis

Each uploaded file can be analyzed independently:
- **Status Tracking**: Shows current status (Queued, Running, Success, Error)
- **Individual Analysis**: Click "Start Analysis" to analyze a single file
- **Results Summary**: View positive/negative topics, emotion variability
- **Full Details**: Expand to see complete analysis JSON and download button

### 3. Per-File Personality Profiles

For each successfully analyzed file:
- **Local Profile**: Generate privacy-preserving local psychological profile
  - No external API calls
  - Comprehensive Big Five and MBTI analysis
  - Download as JSON, CSV, or flagged conversations
- **AI Profile**: Generate AI-powered natural language profile (optional)
  - Uses g4f client for AI generation
  - May require internet connection

### 4. Bulk Actions

Process multiple files efficiently:
- **Analyze All Files**: Start analysis for all uploaded files that haven't been analyzed
- **Generate Local Profiles for All**: Create local profiles for all analyzed files
- **Generate AI Profiles for All**: Generate AI profiles for all analyzed files
- **Merge Personality Profiles**: Combine multiple local profiles into one

### 5. Profile Merging

Combine personality profiles from multiple files:
- **Weighted Averaging**: Big Five traits are averaged by conversation count
- **Aggregated Metrics**: Emotion insights and response times are combined
- **Unified Exports**: Download merged JSON, CSV, and flagged conversations
- **Source Tracking**: Each data point includes source file information

## Session State Structure

```python
st.session_state.files = [
    {
        "filename": "chat1.txt",
        "file_size_bytes": 12345,
        "decode_used": "utf-8",
        "content": "raw file content...",
        "analysis": {
            "matrix": {...},
            "summary": {...},
            "conversation_messages": {...}
        },
        "analysis_status": "success",  # queued, running, success, error
        "analysis_time": 12.3,
        "analysis_error": None,
        "local_profile": {
            "results": {...},
            "profile_text": "...",
            "status": "success",  # none, running, success, error
            "error": None
        },
        "ai_profile": {
            "response": "...",
            "status": "none",
            "error": None
        }
    },
    # ... more files
]

st.session_state.merged_profiles = {
    "results": {...},
    "profile_text": "...",
    "merged_from": ["chat1.txt", "chat2.txt"]
}

st.session_state.username = "JohnDoe"
```

## API Functions

### `merge_local_profiles(profiles_list)`

Merges multiple per-file local profile results into a single aggregated profile.

**Parameters:**
- `profiles_list`: List of tuples `(results_dict, profile_text, filename)`

**Returns:**
- Tuple of `(merged_results_dict, merged_profile_text)`

**Merging Strategy:**
- **Big Five Traits**: Weighted average by conversation count
- **Emotion Counts**: Sum of all counts across files
- **MBTI Distribution**: Sum of all type occurrences
- **Response Times**: Weighted average by conversation count
- **Flagged Conversations**: Union of all flagged items (with source file info)
- **Per-Conversation Data**: Concatenation with source file tracking

### `run_local_analysis(summary, matrix)`

Existing function - unchanged. Generates a local psychological profile for a single analysis.

**Parameters:**
- `summary`: Summary dictionary from `summarize_matrix`
- `matrix`: Matrix dictionary from `run_analysis`

**Returns:**
- Tuple of `(results_dict, profile_text)`

## UI Workflow

### Single File Workflow

1. Upload one file
2. Enter username
3. Click "Upload Files"
4. File card appears with "Start Analysis" button
5. Click "Start Analysis"
6. View analysis summary
7. Click "Generate Local Profile" or "Generate AI Profile"
8. View and download profile

### Multi-File Workflow

1. Upload 2-5 files
2. Enter username
3. Click "Upload Files"
4. Multiple file cards appear
5. Click "Analyze All Files" to process all at once
6. Wait for analyses to complete
7. Click "Generate Local Profiles for All"
8. Wait for profile generation
9. Click "Merge Personality Profiles"
10. View merged profile and download combined exports

## Error Handling

- **File Decoding Errors**: Files that fail to decode are skipped with error message
- **Analysis Errors**: Captured and displayed on file card; allows retry
- **Profile Generation Errors**: Displayed with retry button
- **Merge Errors**: Profiles with errors are skipped; warning shown
- **Debug Mode**: Enable to see detailed logs and stack traces

## Backward Compatibility

The new implementation maintains backward compatibility:
- Single file upload works exactly as before
- `cached_run_analysis` still supports single string input
- Session state structure is new but doesn't break existing functionality
- All existing download buttons and features remain available

## Performance Considerations

- **Caching**: `cached_run_analysis` uses Streamlit caching to avoid recomputation
- **Sequential Processing**: Analyses run sequentially (one at a time) to avoid overwhelming the system
- **Memory Management**: Large files may consume significant memory; 5-file limit helps
- **Response Time**: Total processing time increases linearly with file count

## File Size Limits

- **Per File**: No hard limit, but large files (>10MB) may cause performance issues
- **Total Files**: Maximum 5 files per upload
- **Memory**: Large analyses stored in session state; may require page refresh if memory issues occur

## Export Formats

### Per-File Exports

- **Analysis JSON**: Complete analysis with all metrics and metadata
- **Local Profile JSON**: Personality metrics and aggregated data
- **Conversations CSV**: Individual conversation metrics in spreadsheet format
- **Flagged JSON**: Conversations with unusual patterns

### Merged Exports

- **Merged Profile JSON**: Combined personality metrics from all files
- **Merged Conversations CSV**: All conversations from all files with source tracking
- **Merged Flagged JSON**: All flagged conversations with source file information

## Testing

### Unit Tests

Located in `tests/test_merge_profiles.py`:
- `test_merge_empty_list()`: Verify error handling for empty input
- `test_merge_single_profile()`: Ensure single profile returns correctly
- `test_merge_multiple_profiles()`: Test weighted averaging and aggregation
- `test_merge_with_error_profiles()`: Verify error profiles are skipped
- `test_merge_all_error_profiles()`: Test all-error scenario

Run tests:
```bash
PYTHONPATH=/path/to/repo python tests/test_merge_profiles.py
```

### Manual Testing

Since Streamlit apps require a running server, manual testing is required:

```bash
streamlit run streamlit_app.py
```

Test scenarios:
1. Upload and analyze single file
2. Upload and analyze multiple files
3. Generate profiles individually
4. Use bulk actions
5. Merge profiles
6. Download all export formats
7. Test error scenarios (invalid files, decode errors)

## Security Notes

- **Privacy**: Local profiles use no external APIs
- **AI Profiles**: May send data to external services (g4f)
- **File Content**: Stored in session state; not persisted
- **API Keys**: Configured via secrets or environment variables
- **No Vulnerabilities**: CodeQL scan shows 0 security alerts

## Future Enhancements

Potential improvements (not implemented):
- Parallel analysis processing
- Progress bars for long-running operations
- Selective merge (checkboxes to choose which files)
- Per-file personality profile comparison view
- Export all files at once (ZIP download)
- Retry failed analyses in bulk
- Session persistence across page reloads
