# Profile Fusion v2.0 - Testing Guide

## Overview

Profile Fusion v2.0 has been completely refactored with a modular architecture, comprehensive test coverage, and enhanced features including statistical correlation analysis, personality profile matrices, and natural language summaries.

## Running Tests

### Prerequisites

Install required dependencies:
```bash
pip install pytest scipy numpy pandas streamlit
```

### Running All Tests

Run the complete test suite:
```bash
pytest tests/test_profile_fusion.py tests/test_analysis_io.py tests/test_analysis_stats.py tests/test_profile_fusion_integration.py -v
```

Expected output: **33 tests passing**

### Test Categories

1. **Backward Compatibility Tests** (`tests/test_profile_fusion.py`)
   - 7 tests validating v1.0 API compatibility
   - Tests Big Five, emotions, and MBTI merging

2. **IO and Normalization Tests** (`tests/test_analysis_io.py`)
   - 16 tests for data loading and normalization
   - Tests various input formats and edge cases

3. **Statistical Tests** (`tests/test_analysis_stats.py`)
   - 3 tests for aggregated statistics computation
   - Tests reciprocity, response time, and trait ranking

4. **Integration Tests** (`tests/test_profile_fusion_integration.py`)
   - 7 tests using fixture data
   - Tests complete workflow from loading to output

### Manual Testing

Run the manual test script:
```bash
python test_profile_fusion_manual.py
```

This script simulates the complete Profile Fusion workflow without the Streamlit UI.

## Test Fixtures

Sample personality profiles are available in `tests/fixtures/`:
- `profile1.json` - Sample profile with Big Five, emotions, MBTI, correlations, and topics
- `profile2.json` - Second sample profile for merge testing

## Module Structure

The refactored code is organized into modular components:

```
analysis/
├── __init__.py           # Package initialization
├── io.py                 # Data loading and normalization
├── merge.py              # Merge logic for personality data
├── stats.py              # Aggregated statistics
├── correlations.py       # Correlation analysis with scipy
├── profile_matrix.py     # Profile matrix generation
├── insights.py           # Emotional and topic insights
├── narrative.py          # Natural language summaries
└── ui_components.py      # Streamlit UI components
```

## Running the Streamlit App

Start the Profile Fusion v2.0 app:
```bash
streamlit run pages/Profile_Fusion.py
```

### Testing the App

1. Navigate to the "Profile_Fusion" page
2. Upload 2-5 JSON personality profile files
3. Adjust settings in the sidebar:
   - P-value threshold (default: 0.05)
   - Matrix normalization (None, Min-Max, Z-score)
   - Summary format (bullet or paragraph)
   - Show per-topic analysis
4. Review the analysis sections:
   - Basic Analysis (Big Five, MBTI, Emotions)
   - Profile Matrix with heatmap
   - Aggregated Statistics
   - Correlation Analysis
   - Emotional Patterns
   - Topic-Level Insights
   - Natural Language Summary
5. Download results in multiple formats:
   - Full Analysis (JSON)
   - Profile Matrix (CSV)
   - Summary (TXT)

## Key Features Tested

✅ **Data Ingestion**: Robust loading with error handling  
✅ **Normalization**: Multiple input formats supported  
✅ **Big Five Merging**: Mean, std dev, and count  
✅ **Emotion Merging**: Aggregated counts across files  
✅ **MBTI Merging**: Supports simple and nested formats  
✅ **Aggregated Statistics**: Reciprocity, response time, top emotions  
✅ **Profile Matrix**: Multi-file matrix with normalization  
✅ **Correlations**: Pearson & Spearman with p-values  
✅ **Emotional Insights**: Top emotions with associations  
✅ **Topic Insights**: Highest/lowest reciprocity, outliers  
✅ **Natural Language Summary**: Human-readable narratives  
✅ **Backward Compatibility**: Preserves v1.0 JSON structure  

## Continuous Integration

Tests can be integrated into CI/CD pipelines:
```bash
# GitHub Actions example
- name: Run Profile Fusion tests
  run: |
    pip install -r requirements.txt
    pip install pytest scipy
    pytest tests/test_profile_fusion*.py tests/test_analysis*.py -v
```

## Troubleshooting

### Import Errors

If you see import errors, ensure the repository root is in your Python path:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### Streamlit Warnings in Tests

Streamlit cache warnings in test output are expected and can be ignored:
```
WARNING streamlit.runtime.caching.cache_data_api: No runtime found
```

### Missing Dependencies

Install scipy for correlation analysis:
```bash
pip install scipy
```

## Contributing

When adding new features:
1. Add unit tests to appropriate `test_analysis_*.py` file
2. Add integration tests to `test_profile_fusion_integration.py`
3. Update test fixtures if needed
4. Run full test suite before committing
5. Ensure backward compatibility

## Version History

- **v2.0**: Complete refactor with modular architecture, comprehensive tests, enhanced analytics
- **v1.0**: Original merger implementation
