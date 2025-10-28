# WhatsApp Conversation Analyzer

A Streamlit-based project to analyze WhatsApp chat conversations using sentiment analysis, personality profiling, and emotional metrics.

## Features

- **Sentiment Analysis**: Analyze the emotional tone of conversations using VADER sentiment analysis
- **Personality Profiling**: Calculate Big Five (OCEAN) personality traits and MBTI types
- **Emotion Analysis**: Detect and categorize emotions from text and emojis
- **Response Time Tracking**: Measure conversation dynamics and response patterns
- **Emotional Reciprocity**: Evaluate mutual emotional engagement between participants
- **Topic Classification**: Automatically identify conversation topics
- **Local Psychological Profile Generation**: Generate comprehensive psychological profiles without external AI services
- **AI-Powered Profiles**: Optional integration with g4f for AI-generated summaries

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

1. Upload a WhatsApp chat export file (`.txt` format)
2. Enter the username you want to analyze
3. Click "Start Analysis" to process the conversations
4. View the analysis results including:
   - Positive and negative topics
   - Emotional variability
   - Per-conversation metrics
5. Generate a psychological profile:
   - **Local Profile**: Click "Generate Local Psychological Profile" for a deterministic, privacy-focused analysis
   - **AI Profile**: Click "Generate Psychological Profile with AI" for an AI-generated summary (requires internet)

## Local Profile Generation

The local profile generator (`local_profile_generator.py`) provides a comprehensive analysis pipeline that runs entirely on your machine without external API calls:

### Analysis Steps

1. **Load and Validate**: Ensures data integrity and consistency
2. **Normalize Structure**: Flattens conversation data into a uniform format
3. **Clean Data**: Handles missing values and type conversions
4. **Compute Basic Metrics**: Calculates statistical measures across conversations
5. **Aggregate Personality Data**: Summarizes Big Five personality traits
6. **Correlation Analysis**: Identifies relationships between traits and behaviors
7. **Filter and Segment**: Groups data by topic and MBTI type
8. **Emotion Insights**: Flags unusual emotional patterns
9. **Advanced Analysis**: Optional clustering (requires scikit-learn)
10. **Export Results**: Generates JSON and CSV outputs

### Output

The local profile generation provides:
- **Human-readable summary**: 3-6 bullet points highlighting key findings
- **Structured metrics**: JSON data with all computed statistics
- **Downloadable exports**:
  - Complete analysis results (JSON)
  - Per-conversation data (CSV)
  - Flagged conversations (JSON)

### Optional Dependencies

For enhanced functionality, install:
```bash
pip install pandas numpy scipy scikit-learn
```

The local analysis will work without these libraries but with reduced statistical capabilities (manual computation fallbacks are provided).

## Testing

Run the test suite:
```bash
python -m unittest tests.test_local_profile_generator
```

## Privacy

The local profile generation feature ensures your data remains private by:
- Processing all data locally on your machine
- Not making external network requests
- Using deterministic algorithms (no AI models)
- Providing full control over data exports
