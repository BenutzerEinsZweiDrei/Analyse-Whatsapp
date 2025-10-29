# WhatsApp Conversation Analyzer

A Streamlit-based application to analyze WhatsApp chat conversations using sentiment analysis, personality profiling, and emotional metrics.

> **📢 Update (October 2025)**: The separate Profile Fusion page has been deprecated. The main app now supports uploading and merging 1-5 WhatsApp export files directly! See [Usage](#usage) for details.

> **✨ Enhanced Analysis (October 2025)**: Major improvements to analysis accuracy and depth! New ensemble sentiment/emotion detection, advanced topic extraction, and richer personality profiles with confidence scores and evidence. See [ENHANCED_ANALYSIS.md](ENHANCED_ANALYSIS.md) for details.

## Features

### Core Features
- **Multi-File Support**: Upload and merge 1-5 WhatsApp export files automatically
- **Advanced Sentiment Analysis**: Ensemble approach combining VADER + transformer models (optional) + emoji analysis
- **Enhanced Emotion Detection**: Multi-method emotion classification with 7 emotion categories and confidence scores
- **Personality Profiling**: Calculate Big Five (OCEAN) personality traits and MBTI types with evidence
- **Robust Feature Extraction**: Comprehensive message metadata including linguistic features, emojis, URLs, mentions
- **Advanced Topic Extraction**: KeyBERT, YAKE, and TF-IDF with coherence scores and representative messages
- **Response Time Tracking**: Measure conversation dynamics and response patterns
- **Emotional Reciprocity**: Evaluate mutual emotional engagement between participants
- **Local Psychological Profile Generation**: Generate comprehensive psychological profiles without external AI services
- **AI-Powered Profiles**: Enhanced integration with g4f using structured JSON prompts with validation

### Enhanced Analysis Capabilities
- **Confidence Scores**: All analysis results include confidence metrics
- **Evidence-Based**: Personality traits include supporting message snippets
- **Multilingual Support**: Language detection and multi-language keyword extraction
- **Graceful Degradation**: Works with or without optional ML libraries
- **Enriched Outputs**: Detailed per-author statistics, topic coherence, feature summaries

📖 **See [ENHANCED_ANALYSIS.md](ENHANCED_ANALYSIS.md) for detailed documentation on new features.**

## Architecture

The application has been refactored into a modular structure:

```
├── streamlit_app.py          # Main Streamlit UI (thin wrapper)
├── app/                       # Core application package
│   ├── config.py             # Settings and API key management
│   ├── logging_config.py     # Centralized logging
│   ├── cache.py              # Caching abstraction
│   ├── run_analysis.py       # Main analysis orchestrator
│   ├── data/                 # Data loaders
│   │   └── loaders.py
│   ├── core/                 # Core analysis modules
│   │   ├── parser.py              # WhatsApp message parsing
│   │   ├── preprocessing.py       # Text preprocessing
│   │   ├── feature_extraction.py  # 🆕 Robust feature extraction
│   │   ├── sentiment_enhanced.py  # 🆕 Ensemble sentiment analysis
│   │   ├── emotion_detection.py   # 🆕 Multi-method emotion detection
│   │   ├── topic_extraction.py    # 🆕 Advanced topic/keyword extraction
│   │   ├── keywords.py            # Keyword extraction
│   │   ├── nouns.py               # Noun extraction
│   │   ├── emojis.py              # Emoji analysis
│   │   ├── sentiment.py           # VADER sentiment analysis
│   │   ├── metrics.py             # Response times, reciprocity
│   │   ├── personality.py         # Big Five, MBTI, emotions
│   │   ├── local_profile.py       # Local analysis pipeline
│   │   └── summarizer.py          # Matrix summarization (enhanced)
│   └── services/             # External service clients
│       ├── jina_client.py    # Jina AI classification
│       ├── textrazor_client.py # TextRazor NLP
│       └── g4f_client.py     # g4f AI generation (enhanced)
├── tests/                     # Comprehensive test suite (195 tests)
└── data/                      # JSON assets (stopwords, emojis, etc.)
```

## Installation

### Basic Installation

1. Clone the repository
```bash
git clone https://github.com/BenutzerEinsZweiDrei/Analyse-Whatsapp.git
cd Analyse-Whatsapp
```

2. Install core dependencies:
```bash
pip install -r requirements.txt
```

### Optional: Enhanced Analysis Features

For best accuracy, install optional dependencies (recommended):

```bash
# Tier 2: Basic enhancements (small, fast)
pip install emoji langdetect yake scikit-learn

# Tier 3: Advanced ML models (large downloads, ~500MB)
pip install transformers torch sentence-transformers keybert

# For CPU-only (lighter)
pip install transformers torch --index-url https://download.pytorch.org/whl/cpu
```

**Note:** The app works without optional dependencies but with reduced accuracy. See [ENHANCED_ANALYSIS.md](ENHANCED_ANALYSIS.md) for details.

### API Keys Configuration (Optional)

3. Configure API keys (optional but recommended):

**Option A: Environment Variables**
```bash
cp .env.example .env
# Edit .env and add your API keys
export $(cat .env | xargs)
```

**Option B: Streamlit Secrets**
```bash
mkdir -p .streamlit
cat > .streamlit/secrets.toml << EOF
JINA_API_KEY = "your_jina_api_key"
TEXTRAZOR_API_KEY = "your_textrazor_api_key"
EOF
```

**Getting API Keys:**
- **Jina AI**: Sign up at [https://jina.ai/](https://jina.ai/) for text classification
- **TextRazor**: Sign up at [https://www.textrazor.com/](https://www.textrazor.com/) for advanced NLP

**Note**: The app works without API keys, but topic classification will be disabled.

## Usage

Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

### Single or Multiple File Analysis

The app now supports analyzing **1-5 WhatsApp export files** in a single analysis:

1. Upload one or multiple WhatsApp chat export files (`.txt` format, max 5 files)
   - Multiple files will be automatically merged and deduplicated
   - Useful for split exports or continuation of chats
2. Enter the username you want to analyze
3. Click "Start Analysis" to process the conversations
4. View the analysis results including:
   - Positive and negative topics
   - Emotional variability
   - Per-conversation metrics
   - File origin metadata for each conversation
5. Generate a psychological profile:
   - **Local Profile**: Click "Generate Local Psychological Profile" for a deterministic, privacy-focused analysis
   - **AI Profile**: Click "Generate Psychological Profile with AI" for an AI-generated summary (requires internet)

### Multi-File Merge Features

- **Automatic Merging**: Messages from all files are combined chronologically
- **Deduplication**: Duplicate messages (same timestamp, sender, and text) are automatically removed
- **File Tracking**: Each conversation includes metadata about which file(s) it originated from
- **Encoding Support**: Each file can use different encoding (UTF-8 or Latin-1)

## Exporting WhatsApp Chats

To export a WhatsApp chat:

1. Open WhatsApp on your phone
2. Open the chat you want to analyze
3. Tap the three dots (⋮) or settings icon
4. Select "More" → "Export chat"
5. Choose "Without Media"
6. Save the `.txt` file and upload it to the analyzer

**Tip**: If you have a long chat history, WhatsApp may split it into multiple files. You can now upload all parts at once, and they will be merged automatically!

## Development

### Quick Start with Make

This project includes a Makefile for common development tasks:

```bash
# Show all available commands
make help

# Install production dependencies
make install

# Install development dependencies (includes testing and linting tools)
make install-dev

# Run tests
make test

# Run linters (ruff, black, isort)
make lint

# Auto-format code
make format

# Clean build artifacts and caches
make clean

# Run Streamlit app
make run
```

### Manual Setup

If you prefer not to use Make, here are the equivalent commands:

**Install dependencies:**
```bash
# Production dependencies
pip install -r requirements.txt

# Development dependencies
pip install pytest pytest-cov black ruff isort
```

**Run tests:**
```bash
# Run all tests with pytest
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ -v --cov=app --cov-report=term-missing --cov-report=html

# Run specific test file
python -m pytest tests/test_app_modules.py -v
```

**Linting and formatting:**
```bash
# Check code style
ruff check . --extend-exclude="local_profile_generator.py"
black --check . --extend-exclude="local_profile_generator.py"
isort --check-only --profile black . --skip local_profile_generator.py

# Auto-fix formatting
black . --extend-exclude="local_profile_generator.py"
isort . --profile black --skip local_profile_generator.py
ruff check . --fix --extend-exclude="local_profile_generator.py"
```

**Run the application:**
```bash
streamlit run streamlit_app.py
```

### Project Structure

```
Analyse-Whatsapp/
├── streamlit_app.py          # Main UI entry point (thin wrapper)
├── app/                       # Core application package
│   ├── config.py             # Settings and API key management
│   ├── cache.py              # Caching utilities
│   ├── run_analysis.py       # Main analysis orchestrator
│   ├── core/                 # Core analysis modules
│   │   ├── parsing.py        # WhatsApp message parsing
│   │   ├── preprocessing.py  # Text preprocessing
│   │   ├── feature_extraction.py    # Feature extraction
│   │   ├── sentiment_enhanced.py    # Sentiment analysis
│   │   ├── emotion_detection.py     # Emotion detection
│   │   ├── topic_extraction.py      # Topic modeling
│   │   ├── personality.py           # Personality trait calculation
│   │   ├── local_profile.py         # Local profile generation
│   │   └── summarizer.py            # Result summarization
│   ├── services/             # External service integrations
│   │   ├── ai_provider.py    # AI profile generation (g4f)
│   │   ├── jina_client.py    # Jina AI text classification
│   │   └── textrazor_client.py      # TextRazor NLP
│   ├── utils/                # Utility modules
│   │   ├── logging_config.py        # Centralized logging
│   │   ├── io.py                    # File I/O helpers
│   │   └── types.py                 # Type definitions
│   └── data/                 # Data loaders
│       └── loaders.py
├── tests/                     # Test suite
│   ├── fixtures/             # Test data
│   └── test_*.py             # Test modules
├── .github/workflows/        # CI/CD configuration
├── pyproject.toml            # Project configuration (Black, Ruff, pytest)
├── requirements.txt          # Python dependencies
├── Makefile                  # Development commands
└── README.md                 # This file
```

### Code Structure Guidelines

- **Modularity**: Each module should have a single, well-defined responsibility
- **Type Hints**: All functions should have type hints
- **Docstrings**: All public functions should have docstrings
- **Error Handling**: Use graceful degradation - log warnings instead of crashing
- **Testing**: Add tests for new functionality
- **Security**: Never commit API keys or secrets

## Troubleshooting

### Multiple File Upload Issues

**Problem**: Uploaded files show encoding errors
- **Solution**: The app automatically tries UTF-8 first, then falls back to Latin-1. Check that your files are valid WhatsApp exports.

**Problem**: Duplicate messages after merge
- **Solution**: This is expected behavior. The app automatically deduplicates messages with identical timestamp, sender, and text.

**Problem**: Messages out of chronological order
- **Solution**: The app automatically sorts merged messages by timestamp. If timestamps are missing or invalid in the original export, those messages may appear at the end.

**Problem**: Analysis takes too long with multiple files
- **Solution**: Large combined chats (50,000+ messages) may take several minutes. Consider analyzing shorter time periods or specific conversations.

### File Format Issues

**Problem**: "Could not parse conversations" error
- **Solution**: Ensure your file follows WhatsApp's export format: `DD.MM.YY, HH:MM - Username: Message`

**Problem**: Wrong encoding detected
- **Solution**: Check the file info display after upload. If encoding is incorrect, re-export the chat from WhatsApp.

### General Issues

**Problem**: API key errors for topic classification
- **Solution**: Topic classification requires a Jina AI API key. Without it, topics will show as "no topic" but all other analysis continues.

**Problem**: Memory errors with large chats
- **Solution**: Try splitting very large chats into smaller time periods before export, or increase available system memory.

## Local Profile Generation

The local profile generator (`app/core/local_profile.py`) provides a comprehensive analysis pipeline that runs entirely on your machine without external API calls:

### Analysis Pipeline (11 Steps)

1. **Load and Validate**: Ensures data integrity
2. **Normalize Structure**: Flattens conversation data
3. **Clean Data**: Handles missing values
4. **Compute Basic Metrics**: Statistical measures
5. **Aggregate Personality Data**: Big Five traits
6. **Correlation Analysis**: Trait relationships
7. **Filter and Segment**: Group by topic/MBTI
8. **Emotion Insights**: Flag unusual patterns
9. **Visualizations**: (Optional, placeholder)
10. **Advanced Analysis**: Optional clustering
11. **Export Results**: Generate JSON/CSV outputs

### Output

- **Human-readable summary**: Key findings (3-6 bullet points)
- **Structured metrics**: JSON with all statistics
- **Downloadable exports**: Complete analysis (JSON), per-conversation data (CSV), flagged conversations (JSON)

### Optional Dependencies

For enhanced functionality:
```bash
pip install pandas numpy scipy scikit-learn
```

The local analysis works without these libraries but with reduced statistical capabilities.

## Migration from Profile Fusion

**If you previously used the separate "Profile Fusion" page:**

The Profile Fusion page has been deprecated and integrated into the main app. Here's what changed:

**Before (Old Workflow)**:
1. Analyze each file separately in the main app
2. Download JSON results for each file
3. Go to Profile Fusion page
4. Upload multiple JSON files
5. Merge and analyze

**Now (New Workflow)**:
1. Upload 1-5 WhatsApp .txt files directly in the main app
2. Files are automatically merged, deduplicated, and analyzed
3. Get results immediately - no separate fusion step needed

**Benefits of the new approach**:
- Simpler workflow (one step instead of multiple)
- Faster processing (no need to save/load intermediate JSON files)
- Better deduplication (works at message level, not conversation level)
- File origin tracking (know which file each message came from)
- Maintains all existing analysis features

The Profile Fusion page now shows a deprecation notice directing users to the main app.

## Testing

The project includes comprehensive test coverage:

```
tests/
├── test_app_modules.py              # Unit tests for app/ modules (25 tests)
├── test_local_profile_generator.py  # Local profile tests (12 tests)
├── test_multi_file_merge.py         # Multi-file merge tests (7 tests)
└── test_integration.py              # Integration tests
```

Run tests:
```bash
python -m unittest discover tests/ -v
```

All tests should pass.

## CI/CD

The project uses GitHub Actions for continuous integration:

- **Tests**: Run on Python 3.10, 3.11, 3.12
- **Linting**: ruff, black, isort
- **Coverage**: pytest with coverage reporting

See `.github/workflows/ci.yml` for details.

## Privacy

The local profile generation feature ensures your data remains private by:
- Processing all data locally on your machine
- Not making external network requests
- Using deterministic algorithms (no AI models)
- Providing full control over data exports

## Security

- **No hard-coded API keys**: All keys loaded from environment or Streamlit secrets
- **Graceful degradation**: Works without API keys (some features disabled)
- **Input validation**: All user inputs are validated
- **Error handling**: Comprehensive error handling prevents crashes

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting
5. Commit your changes (`git commit -m 'feat: add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Commit Message Convention

Follow Conventional Commits:
- `feat:` New feature
- `fix:` Bug fix
- `refactor:` Code refactoring
- `docs:` Documentation changes
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

- VADER Sentiment Analysis
- NLTK for NLP
- Empath for lexical analysis
- TextRazor for entity extraction
- Jina AI for text classification
- g4f for AI text generation
- Streamlit for the web interface
