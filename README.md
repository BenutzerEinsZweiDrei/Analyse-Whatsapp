# WhatsApp Conversation Analyzer

A Streamlit-based application to analyze WhatsApp chat conversations using sentiment analysis, personality profiling, and emotional metrics.

## Features

- **Sentiment Analysis**: Analyze the emotional tone of conversations using VADER sentiment analysis
- **Personality Profiling**: Calculate Big Five (OCEAN) personality traits and MBTI types
- **Emotion Analysis**: Detect and categorize emotions from text and emojis
- **Response Time Tracking**: Measure conversation dynamics and response patterns
- **Emotional Reciprocity**: Evaluate mutual emotional engagement between participants
- **Topic Classification**: Automatically identify conversation topics (requires Jina AI API key)
- **Local Psychological Profile Generation**: Generate comprehensive psychological profiles without external AI services
- **AI-Powered Profiles**: Optional integration with g4f for AI-generated summaries

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
│   │   ├── parser.py         # WhatsApp message parsing
│   │   ├── preprocessing.py  # Text preprocessing
│   │   ├── keywords.py       # Keyword extraction
│   │   ├── nouns.py          # Noun extraction
│   │   ├── emojis.py         # Emoji analysis
│   │   ├── sentiment.py      # Sentiment analysis
│   │   ├── metrics.py        # Response times, reciprocity
│   │   ├── personality.py    # Big Five, MBTI, emotions
│   │   ├── local_profile.py  # Local analysis pipeline
│   │   └── summarizer.py     # Matrix summarization
│   └── services/             # External service clients
│       ├── jina_client.py    # Jina AI classification
│       ├── textrazor_client.py # TextRazor NLP
│       └── g4f_client.py     # g4f AI generation
├── tests/                     # Comprehensive test suite
└── data/                      # JSON assets (stopwords, emojis, etc.)
```

## Installation

1. Clone the repository
```bash
git clone https://github.com/BenutzerEinsZweiDrei/Analyse-Whatsapp.git
cd Analyse-Whatsapp
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

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

## Exporting WhatsApp Chats

To export a WhatsApp chat:

1. Open WhatsApp on your phone
2. Open the chat you want to analyze
3. Tap the three dots (⋮) or settings icon
4. Select "More" → "Export chat"
5. Choose "Without Media"
6. Save the `.txt` file and upload it to the analyzer

## Development

### Running Tests

```bash
# Run all tests
python -m unittest discover tests/ -v

# Run specific test file
python -m unittest tests.test_app_modules -v

# Run with pytest (if installed)
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=app --cov-report=term-missing
```

### Linting and Formatting

```bash
# Install development dependencies
pip install ruff black isort pytest pytest-cov

# Run linter
ruff check .

# Format code
black .

# Sort imports
isort .
```

### Code Structure Guidelines

- **Modularity**: Each module should have a single, well-defined responsibility
- **Type Hints**: All functions should have type hints
- **Docstrings**: All public functions should have docstrings
- **Error Handling**: Use graceful degradation - log warnings instead of crashing
- **Testing**: Add tests for new functionality
- **Security**: Never commit API keys or secrets

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

## Testing

The project includes comprehensive test coverage:

```
tests/
├── test_app_modules.py              # Unit tests for app/ modules (25 tests)
├── test_local_profile_generator.py  # Local profile tests (12 tests)
└── test_integration.py              # Integration tests
```

Run tests:
```bash
python -m unittest discover tests/ -v
```

All 37 tests should pass.

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
