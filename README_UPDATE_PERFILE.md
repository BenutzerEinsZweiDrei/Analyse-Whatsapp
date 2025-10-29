# README Update for Per-File Analysis Feature

## Add this section to README.md after the "Features" section

---

## Per-File Analysis & Profile Merging (NEW!)

The app now supports advanced per-file workflows for analyzing multiple WhatsApp exports:

### 🎯 Key Capabilities

1. **Individual File Analysis**: Each uploaded file is analyzed independently with its own status tracking
2. **Per-File Personality Profiles**: Generate local or AI profiles for each file separately
3. **Bulk Operations**: Process multiple files with one click
4. **Profile Merging**: Combine personality profiles from multiple files into a unified analysis

### 📋 Workflow Options

#### Single File (Backward Compatible)
```
Upload → Analyze → Generate Profile → Download
```
Works exactly as before - no changes to existing workflow!

#### Multiple Files with Merge
```
Upload 3 files → Analyze All → Generate Local Profiles for All → Merge Profiles → Download Combined Results
```
Efficiently process multiple files and get aggregated insights!

#### Selective Analysis
```
Upload 5 files → Pick 3 to analyze → Generate profiles → Merge selected → Download
```
Full control over which files to process and merge!

### 🔀 Profile Merging Details

When you merge multiple personality profiles:
- **Big Five Traits**: Weighted averages by conversation count
- **Emotion Metrics**: Aggregated across all files
- **MBTI Distribution**: Combined type frequencies
- **Conversation Data**: Unified with source file tracking
- **Exports**: Merged JSON, CSV, and flagged conversations

### 📊 UI Elements

Each uploaded file gets a card showing:
- ✅ File status (Queued, Running, Success, Error)
- 📊 Analysis summary (topics, emotion variability)
- 🧠 Local profile generation button
- 🤖 AI profile generation button
- 📥 Download buttons for all exports

Bulk action buttons:
- 🔄 **Analyze All Files**: Process all uploaded files
- 🧠 **Generate Local Profiles for All**: Create profiles for all analyzed files
- 🤖 **Generate AI Profiles for All**: Generate AI profiles for all analyzed files
- 🔀 **Merge Personality Profiles**: Combine 2+ local profiles

### 📖 Documentation

For complete details, see:
- **PERFILE_ANALYSIS_DOCS.md**: Comprehensive feature documentation
- **UI_FLOW.md**: Visual UI flow diagrams and workflows
- **IMPLEMENTATION_SUMMARY_PERFILE.md**: Technical implementation details

---

## Update Usage Section

Replace the existing Usage section with:

## Usage

### Basic Single-File Analysis

1. Export your WhatsApp chat:
   - Open WhatsApp
   - Go to Settings → Export chat → Without media
   - Save the .txt file

2. Upload and analyze:
   ```bash
   streamlit run streamlit_app.py
   ```

3. In the web interface:
   - Upload your .txt file
   - Enter the username to analyze
   - Click "Upload Files"
   - Click "Start Analysis" on the file card
   - Generate Local or AI profile
   - Download results

### Multi-File Analysis with Merge

1. Export multiple WhatsApp chats (1-5 files)

2. In the web interface:
   - Upload all .txt files
   - Enter the username to analyze
   - Click "Upload Files"
   - Click "Analyze All Files" to process all at once
   - Click "Generate Local Profiles for All"
   - Click "Merge Personality Profiles"
   - Download merged results

### Advanced: Selective Analysis

1. Upload multiple files
2. Click "Start Analysis" only on specific files you want to process
3. Generate profiles for selected files
4. Merge only the profiles you want to combine

---

## Add to Features Section

Update the features list to include:

- **Per-File Analysis**: Analyze each WhatsApp export file independently with status tracking
- **Per-File Profiles**: Generate personality profiles for each file (local and AI-powered)
- **Bulk Actions**: Process multiple files efficiently with one-click bulk operations
- **Profile Merging**: Combine multiple personality profiles into unified analysis with weighted averaging
- **Source Tracking**: All merged data includes source file information for traceability
