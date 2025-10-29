# Per-File Analysis UI Flow

## Visual UI Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                  WhatsApp Conversation Analyzer                  │
│                                                                   │
│  [x] Enable debug mode                                           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  FILE UPLOAD FORM                                                │
│                                                                   │
│  📁 Upload your whatsapp.txt file(s)                             │
│     [Choose Files] (max 5 files)                                 │
│                                                                   │
│  👤 Enter the username to analyze                                │
│     [_______________]                                             │
│                                                                   │
│                          [Upload Files]                          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  UPLOADED FILES                                                  │
│                                                                   │
│  Bulk Actions:                                                   │
│  [🔄 Analyze All] [🧠 Gen Local (3)] [🤖 Gen AI (3)] [🔀 Merge]│
│                                                                   │
│  ─────────────────────────────────────────────────────────────  │
│  FILE CARD 1                                                     │
│  📄 chat1.txt                           Status: ✅ Analyzed      │
│  Size: 12,345 bytes • Encoding: utf-8                           │
│                                                                   │
│  [Start Analysis]  or  [Re-analyze]                              │
│                                                                   │
│  Analysis Summary:                                               │
│  ┌─────────────┬─────────────┬─────────────┐                   │
│  │ Positive: 5 │ Negative: 2 │ Emotion: 0.7│                   │
│  └─────────────┴─────────────┴─────────────┘                   │
│                                                                   │
│  ▼ 📊 View Full Analysis for chat1.txt                          │
│     (Expandable: shows JSON, download button)                    │
│                                                                   │
│  Profile Generation:                                             │
│  [🧠 Generate Local Profile]  [🤖 Generate AI Profile]          │
│    or if ready:                                                  │
│  ✅ Local Profile Ready [📖 View]                                │
│  ✅ AI Profile Ready [📖 View]                                   │
│                                                                   │
│  ▼ 🧠 Local Profile: chat1.txt  (if viewing)                    │
│     [Shows profile text]                                         │
│     [📥 JSON] [📥 CSV] [📥 Flagged]                              │
│     [❌ Close]                                                    │
│  ─────────────────────────────────────────────────────────────  │
│                                                                   │
│  FILE CARD 2                                                     │
│  📄 chat2.txt                           Status: ⏳ Queued        │
│  Size: 8,765 bytes • Encoding: utf-8                            │
│  [Start Analysis]                                                │
│  ─────────────────────────────────────────────────────────────  │
│                                                                   │
│  FILE CARD 3                                                     │
│  📄 chat3.txt                           Status: ❌ Error         │
│  Size: 15,432 bytes • Encoding: utf-8                           │
│  Error: Could not parse conversation format                      │
│  [Start Analysis] (retry)                                        │
│  ─────────────────────────────────────────────────────────────  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  MERGED PERSONALITY PROFILE                                      │
│                                                                   │
│  🔀 Merged Personality Profile                                   │
│  Merged from 2 file(s): chat1.txt, chat2.txt                    │
│                                                                   │
│  ▼ 📖 View Merged Profile                                        │
│     [Shows merged profile text with aggregated metrics]          │
│                                                                   │
│     [📥 Download Merged JSON]                                    │
│     [📥 Download Merged CSV]                                     │
│     [📥 Download Merged Flagged]                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  DEBUG INFORMATION                                               │
│  [Download Analysis Logs]                                        │
└─────────────────────────────────────────────────────────────────┘
```

## State Transitions

### File Analysis Status
```
none/upload → queued → running → success
                              ↘ error → retry → queued
```

### Profile Generation Status (per file)
```
none → queued → running → success
                      ↘ error → retry → queued
```

### Overall Flow
```
1. Upload Files
   ↓
2. Files in session_state (status: queued)
   ↓
3. Click "Analyze All" or individual "Start Analysis"
   ↓
4. Status: running → success
   ↓
5. View summary, click "Generate Local Profile"
   ↓
6. Profile status: running → success
   ↓
7. View profile, download exports
   ↓
8. Repeat for other files or use "Gen Local (N)" bulk action
   ↓
9. Click "Merge" when 2+ profiles ready
   ↓
10. View merged profile and download combined exports
```

## User Workflows

### Workflow A: Single File (Backward Compatible)
1. Upload 1 file
2. Click "Upload Files"
3. Click "Start Analysis"
4. View results
5. Click "Generate Local Profile"
6. View and download profile
**Time**: ~2-3 clicks after upload

### Workflow B: Multiple Files with Bulk Actions
1. Upload 3 files
2. Click "Upload Files"
3. Click "Analyze All Files" (processes all 3)
4. Click "Generate Local Profiles for All" (generates 3 profiles)
5. Click "Merge Personality Profiles"
6. View merged profile and download
**Time**: ~4 clicks after upload (vs 9 clicks if done individually)

### Workflow C: Selective Analysis
1. Upload 5 files
2. Click "Upload Files"
3. Click "Start Analysis" on files 1, 3, 5 only
4. Generate profiles for those 3
5. Merge those 3 profiles
**Time**: Flexible, user controls which files to process

## Key UI Elements

### Status Indicators
- ⏳ Queued (white)
- 🔄 Running (yellow/warning)
- ✅ Success (green)
- ❌ Error (red)

### Action Buttons
- **Analysis Level**: "Start Analysis", "Re-analyze"
- **Profile Level**: "Generate Local Profile", "Generate AI Profile"
- **Bulk Level**: "Analyze All", "Gen Local (N)", "Gen AI (N)", "Merge"
- **View Level**: "View Local Profile", "View AI Profile", "View Full Analysis"
- **Download Level**: "📥 JSON", "📥 CSV", "📥 Flagged"

### Expandable Sections
- 📊 View Full Analysis for [filename]
- 🧠 Local Profile: [filename]
- 🤖 AI Profile: [filename]
- 📖 View Merged Profile

## Session State Management

The app uses Streamlit's session_state to persist data across reruns:

```python
st.session_state = {
    "files": [
        {
            "filename": str,
            "file_size_bytes": int,
            "decode_used": str,
            "content": str,
            "analysis": dict | None,
            "analysis_status": str,  # queued, running, success, error
            "analysis_time": float,
            "analysis_error": str | None,
            "local_profile": {
                "results": dict | None,
                "profile_text": str | None,
                "status": str,  # none, running, success, error
                "error": str | None
            },
            "ai_profile": {
                "response": str | None,
                "status": str,
                "error": str | None
            }
        }
    ],
    "merged_profiles": {
        "results": dict,
        "profile_text": str,
        "merged_from": list[str]
    } | None,
    "username": str,
    "show_local_0": bool,  # View toggles (dynamic keys)
    "show_ai_0": bool,
    "merge_requested": bool
}
```

## Performance Notes

- **Sequential Processing**: Analyses run one at a time (not parallel)
- **Auto-rerun**: App reruns after each status change to update UI
- **Spinners**: Show during long operations (analysis, profile generation, merge)
- **Caching**: `cached_run_analysis` uses Streamlit caching to avoid recomputation

## Accessibility

- Clear status indicators with colors and emojis
- Descriptive button labels
- Expandable sections to reduce clutter
- Caption text for additional context
- Error messages with retry options
