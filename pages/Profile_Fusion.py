"""
DEPRECATED: Profile Fusion Page

This separate fusion page has been deprecated and is no longer needed.

The main Streamlit app (streamlit_app.py) now supports uploading and merging
1-5 WhatsApp export files directly. The fusion functionality has been integrated
into the main analysis pipeline.

To merge multiple WhatsApp exports:
1. Go to the main app (streamlit_app.py)
2. Upload 1-5 .txt files using the file uploader
3. Files will be automatically merged, deduplicated, and analyzed together

This file is kept for reference but is not functional.
Date deprecated: 2025-10-29
"""

import streamlit as st

st.title("⚠️ Page Deprecated")
st.error(
    "**This page is no longer available.**\n\n"
    "The fusion functionality has been integrated into the main app.\n\n"
    "Please use the main WhatsApp Conversation Analyzer page, which now supports "
    "uploading multiple files (1-5) that will be automatically merged and analyzed."
)

st.info(
    "**New Multi-File Upload Process:**\n"
    "1. Go back to the main page\n"
    "2. Upload 1-5 WhatsApp export .txt files\n"
    "3. Files are automatically merged, deduplicated, and analyzed together\n"
    "4. All results include file origin metadata for traceability"
)
