"""
WhatsApp Conversation Analyzer - Streamlit UI

This is the main Streamlit application entry point.
All core logic has been moved to the app/ package.

Updated to support per-file analysis, per-file personality profiles, and merge functionality.
"""

import importlib
import importlib.metadata
import json
import os
import platform
import time

import streamlit as st

# Import from app package
from app.config import get_settings, mask_key
from app.core.local_profile import merge_local_profiles, run_local_analysis
from app.core.preprocessing import init_nltk
from app.core.summarizer import summarize_matrix
from app.logging_config import configure_logging, get_logs, set_debug_mode
from app.run_analysis import cached_run_analysis
from app.services.g4f_client import generate_profile, handle_g4f_error

# ---------------------------
# Initialize Application
# ---------------------------

# Configure logging
logger = configure_logging(debug=False)

# Initialize NLTK resources (downloads if needed)
# This is done at startup to avoid delays during first analysis
try:
    init_nltk()
    logger.info("NLTK resources initialized")
except Exception as e:
    logger.warning(f"NLTK initialization warning: {e}")

# Get settings (API keys from environment or secrets)
settings = get_settings()

# ---------------------------
# Streamlit UI
# ---------------------------

st.title("WhatsApp Conversation Analyzer")

# Initialize session state for per-file tracking
if "files" not in st.session_state:
    st.session_state.files = []  # List of file dicts
if "merged_profiles" not in st.session_state:
    st.session_state.merged_profiles = None
if "username" not in st.session_state:
    st.session_state.username = ""

# Debug toggle in UI
debug_mode = st.checkbox("Enable debug mode (show logs and detailed info)", value=False)
set_debug_mode(debug_mode)

# Show environment info in debug mode
if debug_mode:
    with st.expander("Environment & Dependency Info (debug)"):
        # Core environment
        env_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "cwd": os.getcwd(),
        }
        st.write(env_info)

        # Package versions
        key_packages = [
            "streamlit",
            "nltk",
            "textrazor",
            "empath",
            "emot",
            "g4f",
            "requests",
            "regex",
            "vaderSentiment",
        ]
        pkg_versions = {}
        for pkg in key_packages:
            try:
                pkg_versions[pkg] = importlib.metadata.version(pkg)
            except Exception:
                try:
                    pkg_versions[pkg] = importlib.import_module(pkg).__version__
                except Exception:
                    pkg_versions[pkg] = "not installed / unknown"
        st.write(pkg_versions)

        # Masked API keys
        st.write(
            {
                "jina_key": mask_key(settings.jina_api_key),
                "textrazor_key": mask_key(settings.textrazor_api_key),
            }
        )
        st.write(
            "Note: For production, configure API keys via Streamlit secrets or environment variables."
        )

# File upload and analysis form
st.info(
    "📋 **How to get started:**\n"
    "1. Export your WhatsApp chat (Settings → Export chat → Without media)\n"
    "2. Upload 1-5 .txt files below\n"
    "3. Enter the username you want to analyze (as it appears in the chat)\n"
    "4. Click 'Upload Files' to begin"
)

with st.form("upload_form"):
    uploaded_files = st.file_uploader(
        "Upload your whatsapp.txt file(s)",
        type=["txt"],
        accept_multiple_files=True,
        help="Upload 1-5 WhatsApp chat export files.",
    )
    st.caption("Upload one or more exported WhatsApp chat files in .txt format (max 5 files)")

    username = st.text_input("Enter the username to analyze", value=st.session_state.username)
    st.caption("Enter the exact username as it appears in the chat messages")

    submit_upload = st.form_submit_button("Upload Files")

if submit_upload:
    try:
        if not uploaded_files:
            st.error("Please upload at least one whatsapp.txt file.")
            logger.warning("Upload pressed but no files uploaded.")
        elif len(uploaded_files) > 5:
            st.error("Please upload a maximum of 5 files.")
            logger.warning(f"Too many files uploaded: {len(uploaded_files)}")
        elif not username.strip():
            st.error("Please enter a username.")
            logger.warning("Upload pressed but username is empty.")
        else:
            # Clear previous session state
            st.session_state.files = []
            st.session_state.merged_profiles = None
            st.session_state.username = username.strip()

            # Process uploaded files
            for idx, uploaded_file in enumerate(uploaded_files):
                file_bytes = uploaded_file.read()
                file_size = len(file_bytes)
                filename = getattr(uploaded_file, "name", f"file_{idx+1}.txt")

                logger.info(
                    f"File {idx+1}/{len(uploaded_files)}: filename={filename}, size={file_size} bytes"
                )

                # Try to decode with utf-8 first, fallback to latin-1
                decode_used = "utf-8"
                try:
                    file_content = file_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    try:
                        file_content = file_bytes.decode("latin-1")
                        decode_used = "latin-1"
                        logger.warning(f"File '{filename}' decoded with latin-1 fallback.")
                    except Exception as e:
                        st.error(
                            f"Could not decode file '{filename}'. Please provide a UTF-8 encoded text file."
                        )
                        logger.exception(f"Failed to decode file '{filename}': {e}")
                        continue

                # Create file state object
                file_state = {
                    "filename": filename,
                    "file_size_bytes": file_size,
                    "decode_used": decode_used,
                    "content": file_content,
                    "analysis": None,
                    "analysis_status": "queued",  # queued, running, success, error
                    "analysis_time": 0.0,
                    "analysis_error": None,
                    "local_profile": {
                        "results": None,
                        "profile_text": None,
                        "status": "none",  # none, running, success, error
                        "error": None,
                    },
                    "ai_profile": {
                        "response": None,
                        "status": "none",
                        "error": None,
                    },
                }

                st.session_state.files.append(file_state)
                logger.info(f"Added file '{filename}' to session state")

            st.success(f"✅ Uploaded {len(st.session_state.files)} file(s). Ready for analysis!")
            st.rerun()

    except Exception as e:
        logger.exception(f"Error during file upload: {e}")
        st.error(f"An error occurred: {e}")
        if debug_mode:
            st.exception(e)

# ---------------------------
# Display Per-File Cards and Actions
# ---------------------------

if st.session_state.files:
    st.divider()
    st.subheader("📁 Uploaded Files")

    # Bulk actions
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🔄 Analyze All Files", key="analyze_all"):
            for file_state in st.session_state.files:
                if file_state["analysis_status"] not in ["success"]:
                    file_state["analysis_status"] = "queued"
            st.session_state.process_queue = True
            st.rerun()

    with col2:
        # Count analyzed files without local profiles
        analyzed_without_local = sum(
            1
            for f in st.session_state.files
            if f["analysis_status"] == "success" and f["local_profile"]["status"] != "success"
        )
        if st.button(
            f"🧠 Generate Local Profiles for All ({analyzed_without_local})",
            key="generate_local_all",
            disabled=analyzed_without_local == 0,
        ):
            for file_state in st.session_state.files:
                if (
                    file_state["analysis_status"] == "success"
                    and file_state["local_profile"]["status"] != "success"
                ):
                    file_state["local_profile"]["status"] = "queued"
            st.session_state.process_queue = True
            st.rerun()

    with col3:
        # Count analyzed files without AI profiles
        analyzed_without_ai = sum(
            1
            for f in st.session_state.files
            if f["analysis_status"] == "success" and f["ai_profile"]["status"] != "success"
        )
        if st.button(
            f"🤖 Generate AI Profiles for All ({analyzed_without_ai})",
            key="generate_ai_all",
            disabled=analyzed_without_ai == 0,
        ):
            for file_state in st.session_state.files:
                if (
                    file_state["analysis_status"] == "success"
                    and file_state["ai_profile"]["status"] != "success"
                ):
                    file_state["ai_profile"]["status"] = "queued"
            st.session_state.process_queue = True
            st.rerun()

    with col4:
        # Count files with local profiles
        files_with_local = sum(
            1 for f in st.session_state.files if f["local_profile"]["status"] == "success"
        )
        if st.button(
            f"🔀 Merge personality profiles ({files_with_local}) — Create final merged profile",
            key="merge_profiles",
            disabled=files_with_local < 2,
        ):
            st.session_state.merge_requested = True
            st.rerun()

    st.caption("Use bulk actions to process multiple files at once")

    # Show clear merge CTA when we have 2+ local profiles
    if files_with_local >= 2 and not st.session_state.merged_profiles:
        st.info(
            f"✨ **Now we have {files_with_local} personality profiles — "
            f"merge them into a big final one!**"
        )

    # Guard against re-entrancy during processing and only process if queue flag is set
    should_process = st.session_state.get("process_queue", False) and not st.session_state.get(
        "processing_pass", False
    )

    if should_process:
        # Set processing flag and clear queue flag
        st.session_state.processing_pass = True
        st.session_state.process_queue = False

        # Track if we need to rerun after processing
        state_changed = False

        logger.info("Starting processing pass for queued items")

        # Process queued analyses (batch process all queued items)
        for file_state in st.session_state.files:
            if file_state["analysis_status"] == "queued":
                file_state["analysis_status"] = "running"
                logger.info(f"Starting analysis for '{file_state['filename']}'")
                try:
                    with st.spinner(f"Analyzing {file_state['filename']}..."):
                        start_time = time.time()

                        # Run analysis for single file
                        matrix, conversation_messages = cached_run_analysis(
                            file_state["content"], st.session_state.username
                        )

                        # Summarize results
                        summary = summarize_matrix(matrix)

                        analysis_time = time.time() - start_time

                        # Store results
                        file_state["analysis"] = {
                            "matrix": matrix,
                            "summary": summary,
                            "conversation_messages": conversation_messages,
                        }
                        file_state["analysis_status"] = "success"
                        file_state["analysis_time"] = analysis_time
                        state_changed = True

                        logger.info(
                            f"Analysis completed for '{file_state['filename']}' in {analysis_time:.2f}s"
                        )

                except Exception as e:
                    logger.exception(f"Error analyzing '{file_state['filename']}': {e}")
                    file_state["analysis_status"] = "error"
                    file_state["analysis_error"] = str(e)
                    state_changed = True

        # Process queued local profiles (batch process all queued items)
        for file_state in st.session_state.files:
            if file_state["local_profile"]["status"] == "queued":
                file_state["local_profile"]["status"] = "running"
                logger.info(f"Starting local profile generation for '{file_state['filename']}'")
                try:
                    with st.spinner(f"Generating local profile for {file_state['filename']}..."):
                        summary = file_state["analysis"]["summary"]
                        matrix = file_state["analysis"]["matrix"]

                        results, profile_text = run_local_analysis(summary, matrix)

                        file_state["local_profile"]["results"] = results
                        file_state["local_profile"]["profile_text"] = profile_text
                        file_state["local_profile"]["status"] = "success"
                        state_changed = True

                        logger.info(
                            f"Local profile generation completed for '{file_state['filename']}'"
                        )

                except Exception as e:
                    logger.exception(
                        f"Error generating local profile for '{file_state['filename']}': {e}"
                    )
                    file_state["local_profile"]["status"] = "error"
                    file_state["local_profile"]["error"] = str(e)
                    state_changed = True

        # Process queued AI profiles (batch process all queued items)
        for file_state in st.session_state.files:
            if file_state["ai_profile"]["status"] == "queued":
                file_state["ai_profile"]["status"] = "running"
                logger.info(f"Starting AI profile generation for '{file_state['filename']}'")
                try:
                    with st.spinner(f"Generating AI profile for {file_state['filename']}..."):
                        response = generate_profile(file_state["analysis"]["summary"]["analysis"])

                        file_state["ai_profile"]["response"] = response
                        file_state["ai_profile"]["status"] = "success"
                        state_changed = True

                        logger.info(
                            f"AI profile generation completed for '{file_state['filename']}'"
                        )

                except Exception as e:
                    err_msg = handle_g4f_error(e)
                    logger.exception(
                        f"AI profile generation failed for '{file_state['filename']}': {err_msg}"
                    )
                    file_state["ai_profile"]["status"] = "error"
                    file_state["ai_profile"]["error"] = err_msg
                    state_changed = True

        # Clear processing flag
        st.session_state.processing_pass = False

        logger.info(f"Processing pass completed. State changed: {state_changed}")

        # Rerun once after all processing if state changed
        if state_changed:
            st.rerun()

    # Handle merge request
    if st.session_state.get("merge_requested", False):
        st.session_state.merge_requested = False

        try:
            with st.spinner("Merging personality profiles..."):
                # Collect profiles to merge
                profiles_to_merge = []
                for file_state in st.session_state.files:
                    if file_state["local_profile"]["status"] == "success":
                        profiles_to_merge.append(
                            (
                                file_state["local_profile"]["results"],
                                file_state["local_profile"]["profile_text"],
                                file_state["filename"],
                            )
                        )

                if len(profiles_to_merge) >= 1:
                    merged_results, merged_text = merge_local_profiles(profiles_to_merge)

                    st.session_state.merged_profiles = {
                        "results": merged_results,
                        "profile_text": merged_text,
                        "merged_from": [f[2] for f in profiles_to_merge],
                    }

                    logger.info(f"Merged {len(profiles_to_merge)} profiles")
                    st.success(f"✅ Merged {len(profiles_to_merge)} profiles successfully!")
                else:
                    st.error("Need at least 1 profile to merge.")

        except Exception as e:
            logger.exception(f"Error merging profiles: {e}")
            st.error(f"Failed to merge profiles: {e}")
            if debug_mode:
                st.exception(e)

        st.rerun()

    # Display per-file cards
    for idx, file_state in enumerate(st.session_state.files):
        with st.container():
            st.markdown("---")

            # File header
            col_header, col_status = st.columns([3, 1])
            with col_header:
                st.markdown(f"### 📄 {file_state['filename']}")
                st.caption(
                    f"Size: {file_state['file_size_bytes']:,} bytes • Encoding: {file_state['decode_used']}"
                )

            with col_status:
                status = file_state["analysis_status"]
                if status == "queued":
                    st.info("⏳ Queued")
                elif status == "running":
                    st.warning("🔄 Running...")
                elif status == "success":
                    st.success("✅ Analyzed")
                elif status == "error":
                    st.error("❌ Error")

            # Analysis actions
            col_actions1, col_actions2, col_actions3 = st.columns(3)

            with col_actions1:
                if file_state["analysis_status"] not in ["running"]:
                    if st.button(
                        (
                            "Start Analysis"
                            if file_state["analysis_status"] != "success"
                            else "Re-analyze"
                        ),
                        key=f"analyze_{idx}",
                    ):
                        file_state["analysis_status"] = "queued"
                        st.rerun()

            # Show analysis results if successful
            if file_state["analysis_status"] == "success":
                summary = file_state["analysis"]["summary"]

                # Brief summary
                st.markdown("**Analysis Summary:**")
                col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
                with col_metrics1:
                    st.metric("Positive Topics", len(summary["positive_topics"]))
                with col_metrics2:
                    st.metric("Negative Topics", len(summary["negative_topics"]))
                with col_metrics3:
                    st.metric("Emotion Var.", f"{summary['emotion_variability']:.3f}")

                # Detailed view toggle
                with st.expander(f"📊 View Full Analysis for {file_state['filename']}"):
                    st.write(
                        f"**Positive Topics:** {', '.join(summary['positive_topics']) or 'None'}"
                    )
                    st.write(
                        f"**Negative Topics:** {', '.join(summary['negative_topics']) or 'None'}"
                    )
                    st.json(summary["analysis"])

                    # Download button for this file's analysis
                    debug_info = {
                        "filename": file_state["filename"],
                        "summary": summary,
                        "analysis_time": file_state["analysis_time"],
                    }
                    debug_json = json.dumps(debug_info, ensure_ascii=False, indent=2)
                    st.download_button(
                        label="Download Analysis JSON",
                        data=debug_json,
                        file_name=f"analysis_{file_state['filename']}.json",
                        mime="application/json",
                        key=f"download_analysis_{idx}",
                    )

                # Profile generation buttons
                st.markdown("**Profile Generation:**")
                col_profile1, col_profile2 = st.columns(2)

                with col_profile1:
                    local_status = file_state["local_profile"]["status"]
                    if local_status == "none":
                        if st.button("🧠 Generate Local Profile", key=f"local_{idx}"):
                            file_state["local_profile"]["status"] = "queued"
                            st.rerun()
                    elif local_status == "running":
                        st.info("⏳ Generating...")
                    elif local_status == "success":
                        st.success("✅ Local Profile Ready")
                        if st.button("📖 View Local Profile", key=f"view_local_{idx}"):
                            st.session_state[f"show_local_{idx}"] = True
                            st.rerun()
                    elif local_status == "error":
                        st.error("❌ Error")
                        if st.button("🔄 Retry", key=f"retry_local_{idx}"):
                            file_state["local_profile"]["status"] = "queued"
                            st.rerun()

                with col_profile2:
                    ai_status = file_state["ai_profile"]["status"]
                    if ai_status == "none":
                        if st.button("🤖 Generate AI Profile", key=f"ai_{idx}"):
                            file_state["ai_profile"]["status"] = "queued"
                            st.rerun()
                    elif ai_status == "running":
                        st.info("⏳ Generating...")
                    elif ai_status == "success":
                        st.success("✅ AI Profile Ready")
                        if st.button("📖 View AI Profile", key=f"view_ai_{idx}"):
                            st.session_state[f"show_ai_{idx}"] = True
                            st.rerun()
                    elif ai_status == "error":
                        st.error("❌ Error")
                        if st.button("🔄 Retry", key=f"retry_ai_{idx}"):
                            file_state["ai_profile"]["status"] = "queued"
                            st.rerun()

                # Display local profile if requested
                if st.session_state.get(f"show_local_{idx}", False):
                    with st.expander(f"🧠 Local Profile: {file_state['filename']}", expanded=True):
                        if file_state["local_profile"]["profile_text"]:
                            st.markdown(file_state["local_profile"]["profile_text"])

                            # Download buttons for exports
                            results = file_state["local_profile"]["results"]
                            if "exports" in results:
                                col_dl1, col_dl2, col_dl3 = st.columns(3)

                                with col_dl1:
                                    if "metrics_json" in results["exports"]:
                                        st.download_button(
                                            label="📥 JSON",
                                            data=results["exports"]["metrics_json"],
                                            file_name=f"local_profile_{file_state['filename']}.json",
                                            mime="application/json",
                                            key=f"dl_json_{idx}",
                                        )

                                with col_dl2:
                                    if "per_conversation_csv" in results["exports"]:
                                        st.download_button(
                                            label="📥 CSV",
                                            data=results["exports"]["per_conversation_csv"],
                                            file_name=f"conversations_{file_state['filename']}.csv",
                                            mime="text/csv",
                                            key=f"dl_csv_{idx}",
                                        )

                                with col_dl3:
                                    if "flagged_json" in results["exports"]:
                                        flagged_data = results["exports"]["flagged_json"]
                                        if flagged_data and flagged_data != "[]":
                                            st.download_button(
                                                label="📥 Flagged",
                                                data=flagged_data,
                                                file_name=f"flagged_{file_state['filename']}.json",
                                                mime="application/json",
                                                key=f"dl_flagged_{idx}",
                                            )

                        if st.button("❌ Close", key=f"close_local_{idx}"):
                            st.session_state[f"show_local_{idx}"] = False
                            st.rerun()

                # Display AI profile if requested
                if st.session_state.get(f"show_ai_{idx}", False):
                    with st.expander(f"🤖 AI Profile: {file_state['filename']}", expanded=True):
                        if file_state["ai_profile"]["response"]:
                            st.write(file_state["ai_profile"]["response"])

                        if st.button("❌ Close", key=f"close_ai_{idx}"):
                            st.session_state[f"show_ai_{idx}"] = False
                            st.rerun()

            elif file_state["analysis_status"] == "error":
                st.error(f"**Error:** {file_state.get('analysis_error', 'Unknown error')}")
                if debug_mode:
                    st.code(file_state.get("analysis_error", "No details"))

    # Display merged profile if available
    if st.session_state.merged_profiles:
        st.divider()
        st.subheader("🔀 Merged Personality Profile")

        merged = st.session_state.merged_profiles
        st.info(
            f"Merged from {len(merged['merged_from'])} file(s): {', '.join(merged['merged_from'])}"
        )

        with st.expander("📖 View Merged Profile", expanded=True):
            st.markdown(merged["profile_text"])

            # Download buttons for merged exports
            results = merged["results"]
            if "exports" in results:
                col_merge1, col_merge2, col_merge3 = st.columns(3)

                with col_merge1:
                    if "metrics_json" in results["exports"]:
                        st.download_button(
                            label="📥 Download Merged JSON",
                            data=results["exports"]["metrics_json"],
                            file_name="merged_personality_profile.json",
                            mime="application/json",
                            key="dl_merged_json",
                        )

                with col_merge2:
                    if "per_conversation_csv" in results["exports"]:
                        st.download_button(
                            label="📥 Download Merged CSV",
                            data=results["exports"]["per_conversation_csv"],
                            file_name="merged_conversations.csv",
                            mime="text/csv",
                            key="dl_merged_csv",
                        )

                with col_merge3:
                    if "flagged_json" in results["exports"]:
                        flagged_data = results["exports"]["flagged_json"]
                        if flagged_data and flagged_data != "[]":
                            st.download_button(
                                label="📥 Download Merged Flagged",
                                data=flagged_data,
                                file_name="merged_flagged.json",
                                mime="application/json",
                                key="dl_merged_flagged",
                            )

# Show logs in debug mode
if debug_mode:
    st.divider()
    st.subheader("Debug Logs")
    logs = get_logs()
    st.text_area("Logs", value=logs, height=300, key="debug_logs_area")

# Always offer log download
st.divider()
st.subheader("Debug Information")
st.caption("Technical logs for troubleshooting issues or understanding the analysis process")
logs = get_logs()
if logs.strip():
    st.download_button(
        label="Download Analysis Logs",
        data=logs,
        file_name="analysis_logs.txt",
        mime="text/plain",
        key="download_logs",
    )
    st.caption("Download detailed processing logs including timing information and any warnings")
else:
    st.info(
        "No logs available. Enable the 'Enable debug mode' checkbox above to capture detailed logs."
    )
