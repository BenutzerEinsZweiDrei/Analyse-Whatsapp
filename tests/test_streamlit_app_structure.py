"""
Test that streamlit_app.py has correct structure to avoid crashes.

This test verifies that st.rerun() is not called inside processing loops
which was causing crashes when bulk processing multiple files.
"""

import ast
import os
from pathlib import Path

# Constants for code analysis
CONTEXT_SEARCH_LINES = 30  # Number of lines to look back for loop context
CONTEXT_DISPLAY_CHARS = 200  # Number of characters to display in error messages

# Processing loop indicators that identify problematic rerun locations
PROCESSING_LOOP_INDICATORS = [
    "# Process queued analyses",
    "# Process queued local profiles",
    "# Process queued AI profiles",
    'if file_state["analysis_status"] == "queued"',
    'if file_state["local_profile"]["status"] == "queued"',
    'if file_state["ai_profile"]["status"] == "queued"',
]


class RerunVisitor(ast.NodeVisitor):
    """AST visitor to find st.rerun() calls and their context."""

    def __init__(self):
        self.rerun_calls = []
        self.in_loop = False
        self.loop_depth = 0
        self.in_function = None

    def visit_For(self, node):
        """Visit For loop node."""
        self.loop_depth += 1
        self.in_loop = True
        self.generic_visit(node)
        self.loop_depth -= 1
        if self.loop_depth == 0:
            self.in_loop = False

    def visit_While(self, node):
        """Visit While loop node."""
        self.loop_depth += 1
        self.in_loop = True
        self.generic_visit(node)
        self.loop_depth -= 1
        if self.loop_depth == 0:
            self.in_loop = False

    def visit_FunctionDef(self, node):
        """Visit function definition."""
        previous_function = self.in_function
        self.in_function = node.name
        self.generic_visit(node)
        self.in_function = previous_function

    def visit_Call(self, node):
        """Visit function call node."""
        # Check if this is a st.rerun() call
        if isinstance(node.func, ast.Attribute):
            if (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "st"
                and node.func.attr == "rerun"
            ):
                self.rerun_calls.append(
                    {
                        "line": node.lineno,
                        "in_loop": self.in_loop,
                        "loop_depth": self.loop_depth,
                        "function": self.in_function,
                    }
                )
        self.generic_visit(node)


def test_no_rerun_in_processing_loops():
    """
    Test that st.rerun() is not called inside the file processing loops.

    The processing loops are identified by comments like:
    - "Process queued analyses"
    - "Process queued local profiles"
    - "Process queued AI profiles"

    These loops iterate over st.session_state.files and should not contain
    st.rerun() calls, as that causes crashes.
    """
    # Read the streamlit_app.py file
    app_file = Path(__file__).parent.parent / "streamlit_app.py"
    with open(app_file, encoding="utf-8") as f:
        content = f.read()
        lines = content.split("\n")

    # Parse the AST
    tree = ast.parse(content)

    # Visit all nodes and collect st.rerun() calls
    visitor = RerunVisitor()
    visitor.visit(tree)

    # Check for problematic rerun calls in processing loops
    problematic_calls = []
    for call in visitor.rerun_calls:
        line_num = call["line"]
        # Get the line and surrounding context
        if line_num > 0 and line_num <= len(lines):
            # Look at the previous lines to find the loop context
            context_start = max(0, line_num - CONTEXT_SEARCH_LINES)
            context_lines = lines[context_start:line_num]
            context = "\n".join(context_lines)

            # Check if this rerun is in a processing loop
            is_in_processing_loop = any(
                indicator in context for indicator in PROCESSING_LOOP_INDICATORS
            )

            if is_in_processing_loop and call["in_loop"]:
                problematic_calls.append(
                    {
                        "line": line_num,
                        "context": context[-CONTEXT_DISPLAY_CHARS:],  # Last N chars of context
                    }
                )

    # Assert no problematic calls found
    assert (
        len(problematic_calls) == 0
    ), f"Found {len(problematic_calls)} st.rerun() calls inside processing loops: {problematic_calls}"


def test_state_changed_flag_exists():
    """
    Test that state_changed flag is used to batch rerun calls.
    """
    app_file = Path(__file__).parent.parent / "streamlit_app.py"
    with open(app_file, encoding="utf-8") as f:
        content = f.read()

    # Check for state_changed flag
    assert "state_changed = False" in content, "state_changed flag initialization not found"
    assert "state_changed = True" in content, "state_changed flag update not found"
    assert (
        "if state_changed:" in content
    ), "state_changed conditional check for rerun not found"


def test_bulk_action_buttons_exist():
    """
    Test that bulk action buttons are present in the app.
    """
    app_file = Path(__file__).parent.parent / "streamlit_app.py"
    with open(app_file, encoding="utf-8") as f:
        content = f.read()

    # Check for bulk action buttons
    assert "Analyze All Files" in content, "Analyze All Files button not found"
    assert (
        "Generate Local Profiles for All" in content
    ), "Generate Local Profiles for All button not found"
    assert "Generate AI Profiles for All" in content, "Generate AI Profiles for All button not found"
    assert (
        "Merge personality profiles" in content or "Merge Personality Profiles" in content
    ), "Merge personality profiles button not found"


def test_merge_cta_message_exists():
    """
    Test that merge CTA message is shown when 2+ profiles exist.
    """
    app_file = Path(__file__).parent.parent / "streamlit_app.py"
    with open(app_file, encoding="utf-8") as f:
        content = f.read()

    # Check for merge CTA
    assert (
        "Now we have" in content and "personality profiles" in content
    ), "Merge CTA message not found"
    assert (
        "files_with_local >= 2" in content
    ), "Merge CTA condition check not found"


if __name__ == "__main__":
    # Run tests with proper error handling
    tests = [
        ("no_rerun_in_processing_loops", test_no_rerun_in_processing_loops),
        ("state_changed_flag_exists", test_state_changed_flag_exists),
        ("bulk_action_buttons_exist", test_bulk_action_buttons_exist),
        ("merge_cta_message_exists", test_merge_cta_message_exists),
    ]

    failed = []
    for name, test_func in tests:
        try:
            test_func()
            print(f"✅ test_{name} passed")
        except AssertionError as e:
            print(f"❌ test_{name} failed: {e}")
            failed.append(name)
        except Exception as e:
            print(f"💥 test_{name} error: {e}")
            failed.append(name)

    if not failed:
        print("\n✅ All structure tests passed!")
    else:
        print(f"\n❌ {len(failed)} test(s) failed: {', '.join(failed)}")
        exit(1)
