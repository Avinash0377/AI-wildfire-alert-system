import os
import streamlit as st


def cleanup_temp_file():
    """Delete the temporary uploaded video file if it exists.

    Checks session state for a stored temp file path, removes the file,
    and clears the path from session state.
    """
    if 'temp_file_path' in st.session_state and st.session_state.temp_file_path:
        try:
            os.unlink(st.session_state.temp_file_path)
        except Exception:
            pass
        st.session_state.temp_file_path = None
