"""
Entrypoint of the streamlit app
"""

from collections.abc import Callable
from typing import Final

import streamlit as st


def landing_page():
    st.title("Landing Page")


def setup_page():
    st.title("Setup")
    if "llm_settings_saved" not in st.session_state:
        st.session_state.llm_settings_saved = False
        st.session_state.llm_name = ""
        st.session_state.llm_temperature = 1.0
        st.session_state.llm_api_base_url = ""
        st.session_state.llm_api_key = ""
    with st.form(key="model_setup_form"):
        llm_name = st.text_input("Model Name", value=st.session_state.llm_name)
        llm_temperature = st.number_input(
            "Model Temperature",
            value=st.session_state.llm_temperature,
            step=0.01,
        )

        llm_setup_submit_button = st.form_submit_button(label="Submit")
        if llm_setup_submit_button:
            if len(llm_name) == 0:
                st.error("Model name is blank")
            elif not (0.0 <= llm_temperature <= 2.0):
                st.error("Temperature must be in the range [0, 2]")
            else:
                st.session_state.llm_name = llm_name
                st.session_state.llm_temperature = llm_temperature
                st.session_state.llm_settings_saved = True
                st.success("Model settings saved successfully")


def chat_page():
    st.title("Chat")
    if not st.session_state.get("llm_settings_saved"):
        st.error("Please run model setup first")
        return


def main():
    PAGES: Final[dict[str, Callable]] = {
        "Landing": landing_page,
        "Setup": setup_page,
        "Chat": chat_page,
    }

    st.sidebar.title("Agent Memory Visualiser")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    PAGES[selection]()


if __name__ == "__main__":
    main()
