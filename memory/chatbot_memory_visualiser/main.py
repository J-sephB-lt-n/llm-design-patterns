"""
Entrypoint of the streamlit app
"""

from collections.abc import Callable
from typing import Final

import httpx
import streamlit as st


def landing_page():
    st.title("Landing Page")


def setup_page():
    st.title("Setup")

    def fetch_llm_names(base_url: str, api_key: str) -> list[str] | None:
        """
        Returns a list of model names, assuming that `base_url` is an OpenAI compatible API
        """
        resp = httpx.get(
            url=f"{base_url}/v1/models",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            },
        )
        if resp.status_code != 200:
            return None

        return sorted([x["id"] for x in resp.json()["data"]])

    if "llm_settings_saved" not in st.session_state:
        st.session_state.llm_settings_saved = False
        st.session_state.llm_name = ""
        st.session_state.llm_temperature = 1.0
        st.session_state.llm_api_base_url = ""
        st.session_state.llm_api_key = ""
    with st.form(key="model_setup_form"):
        llm_temperature = st.number_input(
            "Model Temperature",
            value=st.session_state.llm_temperature,
            step=0.01,
        )
        llm_api_base_url = st.text_input(
            "Model API Base URL", value=st.session_state.llm_api_base_url
        )
        llm_api_key = st.text_input(
            "Model API Key", value=st.session_state.llm_api_key, type="password"
        )

        llm_setup_submit_button = st.form_submit_button(label="Submit")
        if llm_setup_submit_button:
            if len(llm_name) == 0:
                st.error("Model name cannot be blank")
            elif not (0.0 <= llm_temperature <= 2.0):
                st.error("Temperature must be in the range [0, 2]")
            elif len(llm_api_base_url) == 0:
                st.error("Model API Base URL cannot be blank")
            elif len(llm_api_key) == 0:
                st.error("Model API Key cannot be blank")
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
