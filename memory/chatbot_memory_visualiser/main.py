"""
Entrypoint of the streamlit app
"""

from collections.abc import Callable
from typing import Final

import httpx
import streamlit as st
from loguru import logger


def landing_page():
    st.title("Landing Page")


def fetch_llm_names(base_url: str, api_key: str) -> list[str] | None:
    """
    Returns a list of model names from an OpenAI-compatible API
    """
    try:
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
    except Exception as error:
        logger.exception("Unexpected error")
        return None


def setup_page():
    st.title("Setup")
    if "llm_api_is_valid" not in st.session_state:
        st.session_state.llm_api_is_valid = False
        st.session_state.llm_api_base_url = ""
        st.session_state.llm_api_key = ""
        st.session_state.available_llm_names = []

    if "llm_params_saved" not in st.session_state:
        st.session_state.llm_params_saved = False
        st.session_state.llm_name = ""
        st.session_state.llm_temperature = 1.0

    with st.form(key="llm_api_setup_form"):
        llm_api_base_url = st.text_input(
            "Model API Base URL", value=st.session_state.llm_api_base_url
        )
        llm_api_key = st.text_input(
            "Model API Key", value=st.session_state.llm_api_key, type="password"
        )
        llm_api_submit_button = st.form_submit_button(label="Test Model API")
        if llm_api_submit_button:
            if not llm_api_base_url or not llm_api_key:
                st.error("Please provide Model API url and API key")
                return

            llm_names: list[str] | None = fetch_llm_names(
                base_url=llm_api_base_url,
                api_key=llm_api_key,
            )
            if llm_names is None:
                st.error("Provided Model API is Invalid")
                return
            st.session_state.llm_api_is_valid = True
            st.session_state.available_llm_names = llm_names
            st.success(f"Model API is Valid (found {len(llm_names)} models)")

        # with st.form(key="model_setup_form"):
        #     llm_temperature = st.number_input(
        #         "Model Temperature",
        #         value=st.session_state.llm_temperature,
        #         step=0.01,
        #     )
        # if llm_api_base_url and llm_api_key:
        #     models_list: list[str] | None = fetch_llm_names(
        #         base_url=llm_api_base_url,
        #         api_key=llm_api_key,
        #     )
        #     if models_list is None:
        #         st.error("Model API is invalid")
        #     else:
        #         st.json(models_list)
        #
        # llm_setup_submit_button = st.form_submit_button(label="Submit")
        # if llm_setup_submit_button:
        #     # if len(llm_name) == 0:
        #     #     st.error("Model name cannot be blank")
        #     if not (0.0 <= llm_temperature <= 2.0):
        #         st.error("Temperature must be in the range [0, 2]")
        #     elif len(llm_api_base_url) == 0:
        #         st.error("Model API Base URL cannot be blank")
        #     elif len(llm_api_key) == 0:
        #         st.error("Model API Key cannot be blank")
        #     else:
        #         # st.session_state.llm_name = llm_name
        #         st.session_state.llm_temperature = llm_temperature
        #         st.session_state.llm_settings_saved = True
        #         st.success("Model settings saved successfully")


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
