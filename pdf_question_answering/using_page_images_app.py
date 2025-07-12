"""
Streamlit app entrypoint

App for PDF question-answering using LLMs (using PDF page images)
"""

from collections.abc import Callable
from typing import Final

import httpx
import openai
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
    except Exception:
        logger.exception("Unexpected error")
        return None


def setup_page():
    st.title("Setup")
    if "llm_api_is_valid" not in st.session_state:
        st.session_state.llm_api_is_valid = False
        st.session_state.llm_api_base_url = ""
        st.session_state.llm_api_key = ""
        st.session_state.llm_client = None
        st.session_state.available_llm_names = []

    if "llm_params_saved" not in st.session_state:
        st.session_state.llm_params_saved = False
        st.session_state.selected_llm_names = None
        st.session_state.llm_temperature = 1.0

    with st.form(key="llm_api_setup_form"):
        llm_api_base_url = st.text_input(
            "Model API Base URL", value=st.session_state.llm_api_base_url
        )
        llm_api_key = st.text_input(
            "Model API Key", value=st.session_state.llm_api_key, type="password"
        )
        llm_api_submit_button = st.form_submit_button(
            label="Submit New Model API Details"
        )
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
            st.session_state.llm_api_base_url = llm_api_base_url
            st.session_state.llm_api_key = llm_api_key
            st.session_state.available_llm_names = llm_names
            st.session_state.llm_params_saved = False  # new model API
            st.session_state.selected_llm_names = None  # new model API
            st.session_state.llm_temperature = 1.0  # new model API
            st.session_state.llm_client = openai.OpenAI(
                base_url=st.session_state.llm_api_base_url,
                api_key=st.session_state.llm_api_key,
            )
            st.success(f"Model API is Valid (found {len(llm_names)} models)")

    if not st.session_state.llm_api_is_valid:
        return

    with st.form(key="model_setup_form"):
        selected_llms = st.multiselect(
            "Model Name",
            st.session_state.available_llm_names,
            default=(
                st.session_state.selected_llm_names
                if st.session_state.selected_llm_names
                else None
            ),
        )
        llm_temperature = st.number_input(
            "Model Temperature",
            value=st.session_state.llm_temperature,
            step=0.01,
        )
        llm_params_submit_button = st.form_submit_button(
            label="Submit New Model Parameters"
        )
        if llm_params_submit_button:
            if not selected_llms:
                st.error("Please select at least one model")
            elif not (0.0 <= llm_temperature <= 2.0):
                st.error("Temperature must be in the range [0, 2]")
            else:
                st.session_state.selected_llm_names = selected_llms
                st.session_state.llm_temperature = llm_temperature
                st.session_state.llm_params_saved = True
                st.success("Model setup completed")


def question_answering_page():
    st.title("Question-Answering")
    if not st.session_state.llm_params_saved:
        st.error("Please complete model setup")
        return
    uploaded_file = st.file_uploader(label="Upload a PDF", type=["pdf"])
    stop_condition = st.radio(
        label="Stopping condition: ",
        options=[
            "Any model finds answer on page",
            "All models find answer on page",
            "Process whole PDF",
        ],
        horizontal=True,
    )
    user_query = st.text_input(label="Enter your question(s)")
    start_button = st.button(label="Start")

    if not start_button:
        return

    if not uploaded_file:
        st.error("Please upload a PDF")
        return
    if not user_query:
        st.error("Please provide at least 1 question")
        return

    st.success("starting")


def main():
    PAGES: Final[dict[str, Callable]] = {
        "Landing": landing_page,
        "Setup": setup_page,
        "Question-Answering": question_answering_page,
    }

    st.sidebar.title("PDF Question-Answering")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    PAGES[selection]()


if __name__ == "__main__":
    main()
