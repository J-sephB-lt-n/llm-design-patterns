"""
Streamlit app entrypoint

App for PDF question-answering using LLMs (using PDF page images)
"""

import base64
import time
from collections.abc import Callable
from typing import Final

import httpx
from pydantic import BaseModel, Field
import pymupdf
import openai
import streamlit as st
from loguru import logger

from structured_outputs.pydantic_schema_dump_retry import (
    structured_output_chat_completion,
)


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
        st.session_state.llm_temperature = 0.0

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


class LlmPageResponse(BaseModel):
    answer_is_on_this_page: bool = Field(
        ...,
        description="`true` if answer to user's query appears on this page.",
    )
    answer: str | None = Field(
        description="The answer to the user's question (if it appears on this page), else null.",
        default=None,
    )


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

    progress_bar = st.progress(0.0)
    progress_text = st.empty()
    with pymupdf.open(stream=uploaded_file, filetype="pdf") as doc:
        for page_num, page in enumerate(doc, start=1):
            progress_bar.progress(page_num / doc.page_count)
            progress_text.text(f"Processing page {page_num}/{doc.page_count}")
            page_text: str = page.get_text(sort=True)
            page_pixmap: pymupdf.Pixmap = page.get_pixmap(
                dpi=72, colorspace=pymupdf.csRGB
            )
            page_image_bytes: bytes = page_pixmap.tobytes("png")
            page_image_b64: bytes = base64.b64encode(page_image_bytes)
            page_image_b64_str: str = page_image_b64.decode("ascii")
            answer_is_on_this_page_count: int = 0
            for llm_name in st.session_state.selected_llm_names:
                with st.spinner(f"Running [{llm_name}]"):
                    llm_response = structured_output_chat_completion(
                        response_model=LlmPageResponse,
                        max_n_retries=2,
                        llm_client=st.session_state.llm_client,
                        chat_kwargs={
                            "model": llm_name,
                            "temperature": st.session_state.llm_temperature,
                        },
                        logger=logger,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{page_image_b64_str}"
                                        },
                                    },
                                    {
                                        "type": "text",
                                        "text": f"""
<user-query>
{user_query}
</user-query>

<page-text>
{page_text}
</page-text>

You have been provided with a page image (and the roughly extracted page text) of a page from a PDF.
Your task is to identify if the answer to the user's query can be found on this page, \
and to return the answer if it does.
                                            """.strip(),
                                    },
                                ],
                            },
                        ],
                    )
                if llm_response.answer_is_on_this_page:
                    st.success(f"[{llm_name}] [page {page_num}] {llm_response.answer}")
                    answer_is_on_this_page_count += 1
            if (
                stop_condition == "Any model finds answer on page"
                and answer_is_on_this_page_count > 0
            ):
                return
            elif (
                stop_condition == "All models find answer on page"
                and answer_is_on_this_page_count == len(st.selected_llm_names)
            ):
                return


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
