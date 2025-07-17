"""
Entrypoint of the streamlit app
"""

import functools
import inspect
from collections.abc import Callable
from typing import Final, get_args, get_origin, Literal

import httpx
import openai
import streamlit as st
from loguru import logger

from app.interfaces.memory_alg_protocol import ChatMessageDetail, MemoryAlg
from app.memory_algs import memory_algs


def landing_page():
    st.title("Landing Page")
    selected_memory_alg_name: str = st.selectbox(
        label="Select a memory algorithm to see a summary:",
        options=memory_algs.keys(),
    )
    st.markdown(
        "**Algorithm summary:**\n\n"
        + memory_algs[selected_memory_alg_name].alg_description
    )


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


def display_chat_history(chat_history: list[ChatMessageDetail]) -> None:
    """
    Render `chat_history` as a chatbot-style conversation in the streamlit app
    """
    for interaction in chat_history:
        with st.expander("Show internal chat messages", expanded=False):
            for msg in interaction.all_messages:
                with st.chat_message(msg.role):
                    st.markdown(msg.content)
        with st.expander("Show token usage", expanded=False):
            st.json(interaction.token_usage)
        for msg in interaction.visible_messages:
            with st.chat_message(msg.role):
                st.markdown(msg.content)
    st.divider()


def init_args_to_streamlit_controls(memory_alg: type[MemoryAlg]) -> dict:
    """
    Parse the keyword arguments out of the __init__() method of a memory algorithm class \
    and create streamlit widgets for the user to specify them
    """
    alg_kwargs: dict[str, Callable] = {}
    class_signature = inspect.signature(memory_alg.__init__)
    for param in class_signature.parameters.values():
        if param.name in ("self", "llm_client", "llm_name", "llm_temperature"):
            continue
        match param.annotation:
            case t if t is int:
                alg_kwargs[param.name] = functools.partial(
                    st.number_input, param.name, value=param.default
                )
            case t if t is str:
                alg_kwargs[param.name] = functools.partial(
                    st.text_input, param.name, value=param.default
                )
            case t if get_origin(t) is Literal:
                choices = list(get_args(t))
                alg_kwargs[param.name] = functools.partial(
                    st.selectbox,
                    param.name,
                    choices=choices,
                    index=choices.index(param.default),
                )
            case _:
                raise ValueError(
                    f"Cannot create streamlit widget for type {param.annotation} of __init__() arg {param.name}"
                )

    return alg_kwargs


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
        st.session_state.llm_name = ""
        st.session_state.llm_temperature = 1.0

    if "memory_alg_is_selected" not in st.session_state:
        st.session_state.memory_alg_is_selected = False
        st.session_state.memory_alg_name = ""
        st.session_state.memory_alg_kwargs = None

    if "memory_alg_setup_is_completed" not in st.session_state:
        st.session_state.memory_alg_setup_is_completed = False
        st.session_state.memory_alg = None

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
            st.session_state.llm_name = ""  # new model API
            st.session_state.llm_temperature = 1.0  # new model API
            st.session_state.llm_client = openai.OpenAI(
                base_url=f"{st.session_state.llm_api_base_url}/v1",
                api_key=st.session_state.llm_api_key,
            )
            st.success(f"Model API is Valid (found {len(llm_names)} models)")

    if not st.session_state.llm_api_is_valid:
        return

    with st.form(key="model_setup_form"):
        llm_name = st.selectbox(
            "Model Name",
            st.session_state.available_llm_names,
            index=(
                st.session_state.available_llm_names.index(st.session_state.llm_name)
                if st.session_state.llm_name
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
            if not llm_name:
                st.error("Please select a model name")
            elif not (0.0 <= llm_temperature <= 2.0):
                st.error("Temperature must be in the range [0, 2]")
            else:
                st.session_state.llm_name = llm_name
                st.session_state.llm_temperature = llm_temperature
                st.session_state.llm_params_saved = True
                st.success("Model parameters saved successfully")

    if not st.session_state.llm_params_saved:
        return

    with st.form(key="memory_alg_select_form"):
        memory_alg_name = st.selectbox(
            "Memory Algorithm",
            memory_algs,
            index=(
                list(memory_algs).index(st.session_state.memory_alg_name)
                if st.session_state.memory_alg_name
                else None
            ),
        )
        submit_memory_alg_select = st.form_submit_button(
            label="Submit Selected Memory Algorithm"
        )
        if submit_memory_alg_select:
            if memory_alg_name is None:
                st.error("Please select a memory algorithm")
            else:
                st.session_state.memory_alg_is_selected = True
                st.session_state.memory_alg_name = memory_alg_name
                st.success(
                    f"Selected memory algorithm [{st.session_state.memory_alg_name}]"
                )
    if not st.session_state.memory_alg_is_selected:
        return

    with st.form(key="memory_alg_setup_form"):
        alg_setup_controls = init_args_to_streamlit_controls(
            memory_algs[st.session_state.memory_alg_name]
        )
        logger.error(alg_setup_controls)
        alg_setup = {}
        if st.session_state.memory_alg_kwargs:
            st.text("Current args:")
            st.json(st.session_state.memory_alg_kwargs)
        for param, input_control in alg_setup_controls.items():
            alg_setup[param] = input_control()
        logger.error(alg_setup)
        submit_memory_alg_params = st.form_submit_button(
            label="Submit Memory Alg Parameters",
        )
        if submit_memory_alg_params:
            for arg_name, arg_value in alg_setup.items():
                if not arg_value:
                    st.error(f"Please provide value for parameter '{arg_name}'")
                    return
            st.session_state.memory_alg_setup_is_completed = True
            st.session_state.memory_alg_kwargs = alg_setup
            st.session_state.memory_alg = memory_algs[memory_alg_name](
                llm_client=st.session_state.llm_client,
                llm_name=st.session_state.llm_name,
                llm_temperature=st.session_state.llm_temperature,
                **alg_setup,
            )
            st.success(
                f"Completed setup of memory algorithm [{st.session_state.memory_alg_name}]"
            )


def chat_page():
    st.title("Chat")
    if not all(
        st.session_state.get(x)
        for x in (
            "llm_api_is_valid",
            "llm_params_saved",
            "memory_alg_is_selected",
            "memory_alg_setup_is_completed",
        )
    ):
        st.error("Please complete setup")
        return

    display_chat_history(st.session_state.memory_alg.chat_history)
    if user_input := st.chat_input("Enter your response"):
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.spinner():
            st.session_state.memory_alg.chat(user_msg=user_input)
        st.rerun()


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
