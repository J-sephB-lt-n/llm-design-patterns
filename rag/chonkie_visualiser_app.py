"""
TODO: module docstring
"""

import sys
from functools import partial
from pathlib import Path
from typing import Any, Callable, Final, Optional

import chonkie
import streamlit as st
import tiktoken
from pydantic import BaseModel, ConfigDict

# I absolutely hate this hack, but I want streamlit to run
#   from the project root folder and I can't find the equivalent of
#   `python -m` for `streamlit run`
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent),
)

from pdf_to_text.docling_pdf_to_text import doc_to_text


class ChunkerDef(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    chunker: type[chonkie.BaseChunker]
    explanation: str
    streamlit_input_controls: dict[str, Callable]
    hardcoded_chunker_kwargs: dict[str, Any]


CHUNKERS: Final[dict[str, Optional[Callable]]] = {
    "Token Chunker": ChunkerDef(
        chunker=chonkie.TokenChunker,
        explanation="Puts a fixed number of tokens in each chunk.",
        streamlit_input_controls={
            "chunk_size": partial(
                st.number_input, label="Chunk Size (n tokens)", min_value=1, value=512
            ),
            "chunk_overlap": partial(
                st.number_input, label="Chunk Overlap (n tokens)", min_value=0, value=0
            ),
        },
        hardcoded_chunker_kwargs={"tokenizer": tiktoken.get_encoding("gpt2")},
    ),
    # "Sentence Chunker": None,
    # "Recursive Chunker": None,
    # "Code Chunker": None,
    # "Semantic Chunker": None,
    # "SDPM Chunker": None,
    # "Late Chunker": None,
    # "Neural Chunker": None,
    # "Slumber Chunker": None,
}


def upload_doc_page():
    st.title("Upload Doc")
    uploaded_file = st.file_uploader(
        label="Upload a document (.pdf or .txt)", type=["pdf", "txt"]
    )
    if uploaded_file:
        file_extension: str = Path(uploaded_file.name).suffix
        if file_extension == ".txt":
            try:
                st.session_state["doc_text_content"] = uploaded_file.read().decode(
                    "utf-8"
                )
            except UnicodeDecodeError:
                st.error("Failed to read file using encoding utf-8")
        elif file_extension == ".pdf":
            with st.spinner("Extracting text from PDF using docling"):
                st.session_state["doc_text_content"] = doc_to_text(
                    doc_file_extension=file_extension,
                    doc_file_content=uploaded_file.getvalue(),
                )
        else:
            raise RuntimeError("Reached a path which should be unreachable")

        st.success(f"Successfully ingested document {uploaded_file.name}")


def view_doc_page():
    st.title("View Uploaded Document")
    if "doc_text_content" not in st.session_state:
        st.text("No document has been uploaded yet")
    else:
        st.text_area(
            label="Document Content",
            value=st.session_state["doc_text_content"],
            height=999,
        )


def chunk_doc_page():
    st.title("Chunk Document")
    selected_chunker_name: str = st.selectbox(
        label="Select a Chonkie text chunker",
        options=CHUNKERS.keys(),
    )
    # if selected_chunker_name:
    chunker_def: ChunkerDef = CHUNKERS[selected_chunker_name]
    st.text(chunker_def.explanation)
    user_chunker_kwargs: dict = {}
    for argname, streamlit_control in chunker_def.streamlit_input_controls.items():
        user_chunker_kwargs[argname] = streamlit_control()

    submit_button = st.button(label="Chunk Document with Selected Chunker")

    if submit_button:
        all_chunker_kwargs: dict = (
            user_chunker_kwargs | chunker_def.hardcoded_chunker_kwargs
        )
        st.write("Final chunker params:")
        st.json(all_chunker_kwargs)
        chunker: chonkie.BaseChunker = chunker_def.chunker(**all_chunker_kwargs)
        chunks = chunker.chunk(st.session_state["doc_text_content"])
        for chunk in chunks:
            st.code(chunk.text, language=None)


PAGES: Final[dict[str, Callable]] = {
    "Upload Doc": upload_doc_page,
    "View Uploaded Doc": view_doc_page,
    "Chunk Doc": chunk_doc_page,
}

if __name__ == "__main__":
    st.set_page_config(
        page_title="Chonkie Visualiser", layout="wide", initial_sidebar_state="expanded"
    )

    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    page = PAGES[selection]
    page()
