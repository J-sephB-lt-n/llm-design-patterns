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

    chunker: type[chonkie.BaseChunker] | Callable
    explanation: str
    streamlit_input_controls: dict[str, Callable]
    hardcoded_chunker_kwargs: dict[str, Any]


CHUNKERS: Final[dict[str, Optional[ChunkerDef]]] = {
    "Token Chunker": ChunkerDef(
        chunker=chonkie.TokenChunker,
        explanation="""
Puts a fixed number of tokens in each chunk.

| Arg           | Description
|---------------|------------
| chunk_size    | Maximum number of tokens per chunk
| chunk_overlap | Number of tokens shared by consecutive chunks
""".strip(),
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
    "Sentence Chunker": ChunkerDef(
        chunker=chonkie.SentenceChunker,
        explanation="""
Puts a fixed number of tokens in each chunk, ensuring complete sentences (won't split in the middle of a sentence)

| Arg                     | Description
|-------------------------|------------
| chunk_size              | Maximum number of tokens per chunk
| chunk_overlap           | Number of tokens shared by consecutive chunks
| min_sentences_per_chunk | Minimum number of sentences allowed per chunk
""".strip(),
        streamlit_input_controls={
            "chunk_size": partial(
                st.number_input, label="Chunk Size (n tokens)", min_value=1, value=512
            ),
            "chunk_overlap": partial(
                st.number_input, label="Chunk Overlap (n tokens)", min_value=0, value=0
            ),
            "min_sentences_per_chunk": partial(
                st.number_input,
                label="Minimum n Sentences in Chunk",
                min_value=1,
                value=1,
            ),
        },
        hardcoded_chunker_kwargs={
            "tokenizer_or_token_counter": tiktoken.get_encoding("gpt2")
        },
    ),
    "Recursive Chunker": ChunkerDef(
        chunker=chonkie.RecursiveChunker.from_recipe,
        explanation="""
Continues to split chunks until they contain less than `chunk_size` tokens.

It works sequentially through a fixed list of delimiters, which by default is:

1. Chunk by paragraph ('\\n\\n')

2. Chunk by newline ('\\n')

3. Chunk by sentence ('. ', '! ', '? ')

4. Chunk by whitespace (' ')

5. Chunk by characters

| Arg                         | Description
|-----------------------------|------------------------
| chunk_size                  | Minimum number of tokens in a chunk
| min_characters_per_chunk    | Minimum number of characters in a chunk
| recipe                      | A predefined set of rules (see https://huggingface.co/datasets/chonkie-ai/recipes)
""".strip(),
        streamlit_input_controls={
            "chunk_size": partial(
                st.number_input,
                label="Maximum number of tokens in a chunk",
                min_value=1,
                value=512,
            ),
            "min_characters_per_chunk": partial(
                st.number_input,
                label="Minimum number of characters in a chunk",
                min_value=1,
                value=1,
            ),
            "name": partial(
                st.selectbox,
                label="Chosen recipe (rule set)",
                options=["default", "markdown"],
            ),
        },
        hardcoded_chunker_kwargs={
            "tokenizer_or_token_counter": tiktoken.get_encoding("gpt2"),
            "lang": "en",
        },
    ),
    "Code Chunker": None,
    "Semantic Chunker": ChunkerDef(
        chunker=chonkie.SemanticChunker,
        explanation="""
Splits text into chunks based on semantic similarity (using dense vectors).
(i.e. semantically similar text will go into the same chunk)

| Arg                         | Description
|-----------------------------|------------
| mode                        | Method of grouping sentences. One of ["cumulative", "window"]
| threshold                   | Similarity threshold used to decide whether sentences are "similar"
| chunk_size                  | Maximum number of tokens in a chunk
| similarity_window           | Number of sentences to consider for similarity threshold calculation
| min_sentences               | Minimum number of sentences in a chunk
| min_chunk_size              | Minimum number of tokens in a chunk
| min_characters_per_sentence | Minimum number of characters in a sentence
| threshold_step              | Step size for similarity threshold calculation
""".strip(),
        streamlit_input_controls={
            "mode": partial(
                st.selectbox,
                label="Mode (method of grouping sentences)",
                options=["cumulative", "window"],
            ),
            "threshold": partial(
                st.number_input,
                label="Threshold (sentence similarity threshold)",
                min_value=0,
                max_value=1,
                value=0.5,
            ),
            # ...
        },
        hardcoded_chunker_kwargs={
            "embedding_model": "minishlab/potion-base-8M",
        },
    ),
    "SDPM Chunker": None,
    "Late Chunker": None,
    "Neural Chunker": None,
    "Slumber Chunker": None,
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
    chunker_def: Optional[ChunkerDef] = CHUNKERS[selected_chunker_name]
    if chunker_def is None:
        st.write(
            f"I have not implemented the '{selected_chunker_name}' chunker in this app yet"
        )
    else:
        st.markdown(chunker_def.explanation)
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
            for chunk_num, chunk in enumerate(chunks, start=1):
                st.text_area(
                    label=f"Chunk {chunk_num:,}",
                    value=chunk.text,
                    height=max(68, 35 * (chunk.text.count("\n") + 1)),
                )


PAGES: Final[dict[str, Callable]] = {
    "Upload Doc": upload_doc_page,
    "View Uploaded Doc": view_doc_page,
    "Chunk Doc": chunk_doc_page,
}

if __name__ == "__main__":
    st.set_page_config(
        page_title="Chonkie Visualiser",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    page = PAGES[selection]
    page()
