"""
Functions for converting a document (e.g. PDF, docx) into markdown text using https://github.com/docling-project/docling
"""

from io import BytesIO
from pathlib import Path
from typing import Final, Literal

from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorOptions,
    EasyOcrOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.utils.model_downloader import download_models


def doc_to_text(
    doc_file_extension: Literal[".docx", ".pdf"],
    doc_file_content: bytes,
    verbose: bool = True,
) -> str:
    """
    Convert contents of document (e.g. PDF, docx) into markdown string using docling

    Note:
        - Can also export to JSON, HTML and others (read the docling docs)
    """
    DOCLING_MODELS_PATH: Final[Path] = (
        Path.home() / ".cache/docling/models"
    )  # the package default
    if not DOCLING_MODELS_PATH.exists():
        if verbose:
            print(f"Downloading docling models to {DOCLING_MODELS_PATH}")
        download_models(DOCLING_MODELS_PATH)

    file_content_buf = BytesIO(doc_file_content)
    doc_stream = DocumentStream(
        name=f"document{doc_file_extension}",
        stream=file_content_buf,
    )
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=PdfPipelineOptions(
                    artifacts_path=DOCLING_MODELS_PATH,
                    accelerator_options=AcceleratorOptions(
                        num_threads=4, device="auto"
                    ),
                    do_table_structure=True,
                    do_ocr=True,
                    generate_picture_images=False,
                    ocr_options=EasyOcrOptions(
                        lang=["fr", "de", "es", "en"],
                        download_enabled=False,
                    ),
                )
            )
        }
    )
    doc = doc_converter.convert(
        source=doc_stream,
        raises_on_error=True,
        max_num_pages=9_999,
        # page_range=
    )

    doc_markdown: str = doc.document.export_to_markdown()

    return doc_markdown
