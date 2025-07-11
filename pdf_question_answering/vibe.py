"""
PDF Question Answering Streamlit App
Converts PDF pages to images and uses multimodal LLM to find answers to user questions.
"""

import base64
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import dotenv
import fitz  # PyMuPDF
import openai
import streamlit as st
from PIL import Image


def load_environment():
    """Load environment variables from .env file."""
    dotenv.load_dotenv(".env")


def initialize_llm_client():
    """Initialize the OpenAI client with environment variables."""
    return openai.OpenAI(
        base_url=os.environ["OPENAI_API_BASE_URL"],
        api_key=os.environ["OPENAI_API_KEY"],
    )


def pdf_to_images(pdf_path: str) -> list[Image.Image]:
    """
    Convert PDF pages to PIL Images using PyMuPDF.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of PIL Images, one per page
    """
    images = []
    pdf_document = fitz.open(pdf_path)

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        # Convert page to image (PNG format)
        # Use a much higher DPI (600) for better text recognition
        pix = page.get_pixmap()

        # Convert to PIL Image
        img_data = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_data))

        images.append(image)

    pdf_document.close()
    return images


def image_to_base64(image: Image.Image) -> str:
    """
    Convert PIL Image to base64 string.

    Args:
        image: PIL Image to convert

    Returns:
        Base64 encoded string
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_data = buffer.getvalue()
    return base64.b64encode(img_data).decode("ascii")


def ask_llm_about_page(
    llm_client, image: Image.Image, question: str
) -> tuple[bool, str]:
    """
    Ask the LLM if a question is answered on a specific page image.

    Args:
        llm_client: OpenAI client instance
        image: PIL Image of the page
        question: User's question

    Returns:
        Tuple of (answer_found: bool, response: str)
    """
    image_b64 = image_to_base64(image)

    prompt = f"""
    Look at this page image and determine if the following question is answered on this page:
    
    Question: {question}
    
    Please respond with a JSON markdown code block in this exact format:
    ```json
    {{
        "answer_found": true/false,
        "response": "Your detailed response here explaining what you found, or `null` if no answer found"
    }}
    ```
    
    Only return the JSON markdown code block, nothing else.
    """

    llm_response = llm_client.chat.completions.create(
        model=os.environ.get("DEFAULT_MODEL_NAME", "gpt-4o"),
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "content": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            },
        ],
    )

    response_text = llm_response.choices[0].message.content

    # Extract JSON from markdown code block
    try:
        # Find the JSON part between ```json and ```
        json_text = response_text.replace("```json", "").replace("```", "").strip()
        result = json.loads(json_text)
        return result.get("answer_found", False), result.get("response", None)
    except (IndexError, json.JSONDecodeError, KeyError) as e:
        st.error(f"Error parsing LLM response: {e}")
        st.code(response_text)
        return False, f"Error parsing response: {str(e)}"


def main():
    """Main Streamlit application."""
    st.title("PDF Question Answering")
    st.write("Upload a PDF and ask a question about its contents.")

    # Load environment variables
    load_environment()

    # File upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    # Question input
    question = st.text_input("Enter your question:")

    if uploaded_file and question:
        # Save uploaded file to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name

        # Convert PDF to images
        with st.spinner("Converting PDF to images..."):
            images = pdf_to_images(pdf_path)
            st.success(f"PDF converted to {len(images)} page images")

        # Initialize LLM client
        llm_client = initialize_llm_client()

        # Process each page
        answer_found = False
        answer_page = -1
        answer_response = ""

        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Create a container for page results
        results_container = st.container()

        for i, image in enumerate(images):
            # if answer_found:
            #     break

            page_num = i + 1
            progress_percent = (i + 1) / len(images)
            progress_bar.progress(progress_percent)
            status_text.text(f"Processing page {page_num}/{len(images)}...")

            with results_container:
                st.subheader(f"Page {page_num}")
                st.image(image, width=400, caption=f"Page {page_num}")

                with st.spinner(f"Analyzing page {page_num}..."):
                    found, response = ask_llm_about_page(llm_client, image, question)

                    if found:
                        st.success("✅ Answer found on this page!")
                        answer_found = True
                        answer_page = page_num
                        answer_response = response
                    else:
                        st.info("❌ No answer on this page")

                    st.write(response)
                    st.divider()

        # Final result
        if answer_found:
            st.success(f"✅ Answer found on page {answer_page}!")
            st.write(f"**Response:** {answer_response}")
        else:
            st.warning("❌ No answer found in the entire document.")

        # Clean up temp file
        os.unlink(pdf_path)


if __name__ == "__main__":
    main()
