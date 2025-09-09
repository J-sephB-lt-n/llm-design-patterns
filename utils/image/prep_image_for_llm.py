"""
Functions which preprocess images to make them more easily consumable by a LLM.
"""

import base64
import io
from PIL import Image, ImageEnhance, ImageFilter


def prep_image_for_llm(
    img_b64: str,
    to_grayscale: bool,
    increase_contrast: bool,
    target_width: int = 1_200,
) -> str:
    """
    Resize and process base64-encoded image `img_b64` to make it better suited for \
inclusion in a (multimodal) Large Language Model prompt.

    Args:
        base64_string: The image encoded as a base64 string.
        to_grayscale (bool): If `True`, will convert image to grayscale.
        increase_contrast (bool): If `True`, will sharpen the image.
        target_width: The width to which the image will be resized. 
                      A value around 1200-1500 is often good for OCR.

    Notes:
        - to_grayscale=True and increase_contrast=True may improve LLM OCR.

    Returns:
        str: The image after processing, encoded as a base64 string.
    """
    raise NotImplementedError
