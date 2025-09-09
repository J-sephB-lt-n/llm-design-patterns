"""
Functions which preprocess images to make them more easily consumable by a LLM.
"""

import base64
import io
from PIL import Image, ImageFilter


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
        img_b64: The image encoded as a base64 string.
        to_grayscale (bool): If `True`, will convert image to grayscale.
        increase_contrast (bool): If `True`, will sharpen the image.
        target_width: The width to which the image will be resized.
                      A value around 1200-1500 is often good for OCR.

    Notes:
        - to_grayscale=True and increase_contrast=True may improve LLM OCR.

    Returns:
        str: The image after processing, encoded as a base64 string.
    """
    image_data = base64.b64decode(img_b64)
    image = Image.open(io.BytesIO(image_data))

    # Handle transparency by placing on a white background
    if image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        # Create a new white background image in RGB mode
        background = Image.new("RGB", image.size, (255, 255, 255))
        # Paste the original image onto the white background
        # The alpha channel of the original image is used as the mask
        background.paste(image, (0, 0), image)
        image = background

    # Standardize to PNG for consistency after processing, as we may have
    # removed transparency.
    original_format = "PNG"

    if image.width > target_width:
        aspect_ratio = image.height / image.width
        new_height = int(target_width * aspect_ratio)
        image = image.resize(
            (target_width, new_height), resample=Image.Resampling.LANCZOS
        )

    if to_grayscale:
        image = image.convert("L")

    if increase_contrast:
        image = image.filter(ImageFilter.SHARPEN)

    buffered = io.BytesIO()
    image.save(buffered, format=original_format)
    processed_img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return processed_img_b64
