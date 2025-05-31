import base64
import os
from pathlib import Path

import dotenv
import openai

if __name__ == "__main__":
    dotenv.load_dotenv(".env")

    llm_client = openai.OpenAI(
        base_url=os.environ["OPENAI_API_BASE_URL"],
        api_key=os.environ["OPENAI_API_KEY"],
    )

    IMAGE_INPUT_PATH: Path = Path("./static/example_image.png")
    with open(IMAGE_INPUT_PATH, "rb") as file:
        image_contents: bytes = file.read()

    image_contents_b64str: str = base64.b64encode(image_contents).decode("ascii")
    image_file_extension: str = IMAGE_INPUT_PATH.stem

    llm_response = llm_client.chat.completions.create(
        model=os.environ.get("DEFAULT_MODEL_NAME", "gpt-4o"),
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "content": "Describe the contents of this image.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image_file_extension};base64,{image_contents_b64str}"
                        },
                    },
                ],
            },
        ],
    )

    print(llm_response.choices[0].message.content)
