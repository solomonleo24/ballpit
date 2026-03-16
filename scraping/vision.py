from openai import OpenAI
import base64

import os

client = OpenAI(
    api_key=os.getenv('API_KEY'),
    base_url=os.getenv('BASE_URL'),
)

# Vision function call to vision model to extract and summarize images/plots
def vision_fn(image_bytes: bytes, context: str) -> str:
    response = client.chat.completions.create(
        model=os.getenv('VISION_MODEL'),
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64.standard_b64encode(image_bytes).decode('utf-8')}",
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "You are analyzing a figure from a research paper.\n\n"
                            f"{context}\n\n"
                            "Describe what this figure shows, what the key finding is, "
                            "and how it relates to the surrounding text."
                        ),
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content