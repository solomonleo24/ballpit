from groq import Groq
import base64

import os
from dotenv import load_dotenv
load_dotenv()

client = Groq(api_key = os.getenv('API_KEY'))  # reads GROQ_API_KEY from environment

# Vision function call to vision model to extract and summarize images/plots
def vision_fn(image_bytes: bytes, context: str) -> str:
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
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