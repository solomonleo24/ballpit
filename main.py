from dotenv import load_dotenv
load_dotenv()

import json
from dataclasses import asdict
from scraping.paper import Paper
from scraping.vision import vision_fn

from docling_core.types.doc.labels import DocItemLabel

PDF_PATH = "Gate-All-Around_Transistors_at_3nm_Device_Physics_.pdf"
JSON_PATH = "chunks.json"

paper = Paper(path=PDF_PATH)
paper.scrape()
paper.summarize_figures(vision_fn=vision_fn, context_window=2)
chunks = paper.chunk()

payload = {
    "metadata": paper.metadata,
    "chunks": [asdict(c) for c in chunks],
}

with open(JSON_PATH, "w") as f:
    json.dump(payload, f, indent=2)

print(f"Saved {len(chunks)} chunks to {JSON_PATH}")