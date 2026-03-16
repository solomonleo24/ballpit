"""
paper.py — PDF scraping and structure-aware chunking for summarization pipelines.

Dependencies:
    pip install docling pymupdf

Usage:
    paper = Paper("path/to/paper.pdf")
    paper.scrape()
    chunks = paper.chunk()

    # Optional: summarize figures with a vision model
    paper.summarize_figures(vision_fn=my_vision_fn)

Docling is used for both PDF conversion (layout analysis, reading-order
detection, table structure, OCR) and hierarchical chunking.  The chunker
produces one chunk per detected document element (paragraph, table, figure
caption, etc.) preserving the section heading breadcrumb on every chunk.
These raw chunks are the input to the tree-building stage.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import fitz  # pymupdf

# ---------------------------------------------------------------------------
# Docling imports
# ---------------------------------------------------------------------------
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker, DocChunk
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class PageRecord:
    """Text extracted from a single PDF page (reconstructed from DoclingDocument)."""
    page_number: int        # 1-indexed
    text: str
    char_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.char_count = len(self.text)


@dataclass
class Chunk:
    """
    A single structural unit from HierarchicalChunker, unpacked into flat
    fields for easy access during tree construction.

    All fields are derived directly from the DocChunk / DocMeta that docling
    produces — nothing is inferred or computed here.

    Tree-building fields
    --------------------
    headings : list[str]
        Section breadcrumb from root to the immediate parent of this element,
        e.g. ["2. Methods", "2.1 Data"].  Length == heading depth.  Empty
        for preamble elements (abstract, title).  This is the insertion path
        into the bigtree.

    label : str
        Docling DocItemLabel for the primary doc_item, e.g. "text",
        "table", "picture", "section_header", "list_item", "caption".
        Lets the tree builder treat tables and figures differently from
        plain paragraphs.

    Provenance fields
    -----------------
    doc_item_refs : list[str]
        JSON-pointer self_refs for every DocItem contributing to this chunk,
        e.g. ["#/texts/4", "#/tables/1"].  Stable keys back into the
        DoclingDocument for round-tripping or image extraction.

    page_numbers : list[int]
        Sorted, deduplicated list of pages touched by this chunk (1-indexed).
        Derived from prov on every doc_item.  Used for citation and for
        mapping chunks back to pages[] in Paper.

    captions : list[str]
        For table and figure chunks, any caption text docling detected.
        Empty for plain text chunks.

    Content fields
    --------------
    text : str
        Raw serialized element text (no heading breadcrumb prepended).
        For picture chunks where summarize_figures() has been called, this
        will contain the vision-model summary rather than empty string.
        Use contextualized_text when feeding an embedding model.

    contextualized_text : str
        Heading breadcrumb prepended to text via chunker.contextualize().
        This is what should be embedded or sent to a generation model.

    char_count : int
        len(text) — cheap proxy for size before any tokenization step.

    Tree node fields (None until populated by the tree builder)
    -----------------------------------------------------------
    node_id : str | None
        Stable unique identifier assigned when the chunk is inserted into
        the bigtree, e.g. "sec2.1.para3".

    depth : int | None
        Distance from root in the bigtree (root = 0).  Reflects actual
        document heading depth, not a fixed level scheme.

    parent_id : str | None
        node_id of the parent node.  None for top-level section nodes.

    prev_sibling_id : str | None
        node_id of the preceding sibling under the same parent.  Used by
        the agent for sliding-context retrieval across paragraph boundaries.

    next_sibling_id : str | None
        node_id of the following sibling under the same parent.

    is_leaf : bool
        True if this node has no children in the bigtree.  Defaults to
        True at construction; set to False by the tree builder when child
        nodes are attached.  Leaf nodes are the primary embedding targets.

    embedding_id : str | None
        Foreign key into the vector store, populated after embedding.
        None until the embedding step runs.
    """

    # --- tree-building (populated by chunk(), used as insertion path) ---
    headings: list[str]
    label: str

    # --- provenance ---
    doc_item_refs: list[str]
    page_numbers: list[int]
    captions: list[str]

    # --- content ---
    text: str
    contextualized_text: str
    char_count: int = field(init=False)

    # --- tree node fields (None until tree builder populates them) ---
    node_id: str | None = field(default=None)
    depth: int | None = field(default=None)
    parent_id: str | None = field(default=None)
    prev_sibling_id: str | None = field(default=None)
    next_sibling_id: str | None = field(default=None)
    is_leaf: bool = field(default=True)
    embedding_id: str | None = field(default=None)

    def __post_init__(self) -> None:
        self.char_count = len(self.text)

    @classmethod
    def from_doc_chunk(cls, raw: DocChunk, contextualized: str) -> "Chunk":
        """
        Build a Chunk from a DocChunk returned by HierarchicalChunker.

        Parameters
        ----------
        raw : DocChunk
            The raw chunk from HierarchicalChunker.chunk().
        contextualized : str
            The output of chunker.contextualize(raw) — heading-prepended text.
        """
        meta = raw.meta

        headings: list[str] = list(meta.headings or [])
        captions: list[str] = list(meta.captions or [])

        doc_items = meta.doc_items or []

        # Primary label from the first doc_item; fall back to "text".
        label: str = (
            doc_items[0].label.value
            if doc_items and hasattr(doc_items[0], "label")
            else DocItemLabel.TEXT.value
        )

        # Stable JSON-pointer refs for every contributing item.
        doc_item_refs: list[str] = [
            item.self_ref
            for item in doc_items
            if hasattr(item, "self_ref")
        ]

        # Collect page numbers from provenance on every doc_item.
        pages: set[int] = set()
        for item in doc_items:
            for prov in getattr(item, "prov", None) or []:
                pno = getattr(prov, "page_no", None)
                if pno is not None:
                    pages.add(pno)
        page_numbers: list[int] = sorted(pages) if pages else [1]

        return cls(
            headings=headings,
            label=label,
            doc_item_refs=doc_item_refs,
            page_numbers=page_numbers,
            captions=captions,
            text=raw.text,
            contextualized_text=contextualized,
        )


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class Paper:
    """
    Scrape a PDF and split its text into hierarchical chunks.

    Docling is used for both PDF conversion (layout analysis, reading-order
    detection, table structure, OCR) and hierarchical chunking.  The chunker
    produces one chunk per detected document element preserving the full
    section heading breadcrumb — these are passed directly to the
    tree-building stage without further transformation.

    Parameters
    ----------
    path : str | Path
        Path to the PDF file.
    do_ocr : bool
        Run OCR on scanned / image-only pages (default False; slower).
    do_table_structure : bool
        Extract table structure for Markdown serialization (default True).
    """

    def __init__(
        self,
        path: str | Path,
        do_ocr: bool = False,
        do_table_structure: bool = True,
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"PDF not found: {self.path}")

        self._do_ocr = do_ocr
        self._do_table_structure = do_table_structure

        # Populated by scrape()
        self.metadata: dict = {}
        self.pages: list[PageRecord] = []
        self.full_text: str = ""
        self._dl_doc: DoclingDocument | None = None   # raw docling document
        self._fitz_doc: fitz.Document | None = None   # pymupdf document for image extraction

        # Populated by chunk()
        self.chunks: list[Chunk] = []

        # Populated by summarize_figures(); keyed by doc_item self_ref
        # e.g. {"#/pictures/0": "This bar chart shows ...", ...}
        self.figure_summaries: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scrape(self) -> "Paper":
        """
        Convert the PDF with Docling and populate page-level text records.

        Docling runs a multi-stage pipeline:
          1. PDF backend parses the binary (native or OCR)
          2. Layout analysis model detects elements in reading order
          3. TableFormer reconstructs table structure
          4. Text is assembled into a DoclingDocument

        Populates:
            self.metadata  — document-level metadata dict
            self.pages     — list of PageRecord (one per page)
            self.full_text — Markdown export of the full document
            self._dl_doc   — underlying DoclingDocument

        Returns self for method chaining.
        """
        pipeline_options = PdfPipelineOptions(
            do_ocr=self._do_ocr,
            do_table_structure=self._do_table_structure,
        )

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        result = converter.convert(str(self.path))
        self._dl_doc = result.document
        self._fitz_doc = fitz.open(str(self.path))

        self.metadata = self._extract_metadata(result)
        self.pages = self._build_page_records(self._dl_doc)
        self.full_text = self._dl_doc.export_to_markdown()

        return self

    def chunk(self) -> list[Chunk]:
        """
        Split the document into hierarchical chunks using Docling's
        HierarchicalChunker and return them as Chunk dataclass instances.

        One Chunk is produced per detected document element (paragraph,
        table, figure caption, list item, etc.) in document order.

        Each Chunk carries everything needed by the tree-building stage:
            .headings           — breadcrumb insertion path into the bigtree
            .label              — element type (text / table / picture / …)
            .doc_item_refs      — JSON-pointer refs back into DoclingDocument
            .page_numbers       — pages touched by this chunk
            .captions           — caption text for tables / figures
            .text               — raw element text (vision summary for pictures
                                  if summarize_figures() was called first)
            .contextualized_text — heading-prepended text for embedding

        Returns
        -------
        list[Chunk]
            Ordered list of Chunk objects.  Also stored as self.chunks.
        """
        if self._dl_doc is None:
            raise RuntimeError("Call scrape() before chunk().")

        chunker = HierarchicalChunker()
        raw_chunks = [DocChunk.model_validate(raw) for raw in chunker.chunk(self._dl_doc)]

        self.chunks = []
        for raw in raw_chunks:
            chunk = Chunk.from_doc_chunk(
                raw=raw,
                contextualized=chunker.contextualize(raw),
            )
            # If figure summaries have been generated, inject them into the
            # chunk's text fields so they flow naturally into the tree builder
            # and embedder without any special-casing downstream.
            if chunk.label == DocItemLabel.PICTURE.value and chunk.doc_item_refs:
                summary = self.figure_summaries.get(chunk.doc_item_refs[0])
                if summary:
                    chunk.text = summary
                    chunk.contextualized_text = self._contextualize_summary(
                        summary=summary,
                        headings=chunk.headings,
                        captions=chunk.captions,
                    )
            self.chunks.append(chunk)

        # HierarchicalChunker silently skips picture items because they have no
        # text content.  Build chunks for them manually so figure summaries are
        # included in the output alongside text chunks.
        items = list(self._dl_doc.iterate_items())
        for i, (item, _) in enumerate(items):
            if getattr(item, "label", None) != DocItemLabel.PICTURE:
                continue

            self_ref = getattr(item, "self_ref", "")
            summary = self.figure_summaries.get(self_ref, "")

            captions = getattr(item, "captions", None) or []
            caption_text: list[str] = [
                getattr(c, "text", "") for c in captions if getattr(c, "text", "")
            ]

            pages: set[int] = set()
            for prov in getattr(item, "prov", None) or []:
                pno = getattr(prov, "page_no", None)
                if pno is not None:
                    pages.add(pno)

            headings = self._breadcrumb_at(items, i)
            contextualized = self._contextualize_summary(
                summary=summary,
                headings=headings,
                captions=caption_text,
            )

            self.chunks.append(Chunk(
                headings=headings,
                label=DocItemLabel.PICTURE.value,
                doc_item_refs=[self_ref] if self_ref else [],
                page_numbers=sorted(pages) if pages else [1],
                captions=caption_text,
                text=summary,
                contextualized_text=contextualized,
            ))

        # Re-sort all chunks into document order by first page number so picture
        # chunks appear in the right position relative to text chunks.
        self.chunks.sort(key=lambda c: c.page_numbers[0])

        return self.chunks

    def summarize_figures(
        self,
        vision_fn: Callable[[bytes, str], str],
        context_window: int = 2,
    ) -> "Paper":
        """
        Generate vision-model summaries for every PICTURE element in the
        document and store them in self.figure_summaries.

        Must be called after scrape().  Can be called before or after chunk();
        if called before, chunk() will automatically inject the summaries into
        picture Chunks.  If called after, re-run chunk() to pick them up.

        For each figure the vision model receives:
            - The cropped figure image as raw bytes
            - A context string containing:
                * Section breadcrumb (root → section → subsection)
                * Up to `context_window` preceding text elements
                * The figure caption (if docling detected one)
                * The immediately following text element

        Parameters
        ----------
        vision_fn : Callable[[bytes, str], str]
            A callable that accepts (image_bytes, context_str) and returns a
            summary string.  Inject whichever vision model you need here:

                def my_vision_fn(image_bytes: bytes, context: str) -> str:
                    # call Claude / GPT-4o / etc.
                    ...

        context_window : int
            Number of preceding text elements to include in the context string
            (default 2).  Increase for denser papers; decrease to reduce token
            usage in the vision call.

        Returns self for method chaining.
        """
        if self._dl_doc is None:
            raise RuntimeError("Call scrape() before summarize_figures().")

        items = list(self._dl_doc.iterate_items())
        # Rolling window of the most recent text element texts seen before the
        # current position.  maxlen evicts old entries automatically so we
        # never accumulate more than context_window items.
        recent_text: deque[str] = deque(maxlen=context_window)

        for i, (item, _) in enumerate(items):
            label = getattr(item, "label", None)

            if label != DocItemLabel.PICTURE:
                # Keep the rolling window fed with text from non-figure items.
                text = getattr(item, "text", None)
                if text:
                    recent_text.append(text)
                continue

            # --- caption -------------------------------------------------------
            captions = getattr(item, "captions", None) or []
            caption_text = " ".join(
                getattr(c, "text", "") for c in captions
            ).strip()

            # --- lookahead: one item after the figure --------------------------
            following = ""
            if i + 1 < len(items):
                next_item = items[i + 1][0]
                following = getattr(next_item, "text", "") or ""

            # --- section breadcrumb -------------------------------------------
            breadcrumb = self._breadcrumb_at(items, i)

            # --- assemble context string --------------------------------------
            context = _build_figure_context(
                caption=caption_text,
                preceding=list(recent_text),
                following=following,
                breadcrumb=breadcrumb,
            )

            # --- call vision model and store result ---------------------------
            image_bytes = self._get_figure_bytes(item)
            self.figure_summaries[item.self_ref] = vision_fn(image_bytes, context)

        return self

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def export_markdown(self) -> str:
        """Return the full document as Markdown (requires scrape())."""
        if self._dl_doc is None:
            raise RuntimeError("Call scrape() first.")
        return self._dl_doc.export_to_markdown()

    def export_json(self) -> str:
        """Return the full DoclingDocument serialised as JSON."""
        if self._dl_doc is None:
            raise RuntimeError("Call scrape() first.")
        return self._dl_doc.model_dump_json(indent=2)

    def summary_stats(self) -> dict:
        """Return a dict of useful stats after scrape() (and optionally chunk())."""
        stats: dict = {
            "path": str(self.path),
            "pages": len(self.pages),
            "total_chars": len(self.full_text),
            "metadata": self.metadata,
        }
        if self.chunks:
            stats["num_chunks"] = len(self.chunks)
        if self.figure_summaries:
            stats["num_figure_summaries"] = len(self.figure_summaries)
        return stats

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _breadcrumb_at(self, items: list, index: int) -> list[str]:
        """
        Walk backwards from `index` through `items` to collect the nearest
        enclosing section headers, returning them ordered root → subsection.

        Stops after collecting 3 headings (root → section → subsection),
        which is sufficient context for a vision model without being noisy.
        """
        headings: list[str] = []
        for item, _ in reversed(items[:index]):
            if getattr(item, "label", None) == DocItemLabel.SECTION_HEADER:
                headings.insert(0, item.text)
                if len(headings) >= 3:
                    break
        return headings

    def _get_figure_bytes(self, item) -> bytes:
        """
        Extract the figure image from the PDF using PyMuPDF.

        Uses the bounding box and page number from Docling's provenance data
        to clip the exact figure region from the page and rasterize it to PNG.

        PyMuPDF uses a y-down coordinate system with the origin at the top-left
        of the page, matching the standard screen convention.  Docling's bbox
        is stored in the same space (l, t, r, b in points), so no coordinate
        flip is needed.

        Raises RuntimeError if provenance data is missing on the item.
        """
        prov_list = getattr(item, "prov", None) or []
        if not prov_list:
            raise RuntimeError(
                f"No provenance found for figure {getattr(item, 'self_ref', '?')}."
            )

        prov = prov_list[0]
        page_no = getattr(prov, "page_no", None)   # 1-indexed
        bbox = getattr(prov, "bbox", None)

        if page_no is None or bbox is None:
            raise RuntimeError(
                f"Incomplete provenance for figure {getattr(item, 'self_ref', '?')}: "
                f"page_no={page_no}, bbox={bbox}."
            )

        page = self._fitz_doc[page_no - 1]  # fitz is 0-indexed

        # Docling bbox is in PDF space: origin bottom-left, y increases upward.
        # PyMuPDF uses origin top-left, y increases downward.
        # Flip the y-axis using the page height to convert.
        page_height = page.rect.height
        clip = fitz.Rect(
            bbox.l,
            page_height - bbox.t,
            bbox.r,
            page_height - bbox.b,
        )
        pixmap = page.get_pixmap(clip=clip, dpi=150)
        return pixmap.tobytes("png")

    @staticmethod
    def _contextualize_summary(
        summary: str,
        headings: list[str],
        captions: list[str],
    ) -> str:
        """
        Build a contextualized version of a vision summary by prepending the
        section breadcrumb and caption, mirroring what HierarchicalChunker
        does for text chunks via contextualize().
        """
        parts: list[str] = []
        if headings:
            parts.append(" > ".join(headings))
        if captions:
            parts.append(" ".join(captions))
        parts.append(summary)
        return "\n\n".join(parts)

    @staticmethod
    def _extract_metadata(result) -> dict:
        """Pull document-level metadata from the conversion result."""
        doc = result.document
        dl_meta = doc.origin or {}
        return {
            "title": getattr(doc, "name", "") or "",
            "filename": getattr(dl_meta, "filename", "") or "",
            "mimetype": getattr(dl_meta, "mimetype", "") or "",
            "page_count": len(doc.pages) if doc.pages else 0,
            "binary_hash": getattr(dl_meta, "binary_hash", None),
        }

    @staticmethod
    def _build_page_records(doc: DoclingDocument) -> list[PageRecord]:
        """
        Reconstruct per-page text from the DoclingDocument by collecting
        all text items whose provenance falls on each page.
        """
        page_numbers = sorted(doc.pages.keys()) if doc.pages else []
        page_texts: dict[int, list[str]] = {p: [] for p in page_numbers}

        for item, _ in doc.iterate_items():
            text = getattr(item, "text", None)
            if not text:
                continue
            prov_list = getattr(item, "prov", None) or []
            pages_hit: set[int] = set()
            for prov in prov_list:
                pno = getattr(prov, "page_no", None)
                if pno is not None:
                    pages_hit.add(pno)
            if not pages_hit and page_numbers:
                pages_hit = {page_numbers[0]}
            for pno in pages_hit:
                if pno in page_texts:
                    page_texts[pno].append(text)

        records: list[PageRecord] = []
        for pno in page_numbers:
            combined = "\n".join(page_texts[pno])
            records.append(PageRecord(page_number=pno, text=combined))
        return records


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _build_figure_context(
    caption: str,
    preceding: list[str],
    following: str,
    breadcrumb: list[str],
) -> str:
    """
    Assemble the context string passed to the vision model alongside the
    figure image.

    The string is structured so the model understands where in the paper the
    figure lives and what the surrounding text says about it:

        Section: Introduction > 1.1 Background

        Preceding text:
        As shown in Figure 3, the model outperforms ...

        Caption: Figure 3. Comparison of accuracy across datasets.

        Following text:
        These results suggest that ...

    Only non-empty sections are included.
    """
    parts: list[str] = []
    if breadcrumb:
        parts.append("Section: " + " > ".join(breadcrumb))
    if preceding:
        parts.append("Preceding text:\n" + "\n".join(preceding))
    if caption:
        parts.append("Caption: " + caption)
    if following:
        parts.append("Following text:\n" + following)
    return "\n\n".join(parts)