import argparse
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode, PictureItem


def main():
    parser = argparse.ArgumentParser(description="Extract a PDF with Docling")
    parser.add_argument("path", type=Path, help="Path to the PDF")
    parser.add_argument("image_dir", type=Path, help="Existing directory for figure images")
    args = parser.parse_args()

    if not args.path.is_file():
        parser.error(f"File not found: {args.path}")

    pipeline_options = PdfPipelineOptions(
        # Keep model weights inside the extension instead of Docling's global cache.
        artifacts_path=Path(__file__).resolve().parents[1] / ".models",
    )
    pipeline_options.do_ocr = False  # Native-text PDFs first; enable OCR when needed.
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = 2.0

    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        },
    )
    with redirect_stdout(sys.stderr):
        document = converter.convert(args.path).document

        figures = []
        for item, _level in document.iterate_items():
            if not isinstance(item, PictureItem) or item.image is None:
                continue
            image = item.get_image(document)
            if image is None:
                continue

            figure_id = f"fig-{len(figures) + 1}"
            path = args.image_dir / f"{figure_id}.png"
            image.save(path)
            # Link the figure by ID; the caller deletes the image files after reading them.
            item.image.uri = Path(figure_id)
            figures.append({
                "id": figure_id,
                "path": str(path),
                "caption": item.caption_text(document),
                "page": item.prov[0].page_no if item.prov else None,
            })

        markdown = document.export_to_markdown(
            image_mode=ImageRefMode.REFERENCED,
            page_break_placeholder="<!-- page break -->",
        )

    print(json.dumps(
        {"markdown": markdown, "figures": figures},
        ensure_ascii=False,
        allow_nan=False,
    ))


if __name__ == "__main__":
    main()

