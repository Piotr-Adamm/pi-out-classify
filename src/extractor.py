# Docling side of out-classify's document extraction. Three modes:
#   version                          print the installed Docling version
#   models <models_dir>              download the model weights, if missing or for another Docling version
#   extract <pdf> <image_dir> <models_dir>   print the PDF as Markdown plus figures (JSON)
# Docling is imported only where it is needed, so `version` stays fast.
import argparse
import json
import re
import shutil
import sys
from contextlib import redirect_stdout
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# Written last into the models directory, holding the Docling version the models were downloaded with.
INSTALLED_MARKER = ".installed"
# Placeholder text for formula n, replaced by its figure link after the Markdown export.
FORMULA_TOKEN = "OUTCLASSIFYFORMULA"


def docling_version() -> str:
    try:
        return version("docling")
    except PackageNotFoundError:
        sys.exit(f"Docling is not installed for {sys.executable}.")


# Downloads the layout and table models (about 670 MB); OCR is off, so no OCR models. The download goes
# into a temporary directory that replaces the old one only when complete, so an interrupted download
# leaves the previous models untouched.
def download(models_dir: Path) -> None:
    from docling.utils.model_downloader import download_models

    current = docling_version()
    marker = models_dir / INSTALLED_MARKER
    if marker.is_file() and marker.read_text().strip() == current:
        return

    partial = models_dir.with_name(models_dir.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    download_models(
        output_dir=partial,
        with_layout=True,
        with_tableformer=True,
        with_code_formula=False,
        with_picture_classifier=False,
        with_rapidocr=False,
    )
    (partial / INSTALLED_MARKER).write_text(current)
    shutil.rmtree(models_dir, ignore_errors=True)
    partial.rename(models_dir)


def extract(path: Path, image_dir: Path, models_dir: Path) -> dict:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.types.doc import DocItemLabel, ImageRefMode, PictureItem

    pipeline_options = PdfPipelineOptions(artifacts_path=models_dir)
    pipeline_options.do_ocr = False  # Native-text PDFs first; enable OCR when needed.
    pipeline_options.generate_picture_images = True
    # Formulas are cropped from the page images.
    pipeline_options.generate_page_images = True
    pipeline_options.images_scale = 2.0

    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        },
    )
    document = converter.convert(path).document

    figures = []
    formula_count = 0
    unextracted_formulas = {}  # formula number -> page, for formulas without an image
    for item, _level in document.iterate_items():
        if item.label == DocItemLabel.FORMULA:
            # Docling would write <!-- formula-not-decoded --> (no formula model) or the PDF's text layer,
            # which flattens exponents and fractions (s2 for s^2). Instead, each formula becomes an image
            # linked like a figure; the token marks its place in the Markdown. The text layer is kept
            # for the classifier, which judges a section by its formulas too.
            formula_count += 1
            text_layer = " ".join((item.orig or item.text or "").split())
            item.text = f"{FORMULA_TOKEN}{formula_count}"
            page = item.prov[0].page_no if item.prov else None
            image = item.get_image(document)
            if image is None:
                unextracted_formulas[formula_count] = page
                continue
            formula_id = f"formula-{formula_count}"
            formula_path = image_dir / f"{formula_id}.png"
            image.save(formula_path)
            figures.append({
                "id": formula_id,
                "kind": "formula",
                "path": str(formula_path),
                "caption": "",
                "page": page,
                "textLayer": text_layer,
            })
            continue

        if not isinstance(item, PictureItem) or item.image is None:
            continue
        image = item.get_image(document)
        if image is None:
            continue

        figure_id = f"fig-{len(figures) + 1}"
        figure_path = image_dir / f"{figure_id}.png"
        image.save(figure_path)
        # Link the figure by ID; the caller deletes the image files after reading them.
        item.image.uri = Path(figure_id)
        figures.append({
            "id": figure_id,
            "kind": "figure",
            "path": str(figure_path),
            "caption": item.caption_text(document),
            "page": item.prov[0].page_no if item.prov else None,
        })

    markdown = document.export_to_markdown(
        image_mode=ImageRefMode.REFERENCED,
        page_break_placeholder="<!-- page break -->",
    )

    def formula_link(match: re.Match) -> str:
        number = int(match.group(1))
        if number in unextracted_formulas:
            page = unextracted_formulas[number]
            return f"[formula on page {page} could not be extracted]" if page else "[formula could not be extracted]"
        return f"![Formula](formula-{number})"

    markdown = re.sub(rf"\$\${FORMULA_TOKEN}(\d+)\$\$", formula_link, markdown)
    return {"markdown": markdown, "figures": figures}


def main():
    parser = argparse.ArgumentParser(description="Docling extraction for out-classify")
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("version", help="Print the installed Docling version")
    models = modes.add_parser("models", help="Download the model weights")
    models.add_argument("models_dir", type=Path)
    extract_mode = modes.add_parser("extract", help="Extract a PDF")
    extract_mode.add_argument("path", type=Path, help="Path to the PDF")
    extract_mode.add_argument("image_dir", type=Path, help="Existing directory for figure images")
    extract_mode.add_argument("models_dir", type=Path, help="Directory with the model weights")
    args = parser.parse_args()

    if args.mode == "version":
        print(docling_version())
        return

    if args.mode == "models":
        # Only the JSON of `extract` belongs on stdout; Docling's own output goes to stderr.
        with redirect_stdout(sys.stderr):
            download(args.models_dir)
        return

    if not args.path.is_file():
        parser.error(f"File not found: {args.path}")
    with redirect_stdout(sys.stderr):
        result = extract(args.path, args.image_dir, args.models_dir)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
