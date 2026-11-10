"""Main pipeline runner for Automated E-Commerce Product Cataloger.

Processes a raw smartphone photo, synthesizes a studio canvas with contact shadows,
extracts visual attributes with Florence-2 VLM, displays a summary in the terminal,
and opens the generated studio image.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.image_pipeline import process_catalog_image
from src.vlm_pipeline import ProductTagExtractor
from src.listing_generator import ListingCopywriter


def run_cataloger(
    input_path: Path | str = PROJECT_ROOT / "data" / "raw" / "my_product.jpg",
    output_studio_path: Path | str = PROJECT_ROOT / "data" / "processed" / "my_product_studio.jpg",
    output_trans_path: Path | str = PROJECT_ROOT / "data" / "processed" / "my_product_transparent.png",
    output_tags_path: Path | str = PROJECT_ROOT / "data" / "processed" / "extracted_tags.json",
    open_image: bool = True,
) -> None:
    """Execute end-to-end studio transformation and Florence-2 VLM extraction."""
    input_file = Path(input_path)
    studio_file = Path(output_studio_path)
    trans_file = Path(output_trans_path)
    tags_file = Path(output_tags_path)

    if not input_file.exists():
        print(f"Error: Input file '{input_file}' not found.")
        print("Please place your product photo in data/raw/my_product.jpg.")
        return

    print("=" * 68)
    print("      AUTOMATED E-COMMERCE PRODUCT CATALOGER PIPELINE")
    print("=" * 68)
    print(f"Input Image : {input_file.resolve()}\n")

    # -------------------------------------------------------------
    # Step 1 & 2: Background Removal, Drop-Shadow & 2000x2000 Canvas
    # -------------------------------------------------------------
    print("[1/3] Processing studio image & synthetic contact shadow...")
    studio_white, studio_trans, _ = process_catalog_image(
        image_input=input_file,
        canvas_size=2000,
        fill_ratio=0.82,
        add_shadow=True,
    )

    studio_file.parent.mkdir(parents=True, exist_ok=True)
    studio_white.save(studio_file, format="JPEG", quality=95)
    studio_trans.save(trans_file, format="PNG")

    print(f"  -> Studio JPEG saved: {studio_file.resolve()}")
    print(f"  -> Transparent PNG:   {trans_file.resolve()}")

    # -------------------------------------------------------------
    # Step 3: Florence-2 VLM Attribute & OCR Extraction
    # -------------------------------------------------------------
    print("\n[2/3] Running Florence-2 Vision-Language extraction (attributes & OCR)...")
    extractor = ProductTagExtractor(model_id="microsoft/Florence-2-base")
    attributes = extractor.extract_tags(studio_file)

    # Save extracted attributes to JSON
    tags_file.parent.mkdir(parents=True, exist_ok=True)
    dump_data = attributes.model_dump() if hasattr(attributes, "model_dump") else attributes.dict()
    with open(tags_file, "w", encoding="utf-8") as f:
        json.dump(dump_data, f, indent=2)

    # -------------------------------------------------------------
    # Step 4: Terminal Summary Display
    # -------------------------------------------------------------
    print("\n" + "=" * 68)
    print("                 EXTRACTED PRODUCT CATALOG TAGS")
    print("=" * 68)
    print(f" Category        : {attributes.category}")
    print(f" Product Type    : {attributes.product_type}")
    print(f" Dominant Colors : {', '.join(attributes.dominant_colors) or 'None'}")
    print(f" Materials       : {', '.join(attributes.materials) or 'None'}")
    print(f" Style Tags      : {', '.join(attributes.style_tags) or 'None'}")

    print("\n Detected OCR Text:")
    if attributes.detected_text:
        for text_line in attributes.detected_text:
            print(f"   * {text_line}")
    else:
        print("   * (No text detected)")

    print("\n Visual Description (Florence-2):")
    print(f"   \"{attributes.detailed_description}\"")
    print("=" * 68)
    print(f"\nMetadata saved to: {tags_file.resolve()}")

    # -------------------------------------------------------------
    # Step 5: Phase 3 Marketplace Listing Copywriting
    # -------------------------------------------------------------
    print("\n[3/4] Synthesizing Amazon & Shopify Marketplace Listings...")
    copywriter = ListingCopywriter(provider="local")
    listing = copywriter.generate_listing(dump_data)
    listing_out_path = PROJECT_ROOT / "data" / "processed" / "generated_listings.json"
    with open(listing_out_path, "w", encoding="utf-8") as f:
        json.dump(listing.to_dict(), f, indent=2)

    print("\n" + "=" * 68)
    print("               MARKETPLACE COMPLIANT LISTINGS")
    print("=" * 68)
    print(f" Brand                : {listing.brand}")
    print(f" Amazon Title ({len(listing.amazon.title)}/200): {listing.amazon.title}")
    print("\n Amazon 5 Benefit Bullets:")
    for i, bp in enumerate(listing.amazon.bullet_points, 1):
        print(f"   [{i}] {bp}")
    print(f"\n Backend Search Terms : {listing.amazon.search_terms}")
    print(f"\n Shopify Title        : {listing.shopify.title}")
    print(f" Shopify Meta ({len(listing.shopify.meta_description)} chars) : {listing.shopify.meta_description}")
    print("=" * 68)
    print(f"Listings saved to: {listing_out_path.resolve()}")

    # -------------------------------------------------------------
    # Step 6: Automatically Open Resulting Image
    # -------------------------------------------------------------
    if open_image:
        print("\n[4/4] Opening studio image in default viewer...")
        try:
            studio_white.show(title="Studio E-Commerce Image")
        except Exception as e:
            print(f"Note: Could not open default viewer ({e}). View image at: {studio_file}")

    print("\nPipeline execution 100% complete! Launch Streamlit app via: streamlit run app.py")


if __name__ == "__main__":
    run_cataloger()
