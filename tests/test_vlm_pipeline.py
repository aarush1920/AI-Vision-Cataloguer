"""Tests for Phase 2 Florence-2 VLM Attribute & Tag Extraction."""

import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vlm_pipeline import ProductTagExtractor, ProductVisualAttributes

try:
    import pytest
except ImportError:
    pytest = None


def test_packaging_aware_classification():
    """Verify packaged items like Bikaji Rasgulla are classified as Food & Beverage despite people in label artwork."""
    extractor = ProductTagExtractor()

    # Case 1: Canned sweet with actor/person printed on packaging
    parsed_bikaji = extractor._parse_attributes_from_text(
        detailed_caption="The image shows a metal tin of Bikaji Rasgulla. In the center is an actor wearing a red kurta and smiling with a bowl.",
        detected_text=["Bikaji", "Aslee Bikaneri", "Rasgulla", "Net Wt 1kg", "Traditional Sweets"],
        raw_outputs={},
    )
    assert parsed_bikaji.category == "Food & Beverage", f"Expected Food & Beverage, got: {parsed_bikaji.category}"
    assert "Rasgulla" in parsed_bikaji.product_type
    assert "Tin" in parsed_bikaji.materials or "Metal" in parsed_bikaji.materials
    print("Test 1: Bikaji Rasgulla correctly classified as Food & Beverage / Rasgulla Tin (NOT Apparel).")

    # Case 2: Health drink container
    parsed_bournvita = extractor._parse_attributes_from_text(
        detailed_caption="A container of BournVita with illustration of a boy wearing sports jersey.",
        detected_text=["BournVita", "5 Star Magic", "2X Strength Nutrients", "Add to milk"],
        raw_outputs={},
    )
    assert parsed_bournvita.category == "Food & Beverage"
    print("Test 2: Drink mix correctly classified as Food & Beverage.")

    # Case 3: Genuine apparel product
    parsed_shirt = extractor._parse_attributes_from_text(
        detailed_caption="A folded navy blue cotton crewneck t-shirt.",
        detected_text=["UrbanStyle", "100% Combed Cotton", "Size M"],
        raw_outputs={},
    )
    assert parsed_shirt.category == "Apparel & Fashion"
    print("Test 3: Actual clothing correctly classified as Apparel & Fashion.")


def get_target_image_path() -> Path:
    """Resolve target processed studio image for inference."""
    # Priority 1: User's real processed studio image
    user_img = PROJECT_ROOT / "data" / "processed" / "my_product_studio.jpg"
    if user_img.exists():
        return user_img

    # Priority 2: Test studio image from Phase 1
    test_studio = PROJECT_ROOT / "data" / "processed" / "test_studio_white_2000.jpg"
    if test_studio.exists():
        return test_studio

    # Priority 3: Raw dummy product
    raw_dummy = PROJECT_ROOT / "data" / "raw" / "dummy_product.jpg"
    if raw_dummy.exists():
        return raw_dummy

    raise FileNotFoundError("No test images found in data/processed or data/raw.")


def test_vlm_tag_extraction():
    """Verify Florence-2 inference and tag extraction on a processed studio image."""
    img_path = get_target_image_path()
    print(f"\n[Test] Running Florence-2 inference on: {img_path}")

    extractor = ProductTagExtractor(model_id="microsoft/Florence-2-base")
    attributes = extractor.extract_tags(img_path)

    # Validate output type and contents
    assert isinstance(attributes, ProductVisualAttributes)
    assert len(attributes.detailed_description) > 0, "Florence-2 caption should not be empty"

    print("\n" + "=" * 60)
    print("EXTRACTED PRODUCT ATTRIBUTES")
    print("=" * 60)
    print(f"Category:             {attributes.category}")
    print(f"Product Type:         {attributes.product_type}")
    print(f"Dominant Colors:      {', '.join(attributes.dominant_colors) or 'None'}")
    print(f"Materials & Finishes: {', '.join(attributes.materials) or 'None'}")
    print(f"Detected Text (OCR):  {', '.join(attributes.detected_text) or 'None'}")
    print(f"Style Tags:           {', '.join(attributes.style_tags) or 'None'}")
    print(f"Caption:              {attributes.detailed_description}")
    print("=" * 60)

    # Save structured JSON output to data/processed
    out_json = PROJECT_ROOT / "data" / "processed" / "extracted_tags.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    dump_data = attributes.model_dump() if hasattr(attributes, "model_dump") else attributes.dict()
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(dump_data, f, indent=2)

    print(f"\nStructured attributes saved to: {out_json}")
    assert out_json.exists()
    assert out_json.stat().st_size > 0


if __name__ == "__main__":
    test_packaging_aware_classification()
    test_vlm_tag_extraction()
