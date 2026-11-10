"""Test suite for Layer 1 Showcase Priority Registry and Layer 2 Weighted Scoring."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vlm_pipeline.tag_extractor import ProductTagExtractor

def test_showcase_priority_registry():
    ext = ProductTagExtractor(client="dummy")

    # 1. Derma Co / Cleanser / Niacinamide
    attr1 = ext._parse_attributes_from_text(
        "A bottle of face wash on a counter",
        ["The Derma Co", "1% Salicylic Acid Foaming Cleanser", "Non Oil Stripping", "Niacinamide"],
        {},
    )
    assert attr1.category == "Beauty & Personal Care", f"Expected Beauty, got {attr1.category}"
    assert "Cleanser" in attr1.product_type or "Face Wash" in attr1.product_type
    assert "Plastic Bottle with Pump Dispenser" in attr1.materials

    # 2. Bournvita / Cadbury with deodorant in caption
    attr2 = ext._parse_attributes_from_text(
        "The image is of a deodorant stick from the brand Bourn Vita 5Star",
        ["BournVita5starmagic2X STRENGTHNUTRIENTSADD TO MILK FOR2X WT"],
        {},
    )
    assert attr2.category == "Food & Beverage", f"Expected Food, got {attr2.category}"
    assert "Malted Health Nutrition Drink Mix / Jar" in attr2.product_type
    assert "Food-Grade Plastic Jar" in attr2.materials

    # 3. Bikaji / Rasgulla
    attr3 = ext._parse_attributes_from_text(
        "Orange metal tin with illustration of a man on label",
        ["Bikaji", "Aslee Bikaneri", "Rasgulla", "1 kg", "Traditional Indian Sweets"],
        {},
    )
    assert attr3.category == "Food & Beverage", f"Expected Food, got {attr3.category}"
    assert "Canned Traditional Sweets / Rasgulla Tin" in attr3.product_type
    assert "Food-Grade Tin Can" in attr3.materials

    # 4. Pilgrim / Serum
    attr4 = ext._parse_attributes_from_text(
        "Glass bottle with dropper pipette",
        ["Pilgrim", "2% Alpha Arbutin", "Face Serum", "30ml"],
        {},
    )
    assert attr4.category == "Beauty & Personal Care", f"Expected Beauty, got {attr4.category}"
    assert "Dropper Bottle Serum" in attr4.product_type
    assert "Glass Bottle with Dropper" in attr4.materials

    # 5. JK Paper / Ream / Notebook
    attr5 = ext._parse_attributes_from_text(
        "A stack of white paper sheets",
        ["JK Copier Paper", "75 GSM", "500 Sheets", "A4 Ream"],
        {},
    )
    assert attr5.category == "Office Supplies & Stationery", f"Expected Office Supplies, got {attr5.category}"
    assert "A4 Copier Paper Ream / Notebook" in attr5.product_type
    assert "Wood-Free Paper" in attr5.materials

    print("All 5 Showcase Priority Registry tests passed successfully!")


def test_layer2_weighted_scoring_no_naive_substrings():
    ext = ProductTagExtractor(client="dummy")

    # Unseen cleanser with 'non oil stripping'
    # Must NOT match Food & Beverage despite containing the word 'oil'
    attr_unseen_cleanser = ext._parse_attributes_from_text(
        "A cylindrical container on a bathroom shelf",
        ["PureGlow", "Gentle Face Wash", "Non Oil Stripping", "Deep Hydrating Formula for Pores", "150ml"],
        {},
    )
    assert attr_unseen_cleanser.category == "Beauty & Personal Care", f"Expected Beauty, got {attr_unseen_cleanser.category}"
    assert "Cleanser" in attr_unseen_cleanser.product_type or "Face Wash" in attr_unseen_cleanser.product_type

    # Unseen food snack with sweet and chocolate
    attr_unseen_food = ext._parse_attributes_from_text(
        "A pouch on a kitchen counter",
        ["ChocoBites", "Delicious Dark Chocolate Snack", "Crispy Edible Wafers with Sweet Malt"],
        {},
    )
    assert attr_unseen_food.category == "Food & Beverage", f"Expected Food, got {attr_unseen_food.category}"

    # Unseen stationery
    attr_unseen_paper = ext._parse_attributes_from_text(
        "A bundle of wrapped office sheets",
        ["CenturyPrint", "Multipurpose A4 Printing Paper", "80 GSM Sheet Pack"],
        {},
    )
    assert attr_unseen_paper.category == "Office Supplies & Stationery", f"Expected Stationery, got {attr_unseen_paper.category}"

    # Unseen apparel with person illustration on packaging
    attr_unseen_apparel = ext._parse_attributes_from_text(
        "photo of a man wearing a red shirt and hat on the packaging wrapper",
        ["ClassicFit", "100% Combed Cotton T-Shirt", "Size Large"],
        {},
    )
    assert attr_unseen_apparel.category == "Apparel & Fashion", f"Expected Apparel, got {attr_unseen_apparel.category}"

    print("All Layer 2 Weighted Scoring tests passed successfully!")


if __name__ == "__main__":
    test_showcase_priority_registry()
    test_layer2_weighted_scoring_no_naive_substrings()
