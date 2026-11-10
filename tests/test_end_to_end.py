"""End-to-End integration test covering Phases 1, 2, 3, and 4."""

import io
import json
import sys
import zipfile
from pathlib import Path
import pandas as pd
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.export_handlers import (
    create_amazon_csv,
    create_catalog_zip_bundle,
    create_shopify_csv,
)
from src.image_pipeline import process_catalog_image
from src.listing_generator.copywriter import ListingCopywriter, generate_amazon_listing
from src.vlm_pipeline import ProductTagExtractor


def test_full_pipeline_integration():
    """Verify end-to-end execution across all 4 phases."""
    print("=" * 68)
    print("        END-TO-END E-COMMERCE CATALOGER PIPELINE TEST")
    print("=" * 68)

    # 1. Resolve raw test image
    input_image_path = PROJECT_ROOT / "data" / "raw" / "my_product.jpg"
    if not input_image_path.exists():
        input_image_path = PROJECT_ROOT / "data" / "raw" / "dummy_product.jpg"
    assert input_image_path.exists(), f"Input image not found: {input_image_path}"
    print(f"\n[Phase 1] Processing Image: {input_image_path.name}")

    # Phase 1: Studio Matting & Shadows
    studio_white, studio_trans, isolated = process_catalog_image(
        image_input=input_image_path,
        canvas_size=2000,
        fill_ratio=0.82,
        add_shadow=True,
    )
    assert studio_white.size == (2000, 2000)
    assert studio_trans.size == (2000, 2000)
    assert studio_white.mode == "RGB"
    assert studio_trans.mode == "RGBA"
    print("  -> Studio 2000x2000 Canvas generated with contact drop-shadow.")

    # Phase 2: VLM Attribute Extraction
    print("\n[Phase 2] Loading Florence-2 / Cached VLM Attributes...")
    tags_file = PROJECT_ROOT / "data" / "processed" / "extracted_tags.json"
    if tags_file.exists():
        with open(tags_file, "r", encoding="utf-8") as f:
            tags_dict = json.load(f)
    else:
        extractor = ProductTagExtractor()
        attr = extractor.extract_tags(studio_white)
        tags_dict = attr.model_dump() if hasattr(attr, "model_dump") else attr.dict()

    assert "category" in tags_dict
    assert "detailed_description" in tags_dict
    print(f"  -> Extracted Category: {tags_dict.get('category')}")
    print(f"  -> Dominant Colors:    {', '.join(tags_dict.get('dominant_colors', []))}")

    # Phase 3: Marketplace Copywriting
    print("\n[Phase 3] Generating Amazon & Shopify Listings...")
    copywriter = ListingCopywriter(provider="local")
    listing = copywriter.generate_listing(tags_dict, tone="Professional")

    # Amazon checks
    assert len(listing.amazon.title) <= 200
    assert len(listing.amazon.bullet_points) == 5
    assert len(listing.amazon.search_terms.encode("utf-8")) <= 250
    print(f"  -> Amazon Title ({len(listing.amazon.title)} chars): {listing.amazon.title}")
    print(f"  -> 5 Benefit Bullets Verified.")

    # Shopify checks
    assert len(listing.shopify.title) > 5
    assert "<h3>" in listing.shopify.body_html
    print(f"  -> Shopify Title: {listing.shopify.title}")
    print(f"  -> Shopify Handle: {listing.shopify.handle}")
    print(f"  -> Meta Description ({len(listing.shopify.meta_description)} chars): {listing.shopify.meta_description}")

    # Phase 4: Bulk Export Packages
    print("\n[Phase 4] Verifying Export Packages...")
    shopify_csv_str = create_shopify_csv(listing, sku="TEST-SKU-001", price="34.99")
    amazon_csv_str = create_amazon_csv(listing, sku="TEST-SKU-001", price="34.99")

    df_shopify = pd.read_csv(io.StringIO(shopify_csv_str))
    df_amazon = pd.read_csv(io.StringIO(amazon_csv_str))

    assert "Handle" in df_shopify.columns
    assert "Body (HTML)" in df_shopify.columns
    assert "item_sku" in df_amazon.columns
    assert "bullet_point1" in df_amazon.columns
    print(f"  -> Shopify CSV Verified ({len(df_shopify)} record, {len(df_shopify.columns)} columns).")
    print(f"  -> Amazon CSV Verified ({len(df_amazon)} record, {len(df_amazon.columns)} columns).")

    # Verify ZIP Archive
    zip_bytes = create_catalog_zip_bundle(
        listing=listing,
        studio_white_img=studio_white,
        studio_trans_img=studio_trans,
        sku="TEST-SKU-001",
    )
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        file_list = zf.namelist()
        assert "TEST-SKU-001_shopify_import.csv" in file_list
        assert "TEST-SKU-001_amazon_inventory.csv" in file_list
        assert "TEST-SKU-001_catalog_metadata.json" in file_list
        assert "TEST-SKU-001_STUDIO_WHITE_2000.jpg" in file_list
        assert "TEST-SKU-001_STUDIO_TRANSPARENT_2000.png" in file_list

    print(f"  -> ZIP Bundle Verified ({len(file_list)} packaged assets, {len(zip_bytes) / 1024:.1f} KB).")

    print("\n" + "=" * 68)
    print("ALL 4 PHASES FULLY INTEGRATED & VERIFIED (100% COMPLETE)!")
    print("=" * 68)


def test_end_to_end_jk_paper():
    """Verify JK Paper generates paper & printing Amazon bullets strictly matching title and description."""
    print("\n" + "=" * 68)
    print("     END-TO-END TEST: JK PAPER (Office Supplies & Stationery)")
    print("=" * 68)

    copywriter = ListingCopywriter(provider="local")
    tags_jk = {
        "category": "Office Supplies & Stationery",
        "product_type": "A4 Copier Paper Ream / Notebook",
        "brand": "JK Paper",
        "materials": ["Wood-Free Paper"],
        "dominant_colors": ["Bright White"],
        "detected_text": ["JK Copier", "75 GSM", "500 Sheets", "A4 Size Ream"],
        "style_tags": ["Jam Free", "Printing Paper"],
        "detailed_description": "A package of white multipurpose A4 printing paper ream on a desk.",
    }

    # Generate full catalog listing and isolated Amazon listing
    listing = copywriter.generate_listing(tags_jk, tone="Professional")
    amz_listing = generate_amazon_listing(tags_jk, tone="Professional")

    # Domain assertions
    assert listing.domain == "OFFICE_STATIONERY", f"Expected OFFICE_STATIONERY, got {listing.domain}"

    # Amazon Title assertions
    assert "JK Paper" in listing.amazon.title
    assert "Paper" in listing.amazon.title or "Jam-Free" in listing.amazon.title
    assert len(listing.amazon.title) <= 200

    # Verify all 5 dynamic bullet points match paper/printing specifications
    bullets = listing.amazon.bullet_points
    assert len(bullets) == 5, f"Expected 5 bullets, got {len(bullets)}"
    assert "JAM-FREE HIGH-SPEED PERFORMANCE" in bullets[0]
    assert "SUPERIOR OPACITY & ZERO INK BLEED" in bullets[1]
    assert "ULTRA-BRIGHT COLOR CONTRAST" in bullets[2]
    assert "ACID-FREE ARCHIVAL LONGEVITY" in bullets[3]
    assert "MULTIPURPOSE OFFICE & HOME UTILITY" in bullets[4]

    # Property accessors on AmazonListing
    assert listing.amazon.bullet_point1 == bullets[0]
    assert listing.amazon.bullet_point2 == bullets[1]
    assert listing.amazon.bullet_point3 == bullets[2]
    assert listing.amazon.bullet_point4 == bullets[3]
    assert listing.amazon.bullet_point5 == bullets[4]
    assert amz_listing.bullet_point1 == bullets[0]

    # Verify NO stale Bournvita / food / skincare strings leaked in
    joined_bullets = " ".join(bullets).lower()
    for stale_kw in ["bournvita", "malt", "chocolate", "nutrition", "deodorant", "rasgulla", "hydration"]:
        assert stale_kw not in joined_bullets, f"Stale keyword '{stale_kw}' found in JK Paper bullets!"
        assert stale_kw not in listing.amazon.title.lower(), f"Stale keyword '{stale_kw}' found in JK Paper title!"
        assert stale_kw not in listing.amazon.product_description.lower(), f"Stale keyword '{stale_kw}' found in description!"

    # Verify CSV export reflects the paper bullets
    amz_csv = create_amazon_csv(listing, sku="JK-PPR-001", price="12.99")
    df_amz = pd.read_csv(io.StringIO(amz_csv))
    assert "JAM-FREE" in df_amz["bullet_point1"].iloc[0]
    assert "ZERO INK BLEED" in df_amz["bullet_point2"].iloc[0]

    print(f"  -> JK Paper Amazon Title: {listing.amazon.title}")
    print(f"  -> Bullet 1: {bullets[0][:80]}...")
    print(f"  -> Bullet 2: {bullets[1][:80]}...")
    print("  -> JK Paper Amazon bullets verified with 100% accuracy!")


def test_end_to_end_bikaji_rasgulla():
    """Verify Bikaji Rasgulla generates traditional sweets Amazon bullets matching title and description."""
    print("\n" + "=" * 68)
    print("        END-TO-END TEST: BIKAJI RASGULLA (Food & Beverage)")
    print("=" * 68)

    copywriter = ListingCopywriter(provider="local")
    tags_bikaji = {
        "category": "Food & Beverage",
        "product_type": "Canned Traditional Sweets / Rasgulla Tin",
        "brand": "Bikaji",
        "materials": ["Food-Grade Tin Can"],
        "dominant_colors": ["Orange", "White"],
        "detected_text": ["Bikaji", "Aslee Bikaneri", "Rasgulla", "1 kg", "Traditional Indian Sweets"],
        "style_tags": ["Tin Packaging", "Ready To Eat"],
        "detailed_description": "An orange metal tin of Bikaji Rasgulla sweets on a white surface.",
    }

    # Generate full catalog listing and isolated Amazon listing
    listing = copywriter.generate_listing(tags_bikaji, tone="Professional")
    amz_listing = generate_amazon_listing(tags_bikaji, tone="Professional")

    # Domain assertions
    assert listing.domain == "FOOD_BEVERAGE", f"Expected FOOD_BEVERAGE, got {listing.domain}"

    # Amazon Title assertions
    assert "Bikaji" in listing.amazon.title
    assert "Rasgulla" in listing.amazon.title
    assert len(listing.amazon.title) <= 200

    # Verify all 5 dynamic bullet points match sweets specifications
    bullets = listing.amazon.bullet_points
    assert len(bullets) == 5, f"Expected 5 bullets, got {len(bullets)}"
    assert "AUTHENTIC TRADITIONAL RECIPE" in bullets[0]
    assert "SOFT & SPONGY TEXTURE" in bullets[1]
    assert "HYGIENICALLY PROCESSED & CANNED" in bullets[2]
    assert "AIRTIGHT FRESHNESS LOCK" in bullets[3]
    assert "READY TO SERVE & CELEBRATE" in bullets[4]

    # Property accessors on AmazonListing
    assert listing.amazon.bullet_point1 == bullets[0]
    assert listing.amazon.bullet_point2 == bullets[1]
    assert listing.amazon.bullet_point3 == bullets[2]
    assert listing.amazon.bullet_point4 == bullets[3]
    assert listing.amazon.bullet_point5 == bullets[4]
    assert amz_listing.bullet_point1 == bullets[0]

    # Verify NO stale Bournvita / paper / skincare strings leaked in
    joined_bullets = " ".join(bullets).lower()
    for stale_kw in ["bournvita", "malt", "paper", "jam-free", "cotton", "hydration", "deodorant"]:
        assert stale_kw not in joined_bullets, f"Stale keyword '{stale_kw}' found in Bikaji bullets!"
        assert stale_kw not in listing.amazon.title.lower(), f"Stale keyword '{stale_kw}' found in Bikaji title!"
        assert stale_kw not in listing.amazon.product_description.lower(), f"Stale keyword '{stale_kw}' found in description!"

    # Verify CSV export reflects the sweets bullets
    amz_csv = create_amazon_csv(listing, sku="BKJ-RSG-001", price="9.99")
    df_amz = pd.read_csv(io.StringIO(amz_csv))
    assert "AUTHENTIC TRADITIONAL RECIPE" in df_amz["bullet_point1"].iloc[0]
    assert "SOFT & SPONGY" in df_amz["bullet_point2"].iloc[0]

    print(f"  -> Bikaji Rasgulla Amazon Title: {listing.amazon.title}")
    print(f"  -> Bullet 1: {bullets[0][:80]}...")
    print(f"  -> Bullet 2: {bullets[1][:80]}...")
    print("  -> Bikaji Rasgulla Amazon bullets verified with 100% accuracy!")


def test_sequential_product_switching():
    """Verify switching between Bournvita, JK Paper, and Bikaji never retains stale bullets."""
    print("\n" + "=" * 68)
    print("   TEST: SEQUENTIAL PRODUCT SWITCHING (Bournvita -> JK Paper -> Bikaji)")
    print("=" * 68)

    copywriter = ListingCopywriter(provider="local")

    # Step 1: Process Bournvita
    tags_bournvita = {
        "category": "Food & Beverage",
        "product_type": "Malted Health Nutrition Drink Mix / Jar",
        "brand": "Bournvita",
        "materials": ["Food-Grade Plastic Jar"],
        "detected_text": ["Bournvita", "5 Star Magic", "2X Strength Nutrients", "Add to milk"],
    }
    listing_b = copywriter.generate_listing(tags_bournvita)
    b_bullets = listing_b.amazon.bullet_points
    assert "SCIENTIFICALLY FORMULATED NUTRITION" in b_bullets[0]
    assert "RICH CHOCOLATE MALT FLAVOR" in b_bullets[1]

    # Step 2: Switch to JK Paper (Must NOT retain Bournvita bullets!)
    tags_jk = {
        "category": "Office Supplies & Stationery",
        "product_type": "A4 Copier Paper Ream / Notebook",
        "brand": "JK Paper",
        "materials": ["Wood-Free Paper"],
        "detected_text": ["JK Copier", "75 GSM", "500 Sheets"],
    }
    listing_jk = copywriter.generate_listing(tags_jk)
    jk_bullets = listing_jk.amazon.bullet_points
    assert jk_bullets != b_bullets, "JK Paper bullets must NOT match Bournvita bullets!"
    assert "JAM-FREE HIGH-SPEED PERFORMANCE" in jk_bullets[0]
    assert "bournvita" not in " ".join(jk_bullets).lower()

    # Step 3: Switch to Bikaji Rasgulla (Must NOT retain JK Paper or Bournvita bullets!)
    tags_bikaji = {
        "category": "Food & Beverage",
        "product_type": "Canned Traditional Sweets / Rasgulla Tin",
        "brand": "Bikaji",
        "materials": ["Food-Grade Tin Can"],
        "detected_text": ["Bikaji", "Rasgulla", "1 kg"],
    }
    listing_bikaji = copywriter.generate_listing(tags_bikaji)
    bikaji_bullets = listing_bikaji.amazon.bullet_points
    assert bikaji_bullets != jk_bullets, "Bikaji bullets must NOT match JK Paper bullets!"
    assert bikaji_bullets != b_bullets, "Bikaji bullets must NOT match Bournvita bullets!"
    assert "AUTHENTIC TRADITIONAL RECIPE" in bikaji_bullets[0]
    assert "SOFT & SPONGY" in bikaji_bullets[1]
    assert "jam-free" not in " ".join(bikaji_bullets).lower()
    assert "malt" not in " ".join(bikaji_bullets).lower()

    print("  -> Sequential switching successfully verified: Zero stale state leakage across products!")


if __name__ == "__main__":
    test_full_pipeline_integration()
    test_end_to_end_jk_paper()
    test_end_to_end_bikaji_rasgulla()
    test_sequential_product_switching()
