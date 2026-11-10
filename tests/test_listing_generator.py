"""Unit and integration tests for Phase 3: Listing Copywriter with dynamic domain routing."""

import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.listing_generator.copywriter import (
    AmazonListing,
    ListingCopywriter,
    ProductCatalogListing,
    ShopifyListing,
    generate_listing,
    generate_universal_fallback,
)


def test_listing_generator_from_json():
    """Verify listing generation meets Amazon and Shopify compliance constraints."""
    tags_file = PROJECT_ROOT / "data" / "processed" / "extracted_tags.json"
    assert tags_file.exists(), f"Tags file not found at: {tags_file}"

    copywriter = ListingCopywriter(provider="local")
    listing = copywriter.generate_from_file(file_path=tags_file)

    # 1. Structure assertions
    assert isinstance(listing, ProductCatalogListing)
    assert isinstance(listing.amazon, AmazonListing)
    assert isinstance(listing.shopify, ShopifyListing)

    # 2. Amazon Compliance Assertions
    # Title must be under 200 characters
    assert len(listing.amazon.title) > 10, "Amazon title should not be trivial"
    assert len(listing.amazon.title) <= 200, f"Amazon title exceeds 200 chars: {len(listing.amazon.title)}"

    # Exactly 5 benefit-driven bullet points
    assert len(listing.amazon.bullet_points) == 5, f"Expected 5 bullets, got {len(listing.amazon.bullet_points)}"
    for i, bp in enumerate(listing.amazon.bullet_points):
        assert ":" in bp, f"Bullet {i+1} must contain a capitalized headline hook followed by ':'"
        assert len(bp) > 30, f"Bullet {i+1} is too short"

    # Backend search terms must be under 250 bytes
    search_terms_bytes = len(listing.amazon.search_terms.encode("utf-8"))
    assert search_terms_bytes <= 250, f"Search terms exceeds 250 bytes: {search_terms_bytes}"
    assert "," not in listing.amazon.search_terms, "Search terms must not contain commas"

    # Product description must be non-empty
    assert len(listing.amazon.product_description) > 50

    # 3. Shopify Compliance Assertions
    assert len(listing.shopify.title) > 5
    assert "<h3>" in listing.shopify.body_html and "<ul>" in listing.shopify.body_html
    assert len(listing.shopify.tags) >= 3

    # Meta description should be near the optimal 140-160 characters
    meta_len = len(listing.shopify.meta_description)
    assert 120 <= meta_len <= 170, f"Shopify meta description length {meta_len} outside expected range"

    # Handle must be a valid URL slug (lowercase alphanumeric and hyphens only)
    assert listing.shopify.handle == listing.shopify.handle.lower()
    assert " " not in listing.shopify.handle

    print("Base JSON listing test passed successfully!")


def test_domain_routing_food_packaged():
    """Verify Food & Beverage packaged goods (Bikaji Rasgulla) inject food-specific benefits."""
    copywriter = ListingCopywriter(provider="local")
    tags_bikaji = {
        "category": "Food & Beverage",
        "product_type": "Rasgulla Tin",
        "dominant_colors": ["Orange", "White"],
        "detected_text": ["Bikaji", "Aslee Bikaneri", "Rasgulla", "1 kg", "Traditional Indian Sweets"],
        "materials": ["Tin", "Metal"],
        "style_tags": ["Tin Packaging", "Ready To Eat"],
        "detailed_description": "An orange metal tin of Bikaji Rasgulla. The label has an illustration of a man in a kurta.",
    }

    listing = copywriter.generate_listing(tags_bikaji)
    assert listing.domain == "FOOD_BEVERAGE"
    assert "hydration" not in listing.amazon.title.lower()
    assert "hydration" not in listing.amazon.bullet_points[0].lower()
    assert "skin" not in listing.amazon.bullet_points[0].lower()

    # Must contain Taste, Hygiene, Authentic Recipe, Freshness
    joined_bullets = " ".join(listing.amazon.bullet_points)
    assert "AUTHENTIC TRADITIONAL RECIPE" in joined_bullets
    assert "HYGIENICALLY PROCESSED" in joined_bullets
    assert "SOFT & SPONGY" in joined_bullets

    assert "Rasgulla" in listing.shopify.title
    assert "Taste & Quality Highlights" in listing.shopify.body_html
    assert "Serving Suggestions" in listing.shopify.body_html

    print("Food domain routing (Bikaji Rasgulla) passed successfully!")


def test_domain_routing_food_bournvita():
    """Verify Bournvita malted health drink mix generates nutrition & taste copy."""
    copywriter = ListingCopywriter(provider="local")
    tags_bournvita = {
        "category": "Food & Beverage",
        "product_type": "Malted Health Nutrition Drink Mix / Jar",
        "dominant_colors": ["Brown", "White", "Blue"],
        "detected_text": ["Bournvita", "5 Star Magic", "2X Strength Nutrients", "Add to milk"],
        "materials": ["Food-Grade Plastic Jar"],
        "style_tags": ["Nutrition", "Health Drink"],
    }

    listing = copywriter.generate_listing(tags_bournvita)
    assert listing.domain == "FOOD_BEVERAGE"
    assert "Bournvita" in listing.amazon.title
    assert "Nutrition" in listing.amazon.title

    joined_bullets = " ".join(listing.amazon.bullet_points)
    assert "SCIENTIFICALLY FORMULATED NUTRITION" in joined_bullets
    assert "RICH CHOCOLATE MALT FLAVOR" in joined_bullets
    assert "HYGIENICALLY PROCESSED" in joined_bullets

    assert "Nutrition" in listing.shopify.body_html
    assert "Preparation Instructions" in listing.shopify.body_html
    assert "deodorant" not in joined_bullets.lower()
    assert "hydration" not in joined_bullets.lower()

    print("Food domain routing (Bournvita malted drink) passed successfully!")


def test_domain_routing_stationery_paper():
    """Verify Office Supplies & Stationery injects paper & printing benefits."""
    copywriter = ListingCopywriter(provider="local")
    tags_paper = {
        "category": "Office Supplies & Stationery",
        "product_type": "A4 Copier Paper Ream / Notebook",
        "dominant_colors": ["Bright White"],
        "detected_text": ["JK Copier", "75 GSM", "500 Sheets", "A4 Size"],
        "materials": ["Wood-Free Paper"],
        "style_tags": ["Jam Free", "Printing Paper"],
    }

    listing = copywriter.generate_listing(tags_paper)
    assert listing.domain == "OFFICE_STATIONERY"
    assert "Jam-Free" in listing.amazon.title or "Paper" in listing.amazon.title

    joined_bullets = " ".join(listing.amazon.bullet_points)
    assert "JAM-FREE HIGH-SPEED PERFORMANCE" in joined_bullets
    assert "SUPERIOR OPACITY & ZERO INK BLEED" in joined_bullets
    assert "ULTRA-BRIGHT COLOR CONTRAST" in joined_bullets
    assert "ACID-FREE ARCHIVAL LONGEVITY" in joined_bullets

    assert "Jam-Free Reliability" in listing.shopify.body_html
    assert "Zero Ink Bleed" in listing.shopify.body_html
    assert "Technical Specifications" in listing.shopify.body_html

    print("Stationery domain routing (JK Paper) passed successfully!")


def test_domain_routing_apparel():
    """Verify Apparel injects Fabric & Fit benefits."""
    copywriter = ListingCopywriter(provider="local")
    tags_apparel = {
        "category": "Apparel & Fashion",
        "product_type": "T-Shirt",
        "dominant_colors": ["Navy Blue"],
        "detected_text": ["Nike", "100% Combed Cotton", "Size L"],
        "materials": ["Cotton"],
        "style_tags": ["Casual"],
        "detailed_description": "A navy blue crewneck t-shirt on a table.",
    }

    listing = copywriter.generate_listing(tags_apparel)
    assert listing.domain == "APPAREL_FASHION"
    joined_bullets = " ".join(listing.amazon.bullet_points)
    assert "BREATHABLE FABRIC" in joined_bullets
    assert "TAILORED ERGONOMIC FIT" in joined_bullets
    assert "Care Instructions" in listing.shopify.body_html

    print("Apparel domain routing passed successfully!")


def test_domain_routing_electronics():
    """Verify Electronics injects Tech Specs & Connectivity benefits."""
    copywriter = ListingCopywriter(provider="local")
    tags_elec = {
        "category": "Electronics",
        "product_type": "Wireless Earbuds",
        "dominant_colors": ["Black"],
        "detected_text": ["Boat", "Bluetooth 5.3", "Active Noise Cancellation", "40H Playtime"],
        "materials": ["Plastic"],
        "style_tags": ["Modern"],
        "detailed_description": "Wireless bluetooth earbuds with charging case.",
    }

    listing = copywriter.generate_listing(tags_elec)
    assert listing.domain == "ELECTRONICS"
    joined_bullets = " ".join(listing.amazon.bullet_points)
    assert "HIGH-FIDELITY AUDIO" in joined_bullets
    assert "WIRELESS CONNECTIVITY" in joined_bullets
    assert "Technical Specifications" in listing.shopify.body_html

    print("Electronics domain routing passed successfully!")


def test_generate_universal_fallback_defaults():
    """Verify generate_universal_fallback handles empty tags with exact safe defaults."""
    listing = generate_universal_fallback({})
    
    # 1. Structure assertions
    assert isinstance(listing, ProductCatalogListing)
    assert listing.domain == "UNIVERSAL"
    assert listing.brand == "Premium"
    assert listing.product_name == "Item"
    
    # 2. Dynamic Title interpolation: [Brand] [Product Type] - [Materials] in [Colors] | Premium Utility
    expected_title = "Premium Item - High-Quality Materials in Standard Finish | Premium Utility"
    assert listing.amazon.title == expected_title
    assert len(listing.amazon.title) <= 200
    
    # 3. 5 Amazon Bullets (Build, Utility, Aesthetics, Specs, Quality)
    bullets = listing.amazon.bullet_points
    assert len(bullets) == 5
    # Bullet 1 (Build): Materials and durability
    assert "DURABLE HIGH-GRADE BUILD" in bullets[0]
    assert "High-Quality Materials" in bullets[0]
    assert "durability" in bullets[0].lower()
    
    # Bullet 2 (Utility): Versatile application of product_type
    assert "VERSATILE MULTI-PURPOSE UTILITY" in bullets[1]
    assert "Item" in bullets[1]
    
    # Bullet 3 (Aesthetics): Colors and design finish
    assert "REFINED AESTHETIC FINISH" in bullets[2]
    assert "Standard Finish" in bullets[2]
    
    # Bullet 4 (Specs): Dynamic slice of ocr_text as verified manufacturer specs
    assert "AUTHENTICATED MANUFACTURER SPECIFICATIONS" in bullets[3]
    assert "Verified Specifications" in bullets[3]
    
    # Bullet 5 (Quality): Generic quality assurance and packaging safety
    assert "STRINGENT QUALITY ASSURANCE & PACKAGING" in bullets[4]
    assert "quality control" in bullets[4].lower()
    assert "packaging" in bullets[4].lower()
    
    # 4. Shopify HTML: Semantic block (<h3>, <p>, <ul><li>)
    assert "<h3>Product Overview</h3>" in listing.shopify.body_html
    assert "<p>" in listing.shopify.body_html
    assert "<ul>" in listing.shopify.body_html
    assert "<li>" in listing.shopify.body_html
    assert "High-Quality Materials" in listing.shopify.body_html
    assert "Standard Finish" in listing.shopify.body_html
    
    # Search terms under 250 bytes
    assert len(listing.amazon.search_terms.encode("utf-8")) <= 250
    assert "," not in listing.amazon.search_terms

    print("Universal fallback defaults test passed successfully!")


def test_generate_universal_fallback_custom_tags():
    """Verify generate_universal_fallback dynamically interpolates custom tags."""
    tags = {
        "brand": "Stanley",
        "product_type": "Screwdriver",
        "materials": "Chrome Vanadium Steel",
        "colors": "Yellow & Black",
        "ocr_text": "Magnetic Tip 6x100mm PH2",
    }
    listing = generate_universal_fallback(tags)
    
    assert listing.domain == "UNIVERSAL"
    assert listing.brand == "Stanley"
    assert listing.product_name == "Screwdriver"
    
    # Amazon Title
    expected_title = "Stanley Screwdriver - Chrome Vanadium Steel in Yellow & Black | Premium Utility"
    assert listing.amazon.title == expected_title
    assert len(listing.amazon.title) <= 200
    
    # 5 Bullets
    bullets = listing.amazon.bullet_points
    assert len(bullets) == 5
    assert "Chrome Vanadium Steel" in bullets[0]
    assert "Screwdriver" in bullets[1]
    assert "Yellow & Black" in bullets[2]
    assert "Magnetic Tip 6x100mm PH2" in bullets[3]
    
    # Shopify HTML
    assert "Stanley" in listing.shopify.body_html
    assert "Screwdriver" in listing.shopify.body_html
    assert "Chrome Vanadium Steel" in listing.shopify.body_html
    assert "Yellow & Black" in listing.shopify.body_html
    assert "Magnetic Tip 6x100mm PH2" in listing.shopify.body_html

    print("Universal fallback custom tags test passed successfully!")


def test_domain_router_outside_predefined_domains():
    """Verify router redirects non-predefined domains (JK Paper, Screwdriver, Unknown) to universal fallback."""
    copywriter = ListingCopywriter(provider="local")
    
    # 1. JK Paper
    tags_jk = {
        "category": "JK Paper",
        "product_type": "Copier Paper",
        "dominant_colors": ["Bright White"],
        "materials": ["Virgin Wood Pulp"],
        "detected_text": ["JK Paper", "75 GSM", "500 Sheets", "A4 Size"],
    }
    listing_jk = copywriter.generate_listing(tags_jk)
    assert listing_jk.domain == "UNIVERSAL"
    assert "JK Paper" in listing_jk.amazon.title
    assert "Copier Paper" in listing_jk.amazon.title
    # Must NOT have beauty or skincare terms
    assert "hydration" not in listing_jk.amazon.title.lower()
    assert "skin" not in " ".join(listing_jk.amazon.bullet_points).lower()
    assert "glow" not in " ".join(listing_jk.amazon.bullet_points).lower()
    
    # 2. Screwdriver / Hardware
    tags_tool = {
        "category": "Hardware & Tools",
        "product_type": "Precision Screwdriver",
        "dominant_colors": ["Silver", "Black"],
        "materials": ["Alloy Steel"],
        "detected_text": ["Bosch", "PH00", "Magnetic"],
    }
    listing_tool = copywriter.generate_listing(tags_tool)
    assert listing_tool.domain == "UNIVERSAL"
    assert "hydration" not in listing_tool.amazon.title.lower()
    assert "skin" not in " ".join(listing_tool.amazon.bullet_points).lower()
    assert "DURABLE HIGH-GRADE BUILD" in listing_tool.amazon.bullet_points[0]
    
    # 3. Missing / Empty Category
    tags_empty = {
        "category": "",
        "product_type": "Utility Knife",
        "dominant_colors": ["Red"],
        "materials": ["Stainless Steel"],
    }
    listing_empty = copywriter.generate_listing(tags_empty)
    assert listing_empty.domain == "UNIVERSAL"
    assert "hydration" not in " ".join(listing_empty.amazon.bullet_points).lower()
    
    # 4. 'Unknown' Category
    tags_unknown = {
        "category": "Unknown",
        "product_type": "Stationery Binder",
    }
    listing_unknown = copywriter.generate_listing(tags_unknown)
    assert listing_unknown.domain == "UNIVERSAL"
    assert listing_unknown.brand == "Premium"
    assert listing_unknown.product_name == "Stationery Binder"
    assert "hydration" not in " ".join(listing_unknown.amazon.bullet_points).lower()

    print("Domain router universal fallback redirection passed successfully!")


if __name__ == "__main__":
    test_listing_generator_from_json()
    test_domain_routing_food_packaged()
    test_domain_routing_food_bournvita()
    test_domain_routing_stationery_paper()
    test_domain_routing_apparel()
    test_domain_routing_electronics()
    test_generate_universal_fallback_defaults()
    test_generate_universal_fallback_custom_tags()
    test_domain_router_outside_predefined_domains()
    print("\nALL PHASE 3 LISTING GENERATOR & DOMAIN ROUTING TESTS PASSED!")
