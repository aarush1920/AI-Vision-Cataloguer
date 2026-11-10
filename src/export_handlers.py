"""Export handlers for generating marketplace CSVs and ZIP bundles."""

from __future__ import annotations

import io
import json
import zipfile
from typing import Any, Dict, Optional
import pandas as pd
from PIL import Image

from src.listing_generator.copywriter import ProductCatalogListing


def create_shopify_csv(
    listing: ProductCatalogListing,
    vendor: Optional[str] = None,
    price: str = "29.99",
    sku: str = "PROD-001",
) -> str:
    """Generate official Shopify Product Import CSV format.
    
    Returns:
        CSV string matching Shopify's product import specification.
    """
    row = {
        "Handle": listing.shopify.handle,
        "Title": listing.shopify.title,
        "Body (HTML)": listing.shopify.body_html,
        "Vendor": vendor or listing.brand,
        "Product Category": listing.category,
        "Type": listing.product_name,
        "Tags": ", ".join(listing.shopify.tags),
        "Published": "TRUE",
        "Option1 Name": "Title",
        "Option1 Value": "Default Title",
        "Variant SKU": sku,
        "Variant Grams": "250",
        "Variant Inventory Tracker": "shopify",
        "Variant Inventory Qty": "100",
        "Variant Inventory Policy": "deny",
        "Variant Fulfillment Service": "manual",
        "Variant Price": price,
        "Variant Compare At Price": "",
        "Variant Requires Shipping": "TRUE",
        "Variant Taxable": "TRUE",
        "Variant Barcode": "",
        "Image Src": "my_product_studio.jpg",
        "Image Position": "1",
        "Image Alt Text": listing.shopify.title,
        "SEO Title": listing.shopify.title,
        "SEO Description": listing.shopify.meta_description,
    }

    df = pd.DataFrame([row])
    return df.to_csv(index=False)


def create_amazon_csv(
    listing: ProductCatalogListing,
    sku: str = "AMZ-PROD-001",
    price: str = "29.99",
) -> str:
    """Generate Amazon Inventory Flat File compliant CSV format.
    
    Returns:
        CSV string matching Amazon product flat-file upload.
    """
    bullets = listing.amazon.bullet_points
    row = {
        "item_sku": sku,
        "item_name": listing.amazon.title,
        "brand_name": listing.brand,
        "feed_product_type": listing.category.lower().replace(" ", "_"),
        "standard_price": price,
        "quantity": "50",
        "main_image_url": "my_product_studio.jpg",
        "bullet_point1": bullets[0] if len(bullets) > 0 else "",
        "bullet_point2": bullets[1] if len(bullets) > 1 else "",
        "bullet_point3": bullets[2] if len(bullets) > 2 else "",
        "bullet_point4": bullets[3] if len(bullets) > 3 else "",
        "bullet_point5": bullets[4] if len(bullets) > 4 else "",
        "product_description": listing.amazon.product_description,
        "generic_keywords": listing.amazon.search_terms,
    }

    df = pd.DataFrame([row])
    return df.to_csv(index=False)


def create_catalog_zip_bundle(
    listing: ProductCatalogListing,
    studio_white_img: Optional[Image.Image] = None,
    studio_trans_img: Optional[Image.Image] = None,
    sku: str = "PROD_001",
) -> bytes:
    """Package images, CSV exports, and metadata JSON into a downloadable ZIP archive.
    
    Returns:
        Bytes of the created ZIP archive.
    """
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # 1. Add Shopify CSV
        shopify_csv = create_shopify_csv(listing, sku=sku)
        zf.writestr(f"{sku}_shopify_import.csv", shopify_csv)

        # 2. Add Amazon CSV
        amazon_csv = create_amazon_csv(listing, sku=sku)
        zf.writestr(f"{sku}_amazon_inventory.csv", amazon_csv)

        # 3. Add JSON metadata
        json_data = json.dumps(listing.to_dict(), indent=2)
        zf.writestr(f"{sku}_catalog_metadata.json", json_data)

        # 4. Add 2000x2000 Studio White Image
        if studio_white_img is not None:
            img_buffer = io.BytesIO()
            studio_white_img.save(img_buffer, format="JPEG", quality=95)
            zf.writestr(f"{sku}_STUDIO_WHITE_2000.jpg", img_buffer.getvalue())

        # 5. Add 2000x2000 Studio Transparent Image
        if studio_trans_img is not None:
            img_buffer = io.BytesIO()
            studio_trans_img.save(img_buffer, format="PNG")
            zf.writestr(f"{sku}_STUDIO_TRANSPARENT_2000.png", img_buffer.getvalue())

    zip_buffer.seek(0)
    return zip_buffer.getvalue()
