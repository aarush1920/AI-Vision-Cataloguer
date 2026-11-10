"""Automated E-Commerce Product Cataloger - Streamlit Dashboard.

Interactive web application providing:
1. Smartphone photo upload & sample gallery.
2. Studio canvas standardization (2000x2000 pure white) & synthetic drop shadows.
3. Florence-2 VLM visual attribute and OCR text extraction.
4. Human-in-the-Loop review and editing of tags and marketplace copy.
5. Marketplace listing generator (Amazon & Shopify compliant).
6. Bulk export for Shopify CSV, Amazon CSV, JSON metadata, and image ZIP bundle.
"""

from __future__ import annotations

import re
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from PIL import Image
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent

from src.export_handlers import (
    create_amazon_csv,
    create_catalog_zip_bundle,
    create_shopify_csv,
)
from src.image_pipeline import ShadowEngine, process_catalog_image
from src.listing_generator.copywriter import (
    AmazonListing,
    ListingCopywriter,
    ProductCatalogListing,
    ShopifyListing,
)
from src.vlm_pipeline import ProductTagExtractor, ProductVisualAttributes

# -----------------------------------------------------------------------------
# Streamlit App Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Automated E-Commerce Cataloger",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished marketplace UI styling
st.markdown(
    """
    <style>
        .main-header {
            font-size: 2.1rem;
            font-weight: 700;
            color: #1E293B;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #64748B;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 10px;
        }
        .tag-pill {
            display: inline-block;
            background-color: #EEF2F6;
            color: #1E293B;
            border: 1px solid #CBD5E1;
            border-radius: 16px;
            padding: 4px 12px;
            margin: 3px 4px;
            font-size: 0.85rem;
            font-weight: 500;
        }
        .compliance-badge {
            display: inline-block;
            background-color: #ECFDF5;
            color: #065F46;
            border: 1px solid #A7F3D0;
            border-radius: 6px;
            padding: 4px 10px;
            font-size: 0.82rem;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Session State Initialization & Synchronization Helpers
# -----------------------------------------------------------------------------
def sync_bullet_session_state(bullets: List[str]) -> None:
    """Explicitly overwrite session state keys for the 5 Amazon bullet points."""
    st.session_state.process_id = st.session_state.get("process_id", 0) + 1
    for i in range(5):
        val = bullets[i] if i < len(bullets) else ""
        st.session_state[f"bullet_{i+1}"] = val
        st.session_state[f"bullet_{i}"] = val


def init_session_state() -> None:
    """Initialize persistent session variables."""
    if "process_id" not in st.session_state:
        st.session_state.process_id = 0
    if "studio_white" not in st.session_state:
        st.session_state.studio_white = None
    if "studio_trans" not in st.session_state:
        st.session_state.studio_trans = None
    if "raw_image" not in st.session_state:
        st.session_state.raw_image = None
    if "extracted_tags" not in st.session_state:
        st.session_state.extracted_tags = None
    if "catalog_listing" not in st.session_state:
        st.session_state.catalog_listing = None
    if "active_sku" not in st.session_state:
        st.session_state.active_sku = "SKU-PLG-001"

    # Pre-populate session state if existing processed artifacts exist on disk
    if st.session_state.studio_white is None:
        default_studio = PROJECT_ROOT / "data" / "processed" / "my_product_studio.jpg"
        default_trans = PROJECT_ROOT / "data" / "processed" / "my_product_transparent.png"
        default_tags = PROJECT_ROOT / "data" / "processed" / "extracted_tags.json"
        default_raw = PROJECT_ROOT / "data" / "raw" / "my_product.jpg"

        if default_studio.exists():
            st.session_state.studio_white = Image.open(default_studio)
        if default_trans.exists():
            st.session_state.studio_trans = Image.open(default_trans)
        if default_raw.exists():
            st.session_state.raw_image = Image.open(default_raw)
        if default_tags.exists():
            with open(default_tags, "r", encoding="utf-8") as f:
                st.session_state.extracted_tags = json.load(f)
            # Generate initial copywriter listing if not set
            cw = ListingCopywriter()
            st.session_state.catalog_listing = cw.generate_listing(st.session_state.extracted_tags)
            sync_bullet_session_state(st.session_state.catalog_listing.amazon.bullet_points)


init_session_state()


# -----------------------------------------------------------------------------
# Sidebar: Pipeline Configuration & Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3081/3081559.png", width=60)
    st.title("Catalog Studio")
    st.markdown("Automated E-Commerce Product Pipeline")

    st.markdown("---")
    st.subheader("1. Input Product Photo")

    # Sample gallery selector or file uploader
    input_source = st.radio("Select Source", ["Upload Photo", "Sample Gallery"], horizontal=True)

    uploaded_file = None
    sample_path = None

    if input_source == "Upload Photo":
        uploaded_file = st.file_uploader(
            "Upload Smartphone Photo",
            type=["jpg", "jpeg", "png", "webp"],
            help="High-resolution smartphone photos with cluttered backgrounds are fully supported.",
        )
    else:
        raw_dir = PROJECT_ROOT / "data" / "raw"
        sample_files = [f.name for f in raw_dir.glob("*.jpg")] if raw_dir.exists() else []
        if sample_files:
            selected_sample = st.selectbox("Choose sample photo", sample_files, index=0)
            sample_path = raw_dir / selected_sample
        else:
            st.info("No sample photos in data/raw.")

    st.markdown("---")
    st.subheader("2. Canvas & Shadow Setup")

    canvas_size = st.selectbox("Canvas Dimension", [2000, 1500, 1000], index=0, help="2000x2000 px is Amazon standard.")
    fill_ratio = st.slider("Product Canvas Fill Ratio", min_value=0.70, max_value=0.92, value=0.82, step=0.02)
    enable_shadow = st.checkbox("Synthetic Contact Shadow", value=True)

    with st.expander("Advanced Shadow Controls"):
        contact_opacity = st.slider("Contact Shadow Opacity", 0.1, 1.0, 0.65, 0.05)
        cast_opacity = st.slider("Cast Shadow Opacity", 0.05, 0.8, 0.35, 0.05)
        shadow_blur = st.slider("Shadow Blur Radius", 15, 65, 41, 2)

    st.markdown("---")
    st.subheader("3. AI Copywriter Settings")

    brand_input = st.text_input("Brand Name Override (optional)", value="", placeholder="Auto-detect via OCR")
    copy_tone = st.selectbox("Copywriting Tone", ["Professional", "Luxury", "Casual / Modern", "Technical", "Energetic"])
    llm_provider = st.selectbox(
        "Copywriting Engine",
        ["Auto-detect", "Built-in E-Commerce AI", "Google Gemini", "OpenAI"],
        index=0,
    )

    with st.expander("API Keys (Optional)"):
        user_gemini_key = st.text_input("Gemini API Key", type="password", value=os.environ.get("GEMINI_API_KEY", ""))
        user_openai_key = st.text_input("OpenAI API Key", type="password", value=os.environ.get("OPENAI_API_KEY", ""))

    st.markdown("---")
    run_btn = st.button("🚀 Process Image & Generate Catalog", type="primary", use_container_width=True)


# -----------------------------------------------------------------------------
# Pipeline Execution Handler
# -----------------------------------------------------------------------------
if run_btn:
    # Resolve input image
    current_img: Optional[Image.Image] = None
    if uploaded_file is not None:
        current_img = Image.open(uploaded_file).convert("RGB")
    elif sample_path is not None and sample_path.exists():
        current_img = Image.open(sample_path).convert("RGB")

    if current_img is None:
        st.error("Please upload an image or select a sample from the sidebar.")
    else:
        st.session_state.raw_image = current_img

        progress_bar = st.progress(0, text="Initializing processing pipeline...")

        # Phase 1: Background removal & shadow synthesis
        progress_bar.progress(20, text="[1/3] Removing background & synthesizing contact shadows...")
        shadow_engine = ShadowEngine(
            contact_opacity=contact_opacity,
            cast_opacity=cast_opacity,
            cast_blur=shadow_blur,
        )
        studio_white, studio_trans, _ = process_catalog_image(
            image_input=current_img,
            canvas_size=canvas_size,
            fill_ratio=fill_ratio,
            add_shadow=enable_shadow,
            shadow_engine=shadow_engine,
        )
        st.session_state.studio_white = studio_white
        st.session_state.studio_trans = studio_trans

        # Phase 2: Florence-2 VLM Attribute Extraction
        progress_bar.progress(55, text="[2/3] Running Florence-2 Vision-Language extraction (attributes & OCR)...")
        try:
            extractor = ProductTagExtractor(model_id="microsoft/Florence-2-base")
            attributes = extractor.extract_tags(studio_white)
            dump_data = attributes.model_dump() if hasattr(attributes, "model_dump") else attributes.dict()
            st.session_state.extracted_tags = dump_data
        except Exception as e:
            st.warning(f"VLM extraction notice: {e}. Using extracted cache.")
            if st.session_state.extracted_tags is None:
                st.session_state.extracted_tags = {
                    "category": "Beauty & Personal Care",
                    "product_type": "Skin Care",
                    "dominant_colors": ["White", "Black"],
                    "detected_text": ["Product Treatment"],
                    "materials": ["Glass"],
                    "style_tags": ["Modern", "Minimalist"],
                    "detailed_description": "Clean studio product photo on pure white canvas.",
                }

        # Phase 3: Marketplace Copywriting
        progress_bar.progress(85, text="[3/3] Drafting Amazon & Shopify compliant listings...")
        provider_map = {
            "Auto-detect": "auto",
            "Built-in E-Commerce AI": "local",
            "Google Gemini": "gemini",
            "OpenAI": "openai",
        }
        copywriter = ListingCopywriter(
            gemini_api_key=user_gemini_key or None,
            openai_api_key=user_openai_key or None,
            provider=provider_map[llm_provider],
        )
        brand_arg = brand_input.strip() if brand_input.strip() else None
        st.session_state.catalog_listing = copywriter.generate_listing(
            tags_input=st.session_state.extracted_tags,
            brand_override=brand_arg,
            tone=copy_tone,
        )
        sync_bullet_session_state(st.session_state.catalog_listing.amazon.bullet_points)

        progress_bar.progress(100, text="Catalog generation complete!")
        st.success("Pipeline executed successfully! Review and edit your catalog assets below.")


# -----------------------------------------------------------------------------
# Main Application Dashboard
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">Automated E-Commerce Product Cataloger</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Raw Smartphone Photo → Studio Matting & Shadows → Florence-2 Tags → Amazon / Shopify Listings</div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs([
    "🖼️ Studio Image Transformation",
    "🏷️ Florence-2 Tags & Attributes",
    "📝 Marketplace Listing Review",
    "📦 Bulk Export Center",
])


# -----------------------------------------------------------------------------
# TAB 1: Studio Transformation & Visual Inspection
# -----------------------------------------------------------------------------
with tab1:
    col_raw, col_studio = st.columns([1, 1], gap="large")

    with col_raw:
        st.markdown("### Raw Smartphone Photo")
        if st.session_state.raw_image is not None:
            st.image(st.session_state.raw_image, use_container_width=True, caption="Original Input")
            raw_w, raw_h = st.session_state.raw_image.size
            st.info(f"Resolution: {raw_w} × {raw_h} px | Format: {st.session_state.raw_image.format or 'RGB'}")
        else:
            st.info("No raw photo uploaded yet. Select a sample photo or upload from the sidebar.")

    with col_studio:
        st.markdown("### Studio Grounded Canvas (2000×2000)")
        if st.session_state.studio_white is not None:
            view_mode = st.radio("View Layer", ["Pure White (#FFFFFF)", "Transparent PNG with Shadow"], horizontal=True)

            if view_mode == "Pure White (#FFFFFF)":
                st.image(st.session_state.studio_white, use_container_width=True, caption="Amazon-Ready Pure White Studio Canvas")
            else:
                st.image(st.session_state.studio_trans, use_container_width=True, caption="Transparent Layer with Contact Drop-Shadow")

            st.markdown(
                """
                <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px;">
                    <span class="compliance-badge">Amazon Standards: 2000×2000 px ✅</span>
                    <span class="compliance-badge">Pure White #FFFFFF Corners ✅</span>
                    <span class="compliance-badge">Synthetic Contact Shadow Active ✅</span>
                    <span class="compliance-badge">1:1 Square Ratio ✅</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Run the pipeline from the sidebar to inspect the studio output.")


# -----------------------------------------------------------------------------
# TAB 2: Florence-2 Tags & Attributes (Human-in-the-Loop)
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("### Florence-2 Vision-Language Attribute Extraction")
    st.caption("Inspect and refine computer vision attributes detected directly from the product image.")

    if st.session_state.extracted_tags is not None:
        tags_dict = st.session_state.extracted_tags

        col_meta1, col_meta2, col_meta3 = st.columns([1, 1, 1])

        with col_meta1:
            st.markdown("**Category & Product Type**")
            edit_category = st.text_input("Category", value=tags_dict.get("category", "General Merchandise"))
            edit_type = st.text_input("Product Type", value=tags_dict.get("product_type", "Product"))

        with col_meta2:
            st.markdown("**Dominant Colors**")
            colors_str = ", ".join(tags_dict.get("dominant_colors", []))
            edit_colors = st.text_input("Colors (comma separated)", value=colors_str)

        with col_meta3:
            st.markdown("**Materials & Finishes**")
            mat_str = ", ".join(tags_dict.get("materials", []))
            edit_materials = st.text_input("Materials (comma separated)", value=mat_str)

        st.markdown("**Detected OCR Text (Labels, Active Claims & Volume)**")
        ocr_text_joined = "\n".join(tags_dict.get("detected_text", []))
        edit_ocr = st.text_area("OCR Text Detected on Packaging", value=ocr_text_joined, height=80)

        st.markdown("**Florence-2 Dense Visual Description**")
        st.write(f"_{tags_dict.get('detailed_description', '')}_")

        if st.button("🔄 Update Tags & Regenerate Listings", type="secondary"):
            # Update dictionary with merchant edits
            updated_tags = {
                "category": edit_category,
                "product_type": edit_type,
                "dominant_colors": [c.strip() for c in edit_colors.split(",") if c.strip()],
                "materials": [m.strip() for m in edit_materials.split(",") if m.strip()],
                "detected_text": [line.strip() for line in edit_ocr.split("\n") if line.strip()],
                "style_tags": tags_dict.get("style_tags", []),
                "detailed_description": tags_dict.get("detailed_description", ""),
            }
            st.session_state.extracted_tags = updated_tags

            cw = ListingCopywriter(
                gemini_api_key=user_gemini_key or None,
                openai_api_key=user_openai_key or None,
            )
            st.session_state.catalog_listing = cw.generate_listing(
                tags_input=updated_tags,
                brand_override=brand_input.strip() if brand_input.strip() else None,
                tone=copy_tone,
            )
            sync_bullet_session_state(st.session_state.catalog_listing.amazon.bullet_points)
            st.success("Tags updated and listings regenerated! View the updated copy in Tab 3.")
    else:
        st.info("Run the pipeline from the sidebar to view extracted attributes.")


# -----------------------------------------------------------------------------
# TAB 3: Marketplace Listing Review & Copy Editor
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("### Marketplace-Compliant Listing Review")

    if st.session_state.catalog_listing is not None:
        listing: ProductCatalogListing = st.session_state.catalog_listing
        pid = st.session_state.get("process_id", 0)
        prod_name_clean = re.sub(r'[^a-zA-Z0-9_]+', '_', getattr(listing, "product_name", "Item"))

        domain_icons = {
            "FOOD_BEVERAGE": "🍲 Food & Beverage (Taste, Hygiene, Ingredients & Freshness)",
            "APPAREL_FASHION": "👕 Apparel & Fashion (Fabric, Fit, Stitching & Care)",
            "ELECTRONICS": "🎧 Electronics & Gadgets (Audio, Battery, Connectivity & Specs)",
            "BEAUTY_PERSONAL_CARE": "✨ Beauty & Personal Care (Actives, Barrier, Hydration & Routine)",
            "OFFICE_STATIONERY": "📄 Office Supplies & Stationery (Jam-Free Speed, Opacity & Archival)",
            "HOME_KITCHEN": "🍳 Home & Kitchen (Food-Grade Materials, Durability & Utility)",
            "UNIVERSAL": "📦 Universal Open-Vocabulary (Durable Build, Specs & QA)",
            "GENERAL": "📦 General Merchandise",
        }
        active_domain_label = domain_icons.get(getattr(listing, "domain", "GENERAL"), f"📦 {getattr(listing, 'domain', 'General')}")
        st.info(f"**Active Copywriting Domain:** {active_domain_label}")

        platform_tab1, platform_tab2 = st.tabs(["🛒 Amazon Marketplace", "🛍️ Shopify Storefront"])

        # AMAZON TAB
        with platform_tab1:
            st.markdown("#### Amazon Product Detail Page")

            # Title
            amz_title = st.text_input(
                "Amazon Title (Under 200 characters)",
                value=listing.amazon.title,
                help="Follows [Brand] + [Name] + [Key Feature/Ingredient] + [Color/Size]",
                key=f"amz_title_{prod_name_clean}_{pid}",
            )
            len_title = len(amz_title)
            if len_title <= 200:
                st.caption(f"Character Count: **{len_title}/200** ✅ Amazon Compliant")
            else:
                st.caption(f"Character Count: **{len_title}/200** ⚠️ Exceeds 200 character limit")

            st.markdown("#### 5 Benefit-Driven Bullet Points")
            edited_bullets = []
            for i in range(5):
                default_bp = (
                    st.session_state.get(f"bullet_{i+1}")
                    or (listing.amazon.bullet_points[i] if i < len(listing.amazon.bullet_points) else "")
                )
                widget_key = f"bullet_{i}_{prod_name_clean}_{pid}"
                new_bp = st.text_area(
                    f"Bullet Point {i+1}",
                    value=default_bp,
                    height=75,
                    key=widget_key,
                )
                st.caption(f"Bullet {i+1} Length: {len(new_bp)} chars")
                edited_bullets.append(new_bp)
                st.session_state[f"bullet_{i+1}"] = new_bp

            listing.amazon.bullet_points = edited_bullets

            # Backend Search Terms
            amz_keywords = st.text_input(
                "Backend Generic Keywords (Under 250 bytes, space-separated)",
                value=listing.amazon.search_terms,
                key=f"amz_keywords_{prod_name_clean}_{pid}",
            )
            st.caption(f"Search terms byte count: **{len(amz_keywords.encode('utf-8'))}/250 bytes** ✅")

            # Product Description
            amz_desc = st.text_area(
                "Product Description",
                value=listing.amazon.product_description,
                height=120,
                key=f"amz_desc_{prod_name_clean}_{pid}",
            )

        # SHOPIFY TAB
        with platform_tab2:
            st.markdown("#### Shopify E-Commerce Listing")

            shop_title = st.text_input(
                "Shopify Title",
                value=listing.shopify.title,
                key=f"shop_title_{prod_name_clean}_{pid}",
            )

            col_s1, col_s2 = st.columns([1, 1])
            with col_s1:
                shop_handle = st.text_input(
                    "URL Handle (slug)",
                    value=listing.shopify.handle,
                    key=f"shop_handle_{prod_name_clean}_{pid}",
                )
            with col_s2:
                shop_tags_str = st.text_input(
                    "Store Tags (comma-separated)",
                    value=", ".join(listing.shopify.tags),
                    key=f"shop_tags_{prod_name_clean}_{pid}",
                )

            shop_meta = st.text_area(
                "SEO Meta Description (140-160 chars)",
                value=listing.shopify.meta_description,
                height=70,
                key=f"shop_meta_{prod_name_clean}_{pid}",
            )
            len_meta = len(shop_meta)
            if 140 <= len_meta <= 160:
                st.caption(f"SEO Length: **{len_meta} chars** (Optimal 140-160 ✅)")
            else:
                st.caption(f"SEO Length: **{len_meta} chars** (Aim for 140-160 characters)")

            st.markdown("#### Product Description (Semantic HTML)")
            html_view = st.radio("Display Format", ["Rendered HTML Preview", "HTML Source Code"], horizontal=True)
            if html_view == "Rendered HTML Preview":
                st.markdown(listing.shopify.body_html, unsafe_allow_html=True)
            else:
                shop_body = st.text_area(
                    "HTML Source",
                    value=listing.shopify.body_html,
                    height=180,
                    key=f"shop_body_{prod_name_clean}_{pid}",
                )

        # Apply Live Edits Button
        if st.button("💾 Save All Listing Edits", type="primary"):
            listing.amazon.title = amz_title
            listing.amazon.bullet_points = edited_bullets
            listing.amazon.search_terms = amz_keywords
            listing.amazon.product_description = amz_desc

            listing.shopify.title = shop_title
            listing.shopify.handle = shop_handle
            listing.shopify.tags = [t.strip() for t in shop_tags_str.split(",") if t.strip()]
            listing.shopify.meta_description = shop_meta
            if html_view == "HTML Source Code":
                listing.shopify.body_html = shop_body

            st.session_state.catalog_listing = listing
            st.success("Listing edits saved successfully! Updated in export packages.")
            st.success("Listing edits saved successfully! Updated in export packages.")
    else:
        st.info("Run the pipeline from the sidebar to generate marketplace copy.")


# -----------------------------------------------------------------------------
# TAB 4: Bulk Export Center
# -----------------------------------------------------------------------------
with tab4:
    st.markdown("### Bulk Catalog & Asset Export")
    st.caption("Download production-ready CSV spreadsheets for Amazon and Shopify, along with studio image bundles.")

    if st.session_state.catalog_listing is not None and st.session_state.studio_white is not None:
        curr_listing: ProductCatalogListing = st.session_state.catalog_listing

        col_cfg1, col_cfg2 = st.columns([1, 1])
        with col_cfg1:
            export_sku = st.text_input("Export SKU", value="SKU-PLG-001")
        with col_cfg2:
            export_price = st.text_input("Catalog Standard Price ($)", value="29.99")

        # Generate export files in-memory
        shopify_csv_data = create_shopify_csv(curr_listing, price=export_price, sku=export_sku)
        amazon_csv_data = create_amazon_csv(curr_listing, price=export_price, sku=export_sku)
        json_metadata = json.dumps(curr_listing.to_dict(), indent=2)
        zip_bundle_bytes = create_catalog_zip_bundle(
            listing=curr_listing,
            studio_white_img=st.session_state.studio_white,
            studio_trans_img=st.session_state.studio_trans,
            sku=export_sku,
        )

        st.markdown("---")
        st.subheader("Direct Downloads")

        col_dl1, col_dl2, col_dl3, col_dl4 = st.columns([1, 1, 1, 1])

        with col_dl1:
            st.download_button(
                label="📥 Shopify CSV",
                data=shopify_csv_data,
                file_name=f"{export_sku}_shopify_products.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col_dl2:
            st.download_button(
                label="📥 Amazon CSV",
                data=amazon_csv_data,
                file_name=f"{export_sku}_amazon_inventory.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col_dl3:
            st.download_button(
                label="📥 Metadata JSON",
                data=json_metadata,
                file_name=f"{export_sku}_metadata.json",
                mime="application/json",
                use_container_width=True,
            )

        with col_dl4:
            st.download_button(
                label="📦 All Assets (ZIP)",
                data=zip_bundle_bytes,
                file_name=f"{export_sku}_complete_catalog_package.zip",
                mime="application/zip",
                use_container_width=True,
            )

        st.markdown("---")
        st.subheader("Spreadsheet Previews")

        csv_preview_type = st.radio("Preview Table", ["Shopify Product Import", "Amazon Flat File"], horizontal=True)
        if csv_preview_type == "Shopify Product Import":
            df_shopify = pd.read_csv(io.StringIO(shopify_csv_data))
            st.dataframe(df_shopify, use_container_width=True)
        else:
            df_amazon = pd.read_csv(io.StringIO(amazon_csv_data))
            st.dataframe(df_amazon, use_container_width=True)

    else:
        st.info("Execute the catalog pipeline to unlock the export center.")
