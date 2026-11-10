"""Listing Copywriter engine for marketplace-compliant Amazon and Shopify copy.

Features dynamic domain-routing (Food & Beverage, Apparel & Fashion, Electronics,
Beauty & Personal Care, Home & Kitchen) and a robust, open-vocabulary universal fallback
engine (`generate_universal_fallback`) for infinite product variety (e.g. JK Paper,
Screwdrivers, Stationery, Hardware, Industrial Goods).
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AmazonListing(BaseModel):
    """Amazon-compliant product listing metadata."""

    title: str = Field(
        ...,
        description="Amazon title under 200 characters following [Brand] + [Name] + [Key Feature] + [Size/Color]",
    )
    bullet_points: List[str] = Field(
        ...,
        description="Exactly 5 benefit-driven bullet points starting with CAPITALIZED HEADLINE: ...",
    )
    search_terms: str = Field(
        ...,
        description="Backend generic search keywords (under 250 bytes, space-separated, no punctuation)",
    )
    product_description: str = Field(
        ...,
        description="Comprehensive, readable product description paragraph for the detail page",
    )

    @property
    def bullet_point1(self) -> str:
        """First benefit-driven bullet point."""
        return self.bullet_points[0] if len(self.bullet_points) > 0 else ""

    @property
    def bullet_point2(self) -> str:
        """Second benefit-driven bullet point."""
        return self.bullet_points[1] if len(self.bullet_points) > 1 else ""

    @property
    def bullet_point3(self) -> str:
        """Third benefit-driven bullet point."""
        return self.bullet_points[2] if len(self.bullet_points) > 2 else ""

    @property
    def bullet_point4(self) -> str:
        """Fourth benefit-driven bullet point."""
        return self.bullet_points[3] if len(self.bullet_points) > 3 else ""

    @property
    def bullet_point5(self) -> str:
        """Fifth benefit-driven bullet point."""
        return self.bullet_points[4] if len(self.bullet_points) > 4 else ""


class ShopifyListing(BaseModel):
    """Shopify-compliant product listing metadata."""

    title: str = Field(
        ...,
        description="Customer-facing catchy product title",
    )
    body_html: str = Field(
        ...,
        description="Structured semantic HTML description with <h3>, <ul>, and <p> tags",
    )
    tags: List[str] = Field(
        ...,
        description="List of taxonomy and search tags for store organization",
    )
    meta_description: str = Field(
        ...,
        description="SEO meta description between 140-160 characters for search engines",
    )
    handle: str = Field(
        ...,
        description="URL-friendly kebab-case slug for the product page",
    )


class ProductCatalogListing(BaseModel):
    """Combined marketplace listing catalog payload."""

    brand: str = Field(default="Premium")
    product_name: str = Field(default="Item")
    category: str = Field(default="General Merchandise")
    domain: str = Field(default="UNIVERSAL")
    amazon: AmazonListing
    shopify: ShopifyListing
    tags_source: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Support serialization across Pydantic v1 and v2."""
        if hasattr(self, "model_dump"):
            return self.model_dump()
        return self.dict()


# ==============================================================================
# Robust, Open-Vocabulary Universal Fallback Engine for Infinite Product Variety
# ==============================================================================

def generate_universal_fallback(
    tags: Union[str, Path, Dict[str, Any]],
    brand_override: Optional[str] = None,
    tone: str = "Professional",
) -> ProductCatalogListing:
    """Generate universal marketplace copy for any product outside predefined domains.
    
    Safely extracts variables using .get() with safe defaults:
      - brand (default: 'Premium')
      - product_type (default: 'Item')
      - materials (default: 'High-Quality Materials')
      - colors (default: 'Standard Finish')
      - ocr_text (default: 'Verified Specifications')
      
    Constructs Amazon Title via strict string interpolation:
      [Brand] [Product Type] - [Materials] in [Colors] | Premium Utility
      (safely truncated under 200 characters).
      
    Constructs 5 Amazon Bullets programmatically:
      - Bullet 1 (Build): Materials and durability
      - Bullet 2 (Utility): Versatile application of product_type
      - Bullet 3 (Aesthetics): Colors and design finish
      - Bullet 4 (Specs): Dynamic slice of ocr_text as verified manufacturer specs
      - Bullet 5 (Quality): Generic quality assurance and packaging safety
      
    Constructs Shopify HTML:
      Semantic HTML block (<h3>, <p>, <ul><li>) interpolating these same tags.
    """
    # 1. Normalize tags input to a dictionary
    tags_dict: Dict[str, Any] = {}
    if isinstance(tags, (str, Path)):
        p = Path(tags)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                tags_dict = json.load(f)
        else:
            tags_dict = {}
    elif isinstance(tags, dict):
        tags_dict = tags
    elif hasattr(tags, "dict"):
        tags_dict = tags.dict()
    elif hasattr(tags, "model_dump"):
        tags_dict = tags.model_dump()

    # 2. Extract Brand (default: 'Premium')
    if brand_override and str(brand_override).strip():
        brand = str(brand_override).strip()
    else:
        raw_brand = tags_dict.get("brand")
        if raw_brand and str(raw_brand).strip() and str(raw_brand).strip().lower() not in ("unknown", "none", "null"):
            brand = str(raw_brand).strip()
        else:
            # Check detected text for prominent brand name or fallback to 'Premium'
            dt = tags_dict.get("detected_text") or tags_dict.get("ocr_text")
            brand = "Premium"
            if isinstance(dt, list) and dt:
                combined_dt = " ".join([str(x) for x in dt])
                for kb in ["JK Paper", "Stanley", "Milton", "Cello", "Bikaji", "Sony", "Nike", "Adidas"]:
                    if re.search(rf"\b{re.escape(kb)}\b", combined_dt, re.IGNORECASE):
                        brand = kb
                        break
                if brand == "Premium" and dt[0].split():
                    first_token = dt[0].split()[0]
                    clean_tok = re.sub(r'[^a-zA-Z0-9]', '', first_token)
                    if len(clean_tok) >= 2 and clean_tok.lower() not in ("verified", "specifications", "sample", "unknown"):
                        brand = clean_tok.title()
            elif isinstance(dt, str) and dt.strip():
                for kb in ["JK Paper", "Stanley", "Milton", "Cello", "Bikaji", "Sony", "Nike", "Adidas"]:
                    if re.search(rf"\b{re.escape(kb)}\b", dt, re.IGNORECASE):
                        brand = kb
                        break
                if brand == "Premium" and dt.strip().split():
                    first_token = dt.strip().split()[0]
                    clean_tok = re.sub(r'[^a-zA-Z0-9]', '', first_token)
                    if len(clean_tok) >= 2 and clean_tok.lower() not in ("verified", "specifications", "sample", "unknown"):
                        brand = clean_tok.title()

    if not brand:
        brand = "Premium"

    # 3. Extract Product Type (default: 'Item')
    raw_type = tags_dict.get("product_type") or tags_dict.get("type")
    if raw_type and str(raw_type).strip() and str(raw_type).strip().lower() not in ("product", "unknown", "item", "none", "null"):
        product_type = str(raw_type).strip()
    else:
        product_type = "Item"

    # 4. Extract Materials (default: 'High-Quality Materials')
    raw_mat = tags_dict.get("materials", "High-Quality Materials")
    if isinstance(raw_mat, list) and raw_mat:
        clean_mats = [str(m).strip() for m in raw_mat if str(m).strip()]
        materials = ", ".join(clean_mats) if clean_mats else "High-Quality Materials"
    elif isinstance(raw_mat, str) and raw_mat.strip():
        materials = raw_mat.strip()
    else:
        materials = "High-Quality Materials"

    if not materials or materials.lower() in ("none", "unknown", "null"):
        materials = "High-Quality Materials"

    # 5. Extract Colors (default: 'Standard Finish')
    raw_colors = tags_dict.get("colors") or tags_dict.get("dominant_colors", "Standard Finish")
    if isinstance(raw_colors, list) and raw_colors:
        clean_cols = [str(c).strip() for c in raw_colors if str(c).strip()]
        colors = ", ".join(clean_cols) if clean_cols else "Standard Finish"
    elif isinstance(raw_colors, str) and raw_colors.strip():
        colors = raw_colors.strip()
    else:
        colors = "Standard Finish"

    if not colors or colors.lower() in ("none", "unknown", "null"):
        colors = "Standard Finish"

    # 6. Extract OCR Text (default: 'Verified Specifications')
    raw_ocr = tags_dict.get("ocr_text") or tags_dict.get("detected_text") or tags_dict.get("ocr", "Verified Specifications")
    if isinstance(raw_ocr, list) and raw_ocr:
        clean_lines = [str(line).strip() for line in raw_ocr if str(line).strip()]
        ocr_text = " | ".join(clean_lines) if clean_lines else "Verified Specifications"
    elif isinstance(raw_ocr, str) and raw_ocr.strip():
        ocr_text = raw_ocr.strip()
    else:
        ocr_text = "Verified Specifications"

    if not ocr_text:
        ocr_text = "Verified Specifications"

    # Clean slice of ocr_text for specs
    clean_specs = (ocr_text[:110].rsplit(" ", 1)[0] + "...") if len(ocr_text) > 110 else ocr_text

    # -------------------------------------------------------------
    # Rule 1: Amazon Title
    # Strict interpolation: [Brand] [Product Type] - [Materials] in [Colors] | Premium Utility
    # Safe truncation under 200 characters
    # -------------------------------------------------------------
    amz_title = f"{brand} {product_type} - {materials} in {colors} | Premium Utility"
    if len(amz_title) > 200:
        amz_title = amz_title[:196].rsplit(" ", 1)[0] + "..."

    # -------------------------------------------------------------
    # Rule 2: 5 Amazon Bullets (Build, Utility, Aesthetics, Specs, Quality)
    # -------------------------------------------------------------
    bullets = [
        # Bullet 1 (Build): Materials and durability
        (
            f"DURABLE HIGH-GRADE BUILD: Precision-engineered from {materials} to deliver superior structural "
            f"resilience, wear resistance, and long-lasting durability for demanding everyday and professional use."
        ),
        # Bullet 2 (Utility): Versatile application of product_type
        (
            f"VERSATILE MULTI-PURPOSE UTILITY: Optimized for high efficiency and seamless integration across diverse "
            f"commercial, industrial, and household environments as a dependable, heavy-duty {product_type}."
        ),
        # Bullet 3 (Aesthetics): Colors and design finish
        (
            f"REFINED AESTHETIC FINISH: Features an elegant {colors} exterior that provides contemporary visual "
            f"appeal, ergonomic tactile handling, and modern presentation in any workplace or personal setup."
        ),
        # Bullet 4 (Specs): Dynamic slice of ocr_text as verified manufacturer specs
        (
            f"AUTHENTICATED MANUFACTURER SPECIFICATIONS: Built in accordance with verified production parameters: "
            f"{clean_specs}."
        ),
        # Bullet 5 (Quality): Generic quality assurance and packaging safety
        (
            f"STRINGENT QUALITY ASSURANCE & PACKAGING: Subjected to multi-point quality control inspections and packed "
            f"in protective packaging to ensure complete transit safety and dependable out-of-the-box readiness."
        ),
    ]

    # Backend Search Terms (Under 250 bytes, space-separated, no commas)
    raw_search = f"{brand.lower()} {product_type.lower()} utility durable specifications {materials.lower()} {colors.lower()} commercial industrial daily essential"
    search_terms = " ".join(dict.fromkeys(re.sub(r'[^a-z0-9\s]', '', raw_search).split()))[:240]

    # Amazon Product Description
    amz_desc = (
        f"The {brand} {product_type} is engineered to deliver exceptional durability, versatile utility, and dependable "
        f"everyday performance. Crafted from premium {materials} and presented in a sleek {colors} finish, this "
        f"item meets rigorous B2B and B2C standards. With confirmed specifications ({clean_specs}) and strict quality "
        f"assurance, it provides lasting value and reliable operation for commercial, office, and home environments."
    )

    # -------------------------------------------------------------
    # Rule 3: Shopify HTML
    # Semantic HTML block (<h3>, <p>, <ul><li>) interpolating these same tags
    # -------------------------------------------------------------
    shopify_title = f"{brand} {product_type} - {materials} ({colors})"
    shopify_html = (
        f"<h3>Product Overview</h3>\n"
        f"<p>The <strong>{brand} {product_type}</strong> provides dependable utility, robust performance, "
        f"and long-lasting reliability. Engineered with <strong>{materials}</strong> and finished in <strong>{colors}</strong>, "
        f"it meets the highest standards for commercial, professional, and personal demands.</p>\n\n"
        f"<h3>Key Features & Highlights</h3>\n"
        f"<ul>\n"
        f"  <li><strong>Engineered Durability:</strong> Manufactured from {materials} for outstanding structural strength and wear resistance.</li>\n"
        f"  <li><strong>Versatile Application:</strong> Designed to provide exceptional utility for diverse {product_type} tasks.</li>\n"
        f"  <li><strong>Design & Finish:</strong> Clean {colors} aesthetic engineered for modern, ergonomic handling.</li>\n"
        f"  <li><strong>Verified Specifications:</strong> Validated against confirmed manufacturer standards: {clean_specs}.</li>\n"
        f"  <li><strong>Quality Guaranteed:</strong> Factory inspected and safely packaged for secure transit.</li>\n"
        f"</ul>\n\n"
        f"<h3>Product Specifications</h3>\n"
        f"<p><strong>Brand:</strong> {brand} | <strong>Product:</strong> {product_type} | <strong>Material:</strong> {materials} | <strong>Finish:</strong> {colors}</p>"
    )

    # Shopify Meta Description (140-160 chars)
    meta_desc = f"Discover the {brand} {product_type}. Crafted from {materials} in {colors} for reliable performance and lasting durability. Shop online now!"
    if len(meta_desc) > 160:
        meta_desc = meta_desc[:157] + "..."
    elif len(meta_desc) < 140:
        meta_desc += " Fast shipping with quality guarantee."

    handle = re.sub(r'[^a-z0-9]+', '-', shopify_title.lower()).strip('-')

    category_label = tags_dict.get("category") or "General Merchandise"

    return ProductCatalogListing(
        brand=brand,
        product_name=product_type,
        category=category_label,
        domain="UNIVERSAL",
        amazon=AmazonListing(
            title=amz_title,
            bullet_points=bullets,
            search_terms=search_terms,
            product_description=amz_desc,
        ),
        shopify=ShopifyListing(
            title=shopify_title,
            body_html=shopify_html,
            tags=[category_label, product_type, brand, "Premium Utility"],
            meta_description=meta_desc,
            handle=handle,
        ),
        tags_source=tags_dict,
    )


# ==============================================================================
# Listing Copywriter Class
# ==============================================================================

class ListingCopywriter:
    """Orchestrates marketplace listing generation with domain routing & universal fallback."""

    # Explicitly supported predefined domains
    PREDEFINED_DOMAINS = {
        "FOOD_BEVERAGE",
        "APPAREL_FASHION",
        "ELECTRONICS",
        "BEAUTY_PERSONAL_CARE",
        "OFFICE_STATIONERY",
    }

    KNOWN_BRANDS = [
        # Food & Grocery
        "Bikaji", "Haldiram", "Cadbury", "Bournvita", "Nestle", "Amul", "Britannia",
        "Parle", "MTR", "Tata", "Lays", "Doritos", "Kellogg", "Kurkure", "Everest",
        "MDH", "Horlicks", "Boost", "Complan", "Dabur", "Patanjali", "Kissan", "Maggi",
        # Beauty & Personal Care
        "Pilgrim", "Vanadinite", "Cetaphil", "Cerave", "Nivea", "Dove", "Neutrogena",
        "Minimalist", "L'Oreal", "The Ordinary", "Garnier", "Biotique", "Mamaearth",
        "Himalaya", "Plum", "Derma Co", "Vaseline", "Ponds", "Tresemme",
        # Electronics
        "Sony", "Apple", "Samsung", "Boat", "JBL", "Bose", "Anker", "Noise", "OnePlus",
        "Sennheiser", "Logitech", "Boult", "Realme", "Xiaomi", "Zebronics", "Portronics",
        # Apparel & Footwear
        "Nike", "Adidas", "Puma", "Levi's", "Zara", "H&M", "Under Armour", "Allen Solly",
        "Peter England", "Van Heusen", "Arrow", "Clarks", "Skechers", "Bata", "Woodland",
        # Home & Office
        "Milton", "Cello", "Prestige", "Hawkins", "Borosil", "Tupperware", "JK Paper", "Stanley",
    ]

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        provider: str = "auto",  # 'auto', 'gemini', 'openai', 'local'
    ) -> None:
        """Initialize copywriter.
        
        Args:
            gemini_api_key: Optional Google Gemini API Key.
            openai_api_key: Optional OpenAI API Key.
            provider: Generation provider ('auto', 'gemini', 'openai', 'local').
        """
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.provider = provider

    def generate_listing(
        self,
        tags_input: Union[str, Path, Dict[str, Any]],
        brand_override: Optional[str] = None,
        tone: str = "Professional",
    ) -> ProductCatalogListing:
        """Generate marketplace copy from extracted VLM visual attributes and OCR.
        
        Routes to predefined domain engines (Food, Apparel, Electronics, Beauty) if matched;
        otherwise safely routes to `generate_universal_fallback` for any other product category.
        """
        tags = self._load_tags(tags_input)
        inferred = self._infer_product_context(tags, brand_override)
        domain = inferred["domain"]

        # If domain falls outside predefined domains, immediately route to universal fallback!
        # Do NOT let it default to Beauty/Skincare.
        if domain not in self.PREDEFINED_DOMAINS and domain != "HOME_KITCHEN":
            logger.info(f"Category '{inferred.get('category')}' falls outside predefined domains. Routing to universal fallback.")
            return generate_universal_fallback(tags, brand_override=inferred.get("brand"), tone=tone)

        # Attempt LLM generation if API keys are configured and requested
        if self.provider in ("auto", "gemini") and self.gemini_api_key:
            try:
                logger.info("Generating listings via Google Gemini API...")
                listing = self._generate_with_gemini(tags, inferred, tone)
                if listing:
                    return listing
            except Exception as e:
                logger.warning(f"Gemini generation failed ({e}). Falling back to local engine.")

        if self.provider in ("auto", "openai") and self.openai_api_key:
            try:
                logger.info("Generating listings via OpenAI API...")
                listing = self._generate_with_openai(tags, inferred, tone)
                if listing:
                    return listing
            except Exception as e:
                logger.warning(f"OpenAI generation failed ({e}). Falling back to local engine.")

        # Default / Fallback: Dynamic domain-routed offline copywriting engine
        logger.info(f"Generating listings using dynamic domain-routed engine (Domain: {domain})...")
        return self._generate_with_local_engine(tags, inferred, tone)

    def generate_universal_fallback(
        self,
        tags: Union[str, Path, Dict[str, Any]],
        brand_override: Optional[str] = None,
        tone: str = "Professional",
    ) -> ProductCatalogListing:
        """Method wrapper for generate_universal_fallback."""
        return generate_universal_fallback(tags, brand_override=brand_override, tone=tone)

    def generate_amazon_listing(
        self,
        tags_input: Union[str, Path, Dict[str, Any]],
        brand_override: Optional[str] = None,
        tone: str = "Professional",
    ) -> AmazonListing:
        """Generate strictly Amazon-compliant listing with dynamically computed bullet points.
        
        Guarantees that bullet_point1 through bullet_point5 are strictly assigned from
        dynamically computed domain_bullets (or universal_bullets) for the current product,
        and that no static/default Bournvita list is ever returned.
        """
        catalog = self.generate_listing(tags_input, brand_override=brand_override, tone=tone)
        return catalog.amazon

    def generate_from_file(
        self,
        file_path: Union[str, Path] = "data/processed/extracted_tags.json",
        output_path: Union[str, Path] = "data/processed/generated_listings.json",
        brand_override: Optional[str] = None,
        tone: str = "Professional",
    ) -> ProductCatalogListing:
        """Load tags from file, generate listings, and save output JSON."""
        listing = self.generate_listing(file_path, brand_override=brand_override, tone=tone)

        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(listing.to_dict(), f, indent=2)

        logger.info(f"Generated listings saved to: {out_file}")
        return listing

    # --------------------------------------------------------------------------
    # Context Inference & Domain Routing
    # --------------------------------------------------------------------------

    def _load_tags(self, tags_input: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        """Normalize tags input into a Python dictionary."""
        if isinstance(tags_input, (str, Path)):
            p = Path(tags_input)
            if not p.exists():
                raise FileNotFoundError(f"Tags file not found: {p}")
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        elif isinstance(tags_input, dict):
            return tags_input
        elif hasattr(tags_input, "dict"):
            return tags_input.dict()
        elif hasattr(tags_input, "model_dump"):
            return tags_input.model_dump()
        else:
            raise ValueError(f"Unsupported tags input type: {type(tags_input)}")

    def _detect_domain(self, category: Optional[str], product_type: Optional[str] = None, ocr_text: str = "", caption: str = "") -> str:
        """Classify the product into a specific copywriting domain strictly based on extracted Category.
        
        Strict Category-to-Domain Routing:
          - Beauty & Personal Care -> BEAUTY_PERSONAL_CARE (skincare templates)
          - Food & Beverage -> FOOD_BEVERAGE (food/taste/hygiene templates)
          - Office Supplies & Stationery -> OFFICE_STATIONERY (paper/printing templates)
          - Apparel & Fashion -> APPAREL_FASHION (fabric & fit templates)
          - Electronics -> ELECTRONICS (tech specs & connectivity templates)
          - Home & Kitchen -> HOME_KITCHEN
          - Otherwise -> UNIVERSAL (open-vocabulary universal fallback)
        """
        cat_clean = (category or "").strip().lower()

        # 1. Beauty & Personal Care
        if any(k in cat_clean for k in ["beauty", "personal care", "skincare", "cosmetic", "haircare"]):
            return "BEAUTY_PERSONAL_CARE"

        # 2. Food & Beverage
        if any(k in cat_clean for k in ["food", "beverage", "grocery", "gourmet", "sweet", "snack"]):
            return "FOOD_BEVERAGE"

        # 3. Office Supplies & Stationery
        if any(k in cat_clean for k in ["office supplies", "stationery", "office"]):
            return "OFFICE_STATIONERY"

        # 4. Apparel & Fashion
        if any(k in cat_clean for k in ["apparel", "fashion", "clothing", "footwear"]):
            return "APPAREL_FASHION"

        # 5. Electronics
        if any(k in cat_clean for k in ["electronic", "gadget", "audio", "computer"]):
            return "ELECTRONICS"

        # 6. Home & Kitchen
        if any(k in cat_clean for k in ["kitchen", "cookware", "home"]):
            return "HOME_KITCHEN"

        # Otherwise: Fall back to generate_universal_fallback
        return "UNIVERSAL"

    def _infer_product_context(self, tags: Dict[str, Any], brand_override: Optional[str] = None) -> Dict[str, Any]:
        """Extract brand, product title candidate, size, domain, and claims from OCR & caption."""
        category = tags.get("category", "")
        product_type = tags.get("product_type", "Product")
        detected_text = " ".join(tags.get("detected_text", []))
        caption = tags.get("detailed_description", "")
        combined_text = f"{detected_text} {caption}"

        # 1. Detect Domain
        domain = self._detect_domain(category, product_type, detected_text, caption)

        # 2. Detect Brand
        brand = brand_override or tags.get("brand")
        if not brand:
            for kb in self.KNOWN_BRANDS:
                pattern = rf"(?:\b|_){re.escape(kb)}"
                if re.search(pattern, combined_text, re.IGNORECASE) or kb.lower() in combined_text.lower():
                    brand = kb
                    break

            if not brand:
                tokens = re.findall(r'[A-Za-z]{3,15}', detected_text)
                blacklist = {
                    "natural", "advanced", "moisturizing", "deep", "nourish", "protect",
                    "pure", "milk", "sweet", "sweets", "traditional", "indian", "rich",
                    "strength", "nutrients", "added", "scientifically", "designed",
                }
                for t in tokens:
                    if t.lower() not in blacklist and t.istitle():
                        brand = t
                        break
                    elif t.isupper() and len(t) >= 4 and t.lower() not in blacklist:
                        brand = t.title()
                        break

            if not brand or len(brand) > 20:
                brand = "Premium"

        # 3. Detect Size / Weight / Packaging
        size_match = re.search(
            r'(\b\d+(?:\.\d+)?\s*(?:kg|g|gm|gms|ml|fl\.?\s*oz\.?|ltr|litres?|pack|pcs|pieces?)\b)',
            combined_text,
            re.IGNORECASE,
        )
        size_str = size_match.group(1).strip() if size_match else ""

        packaging_format = ""
        check_targets = [product_type, " ".join(tags.get("materials", [])), combined_text]
        for p_fmt in ["Jar", "Tin", "Can", "Bottle", "Box", "Pouch", "Pack", "Tube"]:
            if any(re.search(rf"\b{p_fmt}\b", str(tgt), re.IGNORECASE) for tgt in check_targets):
                packaging_format = p_fmt
                break

        product_name = self._refine_product_name(product_type, domain, combined_text, packaging_format)
        claims = self._extract_domain_claims(domain, combined_text)

        colors = tags.get("dominant_colors", [])
        materials = tags.get("materials", [])
        style_tags = tags.get("style_tags", [])

        return {
            "domain": domain,
            "brand": brand,
            "product_name": product_name,
            "category": category,
            "size": size_str,
            "packaging_format": packaging_format,
            "claims": claims,
            "colors": colors,
            "materials": materials,
            "style_tags": style_tags,
            "caption": caption,
            "detected_text": detected_text,
        }

    def _refine_product_name(self, raw_type: str, domain: str, text: str, packaging: str) -> str:
        """Derive an accurate product name based on domain."""
        if domain == "FOOD_BEVERAGE":
            if re.search(r'rasgulla', text, re.IGNORECASE):
                return "Traditional Bengali Rasgulla"
            elif re.search(r'gulab\s*jamun', text, re.IGNORECASE):
                return "Traditional Gulab Jamun"
            elif re.search(r'bhujia|namkeen', text, re.IGNORECASE):
                return "Crispy Bikaneri Bhujia Namkeen"
            elif re.search(r'bournvita', text, re.IGNORECASE):
                return "Nutritional Health Drink Mix"
            return raw_type if raw_type not in ("Product", "Packaged Food") else "Authentic Food Product"

        elif domain == "APPAREL_FASHION":
            if re.search(r't-?shirt', text, re.IGNORECASE):
                return "Classic Crewneck T-Shirt"
            elif re.search(r'hoodie', text, re.IGNORECASE):
                return "Pullover Fleece Hoodie"
            return raw_type if raw_type != "Product" else "Classic Apparel"

        elif domain == "ELECTRONICS":
            if re.search(r'earbud', text, re.IGNORECASE):
                return "True Wireless Bluetooth Earbuds"
            elif re.search(r'headphone', text, re.IGNORECASE):
                return "Over-Ear Wireless Headphones"
            return raw_type if raw_type != "Product" else "Wireless Audio Device"

        elif domain == "BEAUTY_PERSONAL_CARE":
            if re.search(r'serum', text, re.IGNORECASE):
                return "Face Serum"
            elif re.search(r'body\s*lotion', text, re.IGNORECASE):
                return "Hydrating Body Lotion"
            return raw_type if raw_type not in ("Product", "Skin Care") else "Daily Skin Care Treatment"

        elif domain == "HOME_KITCHEN":
            if re.search(r'water\s*bottle|flask', text, re.IGNORECASE):
                return "Insulated Stainless Steel Water Bottle"
            return raw_type if raw_type != "Product" else "Home & Kitchen Essential"

        return raw_type if raw_type != "Product" else "Item"

    def _extract_domain_claims(self, domain: str, text: str) -> List[str]:
        """Extract domain-relevant marketing hooks from OCR and visual text."""
        claims = []
        patterns = []

        if domain == "FOOD_BEVERAGE":
            patterns = [
                r'(Pure\s*(?:Chenna|Chhena|Ghee|Milk))',
                r'((?:Soft|Spongy|Melt-in-Mouth)\s*(?:Texture)?)',
                r'((?:Aromatic|Fragrant)\s*Sugar\s*Syrup)',
                r'(Traditional\s*(?:Bengali|Indian|Bikaneri)?\s*(?:Recipe|Mithai|Taste))',
                r'(100%\s*Vegetarian)',
            ]
        elif domain == "APPAREL_FASHION":
            patterns = [
                r'(100%\s*(?:Combed\s*)?Cotton)',
                r'((?:Breathable|Lightweight)\s*Fabric)',
                r'((?:Tailored|Regular|Slim)\s*Fit)',
            ]
        elif domain == "ELECTRONICS":
            patterns = [
                r'(Bluetooth\s*5\.\d)',
                r'(Active\s*Noise\s*Cancellation|ANC)',
                r'(\d+H\s*(?:Playtime|Battery\s*Life))',
            ]
        elif domain == "BEAUTY_PERSONAL_CARE":
            patterns = [
                r'(\d+%\s*Niacinamide)',
                r'(Rice\s*(?:Water|Acid))',
                r'(Deep\s*(?:Hydration|Nourish|Moisture))',
            ]
        else:
            patterns = [
                r'(Premium\s*(?:Quality|Material|Utility))',
                r'(Durable\s*&\s*Reliable)',
            ]

        for pat in patterns:
            matches = re.findall(pat, text, re.IGNORECASE)
            for m in matches:
                m_clean = re.sub(r'([a-z])([A-Z])', r'\1 \2', m).strip()
                normalized = re.sub(r'[^a-z0-9]', '', m_clean.lower())
                existing = [re.sub(r'[^a-z0-9]', '', c.lower()) for c in claims]
                if normalized not in existing:
                    claims.append(m_clean.title())

        return claims

    # --------------------------------------------------------------------------
    # Dynamic Domain-Routing Local Generators
    # --------------------------------------------------------------------------

    def _generate_with_local_engine(
        self,
        tags: Dict[str, Any],
        ctx: Dict[str, Any],
        tone: str = "Professional",
    ) -> ProductCatalogListing:
        """Route generation to domain-specific templates based on detected domain."""
        domain = ctx["domain"]

        if domain == "FOOD_BEVERAGE":
            return self._generate_food_listing(tags, ctx, tone)
        elif domain == "BEAUTY_PERSONAL_CARE":
            return self._generate_beauty_listing(tags, ctx, tone)
        elif domain == "OFFICE_STATIONERY":
            return self._generate_stationery_listing(tags, ctx, tone)
        elif domain == "APPAREL_FASHION":
            return self._generate_apparel_listing(tags, ctx, tone)
        elif domain == "ELECTRONICS":
            return self._generate_electronics_listing(tags, ctx, tone)
        elif domain == "HOME_KITCHEN":
            return self._generate_home_listing(tags, ctx, tone)
        else:
            # Universal fallback for infinite variety (Screwdriver, Hardware, etc.)
            return generate_universal_fallback(tags, brand_override=ctx.get("brand"), tone=tone)

    # --------------------------------------------------------------------------
    # Predefined Domain 1: FOOD & BEVERAGE
    # --------------------------------------------------------------------------
    def _generate_food_listing(
        self,
        tags: Dict[str, Any],
        ctx: Dict[str, Any],
        tone: str,
    ) -> ProductCatalogListing:
        brand = ctx["brand"]
        name = ctx["product_name"]
        size = ctx["size"]
        pkg = ctx["packaging_format"] or "Tin"
        claims = ctx["claims"]
        feat1 = claims[0] if claims else "Pure Authentic Ingredients"
        feat2 = claims[1] if len(claims) > 1 else "Aromatic Recipe"

        title_suffix = f"({size} {pkg})" if size and pkg else (f"({size})" if size else f"({pkg})")

        # Check if the product is a malted health nutrition drink mix (e.g. Bournvita, Cadbury)
        full_text_food = f"{name} {brand} {pkg} {' '.join(tags.get('detected_text', []))}".lower()
        is_malted_or_drink = any(
            k in full_text_food
            for k in ["malt", "bournvita", "drink", "nutrition", "nutrients", "health drink", "cadbury", "horlicks", "boost", "complan"]
        )

        if is_malted_or_drink:
            amz_title = f"{brand} {name} - Essential Nutrition with Active Vitamins & Minerals {title_suffix}"
            if len(amz_title) > 195:
                amz_title = amz_title[:192].rsplit(" ", 1)[0] + "..."

            bullets = [
                (
                    f"SCIENTIFICALLY FORMULATED NUTRITION: Fortified with essential vitamins (D, C, B-Complex), iron, zinc, "
                    f"and vital micronutrients designed to support physical strength, active bone development, and stamina."
                ),
                (
                    f"RICH CHOCOLATE MALT FLAVOR: Infused with a rich, delicious malted chocolate taste that dissolves smoothly "
                    f"in hot or cold milk, turning daily nourishment into an irresistible treat kids and adults love."
                ),
                (
                    f"HYGIENICALLY PROCESSED & PACKAGED: Manufactured in certified food-grade facilities following stringent "
                    f"automated hygiene protocols with zero manual touch, ensuring uncompromised purity and family safety."
                ),
                (
                    f"AIRTIGHT FRESHNESS LOCK: Hermetically sealed in a food-grade {pkg.lower()} container that seals in original aroma, "
                    f"crisp freshness, and potent nutrient integrity throughout its extended shelf life."
                ),
                (
                    f"DAILY VITALITY & ACTIVE ENERGY: The quintessential morning and evening nutritional supplement to power active "
                    f"school days, athletic training, and demanding workplace schedules. 100% vegetarian."
                ),
            ]

            search_terms = (
                f"health drink chocolate malt nutrition powder milk mix vitamins iron zinc daily energy vitality vegetarian"
            )[:240]

            amz_desc = (
                f"Energize your family's daily wellness routine with {brand} {name}. Specially formulated with vital "
                f"micronutrients including Iron, Zinc, and essential Vitamins, this malted nutrition mix dissolves smoothly "
                f"in hot or chilled milk. Featuring an irresistible chocolate-malt flavor, it supports daily strength, stamina, "
                f"and active growth. Packaged under certified food-grade hygiene standards in an airtight {pkg.lower()} container, "
                f"it guarantees wholesome goodness and delicious taste in every single glass."
            )

            shopify_title = f"{brand} {name} {title_suffix}"
            shopify_html = (
                f"<h3>Why You'll Love It</h3>\n"
                f"<p>Power up your daily health routine with <strong>{brand} {name}</strong>. "
                f"Specially crafted with essential vitamins and minerals, this delicious chocolate malt mix turns every glass of milk into wholesome vitality.</p>\n\n"
                f"<h3>Nutritional & Taste Highlights</h3>\n"
                f"<ul>\n"
                f"  <li><strong>2X Active Nutrients:</strong> Enriched with Vitamin D, Iron, and Zinc for bone strength and immunity.</li>\n"
                f"  <li><strong>Rich Chocolate Malt Taste:</strong> Blends instantly with warm or cold milk for smooth indulgence.</li>\n"
                f"  <li><strong>Automated Food-Grade Hygiene:</strong> Certified packaging ensures complete safety and purity.</li>\n"
                f"  <li><strong>100% Vegetarian:</strong> Wholesome nutrition suitable for the entire family.</li>\n"
                f"</ul>\n\n"
                f"<h3>Preparation Instructions</h3>\n"
                f"<p>Add 2 heaped teaspoons into a cup of hot or cold milk. Stir thoroughly until dissolved. Add sugar if desired and enjoy twice daily.</p>"
            )

            meta_desc = (
                f"Nourish your family with {brand} {name}. Fortified with essential vitamins, iron, and zinc in rich chocolate malt. Buy now for fast delivery!"
            )
            tags_list = ["Food & Beverage", "Health Drink", "Nutrition", brand, "Chocolate Malt", "Vegetarian"]
        else:
            # Traditional Sweets / Packaged Snacks
            amz_title = f"{brand} {name} - Soft & Spongy Traditional Sweets with {feat1} {title_suffix}"
            if len(amz_title) > 195:
                amz_title = amz_title[:192].rsplit(" ", 1)[0] + "..."

            bullets = [
                (
                    f"AUTHENTIC TRADITIONAL RECIPE: Handcrafted following time-honored traditional recipes, delivering an "
                    f"authentic melt-in-the-mouth taste experience reminiscent of festive celebrations and royal Indian cuisine."
                ),
                (
                    f"SOFT & SPONGY TEXTURE: Prepared with {feat1} simmered gently in {feat2} to achieve an irresistible soft, "
                    f"spongy texture that bursts with sweet deliciousness in every bite."
                ),
                (
                    f"HYGIENICALLY PROCESSED & CANNED: Made in state-of-the-art certified food-grade facilities with zero human "
                    f"touch automated packaging, ensuring uncompromised purity, supreme hygiene, and safety for your family."
                ),
                (
                    f"AIRTIGHT FRESHNESS LOCK: Hermetically sealed in a food-grade {pkg.lower()} container that seals in original aroma, "
                    f"moisture, and delectable freshness for an extended shelf life without artificial colors or harmful chemicals."
                ),
                (
                    f"READY TO SERVE & CELEBRATE: The quintessential dessert for festive gifting (Diwali, Rakhi, Holi), "
                    f"wedding banquets, family dinners, or whenever sweet cravings strike. 100% vegetarian."
                ),
            ]

            search_terms = (
                f"traditional indian sweets dessert mithai canned rasgulla chenna sweets festive gift pack "
                f"spongy sweet syrup ready to eat confectionery bikaneri delicacy vegetarian"
            )[:240]

            amz_desc = (
                f"Indulge in the regal sweetness of {brand} {name}. Celebrated across generations for exceptional culinary "
                f"heritage, this delectable dessert is prepared using {feat1} kneaded to silky perfection and simmered in "
                f"{feat2}. Each bite offers an exquisite, melt-in-the-mouth softness that elevates any meal into a joyous feast. "
                f"Hygienically packaged in an airtight {pkg.lower()} to preserve its wholesome goodness and delicate aroma, "
                f"this sweet is the perfect centerpiece for festive gifting, celebrations, and cherished family moments."
            )

            shopify_title = f"{brand} {name} {title_suffix}"
            shopify_html = (
                f"<h3>Why You'll Love It</h3>\n"
                f"<p>Experience the royal taste of authentic Indian confectionery with <strong>{brand} {name}</strong>. "
                f"Lovingly prepared from {feat1} and soaked in {feat2}, these soft and spongy sweets deliver unmatched joy to dessert lovers.</p>\n\n"
                f"<h3>Taste & Quality Highlights</h3>\n"
                f"<ul>\n"
                f"  <li><strong>Authentic Heritage Recipe:</strong> Traditional preparation ensures an authentic festive taste and texture.</li>\n"
                f"  <li><strong>Melt-in-Mouth Sponginess:</strong> Made with premium ingredients for the softest, juiciest bite.</li>\n"
                f"  <li><strong>Hygienically Canned:</strong> Automated packaging in an airtight food-grade container locks in freshness.</li>\n"
                f"  <li><strong>100% Vegetarian:</strong> Wholesome and safe for the entire family.</li>\n"
                f"</ul>\n\n"
                f"<h3>Serving Suggestions</h3>\n"
                f"<p>Serve chilled for an elevated dessert sensation, or warm gently before serving with a scoop of rabri or ice cream. Refrigerate after opening.</p>"
            )

            meta_desc = (
                f"Indulge in authentic {brand} {name} crafted with {feat1} in fragrant syrup. Soft, spongy, and hygienically canned for festive celebrations. Buy now!"
            )
            tags_list = ["Food & Beverage", "Sweets", "Mithai", brand, "Traditional Sweets", "Vegetarian"]

        if len(meta_desc) > 160:
            meta_desc = meta_desc[:157] + "..."
        elif len(meta_desc) < 140:
            meta_desc += " Fast shipping across India."

        handle = re.sub(r'[^a-z0-9]+', '-', shopify_title.lower()).strip('-')

        return ProductCatalogListing(
            brand=brand,
            product_name=name,
            category="Food & Beverage",
            domain="FOOD_BEVERAGE",
            amazon=AmazonListing(
                title=amz_title,
                bullet_points=bullets,
                search_terms=search_terms,
                product_description=amz_desc,
            ),
            shopify=ShopifyListing(
                title=shopify_title,
                body_html=shopify_html,
                tags=tags_list,
                meta_description=meta_desc,
                handle=handle,
            ),
            tags_source=tags,
        )

    # --------------------------------------------------------------------------
    # Predefined Domain 2: OFFICE SUPPLIES & STATIONERY (Paper & Printing)
    # --------------------------------------------------------------------------
    def _generate_stationery_listing(
        self,
        tags: Dict[str, Any],
        ctx: Dict[str, Any],
        tone: str,
    ) -> ProductCatalogListing:
        brand = ctx["brand"]
        name = ctx["product_name"]
        size = ctx["size"]
        pkg = ctx["packaging_format"] or "Ream"
        materials = ", ".join(ctx["materials"]) if ctx["materials"] else "Wood-Free Paper"
        colors = ctx["colors"]
        color_str = colors[0] if colors else "Bright White"

        spec_suffix = f"({size})" if size else (f"({pkg})" if pkg else "")
        amz_title = f"{brand} {name} - High Jam-Free Speed & Bright Contrast Paper {spec_suffix}".strip()
        if len(amz_title) > 195:
            amz_title = amz_title[:192].rsplit(" ", 1)[0] + "..."

        bullets = [
            (
                f"JAM-FREE HIGH-SPEED PERFORMANCE: Precision-engineered with uniform sheet caliper and clean-cut edges to "
                f"deliver consistent 99.99% jam-free operation across high-volume laser printers, inkjets, and digital copiers."
            ),
            (
                f"SUPERIOR OPACITY & ZERO INK BLEED: Manufactured with high sheet density and opacity, enabling crisp "
                f"double-sided printing and copying with zero ink show-through, feathering, or ghosting."
            ),
            (
                f"ULTRA-BRIGHT COLOR CONTRAST: High ISO brightness and whiteness rating delivers vivid color reproduction, "
                f"deep saturated blacks, and razor-sharp monochrome text definition for professional documentation."
            ),
            (
                f"ACID-FREE ARCHIVAL LONGEVITY: Responsibly crafted from {materials} with elemental chlorine-free (ECF) pulp, "
                f"preventing yellowing and degradation to preserve vital records, legal files, and contracts for decades."
            ),
            (
                f"MULTIPURPOSE OFFICE & HOME UTILITY: The ideal daily workhorse paper for corporate presentations, legal briefs, "
                f"academic reports, invoices, commercial proposals, and routine everyday high-speed printing."
            ),
        ]

        search_terms = (
            f"copier paper ream multipurpose printing sheets laser inkjet jam free bright white office stationery notebook"
        )[:240]

        amz_desc = (
            f"Elevate your office productivity and presentation quality with {brand} {name}. Engineered for demanding "
            f"high-volume commercial and domestic environments, this multipurpose paper delivers dependable 99.99% "
            f"jam-free reliability across all laser printers, inkjets, and copiers. Featuring high sheet opacity and advanced "
            f"surface smoothness, it supports crisp double-sided printing with zero ink bleed. Crafted from durable {materials} "
            f"with acid-free archival longevity, {brand} guarantees vivid color graphics and sharp monochrome text for "
            f"every essential report, presentation, and document."
        )

        shopify_title = f"{brand} {name}" + (f" ({size})" if size else "")
        shopify_html = (
            f"<h3>Product Overview</h3>\n"
            f"<p>Boost workspace productivity and document quality with <strong>{brand} {name}</strong>. "
            f"Precision-engineered with {materials} and finished in {color_str}, this high-performance multipurpose paper "
            f"delivers jam-free dependability and crisp contrast for all your office and home printing requirements.</p>\n\n"
            f"<h3>Key Paper & Printing Highlights</h3>\n"
            f"<ul>\n"
            f"  <li><strong>99.99% Jam-Free Reliability:</strong> Precision-cut edges ensure trouble-free feeding through high-speed copiers and printers.</li>\n"
            f"  <li><strong>Zero Ink Bleed:</strong> High sheet opacity enables confident double-sided printing with no show-through.</li>\n"
            f"  <li><strong>Superior Whiteness:</strong> Formulated for rich, vibrant color graphics and razor-sharp monochrome text.</li>\n"
            f"  <li><strong>Archival Durability:</strong> Acid-free wood-free formulation resists yellowing over time.</li>\n"
            f"</ul>\n\n"
            f"<h3>Technical Specifications</h3>\n"
            f"<p><strong>Brand:</strong> {brand} | <strong>Product:</strong> {name} | <strong>Material:</strong> {materials} | <strong>Color:</strong> {color_str}</p>"
        )

        meta_desc = (
            f"Buy {brand} {name}. High-speed jam-free multipurpose copier paper with superior opacity and bright contrast for professional office printing."
        )
        if len(meta_desc) > 160:
            meta_desc = meta_desc[:157] + "..."
        elif len(meta_desc) < 140:
            meta_desc += " Fast shipping with quality guarantee."

        handle = re.sub(r'[^a-z0-9]+', '-', shopify_title.lower()).strip('-')

        return ProductCatalogListing(
            brand=brand,
            product_name=name,
            category="Office Supplies & Stationery",
            domain="OFFICE_STATIONERY",
            amazon=AmazonListing(
                title=amz_title,
                bullet_points=bullets,
                search_terms=search_terms,
                product_description=amz_desc,
            ),
            shopify=ShopifyListing(
                title=shopify_title,
                body_html=shopify_html,
                tags=["Office Supplies & Stationery", "Paper", "Printing", brand, "Copier Paper", "Jam-Free"],
                meta_description=meta_desc,
                handle=handle,
            ),
            tags_source=tags,
        )

    # --------------------------------------------------------------------------
    # Predefined Domain 2: APPAREL & FASHION
    # --------------------------------------------------------------------------
    def _generate_apparel_listing(
        self,
        tags: Dict[str, Any],
        ctx: Dict[str, Any],
        tone: str,
    ) -> ProductCatalogListing:
        brand = ctx["brand"]
        name = ctx["product_name"]
        size = ctx["size"] or "Regular"
        colors = ctx["colors"]
        color_str = colors[0] if colors else "Classic Color"
        claims = ctx["claims"]
        fabric = claims[0] if claims else "100% Combed Cotton"

        amz_title = f"{brand} {name} - {fabric} Breathable Comfort Fit ({color_str}, {size})"
        if len(amz_title) > 195:
            amz_title = amz_title[:192].rsplit(" ", 1)[0] + "..."

        bullets = [
            (
                f"PREMIUM BREATHABLE FABRIC: Crafted from {fabric} that provides an ultra-soft hand feel, natural "
                f"airflow, and superior moisture absorption to keep you cool and dry all day long."
            ),
            (
                f"TAILORED ERGONOMIC FIT: Engineered with a modern relaxed silhouette that drapes naturally across the shoulders "
                f"and torso, ensuring unhindered movement without bunching or excessive tightness."
            ),
            (
                f"REINFORCED DURABLE STITCHING: Built with twin-needle reinforced hems, durable collar ribbing, and taped "
                f"neck seams that prevent stretching, fraying, and unraveling through daily wear."
            ),
            (
                f"VERSATILE EVERYDAY STYLING: A clean, contemporary wardrobe staple that effortlessly transitions from "
                f"casual daytime outings to smart evening layering under jackets and overshirts."
            ),
            (
                f"EASY CARE & FADE RESISTANCE: Pre-shrunk and bio-washed to resist pilling, shrinkage, and color fading. "
                f"Machine-wash cold with similar colors for effortless long-lasting maintenance."
            ),
        ]

        search_terms = (
            f"mens clothing cotton tshirt casual wear daily regular fit breathable crewneck apparel comfortable top"
        )[:240]

        amz_desc = (
            f"Upgrade your daily rotation with the {brand} {name}. Meticulously tailored using {fabric}, this piece "
            f"delivers the ultimate combination of breathability, tactile softness, and everyday resilience. Whether "
            f"worn solo on warm afternoons or layered underneath sweaters and jackets, its precision tailoring and "
            f"reinforced construction ensure timeless style and dependable comfort wash after wash."
        )

        shopify_title = f"{brand} {name} ({color_str})"
        shopify_html = (
            f"<h3>Why You'll Love It</h3>\n"
            f"<p>Effortless comfort meets refined modern style in the <strong>{brand} {name}</strong>. "
            f"Constructed from premium {fabric}, it delivers cloud-soft comfort and exceptional breathability.</p>\n\n"
            f"<h3>Fabric & Fit Details</h3>\n"
            f"<ul>\n"
            f"  <li><strong>Material:</strong> {fabric} for breathable everyday wear.</li>\n"
            f"  <li><strong>Fit:</strong> Tailored modern silhouette that flatters without restricting.</li>\n"
            f"  <li><strong>Finish:</strong> Bio-washed for anti-pilling and lasting color richness.</li>\n"
            f"  <li><strong>Construction:</strong> Double-needle reinforced seams for longevity.</li>\n"
            f"</ul>\n\n"
            f"<h3>Care Instructions</h3>\n"
            f"<p>Machine wash cold on gentle cycle. Tumble dry low or hang dry to preserve garment longevity. Cool iron if necessary.</p>"
        )

        meta_desc = (
            f"Elevate your wardrobe with the {brand} {name}. Ultra-breathable {fabric}, tailored comfort fit, and durable double stitching. Shop now!"
        )
        if len(meta_desc) > 160:
            meta_desc = meta_desc[:157] + "..."
        elif len(meta_desc) < 140:
            meta_desc += " Satisfaction guaranteed with easy returns."

        handle = re.sub(r'[^a-z0-9]+', '-', shopify_title.lower()).strip('-')

        return ProductCatalogListing(
            brand=brand,
            product_name=name,
            category="Apparel & Fashion",
            domain="APPAREL_FASHION",
            amazon=AmazonListing(
                title=amz_title,
                bullet_points=bullets,
                search_terms=search_terms,
                product_description=amz_desc,
            ),
            shopify=ShopifyListing(
                title=shopify_title,
                body_html=shopify_html,
                tags=["Apparel & Fashion", "Clothing", brand, color_str, "Casual Wear"],
                meta_description=meta_desc,
                handle=handle,
            ),
            tags_source=tags,
        )

    # --------------------------------------------------------------------------
    # Predefined Domain 3: ELECTRONICS
    # --------------------------------------------------------------------------
    def _generate_electronics_listing(
        self,
        tags: Dict[str, Any],
        ctx: Dict[str, Any],
        tone: str,
    ) -> ProductCatalogListing:
        brand = ctx["brand"]
        name = ctx["product_name"]
        colors = ctx["colors"]
        color_str = colors[0] if colors else "Black"
        claims = ctx["claims"]
        spec1 = claims[0] if claims else "Bluetooth 5.3"
        spec2 = claims[1] if len(claims) > 1 else "Active Noise Cancellation"

        amz_title = f"{brand} {name} - {spec1} with {spec2} & Low Latency ({color_str})"
        if len(amz_title) > 195:
            amz_title = amz_title[:192].rsplit(" ", 1)[0] + "..."

        bullets = [
            (
                f"IMMERSIVE HIGH-FIDELITY AUDIO: Engineered with dynamic acoustic drivers and {spec2} to produce punchy, "
                f"deep bass, transparent mids, and crystal-clear trebles across your favorite playlists and podcasts."
            ),
            (
                f"SEAMLESS WIRELESS CONNECTIVITY: Powered by advanced {spec1} technology for instant auto-pairing, ultra-low "
                f"audio-video latency, and uninterrupted signal reception up to 33 feet (10 meters)."
            ),
            (
                f"ALL-DAY PLAYTIME & FAST RECHARGE: Long-lasting lithium battery delivers extended playback on a single charge, "
                f"supported by rapid USB-C fast charging that provides hours of listening with just a quick top-up."
            ),
            (
                f"ERGONOMIC & SWEAT-RESISTANT: Featherlight contoured construction provides a snug, secure fit for all-day "
                f"wear during vigorous gym sessions, daily commuting, and hands-free business calls."
            ),
            (
                f"SMART TOUCH & VOICE ASSISTANT: Intuitive capacitive touch sensors allow seamless playback control, volume "
                f"adjustment, call answering, and one-tap access to Siri or Google Assistant."
            ),
        ]

        search_terms = (
            f"wireless bluetooth earbuds headphones portable audio noise cancelling bass sound headset fast charge"
        )[:240]

        amz_desc = (
            f"Immerse yourself in rich acoustic precision with the {brand} {name}. Featuring cutting-edge {spec1} "
            f"connectivity alongside {spec2}, this audio companion ensures studio-grade sound reproduction whether "
            f"you are on a crowded commute, working at your desk, or pushing your limits in the gym. Its ergonomic "
            f"chassis delivers day-long comfort, while responsive touch controls and long-lasting battery life make "
            f"everyday listening effortlessly enjoyable."
        )

        shopify_title = f"{brand} {name} - {spec1}"
        shopify_html = (
            f"<h3>Why You'll Love It</h3>\n"
            f"<p>Experience premium acoustic freedom with the <strong>{brand} {name}</strong>. "
            f"Engineered with {spec1} and {spec2}, it delivers uncompromised sound fidelity and comfort.</p>\n\n"
            f"<h3>Technical Specifications</h3>\n"
            f"<ul>\n"
            f"  <li><strong>Connectivity:</strong> {spec1} with auto-reconnect.</li>\n"
            f"  <li><strong>Audio Architecture:</strong> Dynamic drivers tuned for balanced high-fidelity sound.</li>\n"
            f"  <li><strong>Battery & Charging:</strong> All-day playtime with Type-C rapid charging.</li>\n"
            f"  <li><strong>Controls:</strong> Intuitive touch navigation for music and hands-free calls.</li>\n"
            f"</ul>\n\n"
            f"<h3>What's in the Box</h3>\n"
            f"<p>1x {brand} {name}, 1x USB-C Charging Cable, User Manual, and Warranty Documentation.</p>"
        )

        meta_desc = (
            f"Experience superior sound with {brand} {name}. Powered by {spec1} and {spec2} for crystal-clear acoustics and all-day battery. Order now!"
        )
        if len(meta_desc) > 160:
            meta_desc = meta_desc[:157] + "..."
        elif len(meta_desc) < 140:
            meta_desc += " Fast shipping with 1-year brand warranty."

        handle = re.sub(r'[^a-z0-9]+', '-', shopify_title.lower()).strip('-')

        return ProductCatalogListing(
            brand=brand,
            product_name=name,
            category="Electronics",
            domain="ELECTRONICS",
            amazon=AmazonListing(
                title=amz_title,
                bullet_points=bullets,
                search_terms=search_terms,
                product_description=amz_desc,
            ),
            shopify=ShopifyListing(
                title=shopify_title,
                body_html=shopify_html,
                tags=["Electronics", "Audio", "Wireless", brand, "Bluetooth Gadgets"],
                meta_description=meta_desc,
                handle=handle,
            ),
            tags_source=tags,
        )

    # --------------------------------------------------------------------------
    # Predefined Domain 4: BEAUTY & PERSONAL CARE
    # --------------------------------------------------------------------------
    def _generate_beauty_listing(
        self,
        tags: Dict[str, Any],
        ctx: Dict[str, Any],
        tone: str,
    ) -> ProductCatalogListing:
        brand = ctx["brand"]
        name = ctx["product_name"]
        size = ctx["size"]
        claims = ctx["claims"]
        feat1 = claims[0] if claims else "Active Botanical Complex"
        feat2 = claims[1] if len(claims) > 1 else "Deep Hydrating Formula"

        size_suffix = f"- {size}" if size else ""
        amz_title = f"{brand} {name} with {feat1} & {feat2} for Radiant Skin {size_suffix}".strip()
        if len(amz_title) > 195:
            amz_title = amz_title[:192].rsplit(" ", 1)[0] + "..."

        bullets = [
            (
                f"TARGETED RADIANCE & TEXTURE: Powered by {feat1} to visibly smooth uneven skin texture, reduce the "
                f"appearance of enlarged pores, and restore an enviable, radiant glass-skin complexion."
            ),
            (
                f"DEEP HYDRATION & NOURISHMENT: Enriched with {feat2} that deeply replenishes depleted moisture reserves, "
                f"preventing dryness and leaving skin soft, supple, and revitalized."
            ),
            (
                f"SKIN BARRIER PROTECTION: Fortified with protective antioxidants that help repair and strengthen the delicate "
                f"dermal moisture barrier against urban pollutants and harsh environmental stress."
            ),
            (
                f"LIGHTWEIGHT & FAST ABSORBING: Ultra-light, non-greasy texture absorbs instantly upon contact without sticky "
                f"residue or pore congestion, making it ideal for both morning and evening application."
            ),
            (
                f"CLEAN & DERMATOLOGICALLY TESTED: Formulated without parabens, sulfates, or harsh synthetic additives. "
                f"Cruelty-free, ethically sourced, and safe for all skin types, including sensitive skin."
            ),
        ]

        search_terms = (
            f"skincare face serum moisturizer daily hydration glass skin glow barrier repair pore treatment beauty"
        )[:240]

        amz_desc = (
            f"Rejuvenate your daily skincare routine with the {brand} {name}. Carefully crafted with a balanced infusion "
            f"of {feat1} and {feat2}, this formula sinks deep into the skin to target dullness, dryness, and uneven tone. "
            f"Its fast-absorbing, non-comedogenic consistency locks in essential hydration, supporting long-term skin "
            f"barrier vitality while leaving your complexion visibly luminous, soft, and youthful."
        )

        shopify_title = f"{brand} {name}" + (f" ({size})" if size else "")
        shopify_html = (
            f"<h3>Why You'll Love It</h3>\n"
            f"<p>Revitalize your complexion with the <strong>{brand} {name}</strong>. "
            f"Infused with {feat1} and {feat2}, this treatment delivers lasting hydration and radiant glow.</p>\n\n"
            f"<h3>Key Benefits</h3>\n"
            f"<ul>\n"
            f"  <li><strong>Deep Hydration:</strong> Quenches parched skin for all-day moisture retention.</li>\n"
            f"  <li><strong>Refined Texture:</strong> Visibly minimizes pores and promotes smooth skin tone.</li>\n"
            f"  <li><strong>Barrier Defense:</strong> Strengthens skin resilience against environmental pollutants.</li>\n"
            f"  <li><strong>Clean Beauty:</strong> 100% cruelty-free, paraben-free, and dermatologically tested.</li>\n"
            f"</ul>\n\n"
            f"<h3>How to Apply</h3>\n"
            f"<p>Dispense 3-4 drops onto cleansed face and neck. Gently press and pat until fully absorbed before applying sunscreen or night cream.</p>"
        )

        meta_desc = (
            f"Transform your skin with {brand} {name}. Infused with {feat1} for deep hydration, pore refinement, and a radiant glass-skin glow. Shop now!"
        )
        if len(meta_desc) > 160:
            meta_desc = meta_desc[:157] + "..."
        elif len(meta_desc) < 140:
            meta_desc += " Dermatologist approved with fast shipping."

        handle = re.sub(r'[^a-z0-9]+', '-', shopify_title.lower()).strip('-')

        return ProductCatalogListing(
            brand=brand,
            product_name=name,
            category="Beauty & Personal Care",
            domain="BEAUTY_PERSONAL_CARE",
            amazon=AmazonListing(
                title=amz_title,
                bullet_points=bullets,
                search_terms=search_terms,
                product_description=amz_desc,
            ),
            shopify=ShopifyListing(
                title=shopify_title,
                body_html=shopify_html,
                tags=["Beauty & Personal Care", "Skincare", brand, "Hydration", "Face Care"],
                meta_description=meta_desc,
                handle=handle,
            ),
            tags_source=tags,
        )

    # --------------------------------------------------------------------------
    # Predefined Domain 5: HOME & KITCHEN
    # --------------------------------------------------------------------------
    def _generate_home_listing(
        self,
        tags: Dict[str, Any],
        ctx: Dict[str, Any],
        tone: str,
    ) -> ProductCatalogListing:
        brand = ctx["brand"]
        name = ctx["product_name"]
        size = ctx["size"]
        materials = ctx["materials"]
        mat_str = materials[0] if materials else "Stainless Steel"

        amz_title = f"{brand} {name} - Premium {mat_str} with Leak-Proof Ergonomic Design" + (f" ({size})" if size else "")
        if len(amz_title) > 195:
            amz_title = amz_title[:192].rsplit(" ", 1)[0] + "..."

        bullets = [
            (
                f"FOOD-GRADE & TOXIN-FREE MATERIAL: Constructed from premium {mat_str} that is 100% BPA-free, rust-proof, "
                f"and non-toxic, preserving the natural flavor and purity of contents without metallic aftertaste."
            ),
            (
                f"SUPERIOR THERMAL & LEAK-PROOF INTEGRITY: Precision-engineered airtight seal prevents accidental spills in "
                f"bags, while vacuum insulation keeps contents at ideal temperatures for extended hours."
            ),
            (
                f"ERGONOMIC EVERYDAY DESIGN: Features an easy-grip silhouette and balanced weight distribution suited for "
                f"office desks, gym workouts, outdoor travel, and daily household use."
            ),
            (
                f"DURABLE & SCRATCH RESISTANT: Tough outer finish withstands drops, scratches, and everyday wear, ensuring "
                f"long-term aesthetic elegance and dependable performance."
            ),
            (
                f"EASY TO CLEAN & ECO-CONSCIOUS: Wide mouth design facilitates effortless cleaning and refilling. A reusable, "
                f"sustainable choice to help eliminate disposable single-use plastics."
            ),
        ]

        search_terms = f"kitchen home essential reusable container durable ergonomic leakproof food grade"[:240]

        amz_desc = (
            f"Experience functional elegance with the {brand} {name}. Fabricated with high-grade {mat_str}, it offers "
            f"unmatched durability, safety, and modern design. Whether utilized at home, in the workplace, or during "
            f"travel, its leak-proof construction and premium finish make everyday utility effortless and stylish."
        )

        shopify_title = f"{brand} {name}" + (f" ({size})" if size else "")
        shopify_html = (
            f"<h3>Why You'll Love It</h3>\n"
            f"<p>Built for durability and effortless living, the <strong>{brand} {name}</strong> pairs premium {mat_str} "
            f"craftsmanship with modern design.</p>\n\n"
            f"<h3>Key Features</h3>\n"
            f"<ul>\n"
            f"  <li><strong>Material:</strong> 100% BPA-free food-grade {mat_str}.</li>\n"
            f"  <li><strong>Leak-Proof:</strong> Precision airtight seal ensures zero mess.</li>\n"
            f"  <li><strong>Durability:</strong> Scratch-resistant and built for daily use.</li>\n"
            f"  <li><strong>Eco-Friendly:</strong> Reusable and sustainable alternative.</li>\n"
            f"</ul>"
        )

        meta_desc = (
            f"Discover the {brand} {name}. Crafted with premium food-grade {mat_str} for leak-proof durability and everyday convenience. Order today!"
        )
        if len(meta_desc) > 160:
            meta_desc = meta_desc[:157] + "..."
        elif len(meta_desc) < 140:
            meta_desc += " Fast nationwide shipping available."

        handle = re.sub(r'[^a-z0-9]+', '-', shopify_title.lower()).strip('-')

        return ProductCatalogListing(
            brand=brand,
            product_name=name,
            category="Home & Kitchen",
            domain="HOME_KITCHEN",
            amazon=AmazonListing(
                title=amz_title,
                bullet_points=bullets,
                search_terms=search_terms,
                product_description=amz_desc,
            ),
            shopify=ShopifyListing(
                title=shopify_title,
                body_html=shopify_html,
                tags=["Home & Kitchen", "Reusable", brand, "Dining & Storage"],
                meta_description=meta_desc,
                handle=handle,
            ),
            tags_source=tags,
        )

    # --------------------------------------------------------------------------
    # LLM Integrations
    # --------------------------------------------------------------------------

    def _generate_with_gemini(
        self,
        tags: Dict[str, Any],
        ctx: Dict[str, Any],
        tone: str,
    ) -> Optional[ProductCatalogListing]:
        """Generate structured listing using Google Gemini API."""
        prompt = self._build_llm_prompt(tags, ctx, tone)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_api_key}"

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.3,
            },
        }

        resp = requests.post(url, json=payload, timeout=20)
        if resp.status_code != 200:
            logger.warning(f"Gemini API returned status {resp.status_code}: {resp.text}")
            return None

        result = resp.json()
        raw_text = result["candidates"][0]["content"]["parts"][0]["text"]
        return self._parse_llm_json_response(raw_text, ctx, tags)

    def _generate_with_openai(
        self,
        tags: Dict[str, Any],
        ctx: Dict[str, Any],
        tone: str,
    ) -> Optional[ProductCatalogListing]:
        """Generate structured listing using OpenAI Chat Completion API."""
        prompt = self._build_llm_prompt(tags, ctx, tone)
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert e-commerce copywriter generating strictly compliant Amazon and Shopify product listings.",
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.3,
        }

        resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=20)
        if resp.status_code != 200:
            logger.warning(f"OpenAI API returned status {resp.status_code}: {resp.text}")
            return None

        result = resp.json()
        raw_text = result["choices"][0]["message"]["content"]
        return self._parse_llm_json_response(raw_text, ctx, tags)

    def _build_llm_prompt(self, tags: Dict[str, Any], ctx: Dict[str, Any], tone: str) -> str:
        """Construct prompt with domain instructions enforcing Amazon & Shopify guidelines."""
        domain_guidelines = {
            "FOOD_BEVERAGE": "Focus on Taste, Authentic Recipe, Ingredients, Hygiene Packaging, and Extended Freshness.",
            "APPAREL_FASHION": "Focus on Fabric & Cotton Material, Breathability, Tailored Fit, Stitching Durability, and Wash Care.",
            "ELECTRONICS": "Focus on Audio/Visual Performance, Wireless Connectivity, Battery Playtime, and Smart Touch Controls.",
            "BEAUTY_PERSONAL_CARE": "Focus on Active Ingredients, Deep Nourishment, Skin Barrier Defense, and Clean Gentle Formula.",
            "HOME_KITCHEN": "Focus on Food-Grade Safe Materials, Leak-Proof Design, Ergonomic Durability, and Easy Cleaning.",
        }.get(ctx["domain"], "Focus on premium build quality, versatile multi-purpose utility, aesthetics, verified specifications, and quality assurance.")

        return f"""You are a professional e-commerce product cataloger and conversion copywriter.
Generate an Amazon and Shopify compliant listing using the following product attributes:

Domain: {ctx['domain']}
Domain Instructions: {domain_guidelines}
Brand: {ctx['brand']}
Product Name: {ctx['product_name']}
Category: {ctx['category']}
Detected Claims / Active Ingredients: {', '.join(ctx['claims']) or 'None'}
Size / Volume: {ctx['size'] or 'Standard'}
Colors: {', '.join(ctx['colors']) or 'Standard'}
Materials: {', '.join(ctx['materials']) or 'Standard'}
Visual Caption: {tags.get('detailed_description', '')}
OCR Detected Text: {' '.join(tags.get('detected_text', []))}
Desired Tone: {tone}

Return ONLY valid JSON matching this exact structure:
{{
  "brand": "{ctx['brand']}",
  "product_name": "{ctx['product_name']}",
  "domain": "{ctx['domain']}",
  "amazon": {{
    "title": "Amazon title strictly UNDER 200 characters following [Brand] + [Product Name] + [Key Feature/Ingredient] + [Size/Color]",
    "bullet_points": [
      "CAPITALIZED BENEFIT HOOK: Detailed point 1 (150-250 chars)",
      "CAPITALIZED BENEFIT HOOK: Detailed point 2 (150-250 chars)",
      "CAPITALIZED BENEFIT HOOK: Detailed point 3 (150-250 chars)",
      "CAPITALIZED BENEFIT HOOK: Detailed point 4 (150-250 chars)",
      "CAPITALIZED BENEFIT HOOK: Detailed point 5 (150-250 chars)"
    ],
    "search_terms": "space separated generic search keywords under 250 bytes without punctuation or brand names",
    "product_description": "Clean engaging paragraph describing the product"
  }},
  "shopify": {{
    "title": "Catchy storefront title",
    "body_html": "<h3>Why You'll Love It</h3><p>...</p><h3>Key Highlights</h3><ul><li>...</li></ul><h3>Usage / Serving</h3><p>...</p>",
    "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"],
    "meta_description": "Compelling SEO meta description strictly between 140 and 160 characters",
    "handle": "url-friendly-kebab-case-slug"
  }}
}}
"""

    def _parse_llm_json_response(
        self,
        raw_text: str,
        ctx: Dict[str, Any],
        tags: Dict[str, Any],
    ) -> Optional[ProductCatalogListing]:
        """Parse, validate, and convert LLM JSON output into ProductCatalogListing."""
        try:
            cleaned = re.sub(r'^```json\s*', '', raw_text.strip())
            cleaned = re.sub(r'\s*```$', '', cleaned.strip())
            data = json.loads(cleaned)

            amazon_data = data.get("amazon", {})
            shopify_data = data.get("shopify", {})

            bullets = amazon_data.get("bullet_points", [])
            while len(bullets) < 5:
                bullets.append("PREMIUM QUALITY: Crafted with rigorous attention to detail for superior everyday satisfaction.")
            bullets = bullets[:5]
            amazon_data["bullet_points"] = bullets

            amazon_obj = AmazonListing(**amazon_data)
            shopify_obj = ShopifyListing(**shopify_data)

            return ProductCatalogListing(
                brand=data.get("brand", ctx["brand"]),
                product_name=data.get("product_name", ctx["product_name"]),
                category=ctx["category"],
                domain=data.get("domain", ctx["domain"]),
                amazon=amazon_obj,
                shopify=shopify_obj,
                tags_source=tags,
            )
        except Exception as e:
            logger.error(f"Error parsing LLM JSON: {e}")
            return None


def generate_listing(
    tags_input: Union[str, Path, Dict[str, Any]] = "data/processed/extracted_tags.json",
    brand_override: Optional[str] = None,
    tone: str = "Professional",
) -> ProductCatalogListing:
    """Convenience helper function to generate listings."""
    copywriter = ListingCopywriter()
    return copywriter.generate_listing(tags_input, brand_override=brand_override, tone=tone)


def generate_amazon_listing(
    tags_input: Union[str, Path, Dict[str, Any]] = "data/processed/extracted_tags.json",
    brand_override: Optional[str] = None,
    tone: str = "Professional",
) -> AmazonListing:
    """Convenience helper function to generate Amazon listing metadata."""
    copywriter = ListingCopywriter()
    return copywriter.generate_amazon_listing(tags_input, brand_override=brand_override, tone=tone)
