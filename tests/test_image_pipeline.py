"""Tests for Phase 1 Image Processing Engine."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from PIL import Image, ImageDraw

try:
    import pytest
except ImportError:
    pytest = None

from src.image_pipeline import (
    CanvasStyler,
    ImageSegmenter,
    ShadowEngine,
    process_catalog_image,
)


def create_dummy_product_photo(output_path: Path) -> Path:
    """Generate a realistic mock smartphone product photo.
    
    Creates a simulated cosmetic/skincare bottle on a textured noisy table background.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Background: noisy textured wooden/table surface
    img_w, img_h = 1000, 1200
    np.random.seed(42)
    # Gradient table tone
    gradient_y = np.linspace(180, 140, img_h, dtype=np.float32)[:, np.newaxis]
    bg_base = np.tile(gradient_y, (1, img_w))
    bg_noise = np.random.normal(0, 10, (img_h, img_w))
    bg_gray = np.clip(bg_base + bg_noise, 0, 255).astype(np.uint8)
    bg_bgr = cv2.applyColorMap(bg_gray, cv2.COLORMAP_PINK)

    pil_img = Image.fromarray(cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # 2. Draw product subject: Skincare Bottle (center: x=500, y=700)
    # Bottle Body
    body_box = [370, 420, 630, 960]
    draw.rounded_rectangle(body_box, radius=40, fill=(35, 75, 145), outline=(20, 45, 95), width=3)

    # Bottle Label
    label_box = [400, 560, 600, 840]
    draw.rectangle(label_box, fill=(245, 245, 240), outline=(200, 200, 190), width=2)
    draw.text((435, 620), "ORGANIC\nSERUM", fill=(25, 25, 25))

    # Bottle Neck & Cap
    neck_box = [460, 360, 540, 420]
    draw.rectangle(neck_box, fill=(210, 175, 95), outline=(170, 140, 70), width=2)
    cap_box = [450, 280, 550, 360]
    draw.rounded_rectangle(cap_box, radius=10, fill=(240, 240, 240), outline=(190, 190, 190), width=2)

    pil_img.save(output_path, quality=95)
    return output_path


if pytest is not None:
    session_fixture = pytest.fixture(scope="session")
else:
    def session_fixture(fn):
        return fn


@session_fixture
def dummy_image_path(tmp_path_factory=None) -> Path:
    """Fixture providing a sample smartphone image."""
    data_raw = PROJECT_ROOT / "data" / "raw"
    sample_file = data_raw / "dummy_product.jpg"
    return create_dummy_product_photo(sample_file)


def test_segmenter(dummy_image_path: Path):
    """Verify segmenter removes background and outputs tightly cropped RGBA."""
    segmenter = ImageSegmenter()
    isolated = segmenter.remove_background(dummy_image_path)

    assert isinstance(isolated, Image.Image)
    assert isolated.mode == "RGBA"

    # Confirm non-zero alpha pixels exist
    alpha = np.array(isolated)[:, :, 3]
    assert np.count_nonzero(alpha) > 1000

    # Ensure corners are transparent
    assert alpha[0, 0] == 0
    assert alpha[0, -1] == 0


def test_shadow_engine():
    """Verify shadow generation creates correct dimensional shadow mask."""
    # Create simple circular subject
    sub = Image.new("RGBA", (300, 400), (0, 0, 0, 0))
    draw = ImageDraw.Draw(sub)
    draw.rectangle([50, 50, 250, 380], fill=(50, 100, 200, 255))

    engine = ShadowEngine(cast_opacity=0.4, contact_opacity=0.7)
    shadow = engine.generate_shadow(
        subject_rgba=sub,
        canvas_size=(1000, 1000),
        subject_position=(350, 300),
    )

    assert shadow.size == (1000, 1000)
    assert shadow.mode == "RGBA"

    shadow_np = np.array(shadow)
    # Check that shadow alpha is present near the bottom baseline
    sub_baseline_y = 300 + 400
    region_below = shadow_np[sub_baseline_y - 20 : sub_baseline_y + 80, 350:650, 3]
    assert np.max(region_below) > 30


def test_canvas_styler():
    """Verify CanvasStyler scales to 2000x2000 pure white canvas."""
    sub = Image.new("RGBA", (400, 600), (0, 0, 0, 0))
    draw = ImageDraw.Draw(sub)
    draw.rectangle([50, 50, 350, 550], fill=(200, 40, 40, 255))

    styler = CanvasStyler(canvas_size=2000, fill_ratio=0.8)
    white_img, trans_img = styler.process(sub, add_shadow=True)

    assert white_img.size == (2000, 2000)
    assert white_img.mode == "RGB"
    assert trans_img.size == (2000, 2000)
    assert trans_img.mode == "RGBA"

    # Verify four corners are pure white (#FFFFFF)
    white_np = np.array(white_img)
    assert tuple(white_np[0, 0]) == (255, 255, 255)
    assert tuple(white_np[0, 1999]) == (255, 255, 255)
    assert tuple(white_np[1999, 0]) == (255, 255, 255)
    assert tuple(white_np[1999, 1999]) == (255, 255, 255)


def test_full_pipeline_execution(dummy_image_path: Path):
    """End-to-end verification of Phase 1 pipeline from raw photo to export."""
    data_processed = PROJECT_ROOT / "data" / "processed"
    data_processed.mkdir(parents=True, exist_ok=True)

    studio_white, studio_trans, isolated = process_catalog_image(
        image_input=dummy_image_path,
        canvas_size=2000,
        fill_ratio=0.82,
        add_shadow=True,
    )

    # Save outputs for inspection
    out_white_path = data_processed / "test_studio_white_2000.jpg"
    out_trans_path = data_processed / "test_studio_transparent_2000.png"
    out_iso_path = data_processed / "test_isolated_subject.png"

    studio_white.save(out_white_path, format="JPEG", quality=95)
    studio_trans.save(out_trans_path, format="PNG")
    isolated.save(out_iso_path, format="PNG")

    assert out_white_path.exists()
    assert out_trans_path.exists()
    assert studio_white.size == (2000, 2000)
    assert studio_trans.size == (2000, 2000)

    # Validate pure white corners
    img_arr = np.array(studio_white)
    assert np.all(img_arr[:10, :10] == [255, 255, 255])
    assert np.all(img_arr[-10:, -10:] == [255, 255, 255])


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 1 IMAGE PROCESSING ENGINE TEST SUITE")
    print("=" * 60)

    print("\n[1/4] Generating dummy smartphone product photo...")
    sample_path = PROJECT_ROOT / "data" / "raw" / "dummy_product.jpg"
    create_dummy_product_photo(sample_path)
    print(f"  -> Generated: {sample_path}")

    print("\n[2/4] Testing ImageSegmenter (rembg + edge matting)...")
    test_segmenter(sample_path)
    print("  -> Passed: Subject cleanly isolated, alpha mask validated.")

    print("\n[3/4] Testing ShadowEngine & CanvasStyler...")
    test_shadow_engine()
    test_canvas_styler()
    print("  -> Passed: Synthetic shadow & 2000x2000 canvas verified.")

    print("\n[4/4] Testing Full End-to-End Pipeline...")
    test_full_pipeline_execution(sample_path)
    print("  -> Passed: Exported studio images verified.")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! Output images available in data/processed/")
    print("=" * 60)
