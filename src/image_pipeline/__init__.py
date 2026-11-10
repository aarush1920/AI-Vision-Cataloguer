"""Image Processing Engine package."""

from .segmenter import ImageSegmenter, remove_background
from .shadow_engine import ShadowEngine
from .canvas_styler import CanvasStyler
from PIL import Image
from typing import Union, Tuple
from pathlib import Path
import numpy as np


def process_catalog_image(
    image_input: Union[str, Path, bytes, Image.Image, np.ndarray],
    canvas_size: int = 2000,
    fill_ratio: float = 0.82,
    add_shadow: bool = True,
    segmenter: ImageSegmenter | None = None,
    shadow_engine: ShadowEngine | None = None,
) -> Tuple[Image.Image, Image.Image, Image.Image]:
    """Execute the end-to-end Phase 1 image transformation pipeline.
    
    Args:
        image_input: Raw input image (path, bytes, PIL Image, or numpy array).
        canvas_size: Target square canvas size (default 2000).
        fill_ratio: Percentage of canvas occupied by product (default 0.82).
        add_shadow: Whether to render realistic contact and drop-shadows.
        segmenter: Optional custom ImageSegmenter instance.
        shadow_engine: Optional custom ShadowEngine instance.
        
    Returns:
        Tuple of:
            1. studio_white: 2000x2000 RGB image on pure white (#FFFFFF) background.
            2. studio_transparent: 2000x2000 RGBA image with transparent background and shadow.
            3. isolated_subject: Tightly-cropped RGBA image of the isolated product.
    """
    # 1. Background removal & edge refinement
    seg = segmenter or ImageSegmenter()
    isolated_subject = seg.remove_background(image_input)

    # 2. Composition on standardized canvas with shadow
    styler = CanvasStyler(
        canvas_size=canvas_size,
        fill_ratio=fill_ratio,
        shadow_engine=shadow_engine,
    )
    studio_white, studio_transparent = styler.process(isolated_subject, add_shadow=add_shadow)

    return studio_white, studio_transparent, isolated_subject


__all__ = [
    "ImageSegmenter",
    "ShadowEngine",
    "CanvasStyler",
    "remove_background",
    "process_catalog_image",
]
