"""Image segmentation and background removal using rembg with alpha edge refinement."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image
import rembg


class ImageSegmenter:
    """Removes backgrounds from product images and refines alpha boundaries."""

    def __init__(self, model_name: str = "u2netp") -> None:
        """Initialize segmenter with a specific rembg model session.
        
        Args:
            model_name: Name of rembg ONNX model (e.g. 'u2netp', 'u2net', 'birefnet-general').
                        Defaults to 'u2netp' for fast, lightweight inference.
        """
        self.model_name = model_name
        try:
            self.session = rembg.new_session(model_name)
        except Exception as e:
            if model_name != "u2netp":
                print(f"[ImageSegmenter] Could not load '{model_name}' ({e}). Falling back to 'u2netp'...")
                self.model_name = "u2netp"
                self.session = rembg.new_session("u2netp")
            else:
                raise

    def remove_background(
        self,
        image_input: Union[str, Path, bytes, Image.Image, np.ndarray],
        alpha_matting: bool = True,
        alpha_matting_foreground_threshold: int = 240,
        alpha_matting_background_threshold: int = 10,
        alpha_matting_erode_size: int = 10,
        edge_feather: int = 1,
        auto_crop: bool = True,
    ) -> Image.Image:
        """Isolate the main subject, remove the background, and refine edges.
        
        Args:
            image_input: File path, bytes, PIL Image, or numpy array.
            alpha_matting: Whether to enable rembg's built-in alpha matting.
            alpha_matting_foreground_threshold: Foreground threshold for trimap.
            alpha_matting_background_threshold: Background threshold for trimap.
            alpha_matting_erode_size: Erode size for trimap generation.
            edge_feather: Radius for subtle alpha edge smoothing (removes harsh cutouts).
            auto_crop: Whether to crop tightly to non-empty alpha bounding box.
            
        Returns:
            RGBA PIL Image with transparent background.
        """
        # Standardize input to PIL Image in RGB format
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, bytes):
            pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if image_input.shape[2] == 4:
                pil_img = Image.fromarray(image_input).convert("RGB")
            else:
                pil_img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        # Optimize high-resolution smartphone photos to 2048px max dimension
        # Prevents pymatting Laplacians from exceeding available system RAM
        max_edge = max(pil_img.size)
        if max_edge > 2048:
            scale = 2048.0 / max_edge
            new_size = (max(1, int(pil_img.width * scale)), max(1, int(pil_img.height * scale)))
            pil_img = pil_img.resize(new_size, resample=Image.Resampling.LANCZOS)

        # Execute rembg background removal with resilient fallback
        try:
            isolated = rembg.remove(
                pil_img,
                session=self.session,
                alpha_matting=alpha_matting,
                alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                alpha_matting_background_threshold=alpha_matting_background_threshold,
                alpha_matting_erode_size=alpha_matting_erode_size,
            )
        except Exception as e:
            # Fall back to pure neural network segmentation if pymatting OOMs
            isolated = rembg.remove(
                pil_img,
                session=self.session,
                alpha_matting=False,
            )

        # Refine alpha edges using OpenCV
        refined_rgba = self._refine_alpha_mask(isolated, edge_feather=edge_feather)

        # Optional tight crop to subject bounding box
        if auto_crop:
            bbox = refined_rgba.getbbox()
            if bbox:
                refined_rgba = refined_rgba.crop(bbox)

        return refined_rgba

    def _refine_alpha_mask(self, rgba_image: Image.Image, edge_feather: int = 1) -> Image.Image:
        """Clean noise speckles and soften jagged alpha cutouts."""
        np_img = np.array(rgba_image)
        if np_img.shape[2] != 4:
            return rgba_image

        alpha = np_img[:, :, 3]

        # Suppress tiny alpha artifacts (threshold near-zero alpha)
        alpha[alpha < 8] = 0

        # Remove small disconnected stray pixel noise using morphological opening
        kernel_noise = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        alpha_cleaned = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel_noise)

        # Subtle Gaussian edge feathering for smooth transitions
        if edge_feather > 0:
            ksize = edge_feather * 2 + 1
            alpha_cleaned = cv2.GaussianBlur(alpha_cleaned, (ksize, ksize), 0)

        np_img[:, :, 3] = alpha_cleaned
        return Image.fromarray(np_img)


# Module-level convenience singleton
_default_segmenter: ImageSegmenter | None = None


def remove_background(
    image_input: Union[str, Path, bytes, Image.Image, np.ndarray],
    **kwargs,
) -> Image.Image:
    """Convenience function using a shared segmenter instance."""
    global _default_segmenter
    if _default_segmenter is None:
        _default_segmenter = ImageSegmenter()
    return _default_segmenter.remove_background(image_input, **kwargs)
