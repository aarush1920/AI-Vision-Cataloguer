"""Canvas standardizer and composition engine for e-commerce catalog standards."""

from __future__ import annotations

from typing import Tuple
import numpy as np
from PIL import Image

from .shadow_engine import ShadowEngine


class CanvasStyler:
    """Standardizes product dimensions on a 2000x2000 square canvas with realistic shadows."""

    def __init__(
        self,
        canvas_size: int = 2000,
        fill_ratio: float = 0.82,
        background_color: Tuple[int, int, int] = (255, 255, 255),
        shadow_engine: ShadowEngine | None = None,
    ) -> None:
        """Initialize canvas styler.
        
        Args:
            canvas_size: Target square canvas dimension (default 2000 for Amazon standards).
            fill_ratio: Proportion of canvas occupied by subject (default 0.82 for ~82-85% fill).
            background_color: RGB background color tuple (default pure white (255, 255, 255)).
            shadow_engine: Instance of ShadowEngine or None (default used).
        """
        self.canvas_size = canvas_size
        self.fill_ratio = fill_ratio
        self.background_color = background_color
        self.shadow_engine = shadow_engine or ShadowEngine()

    def process(
        self,
        subject_rgba: Image.Image,
        add_shadow: bool = True,
    ) -> tuple[Image.Image, Image.Image]:
        """Center the subject onto a 2000x2000 canvas with optional synthetic shadow.
        
        Args:
            subject_rgba: Tightly cropped isolated product RGBA PIL Image.
            add_shadow: Whether to render synthetic contact/drop-shadows.
            
        Returns:
            Tuple of:
                1. studio_rgb: 2000x2000 RGB image on pure white background.
                2. studio_rgba: 2000x2000 RGBA image with transparent background (including shadow).
        """
        sub_w, sub_h = subject_rgba.size
        max_dim = max(sub_w, sub_h)

        # Scale subject to occupy target fill ratio of canvas
        target_max_dim = int(self.canvas_size * self.fill_ratio)
        scale_factor = target_max_dim / max(max_dim, 1)

        new_w = max(1, int(sub_w * scale_factor))
        new_h = max(1, int(sub_h * scale_factor))

        resized_subject = subject_rgba.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

        # Position calculations: center horizontally, baseline positioned slightly below midpoint
        # to allow breathing room and shadow footprint
        pos_x = (self.canvas_size - new_w) // 2
        # Center vertically with slight upward shift to accommodate contact shadow
        base_y = (self.canvas_size - new_h) // 2
        pos_y = max(20, base_y - 20)

        # Create transparent canvas
        canvas_rgba = Image.new("RGBA", (self.canvas_size, self.canvas_size), (0, 0, 0, 0))

        # Add synthetic shadow layer if requested
        if add_shadow:
            shadow_layer = self.shadow_engine.generate_shadow(
                subject_rgba=resized_subject,
                canvas_size=(self.canvas_size, self.canvas_size),
                subject_position=(pos_x, pos_y),
            )
            canvas_rgba.alpha_composite(shadow_layer)

        # Composite the foreground subject
        canvas_rgba.alpha_composite(resized_subject, dest=(pos_x, pos_y))

        # Render pure white background version (Amazon standard)
        white_bg = Image.new("RGB", (self.canvas_size, self.canvas_size), self.background_color)
        white_bg.paste(canvas_rgba, mask=canvas_rgba.split()[3])

        return white_bg, canvas_rgba
