"""Realistic synthetic contact and ambient drop-shadow synthesis using OpenCV and NumPy."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


class ShadowEngine:
    """Synthesizes studio-grade contact shadows and soft floor drop-shadows."""

    def __init__(
        self,
        cast_opacity: float = 0.35,
        contact_opacity: float = 0.65,
        cast_blur: int = 41,
        contact_blur: int = 13,
        vertical_scale: float = 0.28,
        offset_y: int = 10,
        color: tuple[int, int, int] = (20, 20, 20),
    ) -> None:
        """Initialize shadow generator configuration.
        
        Args:
            cast_opacity: Opacity multiplier for the diffused ambient floor shadow (0.0 - 1.0).
            contact_opacity: Opacity multiplier for the tight contact ground shadow (0.0 - 1.0).
            cast_blur: Kernel size for ambient shadow Gaussian diffusion (must be odd).
            contact_blur: Kernel size for contact shadow Gaussian blur (must be odd).
            vertical_scale: Perspective vertical squash factor (0.15 - 0.40).
            offset_y: Vertical pixel displacement downward from object baseline.
            color: RGB color tuple for shadow tint (defaults to dark charcoal studio tone).
        """
        self.cast_opacity = cast_opacity
        self.contact_opacity = contact_opacity
        self.cast_blur = cast_blur if cast_blur % 2 == 1 else cast_blur + 1
        self.contact_blur = contact_blur if contact_blur % 2 == 1 else contact_blur + 1
        self.vertical_scale = vertical_scale
        self.offset_y = offset_y
        self.color = color

    def generate_shadow(
        self,
        subject_rgba: Image.Image,
        canvas_size: tuple[int, int],
        subject_position: tuple[int, int],
    ) -> Image.Image:
        """Create a full-canvas RGBA shadow layer aligned under the subject.
        
        Args:
            subject_rgba: Tightly cropped isolated product RGBA PIL Image.
            canvas_size: (width, height) of the target canvas.
            subject_position: (x, y) top-left coordinates of the subject on the canvas.
            
        Returns:
            RGBA PIL Image of the synthesized shadows positioned on the canvas.
        """
        sub_np = np.array(subject_rgba)
        if sub_np.shape[2] != 4:
            raise ValueError("Subject image must have 4 channels (RGBA).")

        alpha_mask = sub_np[:, :, 3]
        sub_h, sub_w = alpha_mask.shape

        canvas_w, canvas_h = canvas_size
        pos_x, pos_y = subject_position

        # 1. Generate soft ambient cast shadow
        cast_shadow = self._build_cast_shadow(alpha_mask, sub_w, sub_h)

        # 2. Generate tight contact grounding shadow
        contact_shadow = self._build_contact_shadow(alpha_mask, sub_w, sub_h)

        # 3. Create full-canvas shadow accumulator in float32
        shadow_canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)

        # Place ambient cast shadow onto canvas
        cast_h, cast_w = cast_shadow.shape
        cast_y = pos_y + sub_h - int(cast_h * 0.35) + self.offset_y
        cast_x = pos_x + (sub_w - cast_w) // 2

        self._paste_mask_onto_canvas(shadow_canvas, cast_shadow, cast_x, cast_y)

        # Place contact shadow onto canvas (anchored at base of product)
        ct_h, ct_w = contact_shadow.shape
        ct_y = pos_y + sub_h - ct_h // 2 + (self.offset_y // 2)
        ct_x = pos_x + (sub_w - ct_w) // 2

        self._paste_mask_onto_canvas(shadow_canvas, contact_shadow, ct_x, ct_y)

        # Clip alpha to maximum 255
        shadow_alpha = np.clip(shadow_canvas, 0, 255).astype(np.uint8)

        # Construct final RGBA shadow image
        shadow_rgba = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
        shadow_rgba[:, :, 0] = self.color[0]
        shadow_rgba[:, :, 1] = self.color[1]
        shadow_rgba[:, :, 2] = self.color[2]
        shadow_rgba[:, :, 3] = shadow_alpha

        return Image.fromarray(shadow_rgba, mode="RGBA")

    def _build_cast_shadow(self, alpha_mask: np.ndarray, w: int, h: int) -> np.ndarray:
        """Create a perspective-squashed, diffusion-blurred cast shadow."""
        # Scale vertically to simulate floor projection
        squashed_h = max(10, int(h * self.vertical_scale))
        squashed = cv2.resize(
            alpha_mask,
            (w, squashed_h),
            interpolation=cv2.INTER_AREA,
        )

        # Apply vertical fade gradient (closer to object base = denser shadow)
        gradient = np.linspace(1.0, 0.15, squashed_h, dtype=np.float32)[:, np.newaxis]
        faded = (squashed.astype(np.float32) * gradient)

        # Add horizontal expansion padding for soft blur diffusion
        pad = self.cast_blur * 2
        padded = np.pad(faded, ((0, pad), (pad, pad)), mode="constant", constant_values=0)

        # Gaussian blur for soft ambient decay
        blurred = cv2.GaussianBlur(padded, (self.cast_blur, self.cast_blur), 0)

        # Multiply by cast opacity
        cast_result = blurred * self.cast_opacity
        return cast_result

    def _build_contact_shadow(self, alpha_mask: np.ndarray, w: int, h: int) -> np.ndarray:
        """Create a thin, dark contact shadow hugging the lowest edge of the product."""
        # Extract bottom 15% of the mask to capture the object contact profile
        slice_h = max(8, int(h * 0.15))
        base_slice = alpha_mask[-slice_h:, :]

        # Squash into a tight ground strip
        contact_h = max(6, int(slice_h * 0.4))
        contact_base = cv2.resize(
            base_slice,
            (w, contact_h),
            interpolation=cv2.INTER_AREA,
        )

        # Pad for contact blur
        pad = self.contact_blur
        padded = np.pad(contact_base, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)

        # Gentle Gaussian blur
        blurred = cv2.GaussianBlur(padded, (self.contact_blur, self.contact_blur), 0)

        # Multiply by contact opacity
        contact_result = (blurred.astype(np.float32)) * self.contact_opacity
        return contact_result

    @staticmethod
    def _paste_mask_onto_canvas(
        canvas: np.ndarray,
        mask: np.ndarray,
        x: int,
        y: int,
    ) -> None:
        """Blend a float shadow mask into the target canvas with boundary clipping."""
        h_c, w_c = canvas.shape
        h_m, w_m = mask.shape

        x1_c = max(0, x)
        y1_c = max(0, y)
        x2_c = min(w_c, x + w_m)
        y2_c = min(h_c, y + h_m)

        if x1_c >= x2_c or y1_c >= y2_c:
            return

        x1_m = x1_c - x
        y1_m = y1_c - y
        x2_m = x1_m + (x2_c - x1_c)
        y2_m = y1_m + (y2_c - y1_c)

        # Use screen/addition blending for shadow accumulation
        current_region = canvas[y1_c:y2_c, x1_c:x2_c]
        incoming_region = mask[y1_m:y2_m, x1_m:x2_m]

        # Alpha screen blend: 255 - (255 - a) * (255 - b) / 255
        blended = 255.0 - ((255.0 - current_region) * (255.0 - incoming_region) / 255.0)
        canvas[y1_c:y2_c, x1_c:x2_c] = blended
