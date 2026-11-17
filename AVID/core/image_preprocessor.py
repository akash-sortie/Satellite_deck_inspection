"""
AVID Stage 1b — Image Pre-processing
======================================
Prepares a raw, high-resolution metallic deck photograph for reliable
feature detection by suppressing glare, shadows, and surface scratches.

Processing chain (order matters):
    1. Grayscale conversion
    2. CLAHE            — local contrast normalization (fights glare/shadows)
    3. Bilateral filter — edge-preserving smoothing (kills scratches)
    4. Top-hat morph    — removes large-scale illumination gradients
    5. Sharpening mask  — recovers hole rim sharpness lost in filtering
    6. Normalize        — final 0-255 uint8 output

Each step is independently tunable via PreprocessConfig.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PreprocessConfig:
    """Tunable parameters for the pre-processing chain."""

    # CLAHE
    clahe_clip_limit: float = 3.0    # higher = more contrast boost; 2-4 typical
    clahe_tile_size: Tuple[int, int] = (16, 16)  # smaller = more local adaptation

    # Bilateral filter (applied after CLAHE)
    bilateral_d: int = 11            # diameter of pixel neighbourhood
    bilateral_sigma_color: float = 60.0   # color similarity bandwidth
    bilateral_sigma_space: float = 60.0   # spatial distance weight

    # Top-hat morphological transform
    apply_tophat: bool = True
    tophat_kernel_size: int = 61     # must be odd; roughly 2× largest glare blob

    # Unsharp mask (sharpening)
    apply_sharpen: bool = True
    sharpen_kernel_size: int = 5
    sharpen_amount: float = 1.5      # 1.0 = no change; >1.0 = sharpen

    # Optional: downscale factor (for development; set 1.0 for production)
    downscale_factor: float = 1.0    # 0.5 = half resolution; saves memory in dev


# ---------------------------------------------------------------------------
# Pre-processing stages
# ---------------------------------------------------------------------------

def _to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert BGR or RGBA image to grayscale."""
    if img.ndim == 2:
        return img.copy()
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _apply_clahe(gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization.
    Normalizes local contrast across metallic surfaces with uneven lighting.
    Critical for making dark-shadowed and over-exposed regions comparable.
    """
    clahe = cv2.createCLAHE(
        clipLimit=cfg.clahe_clip_limit,
        tileGridSize=cfg.clahe_tile_size,
    )
    return clahe.apply(gray)


def _apply_bilateral(gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """
    Bilateral filter: smooths noise while preserving edges (hole rims).
    Superior to Gaussian for this use case because it does not blur
    the sharp circular boundary that Hough circles rely on.
    """
    return cv2.bilateralFilter(
        gray,
        d=cfg.bilateral_d,
        sigmaColor=cfg.bilateral_sigma_color,
        sigmaSpace=cfg.bilateral_sigma_space,
    )


def _apply_tophat(gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """
    White top-hat transform: extracts small bright features by subtracting
    the morphological opening (large bright background = glare regions).
    Result isolates hole rims and small surface detail against a flatter
    background. Add result to original for glare-suppressed image.
    """
    k = cfg.tophat_kernel_size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    # Blend: suppress overall brightness bias from large glare regions
    suppressed = cv2.addWeighted(gray, 0.7, tophat, 0.5, 0)
    return np.clip(suppressed, 0, 255).astype(np.uint8)


def _apply_unsharp_mask(gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """
    Unsharp masking to recover edge sharpness after bilateral smoothing.
    Sharpened = original + amount * (original - blurred)
    """
    k = cfg.sharpen_kernel_size
    blurred = cv2.GaussianBlur(gray, (k, k), 0)
    sharpened = cv2.addWeighted(
        gray, 1 + cfg.sharpen_amount,
        blurred, -cfg.sharpen_amount,
        0
    )
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def _normalize_output(gray: np.ndarray) -> np.ndarray:
    """Stretch histogram to full 0-255 range for maximum detector sensitivity."""
    return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def preprocess_image(
    image: np.ndarray | str | Path,
    cfg: Optional[PreprocessConfig] = None,
    save_debug: bool = False,
    debug_dir: Optional[str | Path] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Run the full pre-processing chain on an image.

    Parameters
    ----------
    image   : BGR numpy array OR path to image file
    cfg     : PreprocessConfig (defaults applied if None)
    save_debug  : if True, saves intermediate images for tuning
    debug_dir   : folder to write debug images into

    Returns
    -------
    processed : uint8 grayscale numpy array, ready for detection
    meta      : dict with image dimensions and applied config
    """
    if cfg is None:
        cfg = PreprocessConfig()

    # --- Load ---
    if isinstance(image, (str, Path)):
        raw = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if raw is None:
            raise IOError(f"Failed to load image: {image}")
        logger.info(f"Loaded image: {Path(image).name}  shape={raw.shape}")
    else:
        raw = image.copy()

    # --- Optional downscale (dev/tuning mode) ---
    if cfg.downscale_factor != 1.0:
        new_w = int(raw.shape[1] * cfg.downscale_factor)
        new_h = int(raw.shape[0] * cfg.downscale_factor)
        raw = cv2.resize(raw, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.debug(f"Downscaled to {raw.shape[:2]}")

    stages = {}

    # --- Chain ---
    gray = _to_grayscale(raw)
    stages["00_gray"] = gray

    after_clahe = _apply_clahe(gray, cfg)
    stages["01_clahe"] = after_clahe

    after_bilateral = _apply_bilateral(after_clahe, cfg)
    stages["02_bilateral"] = after_bilateral

    current = after_bilateral
    if cfg.apply_tophat:
        current = _apply_tophat(current, cfg)
        stages["03_tophat"] = current

    if cfg.apply_sharpen:
        current = _apply_unsharp_mask(current, cfg)
        stages["04_sharpen"] = current

    processed = _normalize_output(current)
    stages["05_final"] = processed

    logger.info(
        f"Pre-processing complete. Output shape: {processed.shape}, "
        f"dtype: {processed.dtype}, "
        f"min/max: {processed.min()}/{processed.max()}"
    )

    # --- Optional debug saves ---
    if save_debug and debug_dir:
        debug_path = Path(debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)
        for name, img in stages.items():
            cv2.imwrite(str(debug_path / f"{name}.png"), img)
        logger.info(f"Debug images written to: {debug_path}")

    meta = {
        "original_shape": raw.shape,
        "processed_shape": processed.shape,
        "downscale_factor": cfg.downscale_factor,
        "config": cfg,
    }
    return processed, meta


# ---------------------------------------------------------------------------
# Synthetic image generator (for Stage 1 validation — no real photos needed)
# ---------------------------------------------------------------------------

def generate_synthetic_deck_image(
    width_px: int = 2400,
    height_px: int = 1600,
    holes: Optional[list] = None,  # list of (x_px, y_px, radius_px)
    add_glare: bool = True,
    add_scratches: bool = True,
    add_noise: bool = True,
    noise_std: float = 12.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a synthetic metallic deck photograph with realistic
    disturbances: glare blobs, directional scratches, Gaussian noise.

    Useful for homography pipeline validation without a real photograph.

    Parameters
    ----------
    holes : list of (cx_px, cy_px, radius_px) tuples for ground-truth holes

    Returns
    -------
    BGR uint8 image
    """
    rng = np.random.default_rng(seed)

    # Metallic gray base
    base_val = 160
    img = np.full((height_px, width_px, 3), base_val, dtype=np.uint8)

    # Subtle metal grain texture
    grain = rng.integers(-20, 20, (height_px, width_px), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + grain[:, :, None], 0, 255).astype(np.uint8)

    # --- Glare blobs (high-brightness ellipses) ---
    if add_glare:
        n_glare = rng.integers(3, 8)
        for _ in range(n_glare):
            cx = int(rng.integers(0, width_px))
            cy = int(rng.integers(0, height_px))
            rx = int(rng.integers(100, 400))
            ry = int(rng.integers(50, 200))
            angle = int(rng.integers(0, 180))
            intensity = int(rng.integers(200, 255))
            overlay = img.copy()
            cv2.ellipse(overlay, (cx, cy), (rx, ry), angle, 0, 360,
                        (intensity, intensity, intensity), -1)
            alpha = rng.uniform(0.3, 0.7)
            img = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)

    # --- Directional scratches (thin dark lines at shallow angles) ---
    if add_scratches:
        n_scratches = rng.integers(10, 30)
        for _ in range(n_scratches):
            x1 = int(rng.integers(0, width_px))
            y1 = int(rng.integers(0, height_px))
            length = int(rng.integers(80, 400))
            angle_rad = rng.uniform(-0.3, 0.3)  # near-horizontal
            x2 = int(x1 + length * np.cos(angle_rad))
            y2 = int(y1 + length * np.sin(angle_rad))
            color = int(rng.integers(80, 130))
            thickness = 1
            cv2.line(img, (x1, y1), (x2, y2), (color, color, color), thickness)

    # --- Drill holes (dark filled circles with bright rim) ---
    if holes:
        for (cx, cy, r) in holes:
            # Dark interior (drill through)
            cv2.circle(img, (int(cx), int(cy)), int(r), (30, 30, 30), -1)
            # Bright metallic rim from drilling
            cv2.circle(img, (int(cx), int(cy)), int(r), (220, 220, 220), 2)
            # Subtle inner shadow gradient (approximate)
            cv2.circle(img, (int(cx), int(cy)), max(int(r) - 3, 1),
                       (15, 15, 15), -1)

    # --- Gaussian noise ---
    if add_noise:
        noise = rng.normal(0, noise_std, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img
