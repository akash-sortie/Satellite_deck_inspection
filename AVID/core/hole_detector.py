"""
AVID Stage 4 — Targeted ROI Hole Detection
==========================================
Uses the homography from Stage 3 to project each CAD hole to its
predicted pixel location, then crops a tight ROI and runs circular
feature detection within that ROI.

This avoids:
  - The YOLO patch-boundary problem
  - Class imbalance (holes are <0.1% of total image area)
  - Sub-pixel resolution loss from global tiling

Detection strategy per ROI:
  Primary   : cv2.HoughCircles — best for clean, drill-through circles
  Fallback  : cv2.SimpleBlobDetector — more robust under occlusion/glare

Each hole is classified as:
  OK          — circle found within position tolerance
  MISALIGNED  — circle found but centroid offset > tolerance
  MISSING     — no circle found in ROI
  UNCERTAIN   — circle found but diameter deviates significantly
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .cad_parser import CADManifest, Feature
from .homography import HomographyResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Status & configuration
# ---------------------------------------------------------------------------

class HoleStatus(Enum):
    OK          = "OK"
    MISALIGNED  = "MISALIGNED"
    MISSING     = "MISSING"
    UNCERTAIN   = "UNCERTAIN"   # found but diameter anomaly


@dataclass
class DetectionConfig:
    # ROI size: factor × expected hole diameter
    roi_scale_factor: float = 3.5

    # Position tolerance: hole is MISALIGNED if offset > this (in mm)
    position_tolerance_mm: float = 1.5

    # Diameter tolerance: flag UNCERTAIN if |detected - expected| > factor × expected
    diameter_tolerance_factor: float = 0.25   # ±25% of nominal diameter

    # Pixels per mm — derived from homography at runtime; set here as fallback
    # Will be computed dynamically from the homography scale.
    px_per_mm_estimate: float = 5.0

    # HoughCircles parameters
    hough_dp: float = 1.2            # inverse resolution ratio
    hough_min_dist_factor: float = 0.5       # min dist = factor × expected_diameter
    hough_param1: float = 80.0       # upper Canny threshold
    hough_param2: float = 25.0       # accumulator threshold (lower = more detections)
    hough_min_radius_factor: float = 0.25  # min_r = factor × expected_r_px
    hough_max_radius_factor: float = 1.75  # max_r = factor × expected_r_px

    # Fallback blob detector
    blob_min_circularity: float = 0.70
    blob_min_convexity: float = 0.80


# ---------------------------------------------------------------------------
# Per-hole detection result
# ---------------------------------------------------------------------------

@dataclass
class HoleResult:
    feature_id: str
    status: HoleStatus

    # CAD nominal values (normalized mm)
    cad_x_mm: float
    cad_y_mm: float
    cad_diameter_mm: float

    # Projected pixel prediction
    predicted_col_px: float
    predicted_row_px: float

    # Detected position (None if MISSING)
    detected_col_px: Optional[float] = None
    detected_row_px: Optional[float] = None
    detected_diameter_px: Optional[float] = None

    # Computed offset (None if MISSING)
    offset_px: Optional[float] = None
    offset_mm: Optional[float] = None    # back-projected via H_inv scale

    # Debug: the ROI crop (not serialized to report)
    roi_crop: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def is_flagged(self) -> bool:
        return self.status != HoleStatus.OK


# ---------------------------------------------------------------------------
# ROI utilities
# ---------------------------------------------------------------------------

def _compute_px_per_mm(homography: HomographyResult) -> float:
    """
    Estimate the local scale factor (pixels per mm) from the homography.
    Uses the Jacobian at the origin for a first-order estimate.
    """
    # Transform two points 1 mm apart in CAD space
    pt1 = np.array([[[0.0, 0.0]]], dtype=np.float64)
    pt2 = np.array([[[1.0, 0.0]]], dtype=np.float64)
    p1_px = cv2.perspectiveTransform(pt1, homography.H)[0][0]
    p2_px = cv2.perspectiveTransform(pt2, homography.H)[0][0]
    scale = float(np.linalg.norm(p2_px - p1_px))
    logger.info(f"Estimated scale: {scale:.3f} px/mm")
    return scale


def _extract_roi(
    image: np.ndarray,
    cx_px: float,
    cy_px: float,
    roi_radius_px: float,
) -> Tuple[np.ndarray, int, int]:
    """
    Extract a square ROI around (cx_px, cy_px) with half-width roi_radius_px.
    Returns (roi_crop, x_offset, y_offset) for back-projecting detections.
    Clamps to image boundaries safely.
    """
    h, w = image.shape[:2]
    r = int(np.ceil(roi_radius_px))

    x1 = max(0, int(cx_px) - r)
    y1 = max(0, int(cy_px) - r)
    x2 = min(w, int(cx_px) + r)
    y2 = min(h, int(cy_px) + r)

    return image[y1:y2, x1:x2].copy(), x1, y1


# ---------------------------------------------------------------------------
# Per-ROI detection
# ---------------------------------------------------------------------------

def _detect_circle_hough(
    roi: np.ndarray,
    expected_r_px: float,
    cfg: DetectionConfig,
) -> Optional[Tuple[float, float, float]]:
    """
    Run HoughCircles within a single ROI.
    Returns (cx_roi, cy_roi, radius_px) in ROI-local coordinates, or None.
    """
    if roi.size == 0:
        return None

    # Local CLAHE for uneven illumination within the ROI
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    roi_eq = clahe.apply(roi)

    min_r = max(3, int(expected_r_px * cfg.hough_min_radius_factor))
    max_r = max(min_r + 2, int(expected_r_px * cfg.hough_max_radius_factor))
    # minDist must be >> 0 (avoid double-detecting same circle) but <<< ROI size.
    # Using expected_diameter ensures we don't suppress the single valid detection.
    min_dist = max(int(expected_r_px * 2 * cfg.hough_min_dist_factor), 6)

    circles = cv2.HoughCircles(
        roi_eq,
        cv2.HOUGH_GRADIENT,
        dp=cfg.hough_dp,
        minDist=min_dist,
        param1=cfg.hough_param1,
        param2=cfg.hough_param2,
        minRadius=min_r,
        maxRadius=max_r,
    )

    if circles is None:
        return None

    circles = np.round(circles[0]).astype(np.float32)

    # If multiple circles found, pick the one closest to ROI center
    roi_cx = roi.shape[1] / 2.0
    roi_cy = roi.shape[0] / 2.0
    best = min(circles, key=lambda c: (c[0] - roi_cx)**2 + (c[1] - roi_cy)**2)

    return float(best[0]), float(best[1]), float(best[2])


def _detect_circle_blob(
    roi: np.ndarray,
    expected_r_px: float,
    cfg: DetectionConfig,
) -> Optional[Tuple[float, float, float]]:
    """
    Fallback blob detector within a ROI.
    Returns (cx_roi, cy_roi, radius_px) or None.
    """
    params = cv2.SimpleBlobDetector_Params()
    area = np.pi * expected_r_px ** 2
    params.filterByArea        = True
    params.minArea             = area * 0.15
    params.maxArea             = area * 3.0
    params.filterByCircularity = True
    params.minCircularity      = cfg.blob_min_circularity
    params.filterByConvexity   = True
    params.minConvexity        = cfg.blob_min_convexity
    params.filterByColor       = True
    params.blobColor           = 0   # dark holes

    detector = cv2.SimpleBlobDetector_create(params)
    kps = detector.detect(roi)

    if not kps:
        return None

    # Pick closest to center
    roi_cx, roi_cy = roi.shape[1] / 2.0, roi.shape[0] / 2.0
    best = min(kps, key=lambda kp: (kp.pt[0]-roi_cx)**2 + (kp.pt[1]-roi_cy)**2)
    return float(best.pt[0]), float(best.pt[1]), float(best.size / 2.0)


# ---------------------------------------------------------------------------
# Status classification
# ---------------------------------------------------------------------------

def _classify_detection(
    detected: Optional[Tuple[float, float, float]],  # (roi_cx, roi_cy, r_px)
    roi_x_offset: int,
    roi_y_offset: int,
    predicted_col: float,
    predicted_row: float,
    expected_diameter_mm: float,
    px_per_mm: float,
    cfg: DetectionConfig,
    homography: HomographyResult,
) -> Tuple[HoleStatus, Optional[float], Optional[float], Optional[float],
           Optional[float], Optional[float]]:
    """
    Returns (status, det_col, det_row, det_diam_px, offset_px, offset_mm).
    """
    tol_px = cfg.position_tolerance_mm * px_per_mm

    if detected is None:
        return HoleStatus.MISSING, None, None, None, None, None

    roi_cx, roi_cy, r_px = detected
    det_col = roi_x_offset + roi_cx
    det_row = roi_y_offset + roi_cy
    det_diam_px = 2 * r_px

    offset_px = float(np.hypot(det_col - predicted_col, det_row - predicted_row))
    offset_mm = offset_px / px_per_mm

    # Diameter check
    expected_r_px = (expected_diameter_mm / 2.0) * px_per_mm
    diam_tol = cfg.diameter_tolerance_factor * expected_diameter_mm * px_per_mm * 2
    diam_dev = abs(det_diam_px - expected_r_px * 2)

    if offset_px > tol_px:
        status = HoleStatus.MISALIGNED
    elif diam_dev > diam_tol:
        status = HoleStatus.UNCERTAIN
    else:
        status = HoleStatus.OK

    return status, det_col, det_row, det_diam_px, offset_px, offset_mm


# ---------------------------------------------------------------------------
# Main Stage 4 entry point
# ---------------------------------------------------------------------------

def detect_holes(
    processed_image: np.ndarray,
    manifest: CADManifest,
    homography: HomographyResult,
    cfg: Optional[DetectionConfig] = None,
    features_to_inspect: Optional[List[str]] = None,
    save_roi_crops: bool = False,
) -> List[HoleResult]:
    """
    Stage 4: Run targeted hole detection on every (or specified) hole in manifest.

    Parameters
    ----------
    processed_image   : grayscale uint8 from Stage 1b
    manifest          : parsed CAD manifest from Stage 1a
    homography        : HomographyResult from Stage 3
    cfg               : DetectionConfig (defaults if None)
    features_to_inspect : list of feature IDs to inspect (None = all holes)
    save_roi_crops    : if True, attach roi_crop arrays to results (for debug)

    Returns
    -------
    List of HoleResult, one per inspected feature.
    """
    if cfg is None:
        cfg = DetectionConfig()

    if not homography.is_valid:
        raise RuntimeError(
            "Homography is invalid. Cannot safely project hole positions. "
            "Resolve fiducial detection issues in Stage 2-3 first."
        )

    # Compute scale
    px_per_mm = _compute_px_per_mm(homography)

    # Select which features to inspect
    target_ids = set(features_to_inspect) if features_to_inspect else None
    features = [
        f for f in manifest.features
        if f.type in ("hole", "insert")
        and (target_ids is None or f.id in target_ids)
    ]

    logger.info(
        f"Stage 4: Inspecting {len(features)} features "
        f"on a {processed_image.shape[1]}×{processed_image.shape[0]} px image."
    )

    # Pre-project all selected features
    all_pts_mm = np.array(
        [[[f.x_norm_mm, f.y_norm_mm]] for f in features], dtype=np.float64
    )
    all_pts_px = cv2.perspectiveTransform(all_pts_mm, homography.H)

    results: List[HoleResult] = []

    for i, feature in enumerate(features):
        pred_col = float(all_pts_px[i][0][0])
        pred_row = float(all_pts_px[i][0][1])

        # ROI size based on nominal diameter
        expected_r_px = (feature.diameter_mm / 2.0) * px_per_mm
        roi_radius_px = expected_r_px * cfg.roi_scale_factor

        roi, rx0, ry0 = _extract_roi(
            processed_image, pred_col, pred_row, roi_radius_px
        )

        # Primary: HoughCircles
        detected = _detect_circle_hough(roi, expected_r_px, cfg)

        # Fallback: blob detector
        if detected is None:
            logger.debug(f"  {feature.id}: Hough failed, trying blob fallback.")
            detected = _detect_circle_blob(roi, expected_r_px, cfg)

        status, det_col, det_row, det_diam_px, offset_px, offset_mm = \
            _classify_detection(
                detected, rx0, ry0,
                pred_col, pred_row,
                feature.diameter_mm, px_per_mm, cfg, homography
            )

        if status != HoleStatus.OK:
            logger.warning(
                f"  [{status.value:12s}] {feature.id}  "
                f"predicted=({pred_col:.1f},{pred_row:.1f}) px  "
                + (f"offset={offset_mm:.2f} mm" if offset_mm is not None else "")
            )
        else:
            logger.debug(f"  [OK         ] {feature.id}")

        results.append(HoleResult(
            feature_id        = feature.id,
            status            = status,
            cad_x_mm          = feature.x_mm,
            cad_y_mm          = feature.y_mm,
            cad_diameter_mm   = feature.diameter_mm,
            predicted_col_px  = pred_col,
            predicted_row_px  = pred_row,
            detected_col_px   = det_col,
            detected_row_px   = det_row,
            detected_diameter_px = det_diam_px,
            offset_px         = offset_px,
            offset_mm         = offset_mm,
            roi_crop          = roi if save_roi_crops else None,
        ))

    n_ok        = sum(1 for r in results if r.status == HoleStatus.OK)
    n_missing   = sum(1 for r in results if r.status == HoleStatus.MISSING)
    n_misalign  = sum(1 for r in results if r.status == HoleStatus.MISALIGNED)
    n_uncertain = sum(1 for r in results if r.status == HoleStatus.UNCERTAIN)

    logger.info(
        f"Stage 4 complete:  "
        f"OK={n_ok}  MISSING={n_missing}  "
        f"MISALIGNED={n_misalign}  UNCERTAIN={n_uncertain}"
    )

    return results
