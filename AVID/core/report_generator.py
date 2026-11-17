"""
AVID Stage 5 — Inspection Report Generator
===========================================
Produces two output artifacts:
  1. CSV report    — structured table of all hole statuses + offsets
  2. Visual overlay — original image annotated with color-coded markers

Color convention (consistent with aerospace inspection standards):
  GREEN  — OK
  YELLOW — UNCERTAIN (found but diameter anomaly)
  RED    — MISALIGNED (found but position offset > tolerance)
  MAGENTA— MISSING (not detected at all)
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd

from .hole_detector import HoleResult, HoleStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ReportConfig:
    # Overlay marker settings
    marker_radius_px: int = 20       # radius of drawn circle (not detected)
    marker_thickness: int = 3
    font_scale: float = 0.5
    font_thickness: int = 1
    draw_predicted_center: bool = True   # small dot at CAD-projected center
    draw_offset_line: bool = True        # line from prediction to detection

    # Color map: BGR tuples
    color_ok:          tuple = (0, 220, 0)      # green
    color_misaligned:  tuple = (0, 0, 220)      # red
    color_missing:     tuple = (220, 0, 220)    # magenta
    color_uncertain:   tuple = (0, 200, 220)    # yellow

    # Output
    jpeg_quality: int = 92


STATUS_COLORS = {
    HoleStatus.OK:         (0, 200, 0),
    HoleStatus.MISALIGNED: (0, 0, 220),
    HoleStatus.MISSING:    (200, 0, 200),
    HoleStatus.UNCERTAIN:  (0, 200, 220),
}


# ---------------------------------------------------------------------------
# CSV report
# ---------------------------------------------------------------------------

def generate_csv_report(
    results: List[HoleResult],
    output_path: str | Path,
    include_pixel_coords: bool = False,
) -> pd.DataFrame:
    """
    Write a structured CSV report of all inspection results.

    Columns always included:
        feature_id, status, cad_x_mm, cad_y_mm, cad_diameter_mm,
        offset_mm, detected_diameter_px

    Optional (include_pixel_coords=True):
        predicted_col_px, predicted_row_px,
        detected_col_px, detected_row_px
    """
    rows = []
    for r in results:
        row = {
            "feature_id":        r.feature_id,
            "status":            r.status.value,
            "cad_x_mm":          round(r.cad_x_mm, 3),
            "cad_y_mm":          round(r.cad_y_mm, 3),
            "cad_diameter_mm":   round(r.cad_diameter_mm, 3),
            "offset_mm":         round(r.offset_mm, 3) if r.offset_mm is not None else "",
            "detected_diam_px":  round(r.detected_diameter_px, 1) if r.detected_diameter_px is not None else "",
        }
        if include_pixel_coords:
            row.update({
                "predicted_col_px": round(r.predicted_col_px, 1),
                "predicted_row_px": round(r.predicted_row_px, 1),
                "detected_col_px":  round(r.detected_col_px, 1) if r.detected_col_px is not None else "",
                "detected_row_px":  round(r.detected_row_px, 1) if r.detected_row_px is not None else "",
            })
        rows.append(row)

    df = pd.DataFrame(rows)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"CSV report written: {output_path}")
    return df


# ---------------------------------------------------------------------------
# Visual overlay
# ---------------------------------------------------------------------------

def generate_visual_overlay(
    original_image: np.ndarray,
    results: List[HoleResult],
    output_path: str | Path,
    cfg: Optional[ReportConfig] = None,
    downscale_for_output: float = 1.0,
) -> np.ndarray:
    """
    Draw color-coded inspection markers on the original image and save.

    Marker design:
    - Outer circle: status color, drawn at predicted position
    - Inner dot: detected centroid (if found)
    - Line: from prediction to detection (shows offset vector)
    - Label: feature_id + offset in mm (MISALIGNED) or "MISSING"
    """
    if cfg is None:
        cfg = ReportConfig()

    # Work on a color copy
    if original_image.ndim == 2:
        overlay = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    else:
        overlay = original_image.copy()

    for r in results:
        color = STATUS_COLORS[r.status]
        pred_pt = (int(r.predicted_col_px), int(r.predicted_row_px))
        r_draw = cfg.marker_radius_px

        # Outer status ring
        cv2.circle(overlay, pred_pt, r_draw, color, cfg.marker_thickness, cv2.LINE_AA)

        # Cross-hair at predicted center
        if cfg.draw_predicted_center:
            ch_size = 5
            cv2.line(overlay,
                     (pred_pt[0]-ch_size, pred_pt[1]),
                     (pred_pt[0]+ch_size, pred_pt[1]),
                     color, 1, cv2.LINE_AA)
            cv2.line(overlay,
                     (pred_pt[0], pred_pt[1]-ch_size),
                     (pred_pt[0], pred_pt[1]+ch_size),
                     color, 1, cv2.LINE_AA)

        if r.status == HoleStatus.MISSING:
            # X mark
            d = r_draw - 4
            cv2.line(overlay,
                     (pred_pt[0]-d, pred_pt[1]-d),
                     (pred_pt[0]+d, pred_pt[1]+d),
                     color, cfg.marker_thickness, cv2.LINE_AA)
            cv2.line(overlay,
                     (pred_pt[0]+d, pred_pt[1]-d),
                     (pred_pt[0]-d, pred_pt[1]+d),
                     color, cfg.marker_thickness, cv2.LINE_AA)
            label = f"{r.feature_id} MISSING"

        else:
            det_pt = (int(r.detected_col_px), int(r.detected_row_px))

            # Dot at detected centroid
            cv2.circle(overlay, det_pt, 4, color, -1, cv2.LINE_AA)

            # Offset vector line
            if cfg.draw_offset_line and r.status != HoleStatus.OK:
                cv2.arrowedLine(overlay, pred_pt, det_pt, color,
                                1, cv2.LINE_AA, tipLength=0.3)

            if r.status == HoleStatus.MISALIGNED:
                label = f"{r.feature_id} {r.offset_mm:.2f}mm"
            elif r.status == HoleStatus.UNCERTAIN:
                label = f"{r.feature_id} DIAM?"
            else:
                label = ""  # Don't clutter OK holes

        # Label text
        if label:
            text_pt = (pred_pt[0] + r_draw + 4, pred_pt[1] + 5)
            # Shadow for readability
            cv2.putText(overlay, label, text_pt,
                        cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale,
                        (0, 0, 0), cfg.font_thickness + 2, cv2.LINE_AA)
            cv2.putText(overlay, label, text_pt,
                        cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale,
                        color, cfg.font_thickness, cv2.LINE_AA)

    # Legend
    _draw_legend(overlay, cfg)

    # Optional downscale for file size
    if downscale_for_output != 1.0:
        new_w = int(overlay.shape[1] * downscale_for_output)
        new_h = int(overlay.shape[0] * downscale_for_output)
        overlay = cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_AREA)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay,
                [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
    logger.info(f"Visual overlay written: {output_path}")

    return overlay


def _draw_legend(img: np.ndarray, cfg: ReportConfig) -> None:
    """Draw a status legend in the top-left corner."""
    entries = [
        (HoleStatus.OK,         "OK"),
        (HoleStatus.MISALIGNED, "MISALIGNED"),
        (HoleStatus.MISSING,    "MISSING"),
        (HoleStatus.UNCERTAIN,  "UNCERTAIN"),
    ]
    x0, y0 = 20, 20
    box_h  = 26
    pad    = 6

    for i, (status, label) in enumerate(entries):
        color = STATUS_COLORS[status]
        y     = y0 + i * box_h
        # Color swatch
        cv2.rectangle(img, (x0, y), (x0 + 20, y + 18), color, -1)
        cv2.rectangle(img, (x0, y), (x0 + 20, y + 18), (80, 80, 80), 1)
        # Label
        cv2.putText(img, label, (x0 + 26, y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, label, (x0 + 26, y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def generate_summary(results: List[HoleResult]) -> dict:
    """
    Compute summary statistics for the inspection run.
    """
    total   = len(results)
    by_status = {s: 0 for s in HoleStatus}
    for r in results:
        by_status[r.status] += 1

    offsets = [r.offset_mm for r in results if r.offset_mm is not None]

    summary = {
        "timestamp":          datetime.now().isoformat(timespec="seconds"),
        "total_features":     total,
        "ok":                 by_status[HoleStatus.OK],
        "missing":            by_status[HoleStatus.MISSING],
        "misaligned":         by_status[HoleStatus.MISALIGNED],
        "uncertain":          by_status[HoleStatus.UNCERTAIN],
        "pass_rate_pct":      round(100 * by_status[HoleStatus.OK] / total, 1) if total else 0,
        "mean_offset_mm":     round(float(np.mean(offsets)), 3) if offsets else None,
        "max_offset_mm":      round(float(np.max(offsets)),  3) if offsets else None,
        "flagged_ids":        [r.feature_id for r in results if r.is_flagged],
    }

    logger.info(
        f"\n{'='*50}\n"
        f"  AVID INSPECTION SUMMARY\n"
        f"{'='*50}\n"
        f"  Total features inspected : {total}\n"
        f"  OK                       : {by_status[HoleStatus.OK]}\n"
        f"  Missing                  : {by_status[HoleStatus.MISSING]}\n"
        f"  Misaligned               : {by_status[HoleStatus.MISALIGNED]}\n"
        f"  Uncertain                : {by_status[HoleStatus.UNCERTAIN]}\n"
        f"  Pass rate                : {summary['pass_rate_pct']}%\n"
        + (f"  Mean offset              : {summary['mean_offset_mm']} mm\n"
           f"  Max  offset              : {summary['max_offset_mm']} mm\n"
           if offsets else "")
        + f"{'='*50}"
    )

    return summary
