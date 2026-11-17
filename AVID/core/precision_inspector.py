"""
AVID ML Stage C+D — Precision Tolerance Checker & 5-Class Labeller
====================================================================
Takes the sub-pixel ellipse fit results and applies aerospace-grade
tolerances to produce a 5-class classification per hole.

Classes:
  OK          — position ≤ 0.2mm, radius deviation ≤ 0.1mm
  MISSING     — CNN said no hole present (Stage A gate failed)
  POS_FAULT   — position offset > 0.2mm, radius OK
  RAD_FAULT   — radius deviation > 0.1mm, position OK
  BOTH_FAULT  — both position and radius out of tolerance
  UNCERTAIN   — fit succeeded but quality metrics are marginal
               (high residual, low edge count, high ovality)

Each result also carries full measurement intelligence:
  - Exact position offset in mm (with direction: dx_mm, dy_mm)
  - Exact radius deviation in mm (detected - nominal)
  - Ovality (ellipse eccentricity — indicates drill bit wear)
  - Fit confidence score
  - Severity score (0-100, for dashboard sorting)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from .subpixel_fitting import EllipseFitResult
from .homography import HomographyResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 5-class label
# ---------------------------------------------------------------------------

class InspectionClass(Enum):
    OK          = "OK"
    MISSING     = "MISSING"
    POS_FAULT   = "POS_FAULT"
    RAD_FAULT   = "RAD_FAULT"
    BOTH_FAULT  = "BOTH_FAULT"
    UNCERTAIN   = "UNCERTAIN"


CLASS_COLORS_BGR = {
    InspectionClass.OK:         (0,  200,  0),      # green
    InspectionClass.MISSING:    (200,  0, 200),      # magenta
    InspectionClass.POS_FAULT:  (0,   0, 230),       # red
    InspectionClass.RAD_FAULT:  (0, 165, 255),       # orange
    InspectionClass.BOTH_FAULT: (0,   0, 180),       # deep red
    InspectionClass.UNCERTAIN:  (0, 200, 220),       # yellow
}

CLASS_SEVERITY = {
    InspectionClass.OK:         0,
    InspectionClass.UNCERTAIN:  20,
    InspectionClass.RAD_FAULT:  50,
    InspectionClass.POS_FAULT:  65,
    InspectionClass.BOTH_FAULT: 90,
    InspectionClass.MISSING:    100,
}


# ---------------------------------------------------------------------------
# Tolerance configuration
# ---------------------------------------------------------------------------

@dataclass
class ToleranceConfig:
    # Aerospace-grade targets (overridable at runtime)
    position_tol_mm: float = 0.2     # ±0.2mm centroid offset
    radius_tol_mm:   float = 0.1     # ±0.1mm radius deviation

    # Soft warning thresholds (UNCERTAIN zone before hard fault)
    position_warn_mm: float = 0.15
    radius_warn_mm:   float = 0.08

    # Ellipse quality thresholds for UNCERTAIN classification
    max_ovality:     float = 0.12    # > 12% eccentricity = suspect
    max_residual_px: float = 1.5     # > 1.5px RMS residual = suspect
    min_edge_points: int   = 15      # < 15 rim edge points = suspect

    # CNN confidence threshold for MISSING classification
    cnn_hole_threshold: float = 0.45


# ---------------------------------------------------------------------------
# Per-hole precision result
# ---------------------------------------------------------------------------

@dataclass
class PrecisionResult:
    feature_id: str
    inspection_class: InspectionClass
    severity: int   # 0-100

    # CAD nominal
    cad_x_mm: float
    cad_y_mm: float
    cad_radius_mm: float

    # Measurements (None if MISSING)
    detected_col_px:  Optional[float] = None
    detected_row_px:  Optional[float] = None
    detected_radius_px: Optional[float] = None

    # Deviations in physical units
    dx_mm: Optional[float] = None       # signed X offset (positive = right)
    dy_mm: Optional[float] = None       # signed Y offset (positive = down)
    position_offset_mm: Optional[float] = None
    radius_deviation_mm: Optional[float] = None  # detected - nominal (signed)

    # Fit quality
    ovality:        Optional[float] = None
    fit_residual_px: Optional[float] = None
    n_edge_points:  Optional[int]   = None
    cnn_confidence: Optional[float] = None

    # Flags
    is_flagged: bool = False
    flag_reasons: List[str] = field(default_factory=list)

    @property
    def pass_fail(self) -> str:
        return "PASS" if self.inspection_class == InspectionClass.OK else "FAIL"


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class PrecisionInspector:
    """
    Combines CNN gate (Stage A) + ellipse fit (Stage B) results
    into final 5-class precision classification (Stages C+D).
    """

    def __init__(
        self,
        homography: HomographyResult,
        px_per_mm: float,
        tolerance: Optional[ToleranceConfig] = None,
    ):
        self.H       = homography
        self.px_per_mm = px_per_mm
        self.tol     = tolerance or ToleranceConfig()

    def classify_hole(
        self,
        feature_id: str,
        cad_x_mm: float,         # original CAD coords (for report)
        cad_y_mm: float,
        cad_x_norm_mm: float,    # normalized (top-left origin)
        cad_y_norm_mm: float,
        cad_diameter_mm: float,
        predicted_col_px: float,
        predicted_row_px: float,
        fit: Optional[EllipseFitResult],
        cnn_confidence: Optional[float],
    ) -> PrecisionResult:
        """
        Classify one hole given its ellipse fit and CNN confidence.
        """
        cad_radius_mm = cad_diameter_mm / 2.0
        flags = []

        # ---  MISSING: CNN says no hole ---
        if cnn_confidence is not None and cnn_confidence < self.tol.cnn_hole_threshold:
            return PrecisionResult(
                feature_id       = feature_id,
                inspection_class = InspectionClass.MISSING,
                severity         = CLASS_SEVERITY[InspectionClass.MISSING],
                cad_x_mm         = cad_x_mm,
                cad_y_mm         = cad_y_mm,
                cad_radius_mm    = cad_radius_mm,
                cnn_confidence   = cnn_confidence,
                is_flagged       = True,
                flag_reasons     = ["CNN confidence below hole threshold"],
            )

        # --- No fit available (ellipse fitting failed) ---
        if fit is None or not fit.success:
            reason = fit.failure_reason if fit else "fit not attempted"
            return PrecisionResult(
                feature_id       = feature_id,
                inspection_class = InspectionClass.MISSING,
                severity         = CLASS_SEVERITY[InspectionClass.MISSING],
                cad_x_mm         = cad_x_mm,
                cad_y_mm         = cad_y_mm,
                cad_radius_mm    = cad_radius_mm,
                cnn_confidence   = cnn_confidence,
                is_flagged       = True,
                flag_reasons     = [f"Ellipse fit failed: {reason}"],
            )

        # --- Compute deviations in mm ---
        # Position: euclidean offset of detected centroid from CAD projection
        dx_px = fit.center_col_px - predicted_col_px
        dy_px = fit.center_row_px - predicted_row_px
        offset_px = np.hypot(dx_px, dy_px)
        offset_mm = offset_px / self.px_per_mm
        dx_mm = dx_px / self.px_per_mm
        dy_mm = dy_px / self.px_per_mm

        # Radius deviation
        detected_radius_mm = fit.radius_px / self.px_per_mm
        radius_dev_mm = detected_radius_mm - cad_radius_mm

        # --- Quality checks for UNCERTAIN ---
        is_uncertain = False
        if fit.ovality > self.tol.max_ovality:
            flags.append(f"High ovality: {fit.ovality:.3f} > {self.tol.max_ovality}")
            is_uncertain = True
        if fit.fit_residual_px > self.tol.max_residual_px:
            flags.append(f"High fit residual: {fit.fit_residual_px:.2f}px")
            is_uncertain = True
        if fit.n_edge_points < self.tol.min_edge_points:
            flags.append(f"Low edge point count: {fit.n_edge_points}")
            is_uncertain = True

        # --- Tolerance classification ---
        pos_fail = abs(offset_mm)     > self.tol.position_tol_mm
        rad_fail = abs(radius_dev_mm) > self.tol.radius_tol_mm

        if is_uncertain and not pos_fail and not rad_fail:
            cls = InspectionClass.UNCERTAIN
        elif pos_fail and rad_fail:
            cls = InspectionClass.BOTH_FAULT
            flags.append(f"Position: {offset_mm:.3f}mm (tol {self.tol.position_tol_mm}mm)")
            flags.append(f"Radius:   {radius_dev_mm:+.3f}mm (tol ±{self.tol.radius_tol_mm}mm)")
        elif pos_fail:
            cls = InspectionClass.POS_FAULT
            flags.append(f"Position: {offset_mm:.3f}mm (tol {self.tol.position_tol_mm}mm)")
        elif rad_fail:
            cls = InspectionClass.RAD_FAULT
            flags.append(f"Radius: {radius_dev_mm:+.3f}mm (tol ±{self.tol.radius_tol_mm}mm)")
        else:
            cls = InspectionClass.OK

        return PrecisionResult(
            feature_id           = feature_id,
            inspection_class     = cls,
            severity             = CLASS_SEVERITY[cls],
            cad_x_mm             = cad_x_mm,
            cad_y_mm             = cad_y_mm,
            cad_radius_mm        = cad_radius_mm,
            detected_col_px      = fit.center_col_px,
            detected_row_px      = fit.center_row_px,
            detected_radius_px   = fit.radius_px,
            dx_mm                = round(dx_mm, 4),
            dy_mm                = round(dy_mm, 4),
            position_offset_mm   = round(offset_mm, 4),
            radius_deviation_mm  = round(radius_dev_mm, 4),
            ovality              = round(fit.ovality, 4),
            fit_residual_px      = round(fit.fit_residual_px, 3),
            n_edge_points        = fit.n_edge_points,
            cnn_confidence       = cnn_confidence,
            is_flagged           = cls != InspectionClass.OK,
            flag_reasons         = flags,
        )

    def inspect_all(
        self,
        manifest,
        projected_positions: Dict[str, tuple],
        ellipse_results: Dict[str, EllipseFitResult],
        cnn_confidences: Optional[Dict[str, float]] = None,
    ) -> List[PrecisionResult]:
        """
        Run classification for all holes in manifest.
        """
        results = []
        for feature in manifest.holes + manifest.inserts:
            fid = feature.id
            pred = projected_positions.get(fid)
            if pred is None:
                continue
            pred_col, pred_row = pred

            fit  = ellipse_results.get(fid)
            conf = cnn_confidences.get(fid) if cnn_confidences else None

            r = self.classify_hole(
                feature_id       = fid,
                cad_x_mm         = feature.x_mm,
                cad_y_mm         = feature.y_mm,
                cad_x_norm_mm    = feature.x_norm_mm,
                cad_y_norm_mm    = feature.y_norm_mm,
                cad_diameter_mm  = feature.diameter_mm,
                predicted_col_px = pred_col,
                predicted_row_px = pred_row,
                fit              = fit,
                cnn_confidence   = conf,
            )
            results.append(r)

        # Summary
        from collections import Counter
        counts = Counter(r.inspection_class.value for r in results)
        total  = len(results)
        ok     = counts.get("OK", 0)
        logger.info(
            f"\n{'='*55}\n"
            f"  AVID PRECISION INSPECTION — SUMMARY\n"
            f"{'='*55}\n"
            f"  Total inspected  : {total}\n"
            f"  OK               : {ok}  ({100*ok/total:.1f}%)\n"
            f"  Missing          : {counts.get('MISSING',0)}\n"
            f"  Position fault   : {counts.get('POS_FAULT',0)}\n"
            f"  Radius fault     : {counts.get('RAD_FAULT',0)}\n"
            f"  Both faults      : {counts.get('BOTH_FAULT',0)}\n"
            f"  Uncertain        : {counts.get('UNCERTAIN',0)}\n"
            f"  Tolerances used  : pos ±{self.tol.position_tol_mm}mm  "
            f"rad ±{self.tol.radius_tol_mm}mm\n"
            f"{'='*55}"
        )
        return results


# ---------------------------------------------------------------------------
# Precision CSV report
# ---------------------------------------------------------------------------

def generate_precision_csv(
    results: List[PrecisionResult],
    output_path,
) -> None:
    """Write a full precision measurement CSV — one row per hole."""
    import pandas as pd

    rows = []
    for r in results:
        rows.append({
            "feature_id":          r.feature_id,
            "class":               r.inspection_class.value,
            "pass_fail":           r.pass_fail,
            "severity":            r.severity,
            "cad_x_mm":            r.cad_x_mm,
            "cad_y_mm":            r.cad_y_mm,
            "cad_radius_mm":       r.cad_radius_mm,
            "position_offset_mm":  r.position_offset_mm if r.position_offset_mm is not None else "",
            "dx_mm":               r.dx_mm if r.dx_mm is not None else "",
            "dy_mm":               r.dy_mm if r.dy_mm is not None else "",
            "radius_deviation_mm": r.radius_deviation_mm if r.radius_deviation_mm is not None else "",
            "detected_radius_mm":  round(r.detected_radius_px / 2.865, 4) if r.detected_radius_px else "",
            "ovality":             r.ovality if r.ovality is not None else "",
            "fit_residual_px":     r.fit_residual_px if r.fit_residual_px is not None else "",
            "edge_points":         r.n_edge_points if r.n_edge_points is not None else "",
            "cnn_confidence":      round(r.cnn_confidence, 3) if r.cnn_confidence is not None else "",
            "flag_reasons":        " | ".join(r.flag_reasons),
        })

    df = pd.DataFrame(rows).sort_values("severity", ascending=False)
    df.to_csv(output_path, index=False)
    logger.info(f"Precision CSV written: {output_path}")


# ---------------------------------------------------------------------------
# Precision visual overlay
# ---------------------------------------------------------------------------

def generate_precision_overlay(
    original_image,
    results: List[PrecisionResult],
    output_path,
    px_per_mm: float = 2.865,
) -> None:
    """
    Draw a precision overlay with:
    - Color-coded circle at detected centroid
    - Arrow vector showing offset direction and magnitude
    - Measurement text: offset_mm and radius_dev_mm
    """
    import cv2
    import numpy as np

    if original_image.ndim == 2:
        overlay = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    else:
        overlay = original_image.copy()

    for r in results:
        color = CLASS_COLORS_BGR[r.inspection_class]
        pred_pt = None

        if r.detected_col_px is not None:
            det_pt = (int(r.detected_col_px), int(r.detected_row_px))
            r_draw = int(r.detected_radius_px) if r.detected_radius_px else 15

            # Draw detected circle
            cv2.circle(overlay, det_pt, r_draw, color, 2, cv2.LINE_AA)
            cv2.circle(overlay, det_pt, 3, color, -1, cv2.LINE_AA)

            # Measurement label
            lines = [r.feature_id]
            if r.position_offset_mm is not None:
                lines.append(f"d={r.position_offset_mm:.3f}mm")
            if r.radius_deviation_mm is not None:
                sign = "+" if r.radius_deviation_mm >= 0 else ""
                lines.append(f"r{sign}{r.radius_deviation_mm:.3f}mm")

            for li, line in enumerate(lines):
                tx = det_pt[0] + r_draw + 4
                ty = det_pt[1] - 10 + li * 14
                cv2.putText(overlay, line, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(overlay, line, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            color, 1, cv2.LINE_AA)

        else:
            # MISSING: draw X at predicted location
            # We don't store predicted px in PrecisionResult — use CAD coords note
            pass

    # Legend
    legend_items = [
        (InspectionClass.OK,         "OK — within tolerance"),
        (InspectionClass.POS_FAULT,  "POS_FAULT — position > 0.2mm"),
        (InspectionClass.RAD_FAULT,  "RAD_FAULT — radius > 0.1mm"),
        (InspectionClass.BOTH_FAULT, "BOTH_FAULT"),
        (InspectionClass.MISSING,    "MISSING"),
        (InspectionClass.UNCERTAIN,  "UNCERTAIN — low fit quality"),
    ]
    for i, (cls, label) in enumerate(legend_items):
        c = CLASS_COLORS_BGR[cls]
        y = 20 + i * 26
        cv2.rectangle(overlay, (15, y), (35, y+18), c, -1)
        cv2.putText(overlay, label, (40, y+13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(overlay, label, (40, y+13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230,230,230), 1, cv2.LINE_AA)

    import pathlib
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])
    logger.info(f"Precision overlay written: {output_path}")
