"""
AVID Stage 1a — CAD Parser
===========================
Ingests a 2-quadrant CAD CSV and produces a validated, normalized
hole/insert manifest in a unified coordinate frame.

CSV Contract (expected columns):
    id          : str   — unique feature identifier (e.g. "H-001")
    x_mm        : float — X coordinate in CAD space (origin at bottom-center)
    y_mm        : float — Y coordinate in CAD space
    diameter_mm : float — nominal drilled diameter
    type        : str   — "hole" | "insert" | "fiducial"

Origin Convention:
    CAD origin (0, 0) = bottom-center of the deck.
    X positive → right, Y positive → up.
    This module converts to a top-left image-friendly frame internally,
    but preserves the original CAD values for reporting.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Feature:
    """Represents one CAD-defined feature (hole, insert, or fiducial)."""
    id: str
    x_mm: float           # original CAD coordinate
    y_mm: float           # original CAD coordinate
    diameter_mm: float
    type: str             # "hole" | "insert" | "fiducial"
    # Populated after deck-frame normalization
    x_norm_mm: float = 0.0   # origin shifted to top-left
    y_norm_mm: float = 0.0


@dataclass
class CADManifest:
    """Complete parsed and validated deck manifest."""
    features: List[Feature]
    deck_width_mm: float   # bounding box of all features + margin
    deck_height_mm: float
    origin_offset_x: float  # x_norm = x_cad + origin_offset_x
    origin_offset_y: float  # y_norm = (deck_height - y_cad) — flips Y axis

    # Convenience views
    @property
    def holes(self) -> List[Feature]:
        return [f for f in self.features if f.type == "hole"]

    @property
    def inserts(self) -> List[Feature]:
        return [f for f in self.features if f.type == "insert"]

    @property
    def fiducials(self) -> List[Feature]:
        return [f for f in self.features if f.type == "fiducial"]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([vars(f) for f in self.features])


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {"id", "x_mm", "y_mm", "diameter_mm", "type"}
VALID_TYPES      = {"hole", "insert", "fiducial"}
MARGIN_MM        = 20.0   # padding added around feature bounding box


def _validate_schema(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CAD CSV missing required columns: {missing}")


def _validate_types(df: pd.DataFrame) -> None:
    bad = set(df["type"].unique()) - VALID_TYPES
    if bad:
        raise ValueError(
            f"Unknown feature types in CSV: {bad}. "
            f"Expected one of {VALID_TYPES}."
        )


def _validate_diameters(df: pd.DataFrame) -> None:
    bad = df[df["diameter_mm"] <= 0]
    if not bad.empty:
        raise ValueError(
            f"Non-positive diameters found for IDs: {bad['id'].tolist()}"
        )


def _validate_unique_ids(df: pd.DataFrame) -> None:
    dupes = df[df["id"].duplicated()]["id"].tolist()
    if dupes:
        raise ValueError(f"Duplicate feature IDs found: {dupes}")


def _validate_fiducial_count(df: pd.DataFrame) -> None:
    n = (df["type"] == "fiducial").sum()
    if n < 4:
        logger.warning(
            f"Only {n} fiducial(s) found. Minimum 4 recommended for "
            "robust homography. Consider adding more physical markers."
        )


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------

def parse_cad_csv(
    csv_path: str | Path,
    deck_width_mm: Optional[float] = None,
    deck_height_mm: Optional[float] = None,
    margin_mm: float = MARGIN_MM,
) -> CADManifest:
    """
    Parse and validate a CAD feature CSV.

    Parameters
    ----------
    csv_path : path to the CAD CSV file
    deck_width_mm : known physical deck width. If None, inferred from
                    feature bounding box + margin.
    deck_height_mm : known physical deck height. If None, inferred.
    margin_mm : padding applied when inferring deck dimensions.

    Returns
    -------
    CADManifest with all features, deck dimensions, and normalized coords.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CAD CSV not found: {path}")

    logger.info(f"Parsing CAD CSV: {path}")
    df = pd.read_csv(path, dtype={"id": str})

    # --- Schema validation ---
    _validate_schema(df)
    df["type"] = df["type"].str.lower().str.strip()
    _validate_types(df)
    _validate_diameters(df)
    _validate_unique_ids(df)
    _validate_fiducial_count(df)

    # --- Numeric coercion and NaN check ---
    for col in ("x_mm", "y_mm", "diameter_mm"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[["x_mm", "y_mm", "diameter_mm"]].isna().any().any():
        raise ValueError("Non-numeric values found in coordinate/diameter columns.")

    logger.info(
        f"  Loaded {len(df)} features: "
        f"{(df.type=='hole').sum()} holes, "
        f"{(df.type=='insert').sum()} inserts, "
        f"{(df.type=='fiducial').sum()} fiducials."
    )

    # --- Deck dimension inference ---
    # CAD origin at bottom-center: x spans [-W/2, +W/2], y spans [0, H]
    x_min, x_max = df["x_mm"].min(), df["x_mm"].max()
    y_min, y_max = df["y_mm"].min(), df["y_mm"].max()

    inferred_w = (x_max - x_min) + 2 * margin_mm
    inferred_h = (y_max - y_min) + 2 * margin_mm

    w = deck_width_mm  if deck_width_mm  is not None else inferred_w
    h = deck_height_mm if deck_height_mm is not None else inferred_h

    logger.info(f"  Deck dimensions: {w:.1f} mm × {h:.1f} mm")

    # --- Normalize coordinates to top-left origin (image convention) ---
    # x_norm = x_cad - x_min + margin
    # y_norm = (y_max + margin) - y_cad   ← Y axis flip
    offset_x = -x_min + margin_mm
    offset_y =  y_max + margin_mm   # used as: y_norm = offset_y - y_cad

    features = []
    for _, row in df.iterrows():
        f = Feature(
            id          = row["id"],
            x_mm        = float(row["x_mm"]),
            y_mm        = float(row["y_mm"]),
            diameter_mm = float(row["diameter_mm"]),
            type        = row["type"],
            x_norm_mm   = float(row["x_mm"]) + offset_x,
            y_norm_mm   = offset_y - float(row["y_mm"]),
        )
        features.append(f)

    return CADManifest(
        features        = features,
        deck_width_mm   = w,
        deck_height_mm  = h,
        origin_offset_x = offset_x,
        origin_offset_y = offset_y,
    )


# ---------------------------------------------------------------------------
# Synthetic data generator (for pipeline validation — Stage 1 of dev order)
# ---------------------------------------------------------------------------

def generate_synthetic_cad(
    n_holes: int = 40,
    n_fiducials: int = 4,
    deck_width_mm: float = 1200.0,
    deck_height_mm: float = 800.0,
    hole_diameter_mm: float = 12.0,
    seed: int = 42,
    output_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Generate a plausible synthetic CAD CSV for pipeline validation.
    Fiducials are placed at the four corners of the deck.
    Holes are placed on a regular grid with slight jitter.

    Returns a DataFrame and optionally writes it to CSV.
    """
    rng = np.random.default_rng(seed)

    rows = []

    # Fiducials at four corners (in CAD space: origin bottom-center)
    half_w = deck_width_mm / 2 - 30
    fid_positions = [
        (-half_w,               30.0),
        ( half_w,               30.0),
        (-half_w, deck_height_mm - 30.0),
        ( half_w, deck_height_mm - 30.0),
    ]
    for i, (fx, fy) in enumerate(fid_positions):
        rows.append({
            "id":          f"FID-{i+1:02d}",
            "x_mm":        round(fx, 3),
            "y_mm":        round(fy, 3),
            "diameter_mm": 20.0,
            "type":        "fiducial",
        })

    # Regular grid of holes with controlled jitter
    cols_count = int(np.sqrt(n_holes * (deck_width_mm / deck_height_mm))) + 1
    rows_count = int(np.ceil(n_holes / cols_count))

    xs = np.linspace(-deck_width_mm/2 + 60, deck_width_mm/2 - 60, cols_count)
    ys = np.linspace(60, deck_height_mm - 60, rows_count)

    hole_id = 1
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            if hole_id > n_holes:
                break
            jitter_x = rng.uniform(-2, 2)
            jitter_y = rng.uniform(-2, 2)
            rows.append({
                "id":          f"H-{hole_id:03d}",
                "x_mm":        round(x + jitter_x, 3),
                "y_mm":        round(y + jitter_y, 3),
                "diameter_mm": hole_diameter_mm,
                "type":        "hole",
            })
            hole_id += 1

    df = pd.DataFrame(rows)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Synthetic CAD CSV written to: {output_path}")

    return df
