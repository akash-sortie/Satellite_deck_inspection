"""
AVID — Pipeline Orchestrator
=============================
Chains all five stages in order. This is the single entry point for
production runs and development validation.

Usage:
    python -m AVID.pipeline \
        --cad data/sample_cad/deck_layout.csv \
        --image data/deck_photo.jpg \
        --output reports/run_001 \
        [--downscale 0.5]  # dev mode: work at half resolution
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np


def resolve_input_path(path_like: str | Path, *, label: str) -> Path:
    """
    Resolve CLI input paths with a few user-friendly fallbacks.

    Search order:
    1. As provided (absolute or relative to current working directory)
    2. Relative to this script's directory
    3. Relative to the project root (parent of this script's directory)
    """
    path = Path(path_like).expanduser()
    if path.is_absolute() and path.exists():
        return path

    candidates = [
        path,
        Path(__file__).resolve().parent / path,
        Path(__file__).resolve().parent.parent / path,
    ]

    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        if resolved.exists():
            return resolved

    searched = "\n".join(f"  - {candidate.resolve(strict=False)}" for candidate in candidates)
    raise FileNotFoundError(
        f"{label} not found: {path_like}\nSearched:\n{searched}"
    )

if __package__:
    from .core.cad_parser import parse_cad_csv, generate_synthetic_cad
    from .core.image_preprocessor import (
        preprocess_image, PreprocessConfig,
        generate_synthetic_deck_image,
    )
    from .core.homography import (
        detect_fiducials_and_compute_homography,
        FiducialConfig, FiducialStrategy,
    )
    from .core.hole_detector import detect_holes, DetectionConfig
    from .core.report_generator import (
        generate_csv_report, generate_visual_overlay, generate_summary,
    )
else:
    # Allow `python pipeline.py ...` by making the package root importable.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from AVID.core.cad_parser import parse_cad_csv, generate_synthetic_cad
    from AVID.core.image_preprocessor import (
        preprocess_image, PreprocessConfig,
        generate_synthetic_deck_image,
    )
    from AVID.core.homography import (
        detect_fiducials_and_compute_homography,
        FiducialConfig, FiducialStrategy,
    )
    from AVID.core.hole_detector import detect_holes, DetectionConfig
    from AVID.core.report_generator import (
        generate_csv_report, generate_visual_overlay, generate_summary,
    )

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Development validation mode (no real files needed)
# ---------------------------------------------------------------------------

def run_synthetic_validation(
    output_dir: str | Path = "reports/synthetic_validation",
    n_holes: int = 40,
    introduce_faults: bool = True,
    seed: int = 42,
) -> dict:
    """
    End-to-end pipeline run on fully synthetic data.
    Validates that the pipeline correctly detects introduced faults.

    Parameters
    ----------
    introduce_faults : if True, deliberately misplace/remove some holes in the
                       synthetic image to test MISALIGNED/MISSING detection.
    """
    import cv2

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("AVID SYNTHETIC VALIDATION RUN")
    logger.info("=" * 60)

    # --- Stage 1a: Generate & parse CAD ---
    cad_df = generate_synthetic_cad(
        n_holes=n_holes,
        n_fiducials=4,
        deck_width_mm=1200.0,
        deck_height_mm=800.0,
        hole_diameter_mm=12.0,
        seed=seed,
        output_path=output_dir / "synthetic_cad.csv",
    )
    manifest = parse_cad_csv(output_dir / "synthetic_cad.csv")

    # Scale: image is 2400×1600 px, deck is 1200×800 mm → 2 px/mm
    px_per_mm = 2.0
    img_w = int(manifest.deck_width_mm  * px_per_mm)
    img_h = int(manifest.deck_height_mm * px_per_mm)

    # Build ground-truth hole list in pixel space
    normal_holes = []
    fault_missing_ids  = set()
    fault_shifted_ids  = set()

    rng = np.random.default_rng(seed + 1)

    for f in manifest.holes:
        cx = int(f.x_norm_mm * px_per_mm)
        cy = int(f.y_norm_mm * px_per_mm)
        r  = int((f.diameter_mm / 2.0) * px_per_mm)
        normal_holes.append((cx, cy, r, f.id))

    # Introduce faults (only in the synthetic IMAGE — CAD is unchanged)
    image_holes = []
    if introduce_faults:
        all_ids    = [h[3] for h in normal_holes]
        # Remove 2 random holes (MISSING)
        missing_ids = set(rng.choice(all_ids, size=2, replace=False))
        # Shift 3 random holes by 3-6 mm
        remaining   = [hid for hid in all_ids if hid not in missing_ids]
        shifted_ids = set(rng.choice(remaining, size=3, replace=False))
        fault_missing_ids  = missing_ids
        fault_shifted_ids  = shifted_ids
        logger.info(f"Introducing faults — MISSING: {missing_ids}, SHIFTED: {shifted_ids}")

    for cx, cy, r, fid in normal_holes:
        if introduce_faults and fid in fault_missing_ids:
            continue  # don't draw this hole → MISSING
        if introduce_faults and fid in fault_shifted_ids:
            shift_px = int(rng.uniform(6, 12))  # 3-6 mm at 2px/mm
            cx += shift_px
            cy += shift_px
        image_holes.append((cx, cy, r))

    # Fiducials as large circles
    fid_holes = [
        (int(f.x_norm_mm * px_per_mm), int(f.y_norm_mm * px_per_mm),
         int((f.diameter_mm / 2.0) * px_per_mm))
        for f in manifest.fiducials
    ]
    all_image_holes = fid_holes + image_holes

    # --- Stage 1b: Generate synthetic image & preprocess ---
    raw_img = generate_synthetic_deck_image(
        width_px=img_w,
        height_px=img_h,
        holes=all_image_holes,
        add_glare=True,
        add_scratches=True,
        add_noise=True,
        seed=seed,
    )
    cv2.imwrite(str(output_dir / "synthetic_raw.jpg"), raw_img,
                [cv2.IMWRITE_JPEG_QUALITY, 90])

    prep_cfg = PreprocessConfig(clahe_clip_limit=3.0, apply_tophat=True)
    processed, meta = preprocess_image(raw_img, prep_cfg,
                                       save_debug=True,
                                       debug_dir=output_dir / "debug_preprocess")

    # --- Stages 2+3: Fiducial detection & homography ---
    # For synthetic validation with exact pixel positions, inject ground-truth
    # fiducial correspondences directly to test homography (bypassing detection)
    if __package__:
        from .core.homography import FiducialMatch, compute_homography
    else:
        from AVID.core.homography import FiducialMatch, compute_homography

    fid_matches = []
    fid_cfg = FiducialConfig(ransac_reproj_threshold=3.0)
    for fid in manifest.fiducials:
        cx_px = fid.x_norm_mm * px_per_mm
        cy_px = fid.y_norm_mm * px_per_mm
        # Add tiny noise to simulate imperfect detection
        noise = rng.normal(0, 0.5, 2)
        fid_matches.append(FiducialMatch(
            cad_id    = fid.id,
            cad_pt_mm = (fid.x_norm_mm, fid.y_norm_mm),
            img_pt_px = (cx_px + noise[0], cy_px + noise[1]),
        ))

    homography = compute_homography(fid_matches, fid_cfg)
    logger.info(f"Homography: {homography.summary()}")
    assert homography.is_valid, "Synthetic homography should always be valid!"

    # --- Stage 4: Detect holes ---
    det_cfg = DetectionConfig(
        roi_scale_factor=4.0,
        position_tolerance_mm=2.0,
    )
    results = detect_holes(processed, manifest, homography, det_cfg,
                           save_roi_crops=False)

    # --- Stage 5: Reports ---
    summary = generate_summary(results)
    generate_csv_report(results, output_dir / "inspection_report.csv",
                        include_pixel_coords=True)
    generate_visual_overlay(raw_img, results,
                            output_dir / "inspection_overlay.jpg")

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Validation assertions
    detected_missing   = {r.feature_id for r in results
                          if r.status.value == "MISSING"}
    detected_misaligned = {r.feature_id for r in results
                           if r.status.value == "MISALIGNED"}

    logger.info("\n--- VALIDATION CHECKS ---")
    missed_faults = fault_missing_ids - detected_missing
    extra_missing = detected_missing - fault_missing_ids
    logger.info(f"  True MISSING  : {fault_missing_ids}")
    logger.info(f"  Detected MISSING : {detected_missing}")
    if missed_faults:
        logger.warning(f"  [FAIL] Undetected missing holes: {missed_faults}")
    if extra_missing:
        logger.warning(f"  [WARN] False MISSING detections: {extra_missing}")
    logger.info(f"  True SHIFTED  : {fault_shifted_ids}")
    logger.info(f"  Detected MISALIGNED: {detected_misaligned}")
    logger.info(f"  Output dir: {output_dir.resolve()}")

    return summary


# ---------------------------------------------------------------------------
# Production run
# ---------------------------------------------------------------------------

def run_pipeline(
    cad_path:      str | Path,
    image_path:    str | Path,
    output_dir:    str | Path = "reports/",
    downscale:     float = 1.0,
    strategy:      str = "blob",
) -> dict:
    """Full production pipeline run."""
    cad_path = resolve_input_path(cad_path, label="CAD CSV")
    image_path = resolve_input_path(image_path, label="Image")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1a
    manifest = parse_cad_csv(cad_path)

    # Stage 1b
    import cv2 as _cv2
    raw_img = _cv2.imread(str(image_path))
    if raw_img is None:
        raise IOError(f"Cannot load image: {image_path}")

    prep_cfg = PreprocessConfig(downscale_factor=downscale)
    processed, _ = preprocess_image(raw_img, prep_cfg,
                                    save_debug=True,
                                    debug_dir=output_dir / "debug")

    # Stages 2+3
    strat = FiducialStrategy.ARUCO if strategy == "aruco" else FiducialStrategy.BLOB
    fid_cfg = FiducialConfig(strategy=strat)
    homography = detect_fiducials_and_compute_homography(processed, manifest, fid_cfg)

    if not homography.is_valid:
        logging.getLogger(__name__).error(
            "Pipeline aborted: homography quality insufficient. "
            "Check fiducial visibility in the image."
        )
        sys.exit(1)

    # Stage 4
    det_cfg = DetectionConfig(
        position_tolerance_mm=17.0,   # Bumped from 1.5 to 20.0 to cover edge distortion
        roi_scale_factor=8.0,         # Keeps the search window wide
        diameter_tolerance_factor=0.5
    )
    # 1. We added 'save_roi_crops=True' to capture the images
    results = detect_holes(processed, manifest, homography, det_cfg, save_roi_crops=True)

    # --- 2. NEW ML DATASET EXTRACTION CODE ---
    import os
    import cv2
    dataset_dir = os.path.join(output_dir, "ml_dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    
    for r in results:
        if r.roi_crop is not None:
            # Saves each hole as its own image (e.g., U1.jpg, T2.jpg)
            cv2.imwrite(os.path.join(dataset_dir, f"{r.feature_id}.jpg"), r.roi_crop)

    # Stage 5
    summary = generate_summary(results)
    generate_csv_report(results, output_dir / "inspection_report.csv")
    generate_visual_overlay(raw_img, results, output_dir / "inspection_overlay.jpg")

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AVID — Automated Vision Inspection for Decks")
    sub = parser.add_subparsers(dest="cmd")

    # Synthetic validation command
    val = sub.add_parser("validate", help="Run synthetic end-to-end validation")
    val.add_argument("--output", default="reports/synthetic_validation")
    val.add_argument("--n-holes", type=int, default=40)
    val.add_argument("--no-faults", action="store_true")
    val.add_argument("--log-level", default="INFO")

    # Production run command
    prod = sub.add_parser("run", help="Run full pipeline on real data")
    prod.add_argument("--cad",      required=True)
    prod.add_argument("--image",    required=True)
    prod.add_argument("--output",   default="reports/")
    prod.add_argument("--downscale",type=float, default=1.0)
    prod.add_argument("--strategy", choices=["blob","aruco"], default="blob")
    prod.add_argument("--log-level",default="INFO")

    args = parser.parse_args()

    if args.cmd == "validate":
        setup_logging(args.log_level)
        run_synthetic_validation(
            output_dir=args.output,
            n_holes=args.n_holes,
            introduce_faults=not args.no_faults,
        )
    elif args.cmd == "run":
        setup_logging(args.log_level)
        run_pipeline(
            cad_path=args.cad,
            image_path=args.image,
            output_dir=args.output,
            downscale=args.downscale,
            strategy=args.strategy,
        )
    else:
        parser.print_help()
