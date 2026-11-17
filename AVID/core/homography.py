"""
AVID Stage 2 & 3 — Fiducial Detection & Homography
====================================================
This is the most critical module in the pipeline. It bridges the
digital-to-physical gap by computing a perspective transform (homography)
that maps CAD millimeter coordinates to image pixel coordinates.

Architecture
------------
Stage 2a : Pull known fiducial CAD coordinates from the manifest.
Stage 2b : Detect fiducial pixel locations in the pre-processed image.
           Two strategies are supported:
             - ARUCO   : cv2.aruco markers (if physical markers are used)
             - BLOB    : SimpleBlobDetector on the largest circular features
                         (fallback when no markers are present)
Stage 3  : Compute homography via RANSAC using the matched point pairs.
           Provide forward (CAD→pixel) and inverse (pixel→CAD) transforms.

Key design decision: RANSAC is used even with as few as 4 points.
Any one bad fiducial detection would catastrophically shift every
downstream projection. RANSAC flags outliers and logs a warning.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .cad_parser import CADManifest, Feature

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class FiducialStrategy(Enum):
    ARUCO = auto()   # cv2.aruco markers (recommended)
    BLOB  = auto()   # blob detection on large circular features (fallback)


@dataclass
class FiducialConfig:
    strategy: FiducialStrategy = FiducialStrategy.BLOB

    # ArUco settings
    aruco_dict_type: int = cv2.aruco.DICT_4X4_50

    # Blob detector settings (for BLOB strategy or fiducial circle detection)
    blob_min_area: float = 500.0      # px²
    blob_max_area: float = 50000.0    # px²
    blob_min_circularity: float = 0.75
    blob_min_convexity: float = 0.85
    blob_min_inertia: float = 0.5

    # RANSAC threshold for homography (px)
    ransac_reproj_threshold: float = 5.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FiducialMatch:
    """A single confirmed correspondence: one CAD fiducial → one image point."""
    cad_id: str
    cad_pt_mm: Tuple[float, float]   # (x_norm_mm, y_norm_mm) — top-left origin
    img_pt_px: Tuple[float, float]   # (col, row) in pixel space


@dataclass
class HomographyResult:
    """
    The product of Stage 3. Contains the homography matrix and
    convenience projection methods.
    """
    H: np.ndarray           # 3×3 homography: CAD mm → image pixels
    H_inv: np.ndarray       # 3×3 inverse:   image pixels → CAD mm
    inlier_matches: List[FiducialMatch]
    outlier_matches: List[FiducialMatch]
    reprojection_errors_px: List[float]
    mean_reproj_error_px: float

    # --- Projection API ---

    def cad_to_pixel(self, x_norm_mm: float, y_norm_mm: float) -> Tuple[float, float]:
        """Project a single normalized CAD point to pixel coordinates."""
        pt = np.array([[[x_norm_mm, y_norm_mm]]], dtype=np.float64)
        result = cv2.perspectiveTransform(pt, self.H)
        col, row = result[0][0]
        return float(col), float(row)

    def pixel_to_cad(self, col_px: float, row_px: float) -> Tuple[float, float]:
        """Back-project a pixel coordinate to normalized CAD mm space."""
        pt = np.array([[[col_px, row_px]]], dtype=np.float64)
        result = cv2.perspectiveTransform(pt, self.H_inv)
        x_mm, y_mm = result[0][0]
        return float(x_mm), float(y_mm)

    def project_all_features(
        self, manifest: CADManifest
    ) -> Dict[str, Tuple[float, float]]:
        """
        Project every feature in the manifest to pixel coordinates.
        Returns {feature_id: (col_px, row_px)}.
        """
        pts_mm = np.array(
            [[[f.x_norm_mm, f.y_norm_mm]] for f in manifest.features],
            dtype=np.float64
        )
        pts_px = cv2.perspectiveTransform(pts_mm, self.H)
        return {
            f.id: (float(pts_px[i][0][0]), float(pts_px[i][0][1]))
            for i, f in enumerate(manifest.features)
        }

    @property
    def is_valid(self) -> bool:
        return (
            len(self.inlier_matches) >= 4
            and self.mean_reproj_error_px < 10.0
        )

    def summary(self) -> str:
        return (
            f"HomographyResult: {len(self.inlier_matches)} inliers, "
            f"{len(self.outlier_matches)} outliers, "
            f"mean reprojection error = {self.mean_reproj_error_px:.2f} px, "
            f"valid = {self.is_valid}"
        )


# ---------------------------------------------------------------------------
# Stage 2b — Fiducial detection strategies
# ---------------------------------------------------------------------------

def _detect_aruco(
    gray: np.ndarray,
    cfg: FiducialConfig,
) -> Dict[int, Tuple[float, float]]:
    """
    Detect ArUco markers and return {marker_id: (cx_px, cy_px)}.
    Marker IDs must correspond to fiducial IDs in the CAD manifest.
    (E.g., marker ID 0 → 'FID-01', etc.)
    """
    aruco_dict   = cv2.aruco.getPredefinedDictionary(cfg.aruco_dict_type)
    aruco_params = cv2.aruco.DetectorParameters()
    detector     = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    corners, ids, _ = detector.detectMarkers(gray)

    detections = {}
    if ids is not None:
        for corner_set, marker_id in zip(corners, ids.flatten()):
            # Centroid = mean of the 4 corner points
            cx = float(corner_set[0][:, 0].mean())
            cy = float(corner_set[0][:, 1].mean())
            detections[int(marker_id)] = (cx, cy)
        logger.info(f"ArUco: detected {len(detections)} markers: {list(detections.keys())}")
    else:
        logger.warning("ArUco: no markers detected in image.")

    return detections


def _build_blob_detector(cfg: FiducialConfig) -> cv2.SimpleBlobDetector:
    """Configure a SimpleBlobDetector for finding circular fiducial markers."""
    params = cv2.SimpleBlobDetector_Params()

    params.filterByArea       = True
    params.minArea            = cfg.blob_min_area
    params.maxArea            = cfg.blob_max_area

    params.filterByCircularity = True
    params.minCircularity      = cfg.blob_min_circularity

    params.filterByConvexity  = True
    params.minConvexity       = cfg.blob_min_convexity

    params.filterByInertia    = True
    params.minInertiaRatio    = cfg.blob_min_inertia

    params.filterByColor      = True
    params.blobColor          = 0   # detect DARK blobs (drill-through = dark)

    return cv2.SimpleBlobDetector_create(params)


def _detect_blobs(
    gray: np.ndarray,
    cfg: FiducialConfig,
    n_expected: int,
) -> List[Tuple[float, float]]:
    """
    Detect candidate circular blobs (fiducials).
    Returns list of (cx_px, cy_px) sorted by blob size descending.

    Real deck images can produce imperfect fiducials, so if the strict pass
    finds too few candidates we progressively relax the blob shape filters.
    """
    candidate_cfgs = [
        ("strict", cfg),
        ("relaxed-shape", FiducialConfig(
            strategy=cfg.strategy,
            aruco_dict_type=cfg.aruco_dict_type,
            blob_min_area=max(100.0, cfg.blob_min_area * 0.5),
            blob_max_area=max(cfg.blob_max_area, 100000.0),
            blob_min_circularity=min(cfg.blob_min_circularity, 0.50),
            blob_min_convexity=min(cfg.blob_min_convexity, 0.30),
            blob_min_inertia=min(cfg.blob_min_inertia, 0.05),
            ransac_reproj_threshold=cfg.ransac_reproj_threshold,
        )),
        ("relaxed-area", FiducialConfig(
            strategy=cfg.strategy,
            aruco_dict_type=cfg.aruco_dict_type,
            blob_min_area=max(50.0, cfg.blob_min_area * 0.25),
            blob_max_area=max(cfg.blob_max_area, 200000.0),
            blob_min_circularity=min(cfg.blob_min_circularity, 0.35),
            blob_min_convexity=min(cfg.blob_min_convexity, 0.20),
            blob_min_inertia=min(cfg.blob_min_inertia, 0.02),
            ransac_reproj_threshold=cfg.ransac_reproj_threshold,
        )),
    ]

    best_keypoints = []
    best_label = "strict"

    for label, candidate_cfg in candidate_cfgs:
        detector = _build_blob_detector(candidate_cfg)
        keypoints = sorted(detector.detect(gray), key=lambda kp: kp.size, reverse=True)

        if len(keypoints) > len(best_keypoints):
            best_keypoints = keypoints
            best_label = label

        if len(keypoints) >= n_expected:
            logger.info(
                f"Blob detector ({label}): found {len(keypoints)} candidate blob(s)."
            )
            return [(kp.pt[0], kp.pt[1]) for kp in keypoints]

    if not best_keypoints:
        logger.warning("Blob detector: no blobs found.")
        return []

    logger.info(
        f"Blob detector ({best_label}): found {len(best_keypoints)} candidate blob(s), "
        f"still below expected {n_expected}."
    )
    return [(kp.pt[0], kp.pt[1]) for kp in best_keypoints]


# ---------------------------------------------------------------------------
# Matching strategy: blob points -> CAD fiducials (spatial nearest-neighbour)
# ---------------------------------------------------------------------------

def _match_blobs_to_cad_fiducials(
    detected_pts: List[Tuple[float, float]],
    cad_fiducials: List[Feature],
    img_shape: Tuple[int, int],  # (height, width)
    manifest_width_mm: float,
    manifest_height_mm: float,
) -> List[FiducialMatch]:
    """
    Match detected blob pixel positions to CAD fiducial entries using
    a rough scale-normalised spatial nearest-neighbour assignment.

    Strategy:
    1. Normalize detected pixel positions to [0,1] in both axes.
    2. Normalize CAD fiducial positions to [0,1] using deck dimensions.
    3. For each CAD fiducial, select the nearest unused blob candidate.
    4. Warn when the nearest candidate is still far away in normalized space.
    """
    if not detected_pts:
        return []

    h_px, w_px = img_shape[:2]

    det_norm = np.array([
        (cx / w_px, cy / h_px) for cx, cy in detected_pts
    ])

    cad_norm = np.array([
        (f.x_norm_mm / manifest_width_mm, f.y_norm_mm / manifest_height_mm)
        for f in cad_fiducials
    ])

    matches = []
    used_det = set()

    for fid, fid_norm in zip(cad_fiducials, cad_norm):
        available = [i for i in range(len(detected_pts)) if i not in used_det]
        if not available:
            break

        dists = np.linalg.norm(det_norm[available] - fid_norm, axis=1)
        best_pos = int(np.argmin(dists))
        best_i = available[best_pos]
        best_dist = float(dists[best_pos])
        cx, cy = detected_pts[best_i]

        if best_dist > 0.20:
            logger.warning(
                f"Nearest blob for CAD fiducial {fid.id} is still far away "
                f"(normalized distance {best_dist:.3f})."
            )

        used_det.add(best_i)
        matches.append(FiducialMatch(
            cad_id=fid.id,
            cad_pt_mm=(fid.x_norm_mm, fid.y_norm_mm),
            img_pt_px=(cx, cy),
        ))
        logger.debug(
            f"  Matched blob ({cx:.1f}, {cy:.1f}) px "
            f"to CAD fiducial {fid.id} "
            f"({fid.x_norm_mm:.1f}, {fid.y_norm_mm:.1f}) mm  dist={best_dist:.4f}"
        )

    return matches


# ---------------------------------------------------------------------------
# Stage 3 - Homography computation
# ---------------------------------------------------------------------------

def compute_homography(
    matches: List[FiducialMatch],
    cfg: FiducialConfig,
) -> HomographyResult:
    """
    Compute a homography matrix from fiducial correspondences using RANSAC.

    Requires at least 4 matches. Uses RANSAC to identify and exclude
    outlier correspondences caused by misdetection.

    Returns HomographyResult with H, H_inv, inliers, outliers, and errors.
    """
    if len(matches) < 4:
        raise ValueError(
            f"Need at least 4 fiducial matches for homography. "
            f"Got: {len(matches)}."
        )

    src_pts = np.array(
        [[m.cad_pt_mm[0], m.cad_pt_mm[1]] for m in matches], dtype=np.float64
    )
    dst_pts = np.array(
        [[m.img_pt_px[0], m.img_pt_px[1]] for m in matches], dtype=np.float64
    )

    H, mask = cv2.findHomography(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=cfg.ransac_reproj_threshold,
    )

    if H is None:
        raise RuntimeError(
            "cv2.findHomography returned None. "
            "Check fiducial detections and ensure points are not co-linear."
        )

    H_inv = np.linalg.inv(H)

    inliers  = [m for m, keep in zip(matches, mask.flatten()) if keep]
    outliers = [m for m, keep in zip(matches, mask.flatten()) if not keep]

    if outliers:
        logger.warning(
            f"RANSAC rejected {len(outliers)} fiducial match(es) as outliers: "
            f"{[m.cad_id for m in outliers]}"
        )

    # --- Reprojection error ---
    reproj_errors = []
    for m in inliers:
        src = np.array([[[m.cad_pt_mm[0], m.cad_pt_mm[1]]]], dtype=np.float64)
        projected = cv2.perspectiveTransform(src, H)[0][0]
        err = float(np.linalg.norm(projected - np.array(m.img_pt_px)))
        reproj_errors.append(err)

    mean_err = float(np.mean(reproj_errors)) if reproj_errors else float("inf")

    result = HomographyResult(
        H=H,
        H_inv=H_inv,
        inlier_matches=inliers,
        outlier_matches=outliers,
        reprojection_errors_px=reproj_errors,
        mean_reproj_error_px=mean_err,
    )

    logger.info(result.summary())
    if not result.is_valid:
        logger.error(
            "Homography does NOT meet quality threshold. "
            "Inspect fiducial detections before proceeding."
        )

    return result


# ---------------------------------------------------------------------------
# High-level orchestration function (Stages 2+3 together)
# ---------------------------------------------------------------------------

def detect_fiducials_and_compute_homography(
    gray_image: np.ndarray,
    manifest: CADManifest,
    cfg: Optional[FiducialConfig] = None,
) -> HomographyResult:
    """
    Full Stage 2+3 pipeline entry point.

    Parameters
    ----------
    gray_image : pre-processed grayscale image from Stage 1b
    manifest   : parsed CAD manifest from Stage 1a
    cfg        : FiducialConfig (defaults if None)

    Returns
    -------
    HomographyResult ready for Stage 4 projections.
    """
    if cfg is None:
        cfg = FiducialConfig()

    cad_fiducials = manifest.fiducials
    if not cad_fiducials:
        raise ValueError(
            "No fiducial features found in CAD manifest. "
            "Mark at least 4 reference points as type='fiducial' in your CSV."
        )

    logger.info(
        f"Stage 2: Detecting {len(cad_fiducials)} fiducials "
        f"using strategy: {cfg.strategy.name}"
    )

    if cfg.strategy == FiducialStrategy.ARUCO:
        # ArUco: direct ID mapping
        aruco_detections = _detect_aruco(gray_image, cfg)
        if len(aruco_detections) < 4:
            raise RuntimeError(
                f"ArUco detected only {len(aruco_detections)} marker(s). "
                "Ensure markers are visible and dictionary matches cfg."
            )
        # Map ArUco IDs to CAD fiducials by index
        # Convention: ArUco marker ID N → cad_fiducials[N]
        matches = []
        for marker_id, (cx, cy) in aruco_detections.items():
            if marker_id < len(cad_fiducials):
                fid = cad_fiducials[marker_id]
                matches.append(FiducialMatch(
                    cad_id    = fid.id,
                    cad_pt_mm = (fid.x_norm_mm, fid.y_norm_mm),
                    img_pt_px = (cx, cy),
                ))

    elif cfg.strategy == FiducialStrategy.BLOB:
        logger.info("Bypassing Blob Detector. Injecting manual fiducial pixel coordinates.")
        
        # --- MANUAL PIXEL OVERRIDE ---
        # UPDATE THESE 4 COORDINATES using your pixel_picker.py script!
        # They MUST be in the exact same order as the 4 'fiducial' rows in your real_deck.csv
        manual_pixels = [
            (419, 2270),   # Pixel (x, y) for Fiducial 1 (e.g., T1)
            (1822, 2282),  # Pixel (x, y) for Fiducial 2 (e.g., T5)
            (1777, 127), # Pixel (x, y) for Fiducial 3 (e.g., T10)
            (622, 139)   # Pixel (x, y) for Fiducial 4 (e.g., T14)
        ]
        
        if len(manual_pixels) != len(cad_fiducials):
            raise RuntimeError("The number of manual_pixels must match the number of CAD fiducials!")

        matches = []
        for i in range(len(cad_fiducials)):
            fid = cad_fiducials[i]
            matches.append(FiducialMatch(
                cad_id    = fid.id,
                cad_pt_mm = (fid.x_norm_mm, fid.y_norm_mm),
                img_pt_px = manual_pixels[i],
            ))
        # -----------------------------
    else:
        raise ValueError(f"Unknown FiducialStrategy: {cfg.strategy}")

    logger.info(f"Stage 2 complete: {len(matches)} correspondences established.")

    logger.info("Stage 3: Computing homography via RANSAC...")
    homography = compute_homography(matches, cfg)

    return homography
