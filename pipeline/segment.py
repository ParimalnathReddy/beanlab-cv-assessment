"""
segment.py — Instance segmentation of individual beans via distance-transform watershed.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class BeanInstance:
    mask_id:  int
    area_px:  int
    centroid: tuple   # (row, col)


def segment_beans(binary_mask: np.ndarray) -> np.ndarray:
    """
    Separate individual beans from a foreground binary mask using
    distance-transform watershed.

    Args:
        binary_mask: uint8 or bool array, shape (H, W).
                     Foreground pixels (beans) are non-zero.

    Returns:
        Integer label map, shape (H, W):
            0  = background
           -1  = watershed boundary
            1  = sure background (artifact of the algorithm)
            2+ = bean instances
    """
    mask_u8 = binary_mask.astype(np.uint8)

    # Distance transform: each pixel's value = distance to nearest background
    dist      = cv2.distanceTransform(mask_u8, cv2.DIST_L2, maskSize=5)
    dist_norm = dist / (dist.max() + 1e-8)

    # Threshold to find seed regions for each bean instance.
    # 0.7 means only pixels within 30% of the local peak are retained.
    # This is intentionally conservative to avoid false splits on
    # elongated single beans that have two local distance maxima.
    _, sure_fg = cv2.threshold(
        (dist_norm * 255).astype(np.uint8),
        int(0.7 * 255), 255, cv2.THRESH_BINARY
    )

    # Definite background via dilation
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    sure_bg = cv2.dilate(mask_u8 * 255, kernel, iterations=3)

    # Unknown region (neither sure foreground nor sure background)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Label connected foreground regions as initial markers
    _, markers = cv2.connectedComponents(sure_fg)
    markers   += 1
    markers[unknown == 255] = 0

    # Watershed requires a 3-channel image
    bgr     = cv2.cvtColor(mask_u8 * 255, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(bgr, markers)

    return markers


def extract_instances(markers: np.ndarray) -> List[BeanInstance]:
    """Convert a watershed label map to a list of BeanInstance objects."""
    instances = []
    for label_id in np.unique(markers):
        if label_id <= 1:          # skip background (1) and border (-1)
            continue
        region = (markers == label_id)
        area   = int(region.sum())
        cy, cx = [int(v) for v in np.argwhere(region).mean(axis=0)]
        instances.append(BeanInstance(
            mask_id=label_id, area_px=area, centroid=(cy, cx)
        ))
    return instances


def count_beans(binary_mask: np.ndarray) -> int:
    """Convenience wrapper: segment and return the bean instance count."""
    markers = segment_beans(binary_mask)
    return len(extract_instances(markers))
