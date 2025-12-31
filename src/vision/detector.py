"""
Template matching and detection framework.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict, NamedTuple
from dataclasses import dataclass
from enum import Enum


class MatchResult(NamedTuple):
    """Result of a template match."""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    name: str

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Returns (x1, y1, x2, y2)."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)


@dataclass
class Template:
    """A template image for matching."""
    name: str
    image: np.ndarray
    mask: Optional[np.ndarray] = None
    threshold: float = 0.8

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def height(self) -> int:
        return self.image.shape[0]


class TemplateMatcher:
    """Handles template matching operations."""

    def __init__(self, templates_dir: Optional[Path] = None):
        self.templates_dir = templates_dir or Path(__file__).parent.parent.parent / "assets" / "templates"
        self.templates: Dict[str, Template] = {}
        self._cache: Dict[str, np.ndarray] = {}

    def load_template(
        self,
        name: str,
        path: Optional[Path] = None,
        threshold: float = 0.8,
        grayscale: bool = True
    ) -> Optional[Template]:
        """Load a template image from file.

        Args:
            name: Unique identifier for this template
            path: Path to image file (or looks in templates_dir/name.png)
            threshold: Minimum confidence for a match
            grayscale: Convert to grayscale for matching

        Returns:
            Template object or None if load failed
        """
        if path is None:
            path = self.templates_dir / f"{name}.png"

        if not path.exists():
            print(f"Warning: Template not found: {path}")
            return None

        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None

        # Handle alpha channel as mask
        mask = None
        if img.shape[-1] == 4:
            mask = img[:, :, 3]
            img = img[:, :, :3]

        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if mask is not None:
                # Threshold mask
                _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        template = Template(name=name, image=img, mask=mask, threshold=threshold)
        self.templates[name] = template
        return template

    def load_templates_from_dir(
        self,
        subdir: str,
        threshold: float = 0.8,
        grayscale: bool = True
    ) -> int:
        """Load all templates from a subdirectory.

        Returns:
            Number of templates loaded
        """
        dir_path = self.templates_dir / subdir
        if not dir_path.exists():
            return 0

        count = 0
        for img_path in dir_path.glob("*.png"):
            name = f"{subdir}/{img_path.stem}"
            if self.load_template(name, img_path, threshold, grayscale):
                count += 1

        return count

    def match_template(
        self,
        image: np.ndarray,
        template: Template,
        method: int = cv2.TM_CCOEFF_NORMED
    ) -> List[MatchResult]:
        """Find all matches of a template in an image.

        Args:
            image: Image to search in (BGR or grayscale)
            template: Template to find
            method: OpenCV matching method

        Returns:
            List of match results above threshold
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3 and len(template.image.shape) == 2:
            search_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            search_img = image

        # Perform matching
        if template.mask is not None:
            result = cv2.matchTemplate(search_img, template.image, method, mask=template.mask)
        else:
            result = cv2.matchTemplate(search_img, template.image, method)

        # Find all locations above threshold
        locations = np.where(result >= template.threshold)
        matches = []

        for pt in zip(*locations[::-1]):  # Switch x and y
            confidence = result[pt[1], pt[0]]
            matches.append(MatchResult(
                x=pt[0],
                y=pt[1],
                width=template.width,
                height=template.height,
                confidence=float(confidence),
                name=template.name
            ))

        # Non-maximum suppression to remove overlapping detections
        matches = self._nms(matches, overlap_thresh=0.5)

        return matches

    def match_template_best(
        self,
        image: np.ndarray,
        template: Template,
        method: int = cv2.TM_CCOEFF_NORMED
    ) -> Optional[MatchResult]:
        """Find the best match of a template.

        Returns:
            Best match or None if below threshold
        """
        if len(image.shape) == 3 and len(template.image.shape) == 2:
            search_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            search_img = image

        if template.mask is not None:
            result = cv2.matchTemplate(search_img, template.image, method, mask=template.mask)
        else:
            result = cv2.matchTemplate(search_img, template.image, method)

        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val >= template.threshold:
            return MatchResult(
                x=max_loc[0],
                y=max_loc[1],
                width=template.width,
                height=template.height,
                confidence=float(max_val),
                name=template.name
            )
        return None

    def find_by_name(
        self,
        image: np.ndarray,
        template_name: str,
        best_only: bool = True
    ) -> Optional[MatchResult] | List[MatchResult]:
        """Find a template by name.

        Args:
            image: Image to search
            template_name: Name of loaded template
            best_only: Return only best match

        Returns:
            MatchResult(s) or None
        """
        if template_name not in self.templates:
            return None if best_only else []

        template = self.templates[template_name]

        if best_only:
            return self.match_template_best(image, template)
        return self.match_template(image, template)

    def _nms(
        self,
        matches: List[MatchResult],
        overlap_thresh: float = 0.5
    ) -> List[MatchResult]:
        """Non-maximum suppression to remove overlapping boxes."""
        if len(matches) == 0:
            return []

        # Sort by confidence
        matches = sorted(matches, key=lambda x: x.confidence, reverse=True)

        keep = []
        for match in matches:
            # Check overlap with kept matches
            dominated = False
            for kept in keep:
                iou = self._compute_iou(match.bbox, kept.bbox)
                if iou > overlap_thresh:
                    dominated = True
                    break

            if not dominated:
                keep.append(match)

        return keep

    def _compute_iou(
        self,
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int]
    ) -> float:
        """Compute intersection over union of two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0


class ColorDetector:
    """Detect game elements by color."""

    # D2 color constants (BGR format)
    COLORS = {
        # Item label background colors
        "unique": (0, 173, 181),      # Gold/tan
        "set": (0, 255, 0),           # Green
        "rare": (0, 255, 255),        # Yellow
        "magic": (255, 100, 100),     # Blue
        "normal": (255, 255, 255),    # White
        "socketed": (150, 150, 150),  # Gray

        # Health/mana orb colors
        "health_red": (0, 0, 200),
        "mana_blue": (200, 0, 0),

        # Minimap colors
        "minimap_enemy": (0, 0, 255),   # Red dots
        "minimap_player": (255, 255, 255),  # White
    }

    @staticmethod
    def detect_color_mask(
        image: np.ndarray,
        color_bgr: Tuple[int, int, int],
        tolerance: int = 30
    ) -> np.ndarray:
        """Create a binary mask for pixels near the target color.

        Args:
            image: BGR image
            color_bgr: Target color in BGR
            tolerance: Color matching tolerance

        Returns:
            Binary mask
        """
        lower = np.array([max(0, c - tolerance) for c in color_bgr])
        upper = np.array([min(255, c + tolerance) for c in color_bgr])
        return cv2.inRange(image, lower, upper)

    @staticmethod
    def detect_color_hsv(
        image: np.ndarray,
        hue_range: Tuple[int, int],
        sat_range: Tuple[int, int] = (50, 255),
        val_range: Tuple[int, int] = (50, 255)
    ) -> np.ndarray:
        """Detect colors using HSV color space.

        Args:
            image: BGR image
            hue_range: (min_hue, max_hue) in 0-180 range
            sat_range: (min_sat, max_sat)
            val_range: (min_val, max_val)

        Returns:
            Binary mask
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([hue_range[0], sat_range[0], val_range[0]])
        upper = np.array([hue_range[1], sat_range[1], val_range[1]])
        return cv2.inRange(hsv, lower, upper)

    @staticmethod
    def find_color_regions(
        mask: np.ndarray,
        min_area: int = 100
    ) -> List[Tuple[int, int, int, int]]:
        """Find contiguous regions in a binary mask.

        Returns:
            List of bounding boxes (x, y, w, h)
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(contour)
                regions.append((x, y, w, h))

        return regions

    @staticmethod
    def get_color_ratio(
        image: np.ndarray,
        color_bgr: Tuple[int, int, int],
        tolerance: int = 30
    ) -> float:
        """Get the ratio of pixels matching a color.

        Returns:
            Ratio from 0.0 to 1.0
        """
        mask = ColorDetector.detect_color_mask(image, color_bgr, tolerance)
        return np.count_nonzero(mask) / mask.size
