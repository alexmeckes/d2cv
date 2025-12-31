"""
OCR for reading item names, stats, and other game text.
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass
import re

# Try to import easyocr, fall back gracefully if not available
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# Try pytesseract as fallback
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


@dataclass
class OCRResult:
    """Result from OCR recognition."""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h


class GameOCR:
    """OCR optimized for Diablo 2 game text."""

    # Known rune names for fuzzy matching
    RUNE_NAMES = [
        "el", "eld", "tir", "nef", "eth", "ith", "tal", "ral", "ort", "thul",
        "amn", "sol", "shael", "dol", "hel", "io", "lum", "ko", "fal", "lem",
        "pul", "um", "mal", "ist", "gul", "vex", "ohm", "lo", "sur", "ber",
        "jah", "cham", "zod"
    ]

    # Common item type keywords
    ITEM_TYPES = [
        "sword", "axe", "mace", "scepter", "wand", "staff", "bow", "crossbow",
        "dagger", "javelin", "spear", "polearm", "helm", "armor", "shield",
        "gloves", "boots", "belt", "ring", "amulet", "charm", "jewel",
        "circlet", "coronet", "tiara", "diadem"
    ]

    def __init__(self, use_easyocr: bool = True):
        """Initialize OCR engine.

        Args:
            use_easyocr: Prefer easyocr over tesseract (better for stylized fonts)
        """
        self.reader = None
        self.use_easyocr = use_easyocr and EASYOCR_AVAILABLE

        if self.use_easyocr:
            # Initialize easyocr (downloads model on first use)
            self.reader = easyocr.Reader(['en'], gpu=False)
        elif not TESSERACT_AVAILABLE:
            print("Warning: No OCR engine available. Install easyocr or pytesseract.")

    def read_text(
        self,
        image: np.ndarray,
        preprocess: bool = True
    ) -> List[OCRResult]:
        """Read text from an image region.

        Args:
            image: BGR image to read
            preprocess: Apply preprocessing for better recognition

        Returns:
            List of OCR results
        """
        if preprocess:
            image = self._preprocess(image)

        if self.use_easyocr and self.reader:
            return self._read_easyocr(image)
        elif TESSERACT_AVAILABLE:
            return self._read_tesseract(image)
        else:
            return []

    def read_item_label(self, label_image: np.ndarray) -> Optional[str]:
        """Read text from an item label.

        Optimized for the single-line item names shown on ground.
        """
        # Preprocess for item labels
        processed = self._preprocess_label(label_image)

        results = self.read_text(processed, preprocess=False)

        if results:
            # Combine all text, clean it up
            text = " ".join(r.text for r in results)
            return self._clean_item_text(text)

        return None

    def read_item_tooltip(self, tooltip_image: np.ndarray) -> List[str]:
        """Read multi-line item tooltip (when hovering over item).

        Returns list of lines (name, stats, etc.)
        """
        # Split into lines based on text color/position
        results = self.read_text(tooltip_image)

        # Sort by Y position to get line order
        results.sort(key=lambda r: r.bbox[1])

        lines = []
        current_line = []
        last_y = -1

        for result in results:
            y = result.bbox[1]
            if last_y >= 0 and abs(y - last_y) > 15:  # New line threshold
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = []
            current_line.append(result.text)
            last_y = y

        if current_line:
            lines.append(" ".join(current_line))

        return [self._clean_item_text(line) for line in lines]

    def identify_rune(self, text: str) -> Optional[str]:
        """Identify a rune name from OCR text.

        Uses fuzzy matching to handle OCR errors.
        """
        text_lower = text.lower().strip()

        # Direct match
        for rune in self.RUNE_NAMES:
            if rune in text_lower:
                return rune.capitalize()

        # Check for "RUNE" keyword
        if "rune" in text_lower:
            # Extract word before "rune"
            match = re.search(r'(\w+)\s*rune', text_lower)
            if match:
                potential_rune = match.group(1)
                # Fuzzy match
                best_match = self._fuzzy_match_rune(potential_rune)
                if best_match:
                    return best_match.capitalize()

        return None

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """General preprocessing for OCR."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Increase contrast
        gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)

        # Denoise
        gray = cv2.fastNlMeansDenoising(gray, h=10)

        # Threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    def _preprocess_label(self, image: np.ndarray) -> np.ndarray:
        """Preprocessing specific to item labels."""
        # Item labels have colored backgrounds
        # Extract the text (usually lighter) from background

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Scale up for better OCR
        scale = 2
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Threshold to get text
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        # Invert if needed (white text on dark = invert)
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)

        return binary

    def _read_easyocr(self, image: np.ndarray) -> List[OCRResult]:
        """Read using easyocr."""
        results = []

        # easyocr expects BGR or grayscale
        raw_results = self.reader.readtext(image)

        for bbox, text, confidence in raw_results:
            # bbox is [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
            x1 = int(min(p[0] for p in bbox))
            y1 = int(min(p[1] for p in bbox))
            x2 = int(max(p[0] for p in bbox))
            y2 = int(max(p[1] for p in bbox))

            results.append(OCRResult(
                text=text,
                confidence=confidence,
                bbox=(x1, y1, x2 - x1, y2 - y1)
            ))

        return results

    def _read_tesseract(self, image: np.ndarray) -> List[OCRResult]:
        """Read using pytesseract."""
        results = []

        # Get detailed data
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            conf = int(data['conf'][i])

            if text and conf > 0:
                results.append(OCRResult(
                    text=text,
                    confidence=conf / 100.0,
                    bbox=(
                        data['left'][i],
                        data['top'][i],
                        data['width'][i],
                        data['height'][i]
                    )
                ))

        return results

    def _clean_item_text(self, text: str) -> str:
        """Clean up OCR'd item text."""
        # Remove common OCR artifacts
        text = re.sub(r'[|\\/_\[\]{}]', '', text)

        # Fix common OCR mistakes
        replacements = {
            '0': 'O',  # Zero to O in names
            '1': 'I',  # One to I in names
            '5': 'S',  # Five to S
        }

        # Only apply to likely name portions (not stats)
        if not any(c.isdigit() for c in text[-3:]):
            for old, new in replacements.items():
                if old in text and not text.replace(old, '').isdigit():
                    text = text.replace(old, new)

        return text.strip()

    def _fuzzy_match_rune(self, text: str) -> Optional[str]:
        """Fuzzy match a potential rune name."""
        text = text.lower()

        # Simple edit distance matching
        best_match = None
        best_distance = 3  # Max allowed edits

        for rune in self.RUNE_NAMES:
            distance = self._levenshtein(text, rune)
            if distance < best_distance:
                best_distance = distance
                best_match = rune

        return best_match

    def _levenshtein(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein edit distance."""
        if len(s1) < len(s2):
            return self._levenshtein(s2, s1)

        if len(s2) == 0:
            return len(s1)

        prev_row = range(len(s2) + 1)

        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row

        return prev_row[-1]
