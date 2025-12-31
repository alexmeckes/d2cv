import mss
import numpy as np
import cv2
from typing import Optional, Tuple
from dataclasses import dataclass

from .window import WindowManager, WindowInfo


@dataclass
class Screenshot:
    """Container for a captured screenshot with metadata."""
    image: np.ndarray  # BGR format (OpenCV standard)
    timestamp: float
    window_info: WindowInfo

    @property
    def rgb(self) -> np.ndarray:
        """Convert to RGB format."""
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

    @property
    def gray(self) -> np.ndarray:
        """Convert to grayscale."""
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]

    def get_region(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Extract a region from the screenshot."""
        return self.image[y:y+h, x:x+w]


class ScreenCapture:
    """Fast screen capture using mss."""

    def __init__(self, window_manager: Optional[WindowManager] = None):
        self.window_manager = window_manager or WindowManager()
        self._sct = mss.mss()

    def capture(self, refresh_window: bool = False) -> Optional[Screenshot]:
        """Capture the game window.

        Args:
            refresh_window: Whether to refresh window position before capture.

        Returns:
            Screenshot object or None if window not found.
        """
        import time

        window = self.window_manager.get_window(refresh=refresh_window)
        if window is None:
            return None

        # Define capture region
        monitor = {
            "left": window.x,
            "top": window.y,
            "width": window.width,
            "height": window.height,
        }

        # Capture
        sct_img = self._sct.grab(monitor)

        # Convert to numpy array (BGRA -> BGR)
        img = np.array(sct_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        return Screenshot(
            image=img,
            timestamp=time.time(),
            window_info=window,
        )

    def capture_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        relative: bool = True
    ) -> Optional[np.ndarray]:
        """Capture a specific region of the screen.

        Args:
            x, y: Top-left corner
            width, height: Dimensions
            relative: If True, coordinates are relative to game window

        Returns:
            BGR image array or None if window not found.
        """
        if relative:
            window = self.window_manager.get_window()
            if window is None:
                return None
            x += window.x
            y += window.y

        monitor = {
            "left": x,
            "top": y,
            "width": width,
            "height": height,
        }

        sct_img = self._sct.grab(monitor)
        img = np.array(sct_img)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def get_fps_benchmark(self, num_frames: int = 100) -> float:
        """Benchmark capture speed.

        Returns:
            Frames per second achieved.
        """
        import time

        start = time.time()
        for _ in range(num_frames):
            self.capture()
        elapsed = time.time() - start

        return num_frames / elapsed
