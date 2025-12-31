import win32gui
import win32con
import win32api
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class WindowInfo:
    hwnd: int
    title: str
    x: int
    y: int
    width: int
    height: int

    @property
    def region(self) -> Tuple[int, int, int, int]:
        """Returns (left, top, right, bottom) for mss capture."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    @property
    def center(self) -> Tuple[int, int]:
        """Returns the center point of the window."""
        return (self.x + self.width // 2, self.y + self.height // 2)


class WindowManager:
    """Manages finding and interacting with the D2 game window."""

    WINDOW_TITLES = [
        "Diablo II",
        "Project Diablo 2",
        "Diablo II: Lord of Destruction",
    ]

    def __init__(self):
        self._cached_window: Optional[WindowInfo] = None

    def find_window(self) -> Optional[WindowInfo]:
        """Find the D2 game window by title."""
        result = None

        def enum_callback(hwnd, _):
            nonlocal result
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                for search_title in self.WINDOW_TITLES:
                    if search_title.lower() in title.lower():
                        rect = win32gui.GetWindowRect(hwnd)
                        result = WindowInfo(
                            hwnd=hwnd,
                            title=title,
                            x=rect[0],
                            y=rect[1],
                            width=rect[2] - rect[0],
                            height=rect[3] - rect[1],
                        )
                        return False  # Stop enumeration
            return True

        win32gui.EnumWindows(enum_callback, None)
        self._cached_window = result
        return result

    def get_window(self, refresh: bool = False) -> Optional[WindowInfo]:
        """Get cached window info or refresh if requested."""
        if refresh or self._cached_window is None:
            return self.find_window()
        return self._cached_window

    def focus_window(self) -> bool:
        """Bring the D2 window to foreground."""
        window = self.get_window()
        if window is None:
            return False

        try:
            win32gui.SetForegroundWindow(window.hwnd)
            return True
        except Exception:
            # Sometimes fails if window is minimized
            try:
                win32gui.ShowWindow(window.hwnd, win32con.SW_RESTORE)
                win32gui.SetForegroundWindow(window.hwnd)
                return True
            except Exception:
                return False

    def is_foreground(self) -> bool:
        """Check if D2 window is currently in foreground."""
        window = self.get_window()
        if window is None:
            return False
        return win32gui.GetForegroundWindow() == window.hwnd

    def get_client_offset(self) -> Tuple[int, int]:
        """Get the offset from window coords to client area (excludes title bar)."""
        window = self.get_window()
        if window is None:
            return (0, 0)

        # Get client rect (relative to window)
        client_rect = win32gui.GetClientRect(window.hwnd)
        # Convert client (0,0) to screen coordinates
        point = win32gui.ClientToScreen(window.hwnd, (0, 0))

        # Offset is difference between window top-left and client top-left
        return (point[0] - window.x, point[1] - window.y)
