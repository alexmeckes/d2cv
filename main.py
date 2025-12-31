#!/usr/bin/env python3
"""
D2CV - Diablo 2 Computer Vision Bot

A computer vision bot for Project Diablo 2 with LLM integration.
"""

import sys
import time
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config, get_config
from src.capture import ScreenCapture, WindowManager
from src.actions import InputController
from src.gui.main_window import MainWindow, run_gui


def test_capture():
    """Test screen capture functionality."""
    print("Testing screen capture...")

    wm = WindowManager()
    window = wm.find_window()

    if window is None:
        print("ERROR: Could not find D2 window!")
        print("Make sure the game is running.")
        return False

    print(f"Found window: {window.title}")
    print(f"  Position: ({window.x}, {window.y})")
    print(f"  Size: {window.width}x{window.height}")

    sc = ScreenCapture(wm)

    # Test capture speed
    print("\nBenchmarking capture speed...")
    fps = sc.get_fps_benchmark(50)
    print(f"Capture speed: {fps:.1f} FPS")

    # Take a screenshot
    screenshot = sc.capture()
    if screenshot:
        import cv2
        cv2.imwrite("debug_screenshot.png", screenshot.image)
        print(f"\nScreenshot saved to debug_screenshot.png")
        print(f"  Size: {screenshot.width}x{screenshot.height}")

    return True


def test_input():
    """Test input simulation (be careful - this moves your mouse!)."""
    print("\nTesting input simulation...")
    print("WARNING: This will move your mouse in 3 seconds!")
    time.sleep(3)

    wm = WindowManager()
    ic = InputController(wm)

    window = wm.find_window()
    if window is None:
        print("ERROR: No D2 window found")
        return False

    # Move mouse to center of window
    center_x = window.width // 2
    center_y = window.height // 2

    print(f"Moving to window center ({center_x}, {center_y})...")
    ic.move_to(center_x, center_y, relative=True)

    print("Input test complete")
    return True


def main():
    parser = argparse.ArgumentParser(description="D2CV - Diablo 2 Computer Vision Bot")
    parser.add_argument("--test-capture", action="store_true", help="Test screen capture")
    parser.add_argument("--test-input", action="store_true", help="Test input simulation")
    parser.add_argument("--no-gui", action="store_true", help="Run without GUI (headless)")
    parser.add_argument("--config", type=str, help="Path to config file")

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config) if args.config else None
    config = load_config(config_path)

    print("=" * 50)
    print("D2CV - Diablo 2 Computer Vision Bot")
    print("=" * 50)
    print(f"Window: {config.window_width}x{config.window_height}")
    print(f"Runs enabled: {', '.join(config.runs_enabled)}")
    print(f"LLM: {'Enabled' if config.llm.enabled else 'Disabled'}")
    print("=" * 50)

    # Run tests if requested
    if args.test_capture:
        test_capture()
        return

    if args.test_input:
        test_input()
        return

    # Launch GUI
    if not args.no_gui:
        print("\nLaunching GUI...")
        sys.exit(run_gui())
    else:
        print("\nHeadless mode not yet implemented.")
        print("Run with GUI for now, or use --test-capture to verify setup.")


if __name__ == "__main__":
    main()
