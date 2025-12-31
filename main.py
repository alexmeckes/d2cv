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


def test_vision():
    """Test vision detection on current screen."""
    print("Testing vision detection...")

    from src.vision import GameStateDetector, ItemDetector, EnemyDetector

    wm = WindowManager()
    window = wm.find_window()

    if window is None:
        print("ERROR: Could not find D2 window!")
        return False

    sc = ScreenCapture(wm)
    screenshot = sc.capture()

    if screenshot is None:
        print("ERROR: Could not capture screen")
        return False

    # Test game state detection
    game_state = GameStateDetector()
    vitals = game_state.get_vitals(screenshot.image)

    print(f"\nVitals:")
    print(f"  Health: {vitals.health_percent:.0%}")
    print(f"  Mana: {vitals.mana_percent:.0%}")
    print(f"  Poisoned: {vitals.is_poisoned}")

    belt = game_state.get_belt_state(screenshot.image)
    print(f"\nBelt:")
    print(f"  Slot 1: {belt.slot_1_count} potions")
    print(f"  Slot 2: {belt.slot_2_count} potions")
    print(f"  Slot 3: {belt.slot_3_count} potions")
    print(f"  Slot 4: {belt.slot_4_count} potions")

    # Test item detection (requires Alt held)
    item_detector = ItemDetector()
    items = item_detector.detect_items(screenshot.image)
    print(f"\nItems on ground: {len(items)}")
    for item in items[:5]:
        print(f"  - {item.rarity.value} at ({item.x}, {item.y})")

    # Test enemy detection
    enemy_detector = EnemyDetector()
    enemies = enemy_detector.detect_by_health_bars(screenshot.image)
    print(f"\nEnemies detected: {len(enemies)}")
    for enemy in enemies[:5]:
        print(f"  - {'Boss' if enemy.is_boss else 'Enemy'} at ({enemy.center[0]}, {enemy.center[1]})")

    # Save annotated screenshot
    import cv2
    annotated = screenshot.image.copy()

    # Draw item boxes
    for item in items:
        color = {
            "unique": (0, 215, 255),
            "set": (0, 255, 0),
            "rare": (0, 255, 255),
            "magic": (255, 100, 100),
        }.get(item.rarity.value, (255, 255, 255))
        cv2.rectangle(annotated, (item.x, item.y),
                      (item.x + item.width, item.y + item.height), color, 2)

    # Draw enemy boxes
    for enemy in enemies:
        cv2.rectangle(annotated, (enemy.x, enemy.y),
                      (enemy.x + enemy.width, enemy.y + enemy.height), (0, 0, 255), 2)

    cv2.imwrite("debug_vision.png", annotated)
    print(f"\nAnnotated screenshot saved to debug_vision.png")

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


def run_headless(max_runs: int = 0):
    """Run bot in headless mode (no GUI)."""
    from src.bot import D2Bot

    print("\nStarting headless mode...")

    bot = D2Bot()

    def on_run_complete(result):
        print(f"[{result.run_name}] {result.status.name} - "
              f"Duration: {result.duration:.1f}s, Items: {result.items_found}")

    bot.on_run_complete = on_run_complete

    if not bot.start():
        print("Failed to start bot")
        return

    try:
        print("Bot running. Press Ctrl+C to stop.")
        while bot.is_running():
            time.sleep(1)

            # Print periodic status
            status = bot.get_status()
            runtime = int(status.runtime_seconds)
            print(f"\r[{runtime//60}:{runtime%60:02d}] "
                  f"Runs: {status.runs_completed} | "
                  f"HP: {status.health_percent:.0%} | "
                  f"MP: {status.mana_percent:.0%}", end="")

    except KeyboardInterrupt:
        print("\n\nStopping bot...")
        bot.stop()

    # Print final stats
    stats = bot.run_manager.get_statistics()
    print("\n" + "=" * 50)
    print("Session Statistics:")
    print(f"  Total runs: {stats['total_runs']}")
    print(f"  Success rate: {stats['success_rate']}")
    print(f"  Total items: {stats['total_items']}")
    print(f"  Total deaths: {stats['total_deaths']}")
    print(f"  Average run time: {stats['average_duration']}")


def main():
    parser = argparse.ArgumentParser(description="D2CV - Diablo 2 Computer Vision Bot")
    parser.add_argument("--test-capture", action="store_true", help="Test screen capture")
    parser.add_argument("--test-vision", action="store_true", help="Test vision detection")
    parser.add_argument("--test-input", action="store_true", help="Test input simulation")
    parser.add_argument("--no-gui", action="store_true", help="Run without GUI (headless)")
    parser.add_argument("--max-runs", type=int, default=0, help="Max runs in headless mode (0=infinite)")
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

    if args.test_vision:
        test_vision()
        return

    if args.test_input:
        test_input()
        return

    # Run bot
    if args.no_gui:
        run_headless(args.max_runs)
    else:
        print("\nLaunching GUI...")
        sys.exit(run_gui())


if __name__ == "__main__":
    main()
