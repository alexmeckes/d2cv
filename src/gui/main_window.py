import sys
from typing import Optional, Callable
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QGroupBox, QGridLayout, QFrame
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap, QFont
import numpy as np
import cv2


class BotWorker(QThread):
    """Worker thread for running the bot loop."""
    status_update = pyqtSignal(str)
    screenshot_update = pyqtSignal(np.ndarray)
    stats_update = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.running = False
        self.bot_callback: Optional[Callable] = None

    def set_callback(self, callback: Callable):
        self.bot_callback = callback

    def run(self):
        self.running = True
        if self.bot_callback:
            self.bot_callback(self)

    def stop(self):
        self.running = False


class MainWindow(QMainWindow):
    """Main GUI window for D2CV bot."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("D2CV - Diablo 2 Bot")
        self.setMinimumSize(900, 700)

        # Bot state
        self.is_running = False
        self.worker: Optional[BotWorker] = None

        # Stats
        self.stats = {
            "runs": 0,
            "deaths": 0,
            "items_found": 0,
            "runtime": 0,
        }

        self._setup_ui()
        self._setup_timers()

    def _setup_ui(self):
        """Create the UI layout."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # Left panel - controls and stats
        left_panel = QVBoxLayout()

        # Status
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)
        self.status_label = QLabel("Idle")
        self.status_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(self.status_label)
        left_panel.addWidget(status_group)

        # Controls
        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout(controls_group)

        self.start_btn = QPushButton("Start Bot")
        self.start_btn.setMinimumHeight(50)
        self.start_btn.clicked.connect(self._toggle_bot)
        controls_layout.addWidget(self.start_btn)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setEnabled(False)
        controls_layout.addWidget(self.pause_btn)

        left_panel.addWidget(controls_group)

        # Stats
        stats_group = QGroupBox("Statistics")
        stats_layout = QGridLayout(stats_group)

        self.runs_label = QLabel("0")
        self.deaths_label = QLabel("0")
        self.items_label = QLabel("0")
        self.runtime_label = QLabel("0:00:00")

        stats_layout.addWidget(QLabel("Runs:"), 0, 0)
        stats_layout.addWidget(self.runs_label, 0, 1)
        stats_layout.addWidget(QLabel("Deaths:"), 1, 0)
        stats_layout.addWidget(self.deaths_label, 1, 1)
        stats_layout.addWidget(QLabel("Items:"), 2, 0)
        stats_layout.addWidget(self.items_label, 2, 1)
        stats_layout.addWidget(QLabel("Runtime:"), 3, 0)
        stats_layout.addWidget(self.runtime_label, 3, 1)

        left_panel.addWidget(stats_group)

        # Health/Mana display
        vitals_group = QGroupBox("Vitals")
        vitals_layout = QGridLayout(vitals_group)

        self.hp_bar = QLabel("HP: ---%")
        self.mp_bar = QLabel("MP: ---%")

        vitals_layout.addWidget(self.hp_bar, 0, 0)
        vitals_layout.addWidget(self.mp_bar, 1, 0)

        left_panel.addWidget(vitals_group)
        left_panel.addStretch()

        layout.addLayout(left_panel, 1)

        # Right panel - screenshot and log
        right_panel = QVBoxLayout()

        # Screenshot preview
        preview_group = QGroupBox("Game View")
        preview_layout = QVBoxLayout(preview_group)

        self.preview_label = QLabel("No screenshot")
        self.preview_label.setMinimumSize(640, 360)
        self.preview_label.setMaximumSize(640, 360)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333;")
        preview_layout.addWidget(self.preview_label)

        right_panel.addWidget(preview_group)

        # Log output
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        self.log_text.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.log_text)

        right_panel.addWidget(log_group)

        layout.addLayout(right_panel, 2)

    def _setup_timers(self):
        """Set up update timers."""
        self.runtime_timer = QTimer()
        self.runtime_timer.timeout.connect(self._update_runtime)

    def _toggle_bot(self):
        """Start or stop the bot."""
        if self.is_running:
            self._stop_bot()
        else:
            self._start_bot()

    def _start_bot(self):
        """Start the bot."""
        self.is_running = True
        self.start_btn.setText("Stop Bot")
        self.pause_btn.setEnabled(True)
        self.status_label.setText("Running")
        self.status_label.setStyleSheet("color: #00ff00;")
        self.runtime_timer.start(1000)
        self.log("Bot started")

    def _stop_bot(self):
        """Stop the bot."""
        self.is_running = False
        self.start_btn.setText("Start Bot")
        self.pause_btn.setEnabled(False)
        self.status_label.setText("Stopped")
        self.status_label.setStyleSheet("color: #ff6666;")
        self.runtime_timer.stop()

        if self.worker:
            self.worker.stop()
            self.worker.wait()
            self.worker = None

        self.log("Bot stopped")

    def _update_runtime(self):
        """Update runtime display."""
        self.stats["runtime"] += 1
        hours = self.stats["runtime"] // 3600
        minutes = (self.stats["runtime"] % 3600) // 60
        seconds = self.stats["runtime"] % 60
        self.runtime_label.setText(f"{hours}:{minutes:02d}:{seconds:02d}")

    def update_screenshot(self, image: np.ndarray):
        """Update the screenshot preview."""
        # Resize to fit preview
        h, w = image.shape[:2]
        scale = min(640 / w, 360 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))

        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Create QImage and display
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.preview_label.setPixmap(QPixmap.fromImage(qimg))

    def update_status(self, status: str):
        """Update the status label."""
        self.status_label.setText(status)

    def update_vitals(self, hp_percent: float, mp_percent: float):
        """Update HP/MP display."""
        self.hp_bar.setText(f"HP: {hp_percent:.0%}")
        self.mp_bar.setText(f"MP: {mp_percent:.0%}")

        # Color based on health level
        if hp_percent < 0.3:
            self.hp_bar.setStyleSheet("color: #ff3333;")
        elif hp_percent < 0.6:
            self.hp_bar.setStyleSheet("color: #ffaa00;")
        else:
            self.hp_bar.setStyleSheet("color: #33ff33;")

    def update_stats(self, runs: int = None, deaths: int = None, items: int = None):
        """Update statistics display."""
        if runs is not None:
            self.stats["runs"] = runs
            self.runs_label.setText(str(runs))
        if deaths is not None:
            self.stats["deaths"] = deaths
            self.deaths_label.setText(str(deaths))
        if items is not None:
            self.stats["items_found"] = items
            self.items_label.setText(str(items))

    def log(self, message: str):
        """Add a message to the log."""
        self.log_text.append(message)
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def closeEvent(self, event):
        """Handle window close."""
        self._stop_bot()
        event.accept()


def run_gui():
    """Launch the GUI application."""
    app = QApplication(sys.argv)

    # Dark theme
    app.setStyleSheet("""
        QMainWindow, QWidget {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        QGroupBox {
            border: 1px solid #444;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
        QPushButton {
            background-color: #3c3c3c;
            border: 1px solid #555;
            padding: 8px;
            border-radius: 4px;
        }
        QPushButton:hover {
            background-color: #4a4a4a;
        }
        QPushButton:pressed {
            background-color: #555;
        }
        QTextEdit {
            background-color: #1a1a1a;
            border: 1px solid #333;
        }
        QLabel {
            color: #cccccc;
        }
    """)

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    run_gui()
