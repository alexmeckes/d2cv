"""
Main GUI window for D2CV bot.
"""

import sys
from typing import Optional, Callable, List
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QGroupBox, QGridLayout, QFrame,
    QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar, QTabWidget,
    QScrollArea, QSplitter, QComboBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor
import numpy as np
import cv2


class BotWorker(QThread):
    """Worker thread for running the bot loop."""
    status_update = pyqtSignal(str)
    screenshot_update = pyqtSignal(np.ndarray)
    stats_update = pyqtSignal(dict)
    run_complete = pyqtSignal(dict)
    item_found = pyqtSignal(dict)

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


class RunHistoryWidget(QWidget):
    """Widget displaying run history."""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Run history table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Run", "Status", "Duration", "Items", "Deaths"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table)

    def add_run(self, run_data: dict):
        """Add a run to the history."""
        row = self.table.rowCount()
        self.table.insertRow(row)

        # Run type
        self.table.setItem(row, 0, QTableWidgetItem(run_data.get("run_name", "Unknown")))

        # Status
        status = run_data.get("status", "Unknown")
        status_item = QTableWidgetItem(status)
        if status == "COMPLETED":
            status_item.setForeground(QColor("#33ff33"))
        elif status in ("FAILED", "ABORTED"):
            status_item.setForeground(QColor("#ff3333"))
        self.table.setItem(row, 1, status_item)

        # Duration
        duration = run_data.get("duration", 0)
        duration_str = f"{int(duration)}s"
        self.table.setItem(row, 2, QTableWidgetItem(duration_str))

        # Items
        items = run_data.get("items_found", 0)
        self.table.setItem(row, 3, QTableWidgetItem(str(items)))

        # Deaths
        deaths = run_data.get("deaths", 0)
        self.table.setItem(row, 4, QTableWidgetItem(str(deaths)))

        # Scroll to latest
        self.table.scrollToBottom()


class ItemsFoundWidget(QWidget):
    """Widget displaying found items."""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Items table
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Item", "Rarity", "Area", "Action"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table)

    def add_item(self, item_data: dict):
        """Add an item to the list."""
        row = 0  # Insert at top
        self.table.insertRow(row)

        # Item name
        name = item_data.get("name", "Unknown")
        self.table.setItem(row, 0, QTableWidgetItem(name))

        # Rarity with color
        rarity = item_data.get("rarity", "normal")
        rarity_item = QTableWidgetItem(rarity.title())
        rarity_colors = {
            "unique": "#c7b377",  # Gold
            "set": "#00ff00",     # Green
            "rare": "#ffff00",    # Yellow
            "magic": "#6666ff",   # Blue
            "rune": "#ffa500",    # Orange
        }
        if rarity.lower() in rarity_colors:
            rarity_item.setForeground(QColor(rarity_colors[rarity.lower()]))
        self.table.setItem(row, 1, rarity_item)

        # Area
        area = item_data.get("area", "Unknown")
        self.table.setItem(row, 2, QTableWidgetItem(area))

        # Action (picked up / skipped)
        picked = item_data.get("picked_up", True)
        action = "Picked" if picked else "Skipped"
        action_item = QTableWidgetItem(action)
        if picked:
            action_item.setForeground(QColor("#33ff33"))
        else:
            action_item.setForeground(QColor("#ff6666"))
        self.table.setItem(row, 3, action_item)

        # Limit table size
        while self.table.rowCount() > 100:
            self.table.removeRow(self.table.rowCount() - 1)


class StatsPanel(QWidget):
    """Enhanced statistics panel."""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        # Session stats
        session_group = QGroupBox("Session")
        session_layout = QGridLayout(session_group)

        self.runtime_label = QLabel("0:00:00")
        self.runtime_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        session_layout.addWidget(QLabel("Runtime:"), 0, 0)
        session_layout.addWidget(self.runtime_label, 0, 1)

        self.runs_label = QLabel("0")
        self.success_label = QLabel("0%")
        session_layout.addWidget(QLabel("Runs:"), 1, 0)
        session_layout.addWidget(self.runs_label, 1, 1)
        session_layout.addWidget(QLabel("Success:"), 2, 0)
        session_layout.addWidget(self.success_label, 2, 1)

        self.rph_label = QLabel("0.0")
        session_layout.addWidget(QLabel("Runs/Hour:"), 3, 0)
        session_layout.addWidget(self.rph_label, 3, 1)

        layout.addWidget(session_group)

        # Item stats
        items_group = QGroupBox("Items Found")
        items_layout = QGridLayout(items_group)

        self.unique_label = QLabel("0")
        self.unique_label.setStyleSheet("color: #c7b377; font-weight: bold;")
        self.set_label = QLabel("0")
        self.set_label.setStyleSheet("color: #00ff00; font-weight: bold;")
        self.rare_label = QLabel("0")
        self.rare_label.setStyleSheet("color: #ffff00;")
        self.rune_label = QLabel("0")
        self.rune_label.setStyleSheet("color: #ffa500; font-weight: bold;")

        items_layout.addWidget(QLabel("Uniques:"), 0, 0)
        items_layout.addWidget(self.unique_label, 0, 1)
        items_layout.addWidget(QLabel("Sets:"), 1, 0)
        items_layout.addWidget(self.set_label, 1, 1)
        items_layout.addWidget(QLabel("Rares:"), 2, 0)
        items_layout.addWidget(self.rare_label, 2, 1)
        items_layout.addWidget(QLabel("Runes:"), 3, 0)
        items_layout.addWidget(self.rune_label, 3, 1)

        layout.addWidget(items_group)

        # Deaths
        deaths_group = QGroupBox("Deaths")
        deaths_layout = QGridLayout(deaths_group)

        self.deaths_label = QLabel("0")
        self.deaths_label.setFont(QFont("Arial", 14))
        self.chickens_label = QLabel("0")

        deaths_layout.addWidget(QLabel("Total Deaths:"), 0, 0)
        deaths_layout.addWidget(self.deaths_label, 0, 1)
        deaths_layout.addWidget(QLabel("Chickens:"), 1, 0)
        deaths_layout.addWidget(self.chickens_label, 1, 1)

        layout.addWidget(deaths_group)
        layout.addStretch()

    def update_stats(self, stats: dict):
        """Update all statistics."""
        # Runtime
        runtime = stats.get("runtime_seconds", 0)
        hours = int(runtime) // 3600
        minutes = (int(runtime) % 3600) // 60
        seconds = int(runtime) % 60
        self.runtime_label.setText(f"{hours}:{minutes:02d}:{seconds:02d}")

        # Runs
        runs = stats.get("runs_completed", 0)
        success = stats.get("successful_runs", 0)
        self.runs_label.setText(str(runs))
        if runs > 0:
            rate = success / runs * 100
            self.success_label.setText(f"{rate:.0f}%")
        self.rph_label.setText(f"{stats.get('runs_per_hour', 0):.1f}")

        # Items
        items = stats.get("items_by_rarity", {})
        self.unique_label.setText(str(items.get("unique", 0)))
        self.set_label.setText(str(items.get("set", 0)))
        self.rare_label.setText(str(items.get("rare", 0)))
        self.rune_label.setText(str(items.get("rune", 0)))

        # Deaths
        self.deaths_label.setText(str(stats.get("total_deaths", 0)))
        self.chickens_label.setText(str(stats.get("chickens", 0)))


class MainWindow(QMainWindow):
    """Main GUI window for D2CV bot."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("D2CV - Diablo 2 Bot")
        self.setMinimumSize(1100, 800)

        # Bot state
        self.is_running = False
        self.worker: Optional[BotWorker] = None
        self.runtime_seconds = 0

        self._setup_ui()
        self._setup_timers()
        self._load_current_build()

    def _setup_ui(self):
        """Create the UI layout."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Left panel - controls and vitals
        left_panel = QVBoxLayout()

        # Status
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)
        self.status_label = QLabel("Idle")
        self.status_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(self.status_label)

        self.current_run_label = QLabel("")
        self.current_run_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(self.current_run_label)

        left_panel.addWidget(status_group)

        # Build selector
        build_group = QGroupBox("Character Build")
        build_layout = QVBoxLayout(build_group)

        self.build_combo = QComboBox()
        self.build_combo.addItem("Blizzard Sorceress", "blizzard")
        self.build_combo.addItem("Elemental Druid", "elemental_druid")
        self.build_combo.currentIndexChanged.connect(self._on_build_changed)
        build_layout.addWidget(self.build_combo)

        # Show current build info
        self.build_info_label = QLabel("Cold damage - skips cold immunes")
        self.build_info_label.setStyleSheet("color: #888; font-size: 10px;")
        build_layout.addWidget(self.build_info_label)

        left_panel.addWidget(build_group)

        # Controls
        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout(controls_group)

        self.start_btn = QPushButton("Start Bot")
        self.start_btn.setMinimumHeight(50)
        self.start_btn.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.start_btn.clicked.connect(self._toggle_bot)
        controls_layout.addWidget(self.start_btn)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setEnabled(False)
        controls_layout.addWidget(self.pause_btn)

        left_panel.addWidget(controls_group)

        # Vitals
        vitals_group = QGroupBox("Vitals")
        vitals_layout = QVBoxLayout(vitals_group)

        # Health bar
        hp_layout = QHBoxLayout()
        hp_layout.addWidget(QLabel("HP:"))
        self.hp_bar = QProgressBar()
        self.hp_bar.setRange(0, 100)
        self.hp_bar.setValue(100)
        self.hp_bar.setFormat("%p%")
        self.hp_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555;
                border-radius: 3px;
                text-align: center;
                background-color: #1a1a1a;
            }
            QProgressBar::chunk {
                background-color: #cc3333;
            }
        """)
        hp_layout.addWidget(self.hp_bar)
        vitals_layout.addLayout(hp_layout)

        # Mana bar
        mp_layout = QHBoxLayout()
        mp_layout.addWidget(QLabel("MP:"))
        self.mp_bar = QProgressBar()
        self.mp_bar.setRange(0, 100)
        self.mp_bar.setValue(100)
        self.mp_bar.setFormat("%p%")
        self.mp_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555;
                border-radius: 3px;
                text-align: center;
                background-color: #1a1a1a;
            }
            QProgressBar::chunk {
                background-color: #3366cc;
            }
        """)
        mp_layout.addWidget(self.mp_bar)
        vitals_layout.addLayout(mp_layout)

        left_panel.addWidget(vitals_group)

        # Stats panel
        self.stats_panel = StatsPanel()
        left_panel.addWidget(self.stats_panel)

        main_layout.addLayout(left_panel, 1)

        # Right panel - preview and tabs
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

        # Tabs for log, runs, items
        tabs = QTabWidget()

        # Log tab
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.log_text)
        tabs.addTab(log_widget, "Log")

        # Run history tab
        self.run_history = RunHistoryWidget()
        tabs.addTab(self.run_history, "Run History")

        # Items found tab
        self.items_found = ItemsFoundWidget()
        tabs.addTab(self.items_found, "Items Found")

        right_panel.addWidget(tabs)

        main_layout.addLayout(right_panel, 2)

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
        self.runtime_seconds = 0
        self.start_btn.setText("Stop Bot")
        self.start_btn.setStyleSheet("background-color: #663333;")
        self.pause_btn.setEnabled(True)
        self.status_label.setText("Running")
        self.status_label.setStyleSheet("color: #00ff00;")
        self.runtime_timer.start(1000)
        self.log("Bot started")

    def _stop_bot(self):
        """Stop the bot."""
        self.is_running = False
        self.start_btn.setText("Start Bot")
        self.start_btn.setStyleSheet("")
        self.pause_btn.setEnabled(False)
        self.status_label.setText("Stopped")
        self.status_label.setStyleSheet("color: #ff6666;")
        self.current_run_label.setText("")
        self.runtime_timer.stop()

        if self.worker:
            self.worker.stop()
            self.worker.wait()
            self.worker = None

        self.log("Bot stopped")

    def _update_runtime(self):
        """Update runtime display."""
        self.runtime_seconds += 1
        self.stats_panel.update_stats({"runtime_seconds": self.runtime_seconds})

    def _on_build_changed(self, index: int):
        """Handle build selection change."""
        build_id = self.build_combo.currentData()

        # Update info label
        build_info = {
            "blizzard": "Cold damage - skips cold immunes",
            "elemental_druid": "Fire damage - skips fire immunes",
        }
        self.build_info_label.setText(build_info.get(build_id, ""))

        # Update config
        self._save_build_to_config(build_id)
        self.log(f"Build changed to: {self.build_combo.currentText()}")

    def _save_build_to_config(self, build_id: str):
        """Save build selection to config file."""
        import yaml
        from pathlib import Path

        config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"

        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            # Update build settings
            if build_id == "elemental_druid":
                config["character"]["class"] = "druid"
                config["character"]["build"] = "elemental_druid"
            else:
                config["character"]["class"] = "sorceress"
                config["character"]["build"] = "blizzard"

            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        except Exception as e:
            self.log(f"Failed to save config: {e}")

    def _load_current_build(self):
        """Load current build from config and set combo box."""
        try:
            from src.config import get_config
            config = get_config()
            current_build = config.get("character.build", "blizzard")

            # Find and select matching item
            for i in range(self.build_combo.count()):
                if self.build_combo.itemData(i) == current_build:
                    self.build_combo.setCurrentIndex(i)
                    break
        except Exception:
            pass  # Use default

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

    def update_status(self, status: str, current_run: str = None):
        """Update the status label."""
        self.status_label.setText(status)
        if current_run:
            self.current_run_label.setText(f"Current: {current_run}")
        else:
            self.current_run_label.setText("")

    def update_vitals(self, hp_percent: float, mp_percent: float):
        """Update HP/MP display."""
        self.hp_bar.setValue(int(hp_percent * 100))
        self.mp_bar.setValue(int(mp_percent * 100))

        # Color health bar based on level
        if hp_percent < 0.3:
            self.hp_bar.setStyleSheet("""
                QProgressBar { border: 1px solid #555; border-radius: 3px; text-align: center; background-color: #1a1a1a; }
                QProgressBar::chunk { background-color: #ff3333; }
            """)
        elif hp_percent < 0.6:
            self.hp_bar.setStyleSheet("""
                QProgressBar { border: 1px solid #555; border-radius: 3px; text-align: center; background-color: #1a1a1a; }
                QProgressBar::chunk { background-color: #cc6633; }
            """)
        else:
            self.hp_bar.setStyleSheet("""
                QProgressBar { border: 1px solid #555; border-radius: 3px; text-align: center; background-color: #1a1a1a; }
                QProgressBar::chunk { background-color: #cc3333; }
            """)

    def update_stats(self, stats: dict):
        """Update all statistics."""
        self.stats_panel.update_stats(stats)

    def add_run_result(self, run_data: dict):
        """Add a completed run to history."""
        self.run_history.add_run(run_data)

    def add_item_found(self, item_data: dict):
        """Add a found item to the list."""
        self.items_found.add_item(item_data)

    def log(self, message: str):
        """Add a message to the log."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
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
            font-weight: bold;
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
        QPushButton:disabled {
            background-color: #2a2a2a;
            color: #666;
        }
        QTextEdit {
            background-color: #1a1a1a;
            border: 1px solid #333;
        }
        QLabel {
            color: #cccccc;
        }
        QTabWidget::pane {
            border: 1px solid #444;
            background-color: #2b2b2b;
        }
        QTabBar::tab {
            background-color: #333;
            border: 1px solid #444;
            padding: 8px 16px;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background-color: #444;
            border-bottom-color: #2b2b2b;
        }
        QTableWidget {
            background-color: #1a1a1a;
            border: 1px solid #333;
            gridline-color: #333;
        }
        QTableWidget::item {
            padding: 4px;
        }
        QTableWidget::item:alternate {
            background-color: #252525;
        }
        QHeaderView::section {
            background-color: #333;
            border: 1px solid #444;
            padding: 4px;
        }
        QProgressBar {
            border: 1px solid #555;
            border-radius: 3px;
            text-align: center;
        }
        QComboBox {
            background-color: #3c3c3c;
            border: 1px solid #555;
            padding: 6px;
            border-radius: 4px;
            min-width: 120px;
        }
        QComboBox:hover {
            background-color: #4a4a4a;
        }
        QComboBox::drop-down {
            border: none;
            width: 20px;
        }
        QComboBox QAbstractItemView {
            background-color: #3c3c3c;
            border: 1px solid #555;
            selection-background-color: #555;
        }
    """)

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    run_gui()
