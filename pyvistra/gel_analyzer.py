"""
Gel Image Analyzer Widget

Provides molecular weight estimation from gel electrophoresis images.
Uses rectangle ROIs to define lanes, with one lane designated as the
molecular weight ladder/standards.
"""

import json
import os
import numpy as np
from scipy.signal import find_peaks
from vispy import scene

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QSpinBox, QTextEdit, QGroupBox, QFileDialog,
    QSizePolicy
)
from qtpy.QtCore import Qt

from .manager import manager
from .rois import RectangleROI


def get_builtin_ladders_path():
    """Return path to the built-in ladders.json file."""
    return os.path.join(os.path.dirname(__file__), "data", "ladders.json")


def load_ladders(path=None):
    """Load ladder presets from JSON file."""
    if path is None:
        path = get_builtin_ladders_path()
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data.get("ladders", [])
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading ladders: {e}")
        return []


class PeakMarkers:
    """Ephemeral visual markers for detected peaks within ROI bounds."""

    def __init__(self, view):
        self.view = view
        self.line_visuals = []

    def update(self, roi_peaks_data):
        """
        Update peak markers.

        Args:
            roi_peaks_data: List of dicts with keys:
                - 'roi': RectangleROI
                - 'peaks': array of y-positions (in ROI-local coords)
                - 'color': color for markers
                - 'is_ladder': bool
        """
        self.clear()

        for data in roi_peaks_data:
            roi = data['roi']
            peaks = data['peaks']
            color = data['color']

            # Get ROI bounds
            p1 = roi.data.get('p1', (0, 0))
            p2 = roi.data.get('p2', (0, 0))
            x_min = min(p1[0], p2[0])
            x_max = max(p1[0], p2[0])
            y_min = min(p1[1], p2[1])

            # Create horizontal lines for each peak
            for peak_y_local in peaks:
                # Convert local y to global y
                y_global = y_min + peak_y_local

                # Create line visual
                line_pos = np.array([
                    [x_min, y_global, 0],
                    [x_max, y_global, 0]
                ], dtype=np.float32)

                line_visual = scene.visuals.Line(
                    pos=line_pos,
                    color=color,
                    width=2,
                    parent=self.view.scene
                )
                self.line_visuals.append(line_visual)

    def clear(self):
        """Remove all peak marker visuals."""
        for visual in self.line_visuals:
            visual.parent = None
        self.line_visuals = []


class GelAnalyzerWidget(QWidget):
    """
    Widget for analyzing gel electrophoresis images.

    Detects bands in lanes (Rectangle ROIs), uses one lane as a
    molecular weight ladder, and estimates MW of bands in other lanes.
    """

    def __init__(self, roi_manager):
        super().__init__()
        self.roi_manager = roi_manager
        self.ladders = load_ladders()
        self.peak_markers = None  # Created per-window

        self.setWindowTitle("Gel Analyzer")
        self.resize(320, 450)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Ladder Preset Section
        ladder_group = QGroupBox("Ladder Preset")
        ladder_layout = QVBoxLayout(ladder_group)

        self.ladder_combo = QComboBox()
        self.ladder_combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        for ladder in self.ladders:
            self.ladder_combo.addItem(ladder['name'], userData=ladder)
        ladder_layout.addWidget(self.ladder_combo)

        btn_load_custom = QPushButton("Load Custom Ladder...")
        btn_load_custom.clicked.connect(self._load_custom_ladder)
        ladder_layout.addWidget(btn_load_custom)

        layout.addWidget(ladder_group)

        # Lane Selection Section
        lane_group = QGroupBox("Standard Lane")
        lane_layout = QHBoxLayout(lane_group)

        lane_layout.addWidget(QLabel("Ladder ROI:"))
        self.lane_combo = QComboBox()
        self.lane_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        lane_layout.addWidget(self.lane_combo)

        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self._refresh_lanes)
        lane_layout.addWidget(btn_refresh)

        layout.addWidget(lane_group)

        # Peak Detection Section
        detect_group = QGroupBox("Peak Detection")
        detect_layout = QVBoxLayout(detect_group)

        prom_layout = QHBoxLayout()
        prom_layout.addWidget(QLabel("Prominence:"))
        self.prominence_spin = QSpinBox()
        self.prominence_spin.setRange(100, 50000)
        self.prominence_spin.setValue(1000)
        self.prominence_spin.setSingleStep(100)
        prom_layout.addWidget(self.prominence_spin)
        detect_layout.addLayout(prom_layout)

        dist_layout = QHBoxLayout()
        dist_layout.addWidget(QLabel("Min Distance (px):"))
        self.distance_spin = QSpinBox()
        self.distance_spin.setRange(1, 100)
        self.distance_spin.setValue(5)
        dist_layout.addWidget(self.distance_spin)
        detect_layout.addLayout(dist_layout)

        btn_detect = QPushButton("Detect Peaks")
        btn_detect.clicked.connect(self._detect_peaks)
        detect_layout.addWidget(btn_detect)

        btn_clear = QPushButton("Clear Markers")
        btn_clear.clicked.connect(self._clear_markers)
        detect_layout.addWidget(btn_clear)

        layout.addWidget(detect_group)

        # Results Section
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(120)
        results_layout.addWidget(self.results_text)

        btn_copy = QPushButton("Copy to Clipboard")
        btn_copy.clicked.connect(self._copy_results)
        results_layout.addWidget(btn_copy)

        btn_export = QPushButton("Export CSV...")
        btn_export.clicked.connect(self._export_csv)
        results_layout.addWidget(btn_export)

        layout.addWidget(results_group)

        layout.addStretch()

    def showEvent(self, event):
        """Refresh lane list when shown."""
        super().showEvent(event)
        self._refresh_lanes()

    def closeEvent(self, event):
        """Clean up peak markers when closed."""
        self._clear_markers()
        super().closeEvent(event)

    def _load_custom_ladder(self):
        """Load a custom ladder preset from JSON file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Ladder Preset", "", "JSON Files (*.json)"
        )
        if not path:
            return

        try:
            with open(path, "r") as f:
                data = json.load(f)

            # Support both single ladder and list format
            if "ladders" in data:
                new_ladders = data["ladders"]
            elif "name" in data and "weights_kda" in data:
                new_ladders = [data]
            else:
                print("Invalid ladder format")
                return

            for ladder in new_ladders:
                self.ladders.append(ladder)
                self.ladder_combo.addItem(ladder['name'], userData=ladder)

            # Select the first newly added ladder
            self.ladder_combo.setCurrentIndex(self.ladder_combo.count() - 1)
            print(f"Loaded {len(new_ladders)} ladder(s) from {path}")

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error loading ladder file: {e}")

    def _refresh_lanes(self):
        """Refresh the lane combo box with current Rectangle ROIs."""
        self.lane_combo.clear()

        window = self.roi_manager.active_window
        if not window:
            return

        for i, roi in enumerate(window.rois):
            if isinstance(roi, RectangleROI):
                self.lane_combo.addItem(f"{i}: {roi.name}", userData=i)

    def _get_current_image(self):
        """Get the current 2D grayscale image from the active window."""
        window = self.roi_manager.active_window
        if not window:
            return None

        cache = window.renderer.current_slice_cache
        if cache is None:
            return None

        # Convert to grayscale if multi-channel
        if cache.ndim == 3:
            # Mean across channels
            gray = cache.mean(axis=0)
        else:
            gray = cache

        return gray

    def _detect_peaks(self):
        """Detect peaks in all lanes and update markers."""
        window = self.roi_manager.active_window
        if not window:
            self.results_text.setText("No active window")
            return

        # Get current ladder selection
        ladder_data = self.ladder_combo.currentData()
        if ladder_data is None:
            self.results_text.setText("No ladder preset selected")
            return

        # Get standard lane index
        std_idx = self.lane_combo.currentData()
        if std_idx is None:
            self.results_text.setText("No standard lane selected")
            return

        # Get image
        gray = self._get_current_image()
        if gray is None:
            self.results_text.setText("No image data available")
            return

        # Invert intensity (assuming dark bands on light background)
        max_val = gray.max()
        gray_inv = max_val - gray

        # Get detection parameters
        prominence = self.prominence_spin.value()
        min_distance = self.distance_spin.value()

        # Get all rectangle ROIs
        rect_rois = []
        for i, roi in enumerate(window.rois):
            if isinstance(roi, RectangleROI):
                rect_rois.append((i, roi))

        if len(rect_rois) < 1:
            self.results_text.setText("No rectangle ROIs found")
            return

        # Detect peaks in each lane
        profiles = []
        peaks_list = []
        roi_peaks_data = []

        for i, roi in rect_rois:
            region = roi.get_region(gray_inv)
            if region.size == 0:
                profiles.append(np.array([]))
                peaks_list.append(np.array([]))
                continue

            # Get 1D profile (mean across lane width)
            profile = region.mean(axis=1)
            profiles.append(profile)

            # Find peaks
            peaks, _ = find_peaks(
                profile,
                prominence=prominence,
                distance=min_distance
            )
            peaks_list.append(peaks)

            # Prepare marker data
            is_ladder = (i == std_idx)
            color = 'cyan' if is_ladder else 'orange'
            roi_peaks_data.append({
                'roi': roi,
                'peaks': peaks,
                'color': color,
                'is_ladder': is_ladder
            })

        # Update peak markers
        if self.peak_markers is None:
            self.peak_markers = PeakMarkers(window.view)
        self.peak_markers.update(roi_peaks_data)
        window.canvas.update()

        # Calculate molecular weights
        self._calculate_mw(
            rect_rois, peaks_list, std_idx,
            ladder_data['weights_kda']
        )

    def _calculate_mw(self, rect_rois, peaks_list, std_idx, ladder_weights):
        """Calculate molecular weights using log-linear interpolation."""
        results = []

        # Find standard lane peaks
        std_peaks = None
        for i, (idx, roi) in enumerate(rect_rois):
            if idx == std_idx:
                std_peaks = peaks_list[i]
                break

        if std_peaks is None or len(std_peaks) == 0:
            self.results_text.setText("No peaks detected in standard lane")
            return

        # Use as many ladder weights as we have peaks
        n_peaks = min(len(std_peaks), len(ladder_weights))
        if n_peaks < 2:
            self.results_text.setText(
                f"Need at least 2 peaks in standard lane "
                f"(found {len(std_peaks)})"
            )
            return

        # Calibration data (in log scale)
        std_positions = std_peaks[:n_peaks]
        log_weights = np.log(ladder_weights[:n_peaks])

        results.append(f"Standard Lane (ROI {std_idx}):")
        results.append(f"  Detected {len(std_peaks)} peaks")
        results.append(f"  Using {n_peaks} ladder bands for calibration")
        results.append("")

        # Calculate MW for each non-standard lane
        for i, (idx, roi) in enumerate(rect_rois):
            if idx == std_idx:
                continue

            peaks = peaks_list[i]
            if len(peaks) == 0:
                results.append(f"Lane {idx} ({roi.name}): No peaks detected")
                continue

            # Interpolate in log space
            log_mw = np.interp(peaks, std_positions, log_weights)
            mw = np.exp(log_mw)

            results.append(f"Lane {idx} ({roi.name}):")
            mw_strs = [f"{w:.1f}" for w in mw]
            results.append(f"  MW (kDa): {', '.join(mw_strs)}")

        self.results_text.setText("\n".join(results))

    def _clear_markers(self):
        """Clear all peak markers from the canvas."""
        if self.peak_markers:
            self.peak_markers.clear()
            window = self.roi_manager.active_window
            if window:
                window.canvas.update()

    def _copy_results(self):
        """Copy results to clipboard."""
        from qtpy.QtWidgets import QApplication
        text = self.results_text.toPlainText()
        QApplication.clipboard().setText(text)

    def _export_csv(self):
        """Export results as CSV file."""
        text = self.results_text.toPlainText()
        if not text:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "gel_analysis.csv", "CSV Files (*.csv)"
        )
        if not path:
            return

        # Parse results and write CSV
        lines = text.strip().split("\n")
        with open(path, "w") as f:
            f.write("Lane,ROI_Name,MW_kDa\n")
            current_lane = ""
            current_name = ""
            for line in lines:
                line = line.strip()
                if line.startswith("Lane"):
                    # Parse "Lane X (name):"
                    parts = line.rstrip(":").split(" ", 2)
                    if len(parts) >= 2:
                        current_lane = parts[1]
                        if len(parts) > 2 and "(" in parts[2]:
                            current_name = parts[2].strip("():")
                elif line.startswith("MW"):
                    # Parse "MW (kDa): x, y, z"
                    mw_part = line.split(": ", 1)
                    if len(mw_part) == 2:
                        mw_values = mw_part[1].split(", ")
                        for mw in mw_values:
                            f.write(f"{current_lane},{current_name},{mw}\n")

        print(f"Results exported to {path}")


# Singleton instance
_gel_analyzer_instance = None


def get_gel_analyzer(roi_manager):
    """Get or create the GelAnalyzer singleton."""
    global _gel_analyzer_instance
    if _gel_analyzer_instance is None:
        _gel_analyzer_instance = GelAnalyzerWidget(roi_manager)
    return _gel_analyzer_instance


def show_gel_analyzer(roi_manager):
    """Show the Gel Analyzer widget."""
    analyzer = get_gel_analyzer(roi_manager)
    analyzer.show()
    analyzer.raise_()
    analyzer.activateWindow()
