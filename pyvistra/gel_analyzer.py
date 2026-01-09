"""
Gel Image Analyzer Widget

Provides molecular weight/size estimation from gel electrophoresis images.
Works with both protein gels (kDa) and DNA gels (bp).
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
    QSizePolicy, QCheckBox
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


class PeakMarker:
    """A single adjustable peak marker with line and label."""

    def __init__(self, view, roi, y_local, label_text, color, is_ladder=False):
        self.view = view
        self.roi = roi
        self.y_local = y_local  # Position relative to ROI top
        self.label_text = label_text
        self.color = color
        self.is_ladder = is_ladder
        self.selected = False

        # Get ROI bounds
        p1 = roi.data.get('p1', (0, 0))
        p2 = roi.data.get('p2', (0, 0))
        self.x_min = min(p1[0], p2[0])
        self.x_max = max(p1[0], p2[0])
        self.y_roi_min = min(p1[1], p2[1])
        self.y_roi_max = max(p1[1], p2[1])

        # Create line visual
        self.line_visual = scene.visuals.Line(
            pos=self._get_line_pos(),
            color=color,
            width=2,
            parent=self.view.scene
        )

        # Create label visual (small, non-intrusive)
        self.label_visual = scene.visuals.Text(
            text=label_text,
            color=color,
            font_size=8,
            anchor_x='left',
            anchor_y='center',
            parent=self.view.scene
        )
        self._update_label_pos()

    def _get_line_pos(self):
        """Get line position array."""
        y_global = self.y_roi_min + self.y_local
        return np.array([
            [self.x_min, y_global, 0],
            [self.x_max, y_global, 0]
        ], dtype=np.float32)

    def _update_label_pos(self):
        """Update label position to be at right edge of line."""
        y_global = self.y_roi_min + self.y_local
        self.label_visual.pos = (self.x_max + 3, y_global, 0)

    def get_global_y(self):
        """Get global Y coordinate of this marker."""
        return self.y_roi_min + self.y_local

    def set_y_local(self, y_local):
        """Set local Y position and update visuals."""
        # Clamp to ROI bounds
        roi_height = self.y_roi_max - self.y_roi_min
        self.y_local = max(0, min(y_local, roi_height))
        self.line_visual.set_data(pos=self._get_line_pos())
        self._update_label_pos()

    def set_y_global(self, y_global):
        """Set global Y position."""
        self.set_y_local(y_global - self.y_roi_min)

    def hit_test(self, x, y, tolerance=5):
        """Test if point (x, y) is near this marker line."""
        y_global = self.get_global_y()
        if self.x_min <= x <= self.x_max:
            if abs(y - y_global) <= tolerance:
                return True
        return False

    def set_selected(self, selected):
        """Set selection state (changes appearance)."""
        self.selected = selected
        if selected:
            self.line_visual.set_data(color='yellow', width=3)
        else:
            self.line_visual.set_data(color=self.color, width=2)

    def remove(self):
        """Remove visuals from scene."""
        self.line_visual.parent = None
        self.label_visual.parent = None


class PeakMarkers:
    """Manager for ephemeral peak marker visuals with drag support."""

    def __init__(self, view, on_peaks_changed=None):
        self.view = view
        self.markers = []  # List of PeakMarker objects
        self.dragging_marker = None
        self.on_peaks_changed = on_peaks_changed  # Callback when peaks change

    def update(self, roi_peaks_data, ladder_sizes=None, unit="kDa"):
        """
        Update peak markers.

        Args:
            roi_peaks_data: List of dicts with keys:
                - 'roi': RectangleROI
                - 'peaks': array of y-positions (in ROI-local coords)
                - 'color': color for markers
                - 'is_ladder': bool
                - 'roi_idx': int
            ladder_sizes: List of sizes for ladder labels
            unit: Unit string (kDa, bp, etc.)
        """
        self.clear()

        for data in roi_peaks_data:
            roi = data['roi']
            peaks = data['peaks']
            color = data['color']
            is_ladder = data.get('is_ladder', False)

            for i, peak_y_local in enumerate(peaks):
                # Create label text
                if is_ladder and ladder_sizes is not None and i < len(ladder_sizes):
                    label = f"{ladder_sizes[i]} {unit}"
                else:
                    label = ""  # Sample peaks get labels after MW calculation

                marker = PeakMarker(
                    self.view, roi, peak_y_local, label, color, is_ladder
                )
                self.markers.append(marker)

    def update_sample_labels(self, sample_mw_data, unit="kDa"):
        """
        Update labels for sample lane markers after MW calculation.

        Args:
            sample_mw_data: List of (roi, mw_values) tuples
        """
        # Build a lookup by ROI
        roi_mw_map = {id(roi): mws for roi, mws in sample_mw_data}

        mw_idx = {}  # Track index per ROI
        for marker in self.markers:
            if not marker.is_ladder:
                roi_id = id(marker.roi)
                if roi_id in roi_mw_map:
                    idx = mw_idx.get(roi_id, 0)
                    mws = roi_mw_map[roi_id]
                    if idx < len(mws):
                        marker.label_text = f"~{mws[idx]:.1f} {unit}"
                        marker.label_visual.text = marker.label_text
                    mw_idx[roi_id] = idx + 1

    def get_peaks_by_roi(self):
        """Get current peak positions grouped by ROI.

        Returns:
            dict: {roi: [y_local_positions]}
        """
        roi_peaks = {}
        for marker in self.markers:
            roi = marker.roi
            if roi not in roi_peaks:
                roi_peaks[roi] = []
            roi_peaks[roi].append(marker.y_local)
        return roi_peaks

    def hit_test(self, x, y):
        """Find marker at position (x, y)."""
        for marker in self.markers:
            if marker.hit_test(x, y):
                return marker
        return None

    def start_drag(self, marker):
        """Start dragging a marker."""
        if self.dragging_marker:
            self.dragging_marker.set_selected(False)
        self.dragging_marker = marker
        marker.set_selected(True)

    def drag_to(self, y):
        """Move dragging marker to y position."""
        if self.dragging_marker:
            self.dragging_marker.set_y_global(y)

    def end_drag(self):
        """End dragging and trigger callback."""
        if self.dragging_marker:
            self.dragging_marker.set_selected(False)
            self.dragging_marker = None
            if self.on_peaks_changed:
                self.on_peaks_changed()

    def clear(self):
        """Remove all peak marker visuals."""
        for marker in self.markers:
            marker.remove()
        self.markers = []
        self.dragging_marker = None


class GelAnalyzerWidget(QWidget):
    """
    Widget for analyzing gel electrophoresis images.

    Detects bands in lanes (Rectangle ROIs), uses one lane as a
    molecular weight ladder, and estimates MW of bands in other lanes.
    Supports both protein (kDa) and DNA (bp) gels.
    """

    def __init__(self, roi_manager):
        super().__init__()
        self.roi_manager = roi_manager
        self.ladders = load_ladders()
        self.peak_markers = None
        self._mouse_connected = False
        self._cached_roi_data = []  # Store ROI data for recalculation

        self.setWindowTitle("Gel Analyzer")
        self.resize(320, 500)

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
            unit = ladder.get('unit', 'kDa')
            display_name = f"{ladder['name']} ({unit})"
            self.ladder_combo.addItem(display_name, userData=ladder)
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

        # Invert intensity checkbox
        self.invert_check = QCheckBox("Invert intensity (dark bands)")
        self.invert_check.setChecked(True)
        self.invert_check.setToolTip(
            "Check for Coomassie/silver stains (dark bands on light background)\n"
            "Uncheck for fluorescent stains (bright bands on dark background)"
        )
        detect_layout.addWidget(self.invert_check)

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

        # Display Options
        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout(display_group)

        self.hide_rois_check = QCheckBox("Hide lane ROIs")
        self.hide_rois_check.stateChanged.connect(self._toggle_roi_visibility)
        display_layout.addWidget(self.hide_rois_check)

        tip_label = QLabel("Tip: Drag peak lines to adjust positions")
        tip_label.setStyleSheet("color: gray; font-size: 10px;")
        display_layout.addWidget(tip_label)

        layout.addWidget(display_group)

        # Results Section
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(100)
        results_layout.addWidget(self.results_text)

        btn_copy = QPushButton("Copy to Clipboard")
        btn_copy.clicked.connect(self._copy_results)
        results_layout.addWidget(btn_copy)

        btn_export = QPushButton("Export CSV...")
        btn_export.clicked.connect(self._export_csv)
        results_layout.addWidget(btn_export)

        layout.addWidget(results_group)

    def showEvent(self, event):
        """Refresh lane list when shown."""
        super().showEvent(event)
        self._refresh_lanes()
        self._connect_mouse_events()

    def closeEvent(self, event):
        """Clean up peak markers when closed."""
        self._clear_markers()
        self._disconnect_mouse_events()
        # Restore ROI visibility
        if self.hide_rois_check.isChecked():
            self.hide_rois_check.setChecked(False)
        super().closeEvent(event)

    def _connect_mouse_events(self):
        """Connect to canvas mouse events for marker dragging."""
        window = self.roi_manager.active_window
        if window and not self._mouse_connected:
            window.canvas.events.mouse_press.connect(self._on_mouse_press)
            window.canvas.events.mouse_move.connect(self._on_mouse_move)
            window.canvas.events.mouse_release.connect(self._on_mouse_release)
            self._mouse_connected = True

    def _disconnect_mouse_events(self):
        """Disconnect from canvas mouse events."""
        window = self.roi_manager.active_window
        if window and self._mouse_connected:
            try:
                window.canvas.events.mouse_press.disconnect(self._on_mouse_press)
                window.canvas.events.mouse_move.disconnect(self._on_mouse_move)
                window.canvas.events.mouse_release.disconnect(self._on_mouse_release)
            except (TypeError, RuntimeError):
                pass
            self._mouse_connected = False

    def _map_event_to_image(self, event):
        """Convert mouse event to image coordinates."""
        window = self.roi_manager.active_window
        if not window or not window.renderer.layers:
            return None, None
        tr = window.canvas.scene.node_transform(window.renderer.layers[0])
        pos = tr.map(event.pos)
        return pos[0], pos[1]

    def _on_mouse_press(self, event):
        """Handle mouse press for marker selection."""
        if not self.peak_markers or event.button != 1:
            return

        x, y = self._map_event_to_image(event)
        if x is None:
            return

        marker = self.peak_markers.hit_test(x, y)
        if marker:
            self.peak_markers.start_drag(marker)
            window = self.roi_manager.active_window
            if window:
                window.canvas.update()

    def _on_mouse_move(self, event):
        """Handle mouse move for marker dragging."""
        if not self.peak_markers or not self.peak_markers.dragging_marker:
            return

        x, y = self._map_event_to_image(event)
        if y is not None:
            self.peak_markers.drag_to(y)
            window = self.roi_manager.active_window
            if window:
                window.canvas.update()

    def _on_mouse_release(self, event):
        """Handle mouse release to end dragging."""
        if not self.peak_markers:
            return

        if self.peak_markers.dragging_marker:
            self.peak_markers.end_drag()
            window = self.roi_manager.active_window
            if window:
                window.canvas.update()

    def _toggle_roi_visibility(self, state):
        """Toggle visibility of lane ROIs."""
        window = self.roi_manager.active_window
        if not window:
            return

        visible = state != Qt.Checked
        for roi in window.rois:
            if isinstance(roi, RectangleROI):
                roi.set_visible(visible)
        window.canvas.update()

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
            elif "name" in data and ("sizes" in data or "weights_kda" in data):
                new_ladders = [data]
            else:
                print("Invalid ladder format")
                return

            for ladder in new_ladders:
                # Normalize old format
                if "weights_kda" in ladder and "sizes" not in ladder:
                    ladder["sizes"] = ladder["weights_kda"]
                    ladder["unit"] = "kDa"
                self.ladders.append(ladder)
                unit = ladder.get('unit', 'kDa')
                display_name = f"{ladder['name']} ({unit})"
                self.ladder_combo.addItem(display_name, userData=ladder)

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

        # Reconnect mouse events if window changed
        if not self._mouse_connected:
            self._connect_mouse_events()

        # Get current ladder selection
        ladder_data = self.ladder_combo.currentData()
        if ladder_data is None:
            self.results_text.setText("No ladder preset selected")
            return

        # Get ladder info
        ladder_sizes = ladder_data.get('sizes', ladder_data.get('weights_kda', []))
        unit = ladder_data.get('unit', 'kDa')

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

        # Invert intensity if needed
        if self.invert_check.isChecked():
            max_val = gray.max()
            gray_proc = max_val - gray
        else:
            gray_proc = gray

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
        roi_peaks_data = []
        self._cached_roi_data = []

        for idx, roi in rect_rois:
            region = roi.get_region(gray_proc)
            if region.size == 0:
                continue

            # Get 1D profile (mean across lane width)
            profile = region.mean(axis=1)

            # Find peaks
            peaks, _ = find_peaks(
                profile,
                prominence=prominence,
                distance=min_distance
            )

            is_ladder = (idx == std_idx)
            color = 'cyan' if is_ladder else 'orange'

            roi_peaks_data.append({
                'roi': roi,
                'peaks': peaks,
                'color': color,
                'is_ladder': is_ladder,
                'roi_idx': idx
            })

            self._cached_roi_data.append({
                'roi': roi,
                'roi_idx': idx,
                'is_ladder': is_ladder
            })

        # Create/update peak markers
        if self.peak_markers is None:
            self.peak_markers = PeakMarkers(
                window.view,
                on_peaks_changed=self._on_peaks_adjusted
            )
        self.peak_markers.update(roi_peaks_data, ladder_sizes, unit)
        window.canvas.update()

        # Calculate and display results
        self._calculate_and_display(std_idx, ladder_sizes, unit)

    def _on_peaks_adjusted(self):
        """Called when peaks are manually adjusted."""
        ladder_data = self.ladder_combo.currentData()
        if not ladder_data:
            return

        ladder_sizes = ladder_data.get('sizes', ladder_data.get('weights_kda', []))
        unit = ladder_data.get('unit', 'kDa')
        std_idx = self.lane_combo.currentData()

        if std_idx is not None:
            self._calculate_and_display(std_idx, ladder_sizes, unit)

    def _calculate_and_display(self, std_idx, ladder_sizes, unit):
        """Calculate MW and update display."""
        if not self.peak_markers:
            return

        # Get current peak positions from markers
        roi_peaks = self.peak_markers.get_peaks_by_roi()

        # Find standard lane data
        std_roi = None
        std_peaks = None
        for data in self._cached_roi_data:
            if data['roi_idx'] == std_idx:
                std_roi = data['roi']
                std_peaks = sorted(roi_peaks.get(std_roi, []))
                break

        if std_peaks is None or len(std_peaks) == 0:
            self.results_text.setText("No peaks in standard lane")
            return

        # Use as many ladder sizes as we have peaks
        n_peaks = min(len(std_peaks), len(ladder_sizes))
        if n_peaks < 2:
            self.results_text.setText(
                f"Need at least 2 peaks in standard lane "
                f"(found {len(std_peaks)})"
            )
            return

        # Calibration data (in log scale)
        std_positions = np.array(std_peaks[:n_peaks])
        log_sizes = np.log(np.array(ladder_sizes[:n_peaks]))

        results = []
        results.append(f"Standard Lane (ROI {std_idx}):")
        results.append(f"  Detected {len(std_peaks)} peaks")
        results.append(f"  Using {n_peaks} bands for calibration")
        results.append("")

        # Calculate sizes for sample lanes and collect for label update
        sample_mw_data = []

        for data in self._cached_roi_data:
            if data['is_ladder']:
                continue

            roi = data['roi']
            roi_idx = data['roi_idx']
            peaks = sorted(roi_peaks.get(roi, []))

            if len(peaks) == 0:
                results.append(f"Lane {roi_idx} ({roi.name}): No peaks detected")
                continue

            # Interpolate in log space
            log_mw = np.interp(peaks, std_positions, log_sizes)
            mw = np.exp(log_mw)

            sample_mw_data.append((roi, mw))

            results.append(f"Lane {roi_idx} ({roi.name}):")
            mw_strs = [f"{w:.1f}" for w in mw]
            results.append(f"  Size ({unit}): {', '.join(mw_strs)}")

        # Update sample marker labels
        self.peak_markers.update_sample_labels(sample_mw_data, unit)

        self.results_text.setText("\n".join(results))

        window = self.roi_manager.active_window
        if window:
            window.canvas.update()

    def _clear_markers(self):
        """Clear all peak markers from the canvas."""
        if self.peak_markers:
            self.peak_markers.clear()
            self.peak_markers = None
            window = self.roi_manager.active_window
            if window:
                window.canvas.update()
        self._cached_roi_data = []

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

        # Get unit from current ladder
        ladder_data = self.ladder_combo.currentData()
        unit = ladder_data.get('unit', 'kDa') if ladder_data else 'kDa'

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "gel_analysis.csv", "CSV Files (*.csv)"
        )
        if not path:
            return

        # Parse results and write CSV
        lines = text.strip().split("\n")
        with open(path, "w") as f:
            f.write(f"Lane,ROI_Name,Size_{unit}\n")
            current_lane = ""
            current_name = ""
            for line in lines:
                line = line.strip()
                if line.startswith("Lane"):
                    parts = line.rstrip(":").split(" ", 2)
                    if len(parts) >= 2:
                        current_lane = parts[1]
                        if len(parts) > 2 and "(" in parts[2]:
                            current_name = parts[2].strip("():")
                elif line.startswith("Size"):
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
