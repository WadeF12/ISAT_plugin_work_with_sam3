# -*- coding: utf-8 -*-
"""
SAM3 Text-Prompt Batch Auto Annotation Plugin for ISAT

Utilizes ISAT's built-in SAM3 text-prompt API to batch-predict
and auto-save annotations for all images in the current folder.

Core API (from ISAT mainwindow.py):
    mainwindow.predict_image_with_text_prompt_to_annotation(image_name, categories)
"""

import os
from typing import List

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer

from ISAT.widgets.plugin_base import PluginBase


# ==================================================================
# Proportional Region Widget — draggable 4-corner region mapper
# ==================================================================

class ProportionalRegionWidget(QtWidgets.QWidget):
    """A widget displaying a proportional rectangle with 4 draggable corner handles.

    The rectangle aspect ratio matches the target image. Users drag the 4
    corner points to define a quadrilateral region. The proportional
    coordinates (0~1 range) are later mapped to actual image pixels.
    """

    HANDLE_RADIUS = 7
    DEFAULT_AR = 4.0 / 3.0

    def __init__(self, parent=None):
        super().__init__(parent)
        # 4 handles in proportional coords (0~1), clockwise from top-left
        self._handles = [
            QtCore.QPointF(0.0, 0.0),   # top-left
            QtCore.QPointF(1.0, 0.0),   # top-right
            QtCore.QPointF(1.0, 1.0),   # bottom-right
            QtCore.QPointF(0.0, 1.0),   # bottom-left
        ]
        self._active_handle = -1
        self._aspect_ratio = self.DEFAULT_AR
        self.setMinimumHeight(80)
        self.setMouseTracking(True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_aspect_ratio(self, w: int, h: int):
        """Update the target aspect ratio (w / h)."""
        if h > 0:
            self._aspect_ratio = float(w) / float(h)
        self.update()

    def get_proportional_points(self):
        """Return 4 handles as [(rx, ry), ...] in 0~1 proportional coords."""
        return [(p.x(), p.y()) for p in self._handles]

    def reset_to_corners(self):
        """Reset all 4 handles to the four corners."""
        self._handles = [
            QtCore.QPointF(0.0, 0.0),
            QtCore.QPointF(1.0, 0.0),
            QtCore.QPointF(1.0, 1.0),
            QtCore.QPointF(0.0, 1.0),
        ]
        self.update()

    def sizeHint(self):
        return QtCore.QSize(320, 240)

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    def _widget_to_prop(self, wx: float, wy: float) -> QtCore.QPointF:
        """Convert widget pixel coords to proportional (0~1)."""
        pw, ph = self.width(), self.height()
        if pw <= 0 or ph <= 0:
            return QtCore.QPointF(0, 0)
        return QtCore.QPointF(
            max(0.0, min(1.0, wx / pw)),
            max(0.0, min(1.0, wy / ph)),
        )

    def _prop_to_widget(self, px: float, py: float) -> QtCore.QPointF:
        """Convert proportional coords (0~1) to widget pixel coords."""
        return QtCore.QPointF(px * self.width(), py * self.height())

    def _find_handle(self, pos: QtCore.QPoint) -> int:
        """Return index of handle near pos, or -1."""
        hit_dist = (self.HANDLE_RADIUS * 2) ** 2
        for i, h in enumerate(self._handles):
            wp = self._prop_to_widget(h.x(), h.y())
            dx = pos.x() - wp.x()
            dy = pos.y() - wp.y()
            if dx * dx + dy * dy <= hit_dist:
                return i
        return -1

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        w, h = self.width(), self.height()

        # Background rect
        painter.setPen(QtGui.QPen(QtGui.QColor("#666666"), 1))
        painter.setBrush(QtGui.QColor("#2d2d2d"))
        painter.drawRect(0, 0, w - 1, h - 1)

        # Map handles to pixel coords
        pts = [self._prop_to_widget(p.x(), p.y()) for p in self._handles]

        # Semi-transparent filled quadrilateral
        poly = QtGui.QPolygonF(pts)
        fill = QtGui.QColor("#0078D4")
        fill.setAlpha(50)
        painter.setBrush(fill)
        painter.setPen(QtGui.QPen(QtGui.QColor("#0078D4"), 2))
        painter.drawPolygon(poly)

        # Dashed border lines
        dash_pen = QtGui.QPen(QtGui.QColor("#00B7C3"), 2, QtCore.Qt.DashLine)
        for i in range(4):
            painter.setPen(dash_pen)
            painter.drawLine(pts[i], pts[(i + 1) % 4])

        # Corner handles
        for i, pt in enumerate(pts):
            if i == self._active_handle:
                painter.setBrush(QtGui.QColor("#FF6B00"))
                painter.setPen(QtGui.QPen(QtGui.QColor("#FF6B00"), 2))
            else:
                painter.setBrush(QtGui.QColor("#00B7C3"))
                painter.setPen(QtGui.QPen(QtCore.Qt.white, 1))
            painter.drawEllipse(pt, self.HANDLE_RADIUS, self.HANDLE_RADIUS)

        painter.end()

    # ------------------------------------------------------------------
    # Mouse interaction
    # ------------------------------------------------------------------

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self._active_handle = self._find_handle(event.pos())
            if self._active_handle >= 0:
                self.setCursor(QtCore.Qt.ClosedHandCursor)
            self.update()

    def mouseMoveEvent(self, event):
        if self._active_handle >= 0:
            prop = self._widget_to_prop(event.x(), event.y())
            self._handles[self._active_handle] = prop
            self.update()
        else:
            h = self._find_handle(event.pos())
            self.setCursor(
                QtCore.Qt.OpenHandCursor if h >= 0 else QtCore.Qt.ArrowCursor
            )

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self._active_handle = -1
            self.setCursor(QtCore.Qt.ArrowCursor)
            self.update()


# ==================================================================
# Plugin
# ==================================================================

class SAM3TextPromptPlugin(PluginBase):
    """SAM3 text-prompt batch auto annotation plugin."""

    def __init__(self):
        super().__init__()
        self._batch_files: List[str] = []
        self._batch_index = 0
        self._batch_running = False
        self._total_masks = 0
        self._total_objects = 0
        self._range_mode = False
        self._range_category = ""
        self._delete_small_mode = False
        self._delete_small_data = {}
        self._small_scan_mode = False
        self._small_scan_files = []
        self._small_scan_index = 0
        self._small_scan_total = 0
        self._small_scan_affected = 0
        self._small_scan_params = {}
        self._region_mode = False
        self._region_label = ""
        self._region_layer_bottom = True  # True=bottom, False=top
        self._remap_mode = False
        self._remap_source_cat = ""
        self._remap_target_cat = ""
        self._remap_threshold_pct = 0.0
        self.default_prompts = "person, car"

    # ==================================================================
    # PluginBase interface
    # ==================================================================

    def init_plugin(self, mainwindow):
        self.mainwindow = mainwindow
        self.init_ui()

    def enable_plugin(self):
        self.mainwindow.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.dock)
        self.dock.show()
        self.enabled = True

    def disable_plugin(self):
        self.mainwindow.removeDockWidget(self.dock)
        self.enabled = False

    def get_plugin_author(self) -> str:
        try:
            from ISAT_plugin_sam3_text_prompt import __author__
        except ImportError:
            __author__ = "unknown"
        return __author__

    def get_plugin_version(self) -> str:
        try:
            from ISAT_plugin_sam3_text_prompt import __version__
        except ImportError:
            __version__ = "unknown"
        return __version__

    def get_plugin_description(self) -> str:
        return "SAM3 text-prompt batch auto annotation."

    # ==================================================================
    # UI
    # ==================================================================

    def init_ui(self):
        self.dock = QtWidgets.QDockWidget(self.mainwindow)
        self.dock.setWindowTitle("SAM3 Text Prompt Batch")

        main_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(main_widget)
        main_layout.setContentsMargins(6, 6, 6, 6)

        # ---- row 1: current image actions ----
        row = QtWidgets.QWidget()
        row.setMaximumHeight(36)
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)

        self.predict_current_btn = QtWidgets.QPushButton("Predict Current")
        self.predict_current_btn.setToolTip("SAM3 text-prompt on current image")
        self.predict_current_btn.clicked.connect(self.predict_current)

        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setToolTip("Stop batch")
        self.stop_btn.clicked.connect(self.stop)
        self.stop_btn.setEnabled(False)

        layout.addWidget(self.predict_current_btn)
        layout.addWidget(self.stop_btn)
        layout.addStretch()
        main_layout.addWidget(row)

        # ---- separator ----
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setFrameShadow(QtWidgets.QFrame.Sunken)
        main_layout.addWidget(sep)

        # ---- row 2: range settings + category + mapping ----
        row = QtWidgets.QWidget()
        row.setMaximumHeight(36)
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QtWidgets.QLabel("From #"))
        self.range_start_spin = QtWidgets.QSpinBox()
        self.range_start_spin.setMinimum(1)
        self.range_start_spin.setMaximum(999999)
        self.range_start_spin.setValue(1)
        self.range_start_spin.setToolTip("Start image index (1-based)")
        layout.addWidget(self.range_start_spin)

        layout.addWidget(QtWidgets.QLabel("To #"))
        self.range_end_spin = QtWidgets.QSpinBox()
        self.range_end_spin.setMinimum(1)
        self.range_end_spin.setMaximum(999999)
        self.range_end_spin.setValue(1)
        self.range_end_spin.setToolTip("End image index (1-based, inclusive)")
        layout.addWidget(self.range_end_spin)

        self.range_count_label = QtWidgets.QLabel("of 0")
        layout.addWidget(self.range_count_label)

        # unified category input (used by ALL operations)
        layout.addWidget(QtWidgets.QLabel("Cat:"))
        self.range_category_edit = QtWidgets.QLineEdit()
        self.range_category_edit.setPlaceholderText("car, person, tree")
        self.range_category_edit.setText(self.default_prompts)
        self.range_category_edit.setToolTip(
            "Categories for SAM3 text-prompt and Annotate Range.\n"
            "SAM3 mode: comma-separated text prompts.\n"
            "Annotate Range mode: single category name."
        )
        layout.addWidget(self.range_category_edit)

        # label mapping: prompt:label pairs
        layout.addWidget(QtWidgets.QLabel("Map:"))
        self.mapping_edit = QtWidgets.QLineEdit()
        self.mapping_edit.setPlaceholderText("grass/leaves:lawn")
        self.mapping_edit.setToolTip(
            "Optional label mapping: 'prompt:label' pairs, comma-separated.\n"
            "e.g. 'grass/leaves:lawn, car:vehicle'\n"
            "SAM3 detects with the prompt (left of ':'), "
            "but saves as the label (right of ':').\n"
            "Categories without a mapping keep their original name."
        )
        layout.addWidget(self.mapping_edit)
        main_layout.addWidget(row)

        # ---- row 3: range actions (all use From # / To #) ----
        row = QtWidgets.QWidget()
        row.setMaximumHeight(36)
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)

        self.predict_all_btn = QtWidgets.QPushButton("Predict Range")
        self.predict_all_btn.setToolTip(
            "SAM3 text-prompt batch on the selected range.\n"
            "OVERWRITES existing annotations."
        )
        self.predict_all_btn.clicked.connect(self.predict_all)
        self.predict_all_btn.setStyleSheet(
            "QPushButton { background-color: #0078D4; color: white; "
            "font-weight: bold; padding: 4px 12px; }"
        )

        self.predict_resume_btn = QtWidgets.QPushButton("Resume Range")
        self.predict_resume_btn.setToolTip(
            "SAM3 text-prompt on the selected range.\n"
            "Skips images that already have annotations."
        )
        self.predict_resume_btn.clicked.connect(self.predict_resume)
        self.predict_resume_btn.setStyleSheet(
            "QPushButton { background-color: #107C10; color: white; "
            "font-weight: bold; padding: 4px 12px; }"
        )

        self.range_annotate_btn = QtWidgets.QPushButton("Annotate Range")
        self.range_annotate_btn.setToolTip(
            "Set full-image annotation for the selected range.\n"
            "Each image gets one polygon covering the whole image."
        )
        self.range_annotate_btn.clicked.connect(self.predict_range)
        self.range_annotate_btn.setStyleSheet(
            "QPushButton { background-color: #CA5010; color: white; "
            "font-weight: bold; padding: 4px 12px; }"
        )

        self.range_delete_btn = QtWidgets.QPushButton("Delete Range")
        self.range_delete_btn.setToolTip(
            "DELETE annotation files (.json) for the selected range.\n"
            "This is irreversible — use with caution."
        )
        self.range_delete_btn.clicked.connect(self.delete_range)
        self.range_delete_btn.setStyleSheet(
            "QPushButton { background-color: #D13438; color: white; "
            "font-weight: bold; padding: 4px 12px; }"
        )

        layout.addWidget(self.predict_all_btn)
        layout.addWidget(self.predict_resume_btn)
        layout.addWidget(self.range_annotate_btn)
        layout.addWidget(self.range_delete_btn)
        main_layout.addWidget(row)

        # ---- separator ----
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setFrameShadow(QtWidgets.QFrame.Sunken)
        main_layout.addWidget(sep)

        # ---- row 4: delete small masks ----
        row = QtWidgets.QWidget()
        row.setMaximumHeight(36)
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QtWidgets.QLabel("Threshold:"))
        self.small_threshold_spin = QtWidgets.QDoubleSpinBox()
        self.small_threshold_spin.setRange(0.01, 100.0)
        self.small_threshold_spin.setValue(1.0)
        self.small_threshold_spin.setDecimals(2)
        self.small_threshold_spin.setSuffix("%")
        self.small_threshold_spin.setToolTip(
            "Masks with area < this % of image area will be deleted."
        )
        layout.addWidget(self.small_threshold_spin)

        layout.addWidget(QtWidgets.QLabel("Cat:"))
        self.small_category_edit = QtWidgets.QLineEdit()
        self.small_category_edit.setPlaceholderText("e.g. person")
        self.small_category_edit.setToolTip(
            "Only delete small masks of this category.\n"
            "Leave empty to delete small masks of ALL categories."
        )
        layout.addWidget(self.small_category_edit)

        self.delete_small_btn = QtWidgets.QPushButton("Delete Small Masks")
        self.delete_small_btn.setToolTip(
            "Scan the selected range (From # – To #) and remove masks\n"
            "whose area is below the threshold percentage of image area.\n"
            "Only affects the specified category (or all if left empty)."
        )
        self.delete_small_btn.clicked.connect(self.delete_small_masks)
        self.delete_small_btn.setStyleSheet(
            "QPushButton { background-color: #881798; color: white; "
            "font-weight: bold; padding: 4px 12px; }"
        )
        layout.addWidget(self.delete_small_btn)
        layout.addStretch()
        main_layout.addWidget(row)

        # ---- separator ----
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setFrameShadow(QtWidgets.QFrame.Sunken)
        main_layout.addWidget(sep)

        # ---- row 4.5: small mask remap ----
        row = QtWidgets.QWidget()
        row.setMaximumHeight(36)
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QtWidgets.QLabel("Remap:"))
        self.remap_source_edit = QtWidgets.QLineEdit()
        self.remap_source_edit.setPlaceholderText("source cat")
        self.remap_source_edit.setToolTip(
            "Source category: masks of this category below the threshold will be remapped."
        )
        layout.addWidget(self.remap_source_edit)

        layout.addWidget(QtWidgets.QLabel("<"))
        self.remap_threshold_spin = QtWidgets.QDoubleSpinBox()
        self.remap_threshold_spin.setRange(0.01, 100.0)
        self.remap_threshold_spin.setValue(1.0)
        self.remap_threshold_spin.setDecimals(2)
        self.remap_threshold_spin.setSuffix("%")
        self.remap_threshold_spin.setToolTip(
            "Masks of the source category with area < this % of image area will be remapped."
        )
        layout.addWidget(self.remap_threshold_spin)

        layout.addWidget(QtWidgets.QLabel("→"))
        self.remap_target_edit = QtWidgets.QLineEdit()
        self.remap_target_edit.setPlaceholderText("target cat")
        self.remap_target_edit.setToolTip(
            "Target category: small masks will be relabeled to this category.\n"
            "Supports Map: mapping."
        )
        layout.addWidget(self.remap_target_edit)

        self.remap_btn = QtWidgets.QPushButton("Remap")
        self.remap_btn.setToolTip(
            "Scan the From # – To # range and remap small masks\n"
            "of the source category to the target category."
        )
        self.remap_btn.clicked.connect(self._remap_small_masks)
        self.remap_btn.setStyleSheet(
            "QPushButton { background-color: #C239B3; color: white; "
            "font-weight: bold; padding: 4px 12px; }"
        )
        layout.addWidget(self.remap_btn)
        layout.addStretch()
        main_layout.addWidget(row)

        # ---- separator ----
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setFrameShadow(QtWidgets.QFrame.Sunken)
        main_layout.addWidget(sep)

        # ---- row 5: region label + layer dropdown ----
        row = QtWidgets.QWidget()
        row.setMaximumHeight(36)
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QtWidgets.QLabel("Region Label:"))
        self.region_label_edit = QtWidgets.QLineEdit()
        self.region_label_edit.setPlaceholderText("e.g. background")
        self.region_label_edit.setToolTip(
            "Category name for the mapped region mask.\n"
            "Supports Map: mapping."
        )
        layout.addWidget(self.region_label_edit)

        layout.addWidget(QtWidgets.QLabel("Layer:"))
        self.region_layer_combo = QtWidgets.QComboBox()
        self.region_layer_combo.addItems(["追加-最底层", "追加-最上层"])
        self.region_layer_combo.setToolTip(
            "Bottom: place region mask behind all existing objects.\n"
            "Top: place region mask on top of all existing objects."
        )
        layout.addWidget(self.region_layer_combo)
        layout.addStretch()
        main_layout.addWidget(row)

        # ---- row 5.5: region aspect ratio config ----
        row = QtWidgets.QWidget()
        row.setMaximumHeight(36)
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QtWidgets.QLabel("W:"))
        self.region_ratio_w_spin = QtWidgets.QSpinBox()
        self.region_ratio_w_spin.setRange(1, 99999)
        self.region_ratio_w_spin.setValue(640)
        self.region_ratio_w_spin.setToolTip("Reference image width for aspect ratio")
        layout.addWidget(self.region_ratio_w_spin)

        layout.addWidget(QtWidgets.QLabel("H:"))
        self.region_ratio_h_spin = QtWidgets.QSpinBox()
        self.region_ratio_h_spin.setRange(1, 99999)
        self.region_ratio_h_spin.setValue(480)
        self.region_ratio_h_spin.setToolTip("Reference image height for aspect ratio")
        layout.addWidget(self.region_ratio_h_spin)

        self.region_ratio_update_btn = QtWidgets.QPushButton("Update Ratio")
        self.region_ratio_update_btn.setToolTip(
            "Apply the W:H ratio to the region rectangle below.\n"
            "Height is fixed; width adjusts to match the ratio."
        )
        self.region_ratio_update_btn.clicked.connect(self._update_region_ratio)
        layout.addWidget(self.region_ratio_update_btn)
        layout.addStretch()
        main_layout.addWidget(row)

        # ---- row 6: proportional region widget (centered) ----
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)

        self.region_widget = ProportionalRegionWidget()
        self.region_widget.setToolTip(
            "Drag the 4 cyan corner points to define the region.\n"
            "The rectangle ratio matches the W:H values above.\n"
            "The defined quadrilateral is mapped onto each image in the range."
        )
        REGION_H = 237
        self.region_widget.setFixedSize(316, REGION_H)  # 316 = 237*(640/480)
        self.region_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )

        layout.addStretch()
        layout.addWidget(self.region_widget)
        layout.addStretch()
        main_layout.addWidget(row)

        # ---- row 7: apply region button ----
        row = QtWidgets.QWidget()
        row.setMaximumHeight(36)
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)

        self.apply_region_btn = QtWidgets.QPushButton("Apply Region")
        self.apply_region_btn.setToolTip(
            "Apply the defined quadrilateral region as a mask\n"
            "to ALL images in the From # – To # range.\n"
            "Appends the mask (does not replace existing annotations)."
        )
        self.apply_region_btn.clicked.connect(self._apply_region)
        self.apply_region_btn.setStyleSheet(
            "QPushButton { background-color: #038387; color: white; "
            "font-weight: bold; padding: 4px 12px; }"
        )
        layout.addWidget(self.apply_region_btn)
        layout.addStretch()
        main_layout.addWidget(row)

        # ---- progress bar ----
        self.processbar = QtWidgets.QProgressBar()
        self.processbar.setMaximum(100)
        self.processbar.setValue(0)
        main_layout.addWidget(self.processbar)

        # ---- status ----
        self.status_label = QtWidgets.QLabel(
            "Ready. Requires SAM3 model loaded in ISAT."
        )
        self.status_label.setWordWrap(True)
        main_layout.addWidget(self.status_label)

        # ---- result table ----
        self.result_table = QtWidgets.QTableWidget()
        self.result_table.setColumnCount(3)
        self.result_table.setHorizontalHeaderLabels(
            ["Filename", "Masks/Objects", "Status"]
        )
        self.result_table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.Stretch
        )
        self.result_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.result_table.setMinimumHeight(120)
        main_layout.addWidget(self.result_table)

        main_layout.addStretch()
        self.dock.setWidget(main_widget)
        self.mainwindow.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.dock)

        if not self.enabled:
            self.disable_plugin()

    # ==================================================================
    # Prediction
    # ==================================================================

    def _get_categories(self) -> List[str]:
        raw = self.range_category_edit.text().strip()
        if not raw:
            return ["object"]
        return [p.strip() for p in raw.split(",") if p.strip()]

    def _parse_mapping(self) -> dict:
        """解析 Map: 输入框，返回 {prompt: label} 映射字典。

        格式: 'prompt1:label1, prompt2:label2'
        不带 ':' 的条目忽略。
        """
        raw = self.mapping_edit.text().strip()
        if not raw:
            return {}
        mapping = {}
        for pair in raw.split(","):
            pair = pair.strip()
            if ":" in pair:
                prompt, label = pair.split(":", 1)
                prompt = prompt.strip()
                label = label.strip()
                if prompt and label:
                    mapping[prompt] = label
        return mapping

    def _map_category(self, category: str, mapping: dict) -> str:
        """将类别名按映射表转换，无映射则保持原名。"""
        return mapping.get(category, category)

    def _check_sam3(self) -> bool:
        if not self.mainwindow.use_segment_anything:
            QtWidgets.QMessageBox.warning(
                self.mainwindow, "Error",
                "SAM not enabled. Please load a SAM3 model in ISAT first."
            )
            return False
        if self.mainwindow.segany.model_source != "sam3":
            QtWidgets.QMessageBox.warning(
                self.mainwindow, "Error",
                f"Current model is not SAM3 (is {self.mainwindow.segany.model_source})."
                f"\nPlease load a sam3 model."
            )
            return False
        return True

    def _disable_buttons(self):
        self.predict_current_btn.setEnabled(False)
        self.predict_all_btn.setEnabled(False)
        self.predict_resume_btn.setEnabled(False)
        self.range_annotate_btn.setEnabled(False)
        self.range_delete_btn.setEnabled(False)
        self.delete_small_btn.setEnabled(False)
        self.remap_btn.setEnabled(False)
        self.apply_region_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def _enable_buttons(self):
        self.predict_current_btn.setEnabled(True)
        self.predict_all_btn.setEnabled(True)
        self.predict_resume_btn.setEnabled(True)
        self.range_annotate_btn.setEnabled(True)
        self.range_delete_btn.setEnabled(True)
        self.delete_small_btn.setEnabled(True)
        self.remap_btn.setEnabled(True)
        self.apply_region_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def predict_current(self):
        """对当前图片进行 SAM3 text-prompt 预测。"""
        if not self._check_sam3():
            return

        categories = self._get_categories()
        filename = self.mainwindow.files_list[self.mainwindow.current_index]
        file_path = os.path.join(self.mainwindow.image_root, filename)

        self._disable_buttons()
        self.status_label.setText(f"Predicting: {filename} ...")
        QtWidgets.QApplication.processEvents()

        try:
            num_masks, num_objects = self._predict_single_image(
                file_path, filename, categories
            )
            self.mainwindow.show_image(self.mainwindow.current_index, zoomfit=False)
            self.status_label.setText(
                f"Done: {filename} — {num_masks} masks, {num_objects} objects."
            )
        except Exception as e:
            self.status_label.setText(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._enable_buttons()

    # ==================================================================
    # Range helper — shared by predict_all / resume / range / delete
    # ==================================================================

    def _resolve_range(self, require_files: bool = True):
        """读取 spin box 范围，校验并返回 (files_subset, start_1, end_1)。

        若校验失败则弹出对话框并返回 (None, None, None)。
        """
        all_files = list(self.mainwindow.files_list)
        total = len(all_files)
        if total == 0:
            if require_files:
                QtWidgets.QMessageBox.information(self.mainwindow, "Info", "No images.")
            return None, None, None

        start_1 = self.range_start_spin.value()
        end_1 = self.range_end_spin.value()

        if start_1 > end_1:
            QtWidgets.QMessageBox.warning(
                self.mainwindow, "Input Error",
                f"Start (#{start_1}) must be ≤ End (#{end_1})."
            )
            return None, None, None

        start_idx = start_1 - 1
        end_idx = end_1 - 1  # inclusive

        if start_idx < 0 or end_idx >= total:
            QtWidgets.QMessageBox.warning(
                self.mainwindow, "Input Error",
                f"Range #{start_1}–#{end_1} is out of bounds (1–{total})."
            )
            return None, None, None

        files = all_files[start_idx:end_idx + 1]
        return files, start_1, end_1

    def predict_all(self):
        if not self._check_sam3():
            return

        files, start_1, end_1 = self._resolve_range()
        if files is None:
            return

        categories = self._get_categories()

        reply = QtWidgets.QMessageBox.question(
            self.mainwindow, "Confirm",
            f"Predict images #{start_1} – #{end_1} ({len(files)} images)\n"
            f"with SAM3 text-prompt.\n"
            f"Categories: {', '.join(categories)}\n\n"
            f"This will OVERWRITE existing annotations. Continue?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return

        self._batch_files = files
        self._batch_index = 0
        self._batch_running = True
        self._total_masks = 0
        self._total_objects = 0
        self._range_mode = False
        self.result_table.setRowCount(0)
        self.processbar.setMaximum(len(files))
        self.processbar.setValue(0)

        self._disable_buttons()
        QTimer.singleShot(50, self._process_next)

    def predict_resume(self):
        """断点续推理——在选定范围内跳过已有标注，只推理未标注的图片。"""
        if not self._check_sam3():
            return

        range_files, start_1, end_1 = self._resolve_range()
        if range_files is None:
            return

        categories = self._get_categories()

        # 检测范围内哪些文件已有标注
        label_root = self.mainwindow.label_root
        annotated = set()
        for filename in range_files:
            base = ".".join(filename.split(".")[:-1])
            json_path = os.path.join(label_root, base + ".json")
            if os.path.isfile(json_path):
                annotated.add(filename)

        # 过滤出未标注的文件
        files = [f for f in range_files if f not in annotated]
        skipped = len(annotated)

        if not files:
            QtWidgets.QMessageBox.information(
                self.mainwindow, "Info",
                f"All {len(range_files)} images in #{start_1}–#{end_1} "
                f"already have annotations. Nothing to do."
            )
            return

        reply = QtWidgets.QMessageBox.question(
            self.mainwindow, "Resume Prediction",
            f"Range: #{start_1} – #{end_1}\n"
            f"Skip {skipped} annotated images.\n"
            f"Predict {len(files)} remaining images with SAM3 text-prompt.\n"
            f"Categories: {', '.join(categories)}\n\n"
            f"Existing annotations will NOT be overwritten. Continue?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return

        self._batch_files = files
        self._batch_index = 0
        self._batch_running = True
        self._total_masks = 0
        self._total_objects = 0
        self._range_mode = False
        self.result_table.setRowCount(0)
        self.processbar.setMaximum(len(files))
        self.processbar.setValue(0)

        self._disable_buttons()
        QTimer.singleShot(50, self._process_next)

    def predict_range(self):
        """Range annotation — 对指定编号范围的图片进行全图标注。

        将起止编号之间的每张图片用整个画面区域标注为指定类别。
        不依赖 SAM3，纯几何标注。
        """
        files, start_1, end_1 = self._resolve_range()
        if files is None:
            return

        category = self.range_category_edit.text().strip()
        if not category:
            QtWidgets.QMessageBox.warning(
                self.mainwindow, "Input Error",
                "Please enter a category name."
            )
            return

        reply = QtWidgets.QMessageBox.question(
            self.mainwindow, "Confirm Range Annotation",
            f"Annotate images #{start_1} – #{end_1} ({len(files)} images)\n"
            f"as full-image category: \"{category}\"\n\n"
            f"Existing annotations for these images will be REPLACED.\nContinue?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return

        self._batch_files = files
        self._batch_index = 0
        self._batch_running = True
        self._total_masks = 0
        self._total_objects = 0
        self.result_table.setRowCount(0)
        self.processbar.setMaximum(len(files))
        self.processbar.setValue(0)

        # 暂存 range 专用的提示词，供 _process_next 区分模式
        self._range_mode = True
        self._range_category = category

        self._disable_buttons()
        QTimer.singleShot(50, self._process_next)

    def delete_range(self):
        """删除指定编号范围图片的标注文件 (.json)。

        不可逆操作，需二次确认。
        """
        files, start_1, end_1 = self._resolve_range()
        if files is None:
            return

        # 统计哪些有标注
        label_root = self.mainwindow.label_root
        existing = []
        for filename in files:
            base = ".".join(filename.split(".")[:-1])
            json_path = os.path.join(label_root, base + ".json")
            if os.path.isfile(json_path):
                existing.append(json_path)

        if not existing:
            QtWidgets.QMessageBox.information(
                self.mainwindow, "Info",
                f"No annotation files found for images #{start_1}–#{end_1}."
            )
            return

        reply = QtWidgets.QMessageBox.warning(
            self.mainwindow, "⚠ Delete Annotations",
            f"Delete {len(existing)} annotation file(s) for\n"
            f"images #{start_1} – #{end_1}?\n\n"
            f"THIS CANNOT BE UNDONE.\n"
            f"Affected files:\n"
            + "\n".join(f"  • {os.path.basename(p)}" for p in existing[:10])
            + (f"\n  ... and {len(existing) - 10} more" if len(existing) > 10 else ""),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,  # default to No for safety
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return

        deleted = 0
        for json_path in existing:
            try:
                os.remove(json_path)
                deleted += 1
            except OSError as e:
                print(f"[Delete] Failed to remove {json_path}: {e}")

        self.status_label.setText(
            f"Deleted {deleted} annotation file(s) for #{start_1}–#{end_1}."
        )
        # 刷新当前视图
        if self.mainwindow.current_index is not None:
            self.mainwindow.show_image(self.mainwindow.current_index, zoomfit=False)

    def delete_small_masks(self):
        """异步扫描并删除指定范围内面积小于阈值的 mask。

        分两阶段异步执行: 扫描 → 确认 → 删除，全程不阻塞 UI。
        """
        files, start_1, end_1 = self._resolve_range()
        if files is None:
            return

        threshold_pct = self.small_threshold_spin.value()
        filter_cat = self.small_category_edit.text().strip()
        filter_cats = set()
        if filter_cat:
            filter_cats = {c.strip() for c in filter_cat.split(",") if c.strip()}

        # 保存参数，启动异步扫描
        self._small_scan_files = files
        self._small_scan_index = 0
        self._small_scan_total = 0
        self._small_scan_affected = 0
        self._small_scan_params = {
            "threshold_pct": threshold_pct,
            "filter_cats": filter_cats,
            "label_root": self.mainwindow.label_root,
            "start_1": start_1,
            "end_1": end_1,
        }
        self._small_scan_mode = True

        self.result_table.setRowCount(0)
        self.processbar.setMaximum(len(files))
        self.processbar.setValue(0)

        self._disable_buttons()
        self.status_label.setText(f"Scanning... 0/{len(files)}")
        QTimer.singleShot(10, self._process_small_scan_next)

    def _process_small_scan_next(self):
        """异步扫描一张图片的小 mask（QTimer 驱动）。"""
        files = self._small_scan_files
        i = self._small_scan_index
        total = len(files)

        if i >= total:
            # 扫描完成 → 显示结果
            self._small_scan_mode = False
            self.status_label.setText("Scan complete.")
            self._show_small_scan_result()
            return

        # 处理当前文件
        filename = files[i]
        self._small_scan_index = i + 1

        params = self._small_scan_params
        label_root = params["label_root"]
        threshold_pct = params["threshold_pct"]
        filter_cats = params["filter_cats"]

        base = ".".join(filename.split(".")[:-1])
        json_path = os.path.join(label_root, base + ".json")
        from PIL import Image
        from ISAT.annotation import Annotation

        if os.path.isfile(json_path):
            file_path = os.path.join(self.mainwindow.image_root, filename)
            try:
                img = Image.open(file_path)
                img_area = img.width * img.height
                annotation = Annotation(file_path, json_path)
                annotation.load_annotation()

                file_has = False
                for obj in annotation.objects:
                    if filter_cats and obj.category not in filter_cats:
                        continue
                    obj_pct = (obj.area / img_area) * 100.0 if img_area > 0 else 0
                    if obj_pct < threshold_pct:
                        self._small_scan_total += 1
                        file_has = True
                if file_has:
                    self._small_scan_affected += 1
            except Exception:
                pass

        # 更新 UI
        if (i + 1) % 10 == 0 or (i + 1) == total:
            self.processbar.setValue(i + 1)
            self.status_label.setText(
                f"Scanning... {i + 1}/{total}  "
                f"(found {self._small_scan_total} small masks so far)"
            )
            QtWidgets.QApplication.processEvents()

        QTimer.singleShot(1, self._process_small_scan_next)

    def _show_small_scan_result(self):
        """扫描完成后弹出确认对话框。"""
        total_candidates = self._small_scan_total
        affected_files = self._small_scan_affected
        params = self._small_scan_params
        threshold_pct = params["threshold_pct"]
        filter_cats = params["filter_cats"]
        filter_cat = ", ".join(sorted(filter_cats)) if filter_cats else ""

        if total_candidates == 0:
            self._enable_buttons()
            QtWidgets.QMessageBox.information(
                self.mainwindow, "Info",
                f"No small masks found in #{params['start_1']}–#{params['end_1']} "
                f"below {threshold_pct}%."
                + (f" (category: {filter_cat})" if filter_cat else "")
            )
            return

        reply = QtWidgets.QMessageBox.warning(
            self.mainwindow, "⚠ Delete Small Masks",
            f"Found {total_candidates} small mask(s) in {affected_files} file(s)\n"
            f"in #{params['start_1']}–#{params['end_1']} below {threshold_pct}%."
            + (f"\nCategory filter: {filter_cat}" if filter_cat else "\n(all categories)")
            + f"\n\nDelete them now? This cannot be undone.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            self._enable_buttons()
            return

        # ---- 启动异步删除阶段 ----
        self._batch_files = self._small_scan_files
        self._batch_index = 0
        self._batch_running = True
        self._total_masks = total_candidates
        self._total_objects = 0
        self._delete_small_mode = True
        self._delete_small_data = {
            "threshold_pct": threshold_pct,
            "filter_cats": filter_cats,
            "label_root": params["label_root"],
            "actual_removed": 0,
            "actual_files": 0,
        }
        self.result_table.setRowCount(0)
        self.processbar.setMaximum(len(self._small_scan_files))
        self.processbar.setValue(0)

        QTimer.singleShot(50, self._process_next)

    def _process_small_delete(self, filename: str):
        """处理单张图片的小 mask 删除（由 _process_next 调用）。"""
        data = self._delete_small_data
        label_root = data["label_root"]
        threshold_pct = data["threshold_pct"]
        filter_cats = data["filter_cats"]

        base = ".".join(filename.split(".")[:-1])
        json_path = os.path.join(label_root, base + ".json")
        if not os.path.isfile(json_path):
            return 0

        file_path = os.path.join(self.mainwindow.image_root, filename)
        from PIL import Image
        from ISAT.annotation import Annotation

        try:
            img = Image.open(file_path)
            img_area = img.width * img.height
        except Exception:
            return 0

        annotation = Annotation(file_path, json_path)
        annotation.load_annotation()

        kept = []
        removed = 0
        for obj in annotation.objects:
            if filter_cats and obj.category not in filter_cats:
                kept.append(obj)
                continue
            obj_pct = (obj.area / img_area) * 100.0 if img_area > 0 else 0
            if obj_pct < threshold_pct:
                removed += 1
            else:
                kept.append(obj)

        if removed > 0:
            annotation.objects = kept
            annotation.save_annotation()

        return removed

    def _process_next(self):
        if not self._batch_running or self._batch_index >= len(self._batch_files):
            self._finish()
            return

        idx = self._batch_index
        self._batch_index += 1
        filename = self._batch_files[idx]
        file_path = os.path.join(self.mainwindow.image_root, filename)

        self.status_label.setText(f"[{idx + 1}/{len(self._batch_files)}] {filename}")
        self.processbar.setValue(idx + 1)
        QtWidgets.QApplication.processEvents()

        try:
            # 分支: delete_small / remap / range 全图标注 / region 比例映射 / SAM3
            if getattr(self, '_delete_small_mode', False):
                removed = self._process_small_delete(filename)
                self._delete_small_data["actual_removed"] += removed
                if removed > 0:
                    self._delete_small_data["actual_files"] += 1
                mode_tag = "[DelSmall]"
                num_masks = removed
                num_objects = removed
            elif getattr(self, '_remap_mode', False):
                remapped = self._process_remap_single_image(file_path, filename)
                mode_tag = "[Remap]"
                num_masks = remapped
                num_objects = remapped
            elif getattr(self, '_range_mode', False):
                num_masks, num_objects = self._predict_range_single_image(
                    file_path, filename
                )
                mode_tag = "[Range]"
            elif getattr(self, '_region_mode', False):
                num_masks, num_objects = self._process_region_single_image(
                    file_path, filename
                )
                mode_tag = "[Region]"
            else:
                categories = self._get_categories()
                num_masks, num_objects = self._predict_single_image(
                    file_path, filename, categories
                )
                mode_tag = "[SAM3]"
            print(f"{mode_tag} {filename}: masks={num_masks}, objects={num_objects}")
            self._total_masks += num_masks
            self._total_objects += num_objects

            row = self.result_table.rowCount()
            self.result_table.insertRow(row)
            self.result_table.setItem(row, 0, QtWidgets.QTableWidgetItem(filename))
            self.result_table.setItem(
                row, 1, QtWidgets.QTableWidgetItem(f"{num_masks}/{num_objects}")
            )
            self.result_table.setItem(
                row, 2,
                QtWidgets.QTableWidgetItem("OK" if num_masks > 0 else "No detection")
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            row = self.result_table.rowCount()
            self.result_table.insertRow(row)
            self.result_table.setItem(row, 0, QtWidgets.QTableWidgetItem(filename))
            self.result_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"ERR: {e}"))

        QTimer.singleShot(100, self._process_next)

    # ==================================================================
    # 核心: 直接调用 SAM3 text-prompt + 手动保存标注
    # ==================================================================

    def _predict_single_image(self, file_path: str, filename: str,
                               categories: list) -> tuple:
        """
        对单张图片用 SAM3 text-prompt 预测并保存 .json。

        直接使用 ISAT 的基础 API，不依赖可能不存在的高级封装方法。
        """
        import numpy as np
        from PIL import Image

        # 解析 label 映射
        mapping = self._parse_mapping()

        # 1) 打开图片
        image = Image.open(file_path).convert("RGB")

        # 2) 加载/创建标注
        from ISAT.annotation import Annotation, Object
        label_path = os.path.join(
            self.mainwindow.label_root,
            ".".join(filename.split(".")[:-1]) + ".json"
        )
        annotation = Annotation(file_path, label_path)
        annotation.load_annotation()
        w, h = image.size

        # 确定起始 group
        group = 1
        for obj in annotation.objects:
            try:
                group = max(group, int(obj.group) + 1)
            except Exception:
                pass

        total_masks = 0
        total_objects = 0

        # z-ordering: later prompts appear on top of earlier ones.
        # Use a large layer stride so each prompt's masks occupy a
        # contiguous block and never interleave with other prompts.
        LAYER_STRIDE = 10000

        for prompt_idx, category in enumerate(categories):
            if not category or category == "__background__":
                continue

            # 应用映射: 用 prompt 检测，用 mapped label 保存
            save_label = self._map_category(category, mapping)

            # 3) SAM3 text-prompt 预测
            masks, scores = self.mainwindow.segany.predictor.predict_with_text_prompt(
                image, category
            )

            layer_base = 1 + prompt_idx * LAYER_STRIDE
            local_obj_count = 0

            for mask in masks:
                total_masks += 1

                # 4) mask → contours
                contours, hierarchy = self.mainwindow.mask_to_polygon(mask)

                for contour in contours:
                    if len(contour) < 3:
                        continue

                    segmentation = []
                    xmin, ymin = w, h
                    xmax, ymax = 0, 0
                    for point in contour:
                        x, y = point[0]
                        x, y = max(0.1, float(x)), max(0.1, float(y))
                        xmin = min(x, xmin)
                        ymin = min(y, ymin)
                        xmax = max(x, xmax)
                        ymax = max(y, ymax)
                        segmentation.append((round(x, 2), round(y, 2)))

                    area = self._polygon_area(segmentation)
                    obj = Object(
                        category=save_label, group=group,
                        segmentation=segmentation, area=area,
                        layer=layer_base + local_obj_count,
                        bbox=(xmin, ymin, xmax, ymax),
                        iscrowd=False, note="",
                    )
                    annotation.objects.append(obj)
                    total_objects += 1
                    local_obj_count += 1

                if self.mainwindow.group_select_mode == "auto":
                    group += 1

        # 5) 保存
        annotation.save_annotation()
        return total_masks, total_objects

    def _predict_range_single_image(self, file_path: str, filename: str) -> tuple:
        """对单张图片进行全图标注——整个画面作为一个多边形，归入 range 类别。

        不依赖 SAM3，纯几何标注。
        """
        from PIL import Image

        image = Image.open(file_path)
        w, h = image.size

        from ISAT.annotation import Annotation, Object
        label_path = os.path.join(
            self.mainwindow.label_root,
            ".".join(filename.split(".")[:-1]) + ".json"
        )
        annotation = Annotation(file_path, label_path)

        # 应用映射
        mapping = self._parse_mapping()
        save_label = self._map_category(self._range_category, mapping)

        # 全图 polygon: 左上 → 右上 → 右下 → 左下
        segmentation = [(0.0, 0.0), (float(w), 0.0),
                        (float(w), float(h)), (0.0, float(h))]
        area = float(w) * float(h)

        obj = Object(
            category=save_label,
            group=1,
            segmentation=segmentation,
            area=area,
            layer=1,
            bbox=(0.0, 0.0, float(w), float(h)),
            iscrowd=False,
            note="range_full_image",
        )
        annotation.objects = [obj]
        annotation.save_annotation()
        return 1, 1  # masks, objects

    @staticmethod
    def _polygon_area(points: list) -> float:
        area = 0
        n = len(points)
        for i in range(n):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % n]
            area += x1 * y2 - x2 * y1
        return abs(area) / 2

    # ==================================================================
    # Proportional Region Mapping
    # ==================================================================

    def _update_region_ratio(self):
        """Update the rectangle widget size to match the W:H ratio inputs.

        Height is fixed (237 px); width = 237 * (W / H).
        """
        w = self.region_ratio_w_spin.value()
        h = self.region_ratio_h_spin.value()
        FIXED_H = 237
        new_w = max(20, int(FIXED_H * w / h)) if h > 0 else FIXED_H
        self.region_widget.setFixedSize(new_w, FIXED_H)
        self.region_widget.set_aspect_ratio(w, h)

    def _apply_region(self):
        """Apply proportional region quadrilateral as mask to From/To range."""
        files, start_1, end_1 = self._resolve_range()
        if files is None:
            return

        label = self.region_label_edit.text().strip()
        if not label:
            QtWidgets.QMessageBox.warning(
                self.mainwindow, "Input Error",
                "Please enter a label for the region mask."
            )
            return

        layer_mode = "最底层" if self.region_layer_combo.currentIndex() == 0 else "最上层"

        reply = QtWidgets.QMessageBox.question(
            self.mainwindow, "Confirm Apply Region",
            f"Apply region mask to images #{start_1} – #{end_1} ({len(files)} images)\n"
            f"Label: \"{label}\"\n"
            f"Layer: {layer_mode}\n\n"
            f"The region mask will be APPENDED to existing annotations.\nContinue?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return

        self._batch_files = files
        self._batch_index = 0
        self._batch_running = True
        self._total_masks = 0
        self._total_objects = 0
        self._region_mode = True
        self._region_label = label
        self._region_layer_bottom = (self.region_layer_combo.currentIndex() == 0)
        self.result_table.setRowCount(0)
        self.processbar.setMaximum(len(files))
        self.processbar.setValue(0)

        self._disable_buttons()
        QTimer.singleShot(50, self._process_next)

    def _process_region_single_image(self, file_path: str, filename: str) -> tuple:
        """Apply proportional region polygon as a mask to a single image."""
        from PIL import Image
        from ISAT.annotation import Annotation, Object

        image = Image.open(file_path)
        w, h = image.size

        label_path = os.path.join(
            self.mainwindow.label_root,
            ".".join(filename.split(".")[:-1]) + ".json"
        )
        annotation = Annotation(file_path, label_path)
        annotation.load_annotation()

        # Apply Map: mapping
        mapping = self._parse_mapping()
        save_label = self._map_category(self._region_label, mapping)

        # Map proportional points (0~1) to pixel coords
        prop_pts = self.region_widget.get_proportional_points()
        segmentation = [
            (round(rx * w, 2), round(ry * h, 2))
            for rx, ry in prop_pts
        ]
        area = self._polygon_area(segmentation)

        # Compute layer — bottom or top relative to existing objects
        if self._region_layer_bottom:
            # Shift all existing objects up by 1 to make room at absolute bottom
            for obj in annotation.objects:
                obj.layer = int(obj.layer) + 1
            layer = 1
        else:
            max_layer = 0
            for obj in annotation.objects:
                try:
                    max_layer = max(max_layer, int(obj.layer))
                except Exception:
                    pass
            layer = max_layer + 1 if annotation.objects else 1

        # Compute bbox
        xs = [p[0] for p in segmentation]
        ys = [p[1] for p in segmentation]
        bbox = (min(xs), min(ys), max(xs), max(ys))

        # Determine group
        group = 1
        for obj in annotation.objects:
            try:
                group = max(group, int(obj.group) + 1)
            except Exception:
                pass

        obj = Object(
            category=save_label,
            group=group,
            segmentation=segmentation,
            area=area,
            layer=layer,
            bbox=bbox,
            iscrowd=False,
            note="",
        )
        annotation.objects.append(obj)
        annotation.save_annotation()
        return 1, 1  # masks, objects

    # ==================================================================
    # Small Mask Remap (relabel small masks to another category)
    # ==================================================================

    def _remap_small_masks(self):
        """Remap small masks of a source category to a target category."""
        files, start_1, end_1 = self._resolve_range()
        if files is None:
            return

        source_cat = self.remap_source_edit.text().strip()
        target_cat = self.remap_target_edit.text().strip()
        threshold_pct = self.remap_threshold_spin.value()

        if not source_cat:
            QtWidgets.QMessageBox.warning(
                self.mainwindow, "Input Error",
                "Please enter a source category."
            )
            return
        if not target_cat:
            QtWidgets.QMessageBox.warning(
                self.mainwindow, "Input Error",
                "Please enter a target category."
            )
            return

        reply = QtWidgets.QMessageBox.question(
            self.mainwindow, "Confirm Small Mask Remap",
            f"Remap masks in images #{start_1} – #{end_1} ({len(files)} images)\n"
            f"Source: \"{source_cat}\" < {threshold_pct}%\n"
            f"→ Target: \"{target_cat}\"\n\n"
            f"Masks with area below the threshold will be relabeled.\nContinue?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return

        self._batch_files = files
        self._batch_index = 0
        self._batch_running = True
        self._total_masks = 0
        self._total_objects = 0
        self._remap_mode = True
        self._remap_source_cat = source_cat
        self._remap_target_cat = target_cat
        self._remap_threshold_pct = threshold_pct
        self.result_table.setRowCount(0)
        self.processbar.setMaximum(len(files))
        self.processbar.setValue(0)

        self._disable_buttons()
        QTimer.singleShot(50, self._process_next)

    def _process_remap_single_image(self, file_path: str, filename: str) -> int:
        """Relabel small masks of source category in a single image. Returns count."""
        from PIL import Image
        from ISAT.annotation import Annotation

        base = ".".join(filename.split(".")[:-1])
        json_path = os.path.join(self.mainwindow.label_root, base + ".json")
        if not os.path.isfile(json_path):
            return 0

        # Apply Map: mapping
        mapping = self._parse_mapping()
        save_target = self._map_category(self._remap_target_cat, mapping)

        try:
            img = Image.open(file_path)
            img_area = img.width * img.height
        except Exception:
            return 0

        annotation = Annotation(file_path, json_path)
        annotation.load_annotation()

        remapped = 0
        for obj in annotation.objects:
            if obj.category != self._remap_source_cat:
                continue
            obj_pct = (obj.area / img_area) * 100.0 if img_area > 0 else 0
            if obj_pct < self._remap_threshold_pct:
                obj.category = save_target
                remapped += 1

        if remapped > 0:
            annotation.save_annotation()

        return remapped

    def _finish(self):
        self._batch_running = False

        if getattr(self, '_delete_small_mode', False):
            data = self._delete_small_data
            self.status_label.setText(
                f"Deleted {data['actual_removed']} small mask(s) "
                f"from {data['actual_files']} file(s)."
            )
        else:
            self.status_label.setText(
                f"Done: {len(self._batch_files)} images, "
                f"{self._total_masks} masks, {self._total_objects} objects."
            )

        self._range_mode = False
        self._delete_small_mode = False
        self._region_mode = False
        self._remap_mode = False
        self._enable_buttons()
        if self.mainwindow.current_index is not None:
            self.mainwindow.show_image(self.mainwindow.current_index, zoomfit=False)

    def stop(self):
        self._batch_running = False
        self._small_scan_mode = False
        self.status_label.setText("Stopping ...")
        self.stop_btn.setEnabled(False)
        self._enable_buttons()

    # ==================================================================
    # Events
    # ==================================================================

    def after_image_open_event(self):
        self.predict_current_btn.setEnabled(True)
        self.predict_all_btn.setEnabled(True)
        self.predict_resume_btn.setEnabled(True)
        self.range_annotate_btn.setEnabled(True)
        self.range_delete_btn.setEnabled(True)
        self.delete_small_btn.setEnabled(True)
        self.remap_btn.setEnabled(True)
        self.apply_region_btn.setEnabled(True)

        # 同步 range spinbox 上限到当前图片总数
        total = len(self.mainwindow.files_list)
        self.range_start_spin.setMaximum(max(1, total))
        self.range_end_spin.setMaximum(max(1, total))
        self.range_count_label.setText(f"of {total}")
