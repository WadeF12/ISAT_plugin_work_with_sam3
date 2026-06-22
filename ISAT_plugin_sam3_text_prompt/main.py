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

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QTimer

from ISAT.widgets.plugin_base import PluginBase


class SAM3TextPromptPlugin(PluginBase):
    """SAM3 text-prompt batch auto annotation plugin."""

    def __init__(self):
        super().__init__()
        self._batch_files: List[str] = []
        self._batch_index = 0
        self._batch_running = False
        self._total_masks = 0
        self._total_objects = 0
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

        # ---- category input ----
        row = QtWidgets.QWidget()
        row.setMaximumHeight(36)
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QtWidgets.QLabel("Categories:"))
        self.category_edit = QtWidgets.QLineEdit()
        self.category_edit.setPlaceholderText("car, person, tree, dog")
        self.category_edit.setText(self.default_prompts)
        self.category_edit.setToolTip(
            "Text prompt categories, comma separated. "
            "Each will be predicted by SAM3 text-prompt."
        )
        layout.addWidget(self.category_edit)
        main_layout.addWidget(row)

        # ---- buttons ----
        row = QtWidgets.QWidget()
        row.setMaximumHeight(36)
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)

        self.predict_current_btn = QtWidgets.QPushButton("Predict Current")
        self.predict_current_btn.setToolTip("SAM3 text-prompt on current image")
        self.predict_current_btn.clicked.connect(self.predict_current)

        self.predict_all_btn = QtWidgets.QPushButton("Predict ALL")
        self.predict_all_btn.setToolTip("Full batch predict — overwrites ALL images")
        self.predict_all_btn.clicked.connect(self.predict_all)
        self.predict_all_btn.setStyleSheet(
            "QPushButton { background-color: #0078D4; color: white; "
            "font-weight: bold; padding: 4px 12px; }"
        )

        self.predict_resume_btn = QtWidgets.QPushButton("Resume")
        self.predict_resume_btn.setToolTip(
            "Resume prediction — skip images that already have annotations"
        )
        self.predict_resume_btn.clicked.connect(self.predict_resume)
        self.predict_resume_btn.setStyleSheet(
            "QPushButton { background-color: #107C10; color: white; "
            "font-weight: bold; padding: 4px 12px; }"
        )

        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setToolTip("Stop batch")
        self.stop_btn.clicked.connect(self.stop)
        self.stop_btn.setEnabled(False)

        layout.addWidget(self.predict_current_btn)
        layout.addWidget(self.predict_all_btn)
        layout.addWidget(self.predict_resume_btn)
        layout.addWidget(self.stop_btn)
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
        raw = self.category_edit.text().strip()
        if not raw:
            return ["object"]
        return [p.strip() for p in raw.split(",") if p.strip()]

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
        self.stop_btn.setEnabled(True)

    def _enable_buttons(self):
        self.predict_current_btn.setEnabled(True)
        self.predict_all_btn.setEnabled(True)
        self.predict_resume_btn.setEnabled(True)
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

    def predict_all(self):
        if not self._check_sam3():
            return

        files = list(self.mainwindow.files_list)
        if not files:
            QtWidgets.QMessageBox.information(self.mainwindow, "Info", "No images.")
            return

        categories = self._get_categories()

        reply = QtWidgets.QMessageBox.question(
            self.mainwindow, "Confirm",
            f"Predict {len(files)} images with SAM3 text-prompt.\n"
            f"Categories: {', '.join(categories)}\n\n"
            f"This will overwrite existing annotations. Continue?",
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

        self._disable_buttons()
        QTimer.singleShot(50, self._process_next)

    def predict_resume(self):
        """断点续推理——跳过已有标注的文件，只推理未标注的图片。"""
        if not self._check_sam3():
            return

        all_files = list(self.mainwindow.files_list)
        if not all_files:
            QtWidgets.QMessageBox.information(self.mainwindow, "Info", "No images.")
            return

        categories = self._get_categories()

        # 检测哪些文件已有标注
        label_root = self.mainwindow.label_root
        annotated = set()
        for filename in all_files:
            base = ".".join(filename.split(".")[:-1])
            json_path = os.path.join(label_root, base + ".json")
            if os.path.isfile(json_path):
                annotated.add(filename)

        # 过滤出未标注的文件
        files = [f for f in all_files if f not in annotated]
        skipped = len(annotated)

        if not files:
            QtWidgets.QMessageBox.information(
                self.mainwindow, "Info",
                f"All {len(all_files)} images already have annotations. Nothing to do."
            )
            return

        reply = QtWidgets.QMessageBox.question(
            self.mainwindow, "Resume Prediction",
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
        self.result_table.setRowCount(0)
        self.processbar.setMaximum(len(files))
        self.processbar.setValue(0)

        self._disable_buttons()
        QTimer.singleShot(50, self._process_next)

    def _process_next(self):
        if not self._batch_running or self._batch_index >= len(self._batch_files):
            self._finish()
            return

        idx = self._batch_index
        self._batch_index += 1
        filename = self._batch_files[idx]
        file_path = os.path.join(self.mainwindow.image_root, filename)
        categories = self._get_categories()

        self.status_label.setText(f"[{idx + 1}/{len(self._batch_files)}] {filename}")
        self.processbar.setValue(idx + 1)
        QtWidgets.QApplication.processEvents()

        try:
            num_masks, num_objects = self._predict_single_image(
                file_path, filename, categories
            )
            print(f"[SAM3] {filename}: masks={num_masks}, objects={num_objects}")
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

        for category in categories:
            if not category or category == "__background__":
                continue

            # 3) SAM3 text-prompt 预测
            masks, scores = self.mainwindow.segany.predictor.predict_with_text_prompt(
                image, category
            )

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
                        category=category, group=group,
                        segmentation=segmentation, area=area,
                        layer=1 + len(annotation.objects) + len([obj for _ in []]),
                        bbox=(xmin, ymin, xmax, ymax),
                        iscrowd=False, note="",
                    )
                    annotation.objects.append(obj)
                    total_objects += 1

                if self.mainwindow.group_select_mode == "auto":
                    group += 1

        # 5) 保存
        annotation.save_annotation()
        return total_masks, total_objects

    @staticmethod
    def _polygon_area(points: list) -> float:
        area = 0
        n = len(points)
        for i in range(n):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % n]
            area += x1 * y2 - x2 * y1
        return abs(area) / 2

    def _finish(self):
        self._batch_running = False
        self._enable_buttons()
        if self.mainwindow.current_index is not None:
            self.mainwindow.show_image(self.mainwindow.current_index, zoomfit=False)
        self.status_label.setText(
            f"Done: {len(self._batch_files)} images, "
            f"{self._total_masks} masks, {self._total_objects} objects."
        )

    def stop(self):
        self._batch_running = False
        self.status_label.setText("Stopping ...")
        self.stop_btn.setEnabled(False)

    # ==================================================================
    # Events
    # ==================================================================

    def after_image_open_event(self):
        self.predict_current_btn.setEnabled(True)
        self.predict_all_btn.setEnabled(True)
        self.predict_resume_btn.setEnabled(True)
