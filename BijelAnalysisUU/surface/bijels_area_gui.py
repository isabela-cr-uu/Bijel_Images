import os
import sys
from typing import Optional, Tuple

import numpy as np
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets

import liffile
from liffile import LifFile

if __name__ == "__main__":
        from bijels_area_estimation import (
            ProcessingParams,
            load_image,
            process_image,
            save_image,
        )
else:
    from BijelAnalysisUU.surface.bijels_area_estimation import (
        ProcessingParams,
        load_image,
        process_image,
        save_image,
    )


def cv_bgr_to_qimage(bgr: np.ndarray) -> QtGui.QImage:
    if bgr is None:
        return QtGui.QImage()
    h, w, ch = bgr.shape
    bytes_per_line = ch * w
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)


def cv_gray_to_qimage(gray: np.ndarray) -> QtGui.QImage:
    if gray is None:
        return QtGui.QImage()
    h, w = gray.shape
    return QtGui.QImage(gray.data, w, h, w, QtGui.QImage.Format_Grayscale8)


def make_pixmap_from_cv(img: np.ndarray) -> QtGui.QPixmap:
    if img.ndim == 2:
        qimg = cv_gray_to_qimage(img)
    else:
        qimg = cv_bgr_to_qimage(img)
    return QtGui.QPixmap.fromImage(qimg)


def plot_hist_on_pixmap(hist: np.ndarray, size: Tuple[int, int] = (400, 150)) -> QtGui.QPixmap:
    w, h = size
    pix = QtGui.QPixmap(w, h)
    pix.fill(QtCore.Qt.white)
    painter = QtGui.QPainter(pix)
    painter.setPen(QtGui.QPen(QtCore.Qt.black))
    if hist is not None and len(hist) > 0:
        max_val = float(np.max(hist)) if np.max(hist) > 0 else 1.0
        bin_w = w / 256.0
        for i in range(256):
            val = float(hist[i]) / max_val
            bar_h = int(val * (h - 10))
            painter.drawLine(int(i * bin_w), h - 1, int(i * bin_w), h - 1 - bar_h)
    painter.end()
    return pix

def label_hist_pixmap(label: QtWidgets.QLabel, hist: np.ndarray, default_height: int = 150) -> QtGui.QPixmap:
    target_w = max(label.width(), 200)
    return plot_hist_on_pixmap(hist, (target_w, default_height))


class ImagePanel(QtWidgets.QWidget):
    def __init__(self, title: str):
        super().__init__()
        self._orig_pixmap: Optional[QtGui.QPixmap] = None
        self.title_label = QtWidgets.QLabel(title)
        self.title_label.setStyleSheet("font-weight: bold;")
        self.top_label = QtWidgets.QLabel()
        self.top_label.setAlignment(QtCore.Qt.AlignCenter)
        self.hist_label = QtWidgets.QLabel()
        self.hist_label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter)
        self.bottom_widget = QtWidgets.QWidget()
        self.bottom_layout = QtWidgets.QVBoxLayout(self.bottom_widget)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.title_label)
        layout.addWidget(self.top_label, stretch=3)
        layout.addWidget(self.hist_label, stretch=1)
        layout.addWidget(self.bottom_widget)

    def set_top_pixmap(self, pixmap: QtGui.QPixmap):
        self._orig_pixmap = pixmap
        self._apply_scaled_pixmap()

    def _apply_scaled_pixmap(self):
        if self._orig_pixmap and not self._orig_pixmap.isNull():
            target_size = self.top_label.size()
            if target_size.width() > 0 and target_size.height() > 0:
                scaled = self._orig_pixmap.scaled(
                    target_size,
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation,
                )
                self.top_label.setPixmap(scaled)
        else:
            self.top_label.clear()

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        self._apply_scaled_pixmap()

class ROISelectionDialog(QtWidgets.QDialog):
    def __init__(self, image_bgr: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Region of Interest (ROI)")
        self.image_bgr = image_bgr.copy()
        self.display_bgr = image_bgr.copy()
        self.points = []
        layout = QtWidgets.QVBoxLayout(self)
        self.image_label = QtWidgets.QLabel()
        self.image_label.setCursor(QtCore.Qt.CrossCursor)
        self.image_label.mousePressEvent = self._on_mouse_press
        layout.addWidget(self.image_label, alignment=QtCore.Qt.AlignCenter)
        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_reset = QtWidgets.QPushButton("Reset")
        self.btn_reset.clicked.connect(self._reset_points)
        self.btn_full = QtWidgets.QPushButton("Use Full Image")
        self.btn_full.clicked.connect(self._use_full_image)
        self.btn_apply = QtWidgets.QPushButton("Apply ROI")
        self.btn_apply.setStyleSheet("font-weight: bold;")
        self.btn_apply.clicked.connect(self.accept)
        btn_layout.addWidget(self.btn_reset)
        btn_layout.addWidget(self.btn_full)
        btn_layout.addWidget(self.btn_apply)
        layout.addLayout(btn_layout)
        self._update_display()

    def _update_display(self):
        h, w = self.image_bgr.shape[:2]
        max_w, max_h = 800, 600
        scale = min(max_w / w, max_h / h, 1.0)
        self.disp_w = max(1, int(w * scale))
        self.disp_h = max(1, int(h * scale))
        qimg = cv_bgr_to_qimage(self.display_bgr)
        pixmap = QtGui.QPixmap.fromImage(qimg).scaled(
            self.disp_w, self.disp_h,
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)
        self.image_label.setFixedSize(pixmap.size())

    def _on_mouse_press(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.LeftButton:
            pos = event.pos()
            orig_h, orig_w = self.image_bgr.shape[:2]
            real_x = int(pos.x() * (orig_w / self.disp_w))
            real_y = int(pos.y() * (orig_h / self.disp_h))
            real_x = max(0, min(orig_w - 1, real_x))
            real_y = max(0, min(orig_h - 1, real_y))
            self.points.append((real_x, real_y))
            self._redraw_polygon()

    def _redraw_polygon(self):
        self.display_bgr = self.image_bgr.copy()
        if len(self.points) > 0:
            pts = np.array(self.points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(self.display_bgr, [pts], isClosed=(len(self.points) > 2), color=(0, 255, 0), thickness=2)
            for pt in self.points:
                cv2.circle(self.display_bgr, pt, 5, (0, 0, 255), -1)
                
        self._update_display()

    def _reset_points(self):
        self.points = []
        self.display_bgr = self.image_bgr.copy()
        self._update_display()

    def _use_full_image(self):
        self.points = []
        self.accept()

    def get_mask(self) -> np.ndarray:
        h, w = self.image_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if len(self.points) >= 3:
            pts = np.array(self.points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)
        else:
            mask.fill(255)  # If fewer than 3 points, return full image mask
            
        return mask

class BijelsAreaApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bijels Area Estimation")
        self.resize(1300, 900)

        self.image_path: Optional[str] = None
        self.input_bgr: Optional[np.ndarray] = None
        self.params = ProcessingParams()
        self.results = None
        self.resolution_nm: float = 0.1202
        self.params.overlay_color = (0, 0, 255)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        vroot = QtWidgets.QVBoxLayout(central)
        vroot.setSpacing(8)

        self.panel_input = ImagePanel("1. Input Image")
        self.panel_eq = ImagePanel("2. Equalized Image")
        self.panel_smooth = ImagePanel("3. Smoothen Image")
        self.panel_quant = ImagePanel("4. Quantized Image")
        self.panel_final = ImagePanel("5. Final Image")

        images_row = QtWidgets.QHBoxLayout()
        images_row.setSpacing(10)
        images_row.addWidget(self.panel_input, stretch=1)
        images_row.addWidget(self.panel_eq, stretch=1)
        images_row.addWidget(self.panel_smooth, stretch=1)
        images_row.addWidget(self.panel_quant, stretch=1)
        images_row.addWidget(self.panel_final, stretch=1)
        vroot.addLayout(images_row, stretch=3)

        info_row = QtWidgets.QHBoxLayout()
        info_row.setSpacing(10)

        self._setup_panel_input_bottom()
        info_row.addWidget(self.panel_input.bottom_widget, stretch=1, alignment=QtCore.Qt.AlignTop)

        self._setup_panel_eq_bottom()
        info_row.addWidget(self.panel_eq.bottom_widget, stretch=1, alignment=QtCore.Qt.AlignTop)

        self._setup_panel_smooth_bottom()
        info_row.addWidget(self.panel_smooth.bottom_widget, stretch=1, alignment=QtCore.Qt.AlignTop)

        self._setup_panel_final_bottom()
        
        self._setup_panel_binar_bottom()
        info_row.addWidget(self.panel_quant.bottom_widget, stretch=1, alignment=QtCore.Qt.AlignTop)

        info_row.addWidget(self.panel_final.bottom_widget, stretch=1, alignment=QtCore.Qt.AlignTop)

        vroot.addLayout(info_row, stretch=2)

        self._setup_menu()
        self._update_ui()

    def _setup_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        load_act = QtWidgets.QAction("Load Image", self)
        load_act.triggered.connect(self.on_load_image)
        file_menu.addAction(load_act)
        save_act = QtWidgets.QAction("Save Final Image", self)
        save_act.triggered.connect(self.on_save_final)
        file_menu.addAction(save_act)

    def _setup_panel_input_bottom(self):
        self.input_info_label = QtWidgets.QLabel("No image loaded")
        self.panel_input.bottom_layout.addWidget(self.input_info_label)

    def _setup_panel_eq_bottom(self):
        self.eq_checkbox = QtWidgets.QCheckBox("Use Histogram Equalization")
        self.eq_checkbox.setChecked(self.params.use_hist_eq)
        self.eq_checkbox.stateChanged.connect(self.on_params_changed)
        self.panel_eq.bottom_layout.addWidget(self.eq_checkbox)

    def _setup_panel_smooth_bottom(self):
        self.kernel_combo = QtWidgets.QComboBox()
        for k in [3, 5, 7, 9]:
            self.kernel_combo.addItem(f"{k}x{k}", (k, k))
        self.kernel_combo.setCurrentIndex(1)
        self.kernel_combo.currentIndexChanged.connect(self.on_params_changed)

        self.sigma_x_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sigma_x_slider.setRange(0, 50)  # 0.0 to 5.0 in steps of 0.1
        self.sigma_x_slider.setValue(max(0, min(50, int(self.params.sigma_x * 10))))
        self.sigma_x_slider.valueChanged.connect(self.on_params_changed)
        self.sigma_x_value = QtWidgets.QLineEdit(f"{self.sigma_x_slider.value()/10.0:.1f}")
        self.sigma_x_value.setMaximumWidth(60)
        self.sigma_x_value.setValidator(QtGui.QDoubleValidator(0.0, 5.0, 1))
        self.sigma_x_value.editingFinished.connect(self._on_sigma_x_text_changed)

        self.sigma_y_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sigma_y_slider.setRange(0, 50)  # 0.0 to 5.0 in steps of 0.1
        self.sigma_y_slider.setValue(max(0, min(50, int(self.params.sigma_y * 10))))
        self.sigma_y_slider.valueChanged.connect(self.on_params_changed)
        self.sigma_y_value = QtWidgets.QLineEdit(f"{self.sigma_y_slider.value()/10.0:.1f}")
        self.sigma_y_value.setMaximumWidth(60)
        self.sigma_y_value.setValidator(QtGui.QDoubleValidator(0.0, 5.0, 1))
        self.sigma_y_value.editingFinished.connect(self._on_sigma_y_text_changed)

        self.gaussian_checkbox = QtWidgets.QCheckBox("Use Gaussian Smoothing")
        self.gaussian_checkbox.setChecked(self.params.use_gaussian)
        self.gaussian_checkbox.stateChanged.connect(self.on_params_changed)

        form = QtWidgets.QFormLayout()
        form.addRow("Kernel Size", self.kernel_combo)
        sx_row = QtWidgets.QHBoxLayout()
        sx_row.addWidget(self.sigma_x_slider)
        sx_row.addWidget(self.sigma_x_value)
        sy_row = QtWidgets.QHBoxLayout()
        sy_row.addWidget(self.sigma_y_slider)
        sy_row.addWidget(self.sigma_y_value)
        form.addRow("Sigma X (0-5)", self._wrap_layout_widget(sx_row))
        form.addRow("Sigma Y (0-5)", self._wrap_layout_widget(sy_row))
        form.addRow(self.gaussian_checkbox)
        self.panel_smooth.bottom_layout.addLayout(form)

    def _setup_panel_binar_bottom(self):
        self.binar_combo = QtWidgets.QComboBox()
        self.panel_quant.bottom_layout.addWidget(self.binar_combo)
        
        self.algorithm_list = []
        self._setup_panel_binar_bottom_manual()
        self._setup_panel_binar_bottom_otsu()
        self._setup_panel_binar_bottom_local()
        
        self.binar_combo.setCurrentIndex(0)
        self._update_binar_bottom_panel()
        
        self.binar_combo.currentIndexChanged.connect(self.on_params_changed)
        

        
    def _setup_panel_binar_bottom_manual(self):
        self.binar_combo.addItem("manual", "manual")
        
        self.manual_text = QtWidgets.QLabel("Quantization Threshold")
        
        self.quant_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.quant_slider.setRange(0, 255)
        self.quant_slider.setValue(128)
        self.params.threshold_info["quant_threshold"] = 128
        self.quant_slider.valueChanged.connect(self.on_params_changed)

        self.quant_value = QtWidgets.QLineEdit(str(self.quant_slider.value()))
        self.quant_value.setMaximumWidth(60)
        self.quant_value.setValidator(QtGui.QIntValidator(0, 255))
        self.quant_value.editingFinished.connect(self._on_quant_text_changed)
        
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.quant_slider)
        row.addWidget(self.quant_value)
        self.manual_input = self._wrap_layout_widget(row)
        
        self.panel_binar_manual = QtWidgets.QFrame()
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.manual_text)
        layout.addWidget(self.manual_input)
        self.panel_binar_manual.setLayout(layout)
        
        self.panel_quant.bottom_layout.addWidget(self.panel_binar_manual)
        self.algorithm_list.append(["manual", self.panel_binar_manual])


    def _setup_panel_binar_bottom_otsu(self):
        self.binar_combo.addItem("Otsu", "Otsu")
        
        self.otsu_label = QtWidgets.QLabel("N/A")
        self.otsu_text = QtWidgets.QLabel("Otsu calculated Threshold")
        
        self.panel_binar_otsu = QtWidgets.QFrame()
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.otsu_text)
        layout.addWidget(self.otsu_label)
        self.panel_binar_otsu.setLayout(layout)
        
        self.panel_quant.bottom_layout.addWidget(self.panel_binar_otsu)
        self.algorithm_list.append(["Otsu", self.panel_binar_otsu])
        
        
        
    def _setup_panel_binar_bottom_local(self):
        self.binar_combo.addItem("local", "local")
        
        try:
            local_block_start = self.params.threshold_info["local_block"]
        except (KeyError):
            local_block_start = 3
            self.params.threshold_info["local_block"] = local_block_start
        self.local_block_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.local_block_slider.setRange(1, 250)  # 3 to 501 in steps of 2

        self.local_block_slider.setValue(int((local_block_start-1) / 2))
        self.local_block_slider.valueChanged.connect(self.on_params_changed)
        
        self.local_block_value = QtWidgets.QLineEdit(f"{local_block_start:3d}")
        self.local_block_value.setMaximumWidth(60)
        self.local_block_value.setValidator(QtGui.QDoubleValidator(3, 501, 0,  
                                            notation=QtGui.QDoubleValidator.StandardNotation))
        self.local_block_value.editingFinished.connect(self._on_local_block_text_changed)
        
        lb_talker = QtWidgets.QHBoxLayout()
        lb_talker.addWidget(self.local_block_slider)
        lb_talker.addWidget(self.local_block_value)
        lb_row = QtWidgets.QHBoxLayout()
        lb_row.addWidget(QtWidgets.QLabel("Block size"))
        lb_row.addWidget(self._wrap_layout_widget(lb_talker))
        
        try:
            local_offset_start = self.params.threshold_info["local_offset"]
        except (KeyError):
            local_offset_start = 0
            self.params.threshold_info["local_offset"] = local_offset_start
        self.local_offset_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.local_offset_slider.setRange(-10, 10)  # -10 to 10

        self.local_offset_slider.setValue(int(local_offset_start))
        self.local_offset_slider.valueChanged.connect(self.on_params_changed)
        
        self.local_offset_value = QtWidgets.QLineEdit(f"{local_offset_start:3d}")
        self.local_offset_value.setMaximumWidth(60)
        self.local_offset_value.setValidator(QtGui.QDoubleValidator(-10, 10, 0,  
                                            notation=QtGui.QDoubleValidator.StandardNotation))
        self.local_offset_value.editingFinished.connect(self._on_local_offset_text_changed)                
        
        lo_talker = QtWidgets.QHBoxLayout()
        lo_talker.addWidget(self.local_offset_slider)
        lo_talker.addWidget(self.local_offset_value)
        lo_row = QtWidgets.QHBoxLayout()
        lo_row.addWidget(QtWidgets.QLabel("Offset"))
        lo_row.addWidget(self._wrap_layout_widget(lo_talker))
        
        self.local_text = QtWidgets.QLabel("Local calculated Threshold, Otsu algorithm")

        self.panel_binar_local = QtWidgets.QFrame()
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.local_text)
        layout.addLayout(lb_row)
        layout.addLayout(lo_row)
        self.panel_binar_local.setLayout(layout)
        self.panel_quant.bottom_layout.addWidget(self.panel_binar_local)
        self.algorithm_list.append(["local", self.panel_binar_local])

    def _update_binar_bottom_panel(self):
        for algo, panel in self.algorithm_list:
            if algo == self.binar_combo.currentData():
                panel.show()
            else:
                panel.hide()
            
    def _setup_panel_final_bottom(self):
        self.update_button = QtWidgets.QPushButton("Force Update Final")
        self.update_button.clicked.connect(self._force_update)
        self.update_checkbox = QtWidgets.QCheckBox("Automatic Updating")
        self.update_checkbox.setChecked(self.params.auto_update)
        self.update_checkbox.stateChanged.connect(self.on_params_changed)
        self.resolution_input = QtWidgets.QLineEdit("0.1202")
        self.resolution_input.setValidator(QtGui.QDoubleValidator(0.0, 1e9, 6))
        self.resolution_input.editingFinished.connect(self.on_params_changed)
        # Overlay choice: choose which mask value to overlay
        self.overlay_zero_radio = QtWidgets.QRadioButton("Overlay mask==0")
        self.overlay_255_radio = QtWidgets.QRadioButton("Overlay mask==255")
        self.overlay_255_radio.setChecked(True)
        self.overlay_zero_radio.toggled.connect(self.on_params_changed)
        # Color selector for overlay
        self.overlay_color_button = QtWidgets.QPushButton("Overlay Color")
        self.overlay_color_button.clicked.connect(self._choose_overlay_color)
        # Show current color swatch
        self.overlay_color_swatch = QtWidgets.QLabel()
        self.overlay_color_swatch.setFixedSize(24, 24)
        self._update_overlay_swatch()
        self.stats_label = QtWidgets.QLabel("Stats: N/A")
        form = QtWidgets.QFormLayout()
        form.addRow(self.update_checkbox, self.update_button)
        form.addRow("Pixel Resolution (µm)", self.resolution_input)
        overlay_row = QtWidgets.QHBoxLayout()
        overlay_row.addWidget(self.overlay_zero_radio)
        overlay_row.addWidget(self.overlay_255_radio)
        form.addRow("Overlay Pixels", self._wrap_layout_widget(overlay_row))
        color_row = QtWidgets.QHBoxLayout()
        color_row.addWidget(self.overlay_color_button)
        color_row.addWidget(self.overlay_color_swatch)
        form.addRow("Overlay Color", self._wrap_layout_widget(color_row))
        form.addRow("Image Stats", self.stats_label)
        self.panel_final.bottom_layout.addLayout(form)

    def on_load_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.lif *.liff)")
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        if ext in [".lif", ".liff"]:
            try:
                with LifFile(path) as lif:
                    if len(lif.images) == 0:
                        QtWidgets.QMessageBox.warning(self, "Load Failed", "No series found in LIF file.")
                        return
                    image_names = []
                    for i, image in enumerate(lif.images):
                        try:
                            shape_str = "x".join(map(str, image.shape))
                        except Exception:
                            shape_str = "unknown shape"
                        series_name = getattr(image, "name", f"Series {i}")
                        image_names.append(f"{series_name} ({shape_str})")
                    item, ok = QtWidgets.QInputDialog.getItem(self, "Select Image", "Choose image series", image_names, 0, False)
                    if not ok:
                        return
                    selected_index = image_names.index(item)
                    image = lif.images[selected_index]
                    arr = image.asarray()
                    arr = np.squeeze(arr)  
                    if arr.dtype != np.uint8:
                        arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    if arr.ndim == 2:
                        img = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                    elif arr.ndim == 3 and arr.shape[-1] in [3, 4]:
                        base = arr[:, :, :3].copy()
                        mag = np.maximum(base[:, :, 0], base[:, :, 2])
                        img = cv2.merge([mag, np.zeros_like(mag, dtype=np.uint8), mag])
                    elif arr.ndim == 3 and arr.shape[0] in [2, 3, 4]:  
                        channels = []
                        for i in range(min(3, arr.shape[0])):
                            channel = arr[i]
                            channel = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                            channels.append(channel)
                        while len(channels) < 3:
                            channels.append(np.zeros_like(channels[0], dtype=np.uint8))
                        blue = channels[0]
                        red = channels[2] if len(channels) > 2 else np.zeros_like(blue, dtype=np.uint8)
                        mag = np.maximum(red, blue)
                        img = cv2.merge([mag, np.zeros_like(mag, dtype=np.uint8), mag])
                    else:
                        while arr.ndim > 2:
                            arr = arr[0]
                        img = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Load Failed", f"Could not load LIF file: {str(e)}")
                return
        else:
            img = load_image(path)
            if img is None:
                QtWidgets.QMessageBox.warning(self, "Load Failed", "Could not load image.")
                return
        roi_dialog = ROISelectionDialog(img, parent=self)
        if roi_dialog.exec_() == QtWidgets.QDialog.Accepted:
            mask = roi_dialog.get_mask()
            # Mask out region outside polygon (set to black background)
            img = cv2.bitwise_and(img, img, mask=mask)
        else:
            return
        self.image_path = path
        self.input_bgr = img
        self._run_pipeline()
        self._update_ui()

    def on_save_final(self):
        if self.results is None:
            QtWidgets.QMessageBox.information(self, "No Result", "Please load an image and update.")
            return
        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Final Image", "final_overlay.png", "PNG (*.png);;JPEG (*.jpg *.jpeg);;TIFF (*.tif *.tiff)"
        )
        if not save_path:
            return
        final_overlay = self.results[5]
        ok = save_image(save_path, final_overlay)
        if ok:
            QtWidgets.QMessageBox.information(self, "Saved", f"Saved to: {save_path}")
        else:
            QtWidgets.QMessageBox.warning(self, "Save Failed", "Could not save image.")

    def on_params_changed(self):
        self.params.make_gray = self.binar_combo.currentData()
        self.params.use_hist_eq = self.eq_checkbox.isChecked()
        self.params.auto_update = self.update_checkbox.isChecked()
        self.params.use_gaussian = self.gaussian_checkbox.isChecked()
        self.params.gaussian_kernel = self.kernel_combo.currentData()
        self.params.sigma_x = self.sigma_x_slider.value() / 10.0
        self.params.sigma_y = self.sigma_y_slider.value() / 10.0
        # Clamp to [0, 5]
        self.params.sigma_x = max(0.0, min(5.0, self.params.sigma_x))
        self.params.sigma_y = max(0.0, min(5.0, self.params.sigma_y))
        # Reflect slider changes to text boxes
        self.sigma_x_value.setText(f"{self.params.sigma_x:.1f}")
        self.sigma_y_value.setText(f"{self.params.sigma_y:.1f}")
        
        
        # Overlay mode: True for zeros, False for 255
        self.params.overlay_use_zero = self.overlay_zero_radio.isChecked()
        
        self.params.threshold_algorithm = self.binar_combo.currentData()
        self.params.threshold_info["local_block"] = int(self.local_block_slider.value()*2+1)
        self.local_block_value.setText(str(self.params.threshold_info["local_block"]))
        self.params.threshold_info["local_offset"] = int(self.local_offset_slider.value())
        self.local_offset_value.setText(str(self.params.threshold_info["local_offset"]))
        self.params.threshold_info["quant_threshold"] = self.quant_slider.value()
        self.quant_value.setText(str(self.params.threshold_info["quant_threshold"]))

        
        try:
            self.resolution_nm = float(self.resolution_input.text())
        except Exception:
            self.resolution_nm = 1.0
            
        #run pipeline only if auto_update is turned on 
        if self.params.auto_update:
            self._run_pipeline()
        self._update_ui()
        
    def _force_update(self):
        self._run_pipeline()
        self._update_ui()

    def _choose_overlay_color(self):
        # QColorDialog returns RGB; convert to BGR for OpenCV usage
        current_bgr = getattr(self.params, 'overlay_color', (0, 0, 255))
        current_rgb = QtGui.QColor(current_bgr[2], current_bgr[1], current_bgr[0])
        color = QtWidgets.QColorDialog.getColor(current_rgb, self, "Select Overlay Color")
        if color.isValid():
            r, g, b, _ = color.getRgb()
            self.params.overlay_color = (b, g, r)  # store as BGR
            self._update_overlay_swatch()
            self._run_pipeline()
            self._update_ui()

    def _update_overlay_swatch(self):
        # Update swatch label background to current overlay color
        bgr = getattr(self.params, 'overlay_color', (0, 0, 255))
        r, g, b = bgr[2], bgr[1], bgr[0]
        self.overlay_color_swatch.setStyleSheet(f"background-color: rgb({r}, {g}, {b}); border: 1px solid #666;")

    # Helpers for two-way binding
    def _wrap_layout_widget(self, layout: QtWidgets.QLayout) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        container.setLayout(layout)
        return container

    def _on_sigma_x_text_changed(self):
        try:
            val = float(self.sigma_x_value.text())
        except Exception:
            return
        val = max(0.0, min(5.0, val))
        self.sigma_x_slider.setValue(int(round(val * 10)))
        # Trigger pipeline update via on_params_changed
        self.on_params_changed()

    def _on_sigma_y_text_changed(self):
        try:
            val = float(self.sigma_y_value.text())
        except Exception:
            return
        val = max(0.0, min(5.0, val))
        self.sigma_y_slider.setValue(int(round(val * 10)))
        self.on_params_changed()
        
    def _on_local_block_text_changed(self):
        try:
            val = int(self.local_block_value.text())
        except Exception:
            return
        val = max(3, min(501, val))
        if val%2==1:
            val+=1
        self.local_block_slider.setValue(int((val-1)/2))
        self.on_params_changed()
        
    def _on_local_offset_text_changed(self):
        try:
            val = int(self.local_offset_value.text())
        except Exception:
            return
        val = max(-10, min(10, val))
        self.local_offset_slider.setValue(int(val))
        self.on_params_changed()

    def _on_quant_text_changed(self):
        try:
            val = int(self.quant_value.text())
        except Exception:
            return
        val = max(0, min(255, val))
        self.quant_slider.setValue(val)
        self.on_params_changed()

    def _run_pipeline(self):
        if self.input_bgr is None:
            self.results = None
            return
        self.results = process_image(self.input_bgr, self.params)

    def _update_ui(self):
        if self.input_bgr is not None and self.results is not None:
            input_color, gray, eq_gray, smooth_gray, quant_mask, final_overlay, stats = self.results

            self.panel_input.set_top_pixmap(make_pixmap_from_cv(input_color))
            input_hist_pix = label_hist_pixmap(self.panel_input.top_label, stats["hist_input_gray"]) 
            self.panel_input.hist_label.setPixmap(input_hist_pix)
            if self.image_path:
                h, w, _ = self.input_bgr.shape
                info = f"File: {os.path.basename(self.image_path)} | Size: {w} x {h}"
            else:
                info = "No image loaded"
            self.input_info_label.setText(info)

            self.panel_eq.set_top_pixmap(make_pixmap_from_cv(eq_gray))
            eq_hist_pix = label_hist_pixmap(self.panel_eq.top_label, stats["hist_eq_gray"]) 
            self.panel_eq.hist_label.setPixmap(eq_hist_pix)

            self.panel_smooth.set_top_pixmap(make_pixmap_from_cv(smooth_gray))
            maskpicture = np.asarray(255*quant_mask, dtype=np.ubyte)
            self.panel_quant.set_top_pixmap(make_pixmap_from_cv(maskpicture))
            if self.params.threshold_algorithm == "Otsu":
                self.otsu_label.setText(str(stats["threshold_used"]))
            self._update_binar_bottom_panel()
                
            self.panel_final.set_top_pixmap(make_pixmap_from_cv(final_overlay))
            total_pixels = stats["total_pixels"]
            count_0 = stats.get("count_0", 0)
            percent_0 = stats.get("percent_0", (count_0 / total_pixels * 100.0) if total_pixels else 0.0)
            # Derive 255 stats if not provided
            count_255 = total_pixels - count_0
            percent_255 = (count_255 / total_pixels * 100.0) if total_pixels else 0.0

            # Use overlay selection to drive displayed stats and area
            overlay_zeros = self.params.overlay_use_zero
            selected_label = "Quantized 0" if overlay_zeros else "Quantized 255"
            selected_count = count_0 if overlay_zeros else count_255
            selected_percent = percent_0 if overlay_zeros else percent_255
            surface_area_um2 = selected_count * (self.resolution_nm ** 2)
            stats_text = (
                f"{selected_label}: {selected_count} | %: {selected_percent:.2f}% | "
                f"Surface area: {surface_area_um2:.4f} µm^2"
            )
            self.stats_label.setText(stats_text)
        else:
            self.panel_input.set_top_pixmap(QtGui.QPixmap())
            self.panel_input.hist_label.clear()
            self.input_info_label.setText("No image loaded")
            self.panel_eq.set_top_pixmap(QtGui.QPixmap())
            self.panel_eq.hist_label.clear()
            self.panel_smooth.set_top_pixmap(QtGui.QPixmap())
            self.panel_quant.set_top_pixmap(QtGui.QPixmap())
            self.panel_final.set_top_pixmap(QtGui.QPixmap())
            self.stats_label.setText("Stats: N/A")


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = BijelsAreaApp()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
