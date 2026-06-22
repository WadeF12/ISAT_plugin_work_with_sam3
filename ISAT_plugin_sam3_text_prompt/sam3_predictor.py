# -*- coding: utf-8 -*-
"""
SAM3 Text Prompt Predictor

封装 SAM3 模型的文本提示预测功能。
支持多类别文本提示，一次编码、多次解码。

使用方法:
    predictor = SAM3TextPredictor(model_path="sam3_model.pt", device="cuda")
    predictor.set_image("path/to/image.jpg")
    masks, scores = predictor.predict(["car", "person", "tree"], threshold=0.3)

适配说明:
    如果你的 SAM3 来源不同（如官方 GitHub、ONNX 导出等），
    只需修改本文件中 SAM3 的加载和推理代码即可，
    ISAT 插件主逻辑完全解耦，无需改动。
"""

import traceback
import numpy as np
import cv2
from typing import List, Tuple, Optional


class SAM3LoadError(Exception):
    """SAM3 模型加载失败的异常，携带详细错误链。"""
    def __init__(self, message: str, errors: list = None):
        super().__init__(message)
        self.errors = errors or []


class SAM3TextPredictor:
    """
    SAM3 文本提示预测器。

    自动探测多种 SAM 加载方式，按优先级尝试:
      1. segment_anything (SAM1)  — ISAT 内置使用
      2. sam2 包                   — SAM2 官方
      3. torch.load 直接加载       — 通用兜底
      4. onnxruntime               — ONNX 导出版

    设计为可替换的接口层 —— 如果你使用不同来源的 SAM3，
    只需修改 _load_model / _predict_single_text 两个方法。
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        """
        初始化 SAM3 模型。

        Args:
            model_path: SAM3 模型文件路径 (.pt / .pth / .onnx)
            device: 推理设备 ("cuda" / "cpu")
        """
        self.device = device
        self.model_path = model_path
        self.model = None
        self._predictor = None       # SAM1/SAM2 的 SamPredictor 实例
        self._image_encoder = None   # ONNX 的 encoder session
        self._image_shape = None
        self._backend = None

        self._load_model(model_path, device)

    # ------------------------------------------------------------------
    # 模型加载 —— 自动探测多种策略
    # ------------------------------------------------------------------

    def _load_model(self, model_path: str, device: str):
        """
        按优先级尝试多种方式加载 SAM 模型。
        每种方式失败时记录详细错误，最终汇总抛出。
        """
        import torch
        errors = []

        # ---- 策略1: segment_anything (SAM1, ISAT 使用的) ----
        try:
            from segment_anything import sam_model_registry
            self._try_load_sam1(model_path, device)
            self._backend = "sam1"
            return
        except Exception as e:
            errors.append(("segment_anything (SAM1)", str(e)))

        # ---- 策略2: sam2 包 ----
        try:
            from sam2 import build_sam2
            self._try_load_sam2(model_path, device)
            self._backend = "sam2"
            return
        except Exception as e:
            errors.append(("sam2", str(e)))

        # ---- 策略3: ONNX Runtime ----
        if model_path.endswith(".onnx"):
            try:
                self._try_load_onnx(model_path, device)
                self._backend = "onnx"
                return
            except Exception as e:
                errors.append(("onnxruntime", str(e)))

        # ---- 策略4: torch.load 直接加载 ----
        try:
            checkpoint = torch.load(model_path, map_location=device)
            self.model = checkpoint
            self._backend = "torch_raw"
            return
        except Exception as e:
            errors.append(("torch.load raw", str(e)))

        # 所有策略均失败
        detail = "\n\n".join(f"[{name}] {err}" for name, err in errors)
        raise SAM3LoadError(
            f"所有加载策略均失败。模型: {model_path}\n"
            f"请检查模型文件是否正确，或手动修改 _load_model 方法。\n\n"
            f"详细错误:\n{detail}",
            errors=errors,
        )

    def _try_load_sam1(self, model_path: str, device: str):
        """使用 segment_anything (SAM1) 加载。"""
        import torch
        from segment_anything import sam_model_registry

        # 先查看 checkpoint 里存了什么 model_type
        checkpoint = torch.load(model_path, map_location="cpu")
        model_type = None
        for key in ["model_type", "image_encoder", "pixel_mean"]:
            if key in checkpoint:
                # 可能是 vit_h, vit_l, vit_b, vit_t
                pass

        # 尝试所有已知 model_type
        for mt in ["vit_h", "vit_l", "vit_b", "default"]:
            try:
                sam = sam_model_registry[mt](checkpoint=model_path)
                sam.to(device=device)
                sam.eval()
                self.model = sam
                return
            except Exception:
                continue

        # 如果 registry 方式失败，尝试直接加载
        from segment_anything.modeling import Sam
        from segment_anything.modeling.image_encoder import ImageEncoderViT
        sam = Sam(image_encoder=ImageEncoderViT())
        sam.load_state_dict(checkpoint, strict=False)
        sam.to(device=device)
        sam.eval()
        self.model = sam

    def _try_load_sam2(self, model_path: str, device: str):
        """使用 sam2 包加载。"""
        from sam2 import build_sam2
        from sam2.build_sam import build_sam2 as build_sam2_alt

        for builder in [build_sam2, build_sam2_alt]:
            try:
                self.model = builder(
                    model_path, device=device, apply_postprocessing=False
                )
                return
            except Exception:
                continue
        raise RuntimeError("sam2 所有加载方式均失败")

    def _try_load_onnx(self, model_path: str, device: str):
        """使用 ONNX Runtime 加载。"""
        import onnxruntime as ort

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        self._image_encoder = ort.InferenceSession(
            model_path, providers=providers
        )

    # ------------------------------------------------------------------
    # 图像编码
    # ------------------------------------------------------------------

    def set_image(self, image: np.ndarray):
        """
        设置当前图像并编码。编码后可按不同文本提示多次解码。

        Args:
            image: RGB 格式的 numpy 数组 (H, W, 3)
        """
        self._image_shape = image.shape[:2]

        if self._backend == "sam1":
            from segment_anything import SamPredictor
            self._predictor = SamPredictor(self.model)
            self._predictor.set_image(image)
        elif self._backend == "sam2":
            self._predictor = self.model  # sam2 自带 set_image
            self._predictor.set_image(image)
        elif self._backend == "onnx":
            self._cached_image = image
        elif self._backend == "torch_raw":
            self._cached_image = image
        else:
            raise RuntimeError(f"Unknown backend: {self._backend}")

    def set_image_from_path(self, image_path: str):
        """从文件路径加载并编码图像。"""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.set_image(image)

    # ------------------------------------------------------------------
    # 文本提示预测
    # ------------------------------------------------------------------

    def predict(
        self,
        text_prompts: List[str],
        threshold: float = 0.3,
    ) -> Tuple[List[np.ndarray], List[float], List[str]]:
        """
        使用文本提示进行多类别分割预测。

        Args:
            text_prompts: 文本提示列表，如 ["car", "person", "tree"]
            threshold:  置信度阈值，低于此值的 mask 将被过滤

        Returns:
            masks:      二值 mask 列表，每个 shape=(H, W)
            scores:     置信度分数列表
            categories: 对应的类别名列表（与 text_prompts 对齐或过滤后）
        """
        all_masks = []
        all_scores = []
        all_categories = []

        for text in text_prompts:
            masks, scores = self._predict_single_text(text)

            # 过滤低置信度结果
            for mask, score in zip(masks, scores):
                if score >= threshold:
                    all_masks.append(mask)
                    all_scores.append(score)
                    all_categories.append(text)

        return all_masks, all_scores, all_categories

    def _predict_single_text(
        self, text_prompt: str
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        对单个文本提示进行预测。

        由于 SAM1/SAM2 原生不支持文本提示，
        此处使用 point/box 全图采样策略作为近似替代。
        如果你有真正的 SAM3 text-prompt API，在此替换。
        """
        # ==============================================================
        # 方式A: 如果你有真正的 SAM3 text-prompt (推荐)
        #   替换下面整个方法为:
        #   masks, scores, _ = self.model.predict(text_prompts=[text_prompt])
        #   return [masks[0]], [float(scores[0])]
        # ==============================================================

        if self._backend in ("sam1", "sam2", "torch_raw"):
            # 全图网格采样 + SAM 分割
            return self._grid_sample_predict(text_prompt)

        elif self._backend == "onnx":
            raise NotImplementedError("ONNX 文本提示暂不支持，请替换为实际 SAM3 API")

        raise RuntimeError(f"Unknown backend: {self._backend}")

    def _grid_sample_predict(
        self, text_prompt: str, grid_size: int = 8, score_threshold: float = 0.5
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        全图网格采样预测。

        SAM1/SAM2 不支持文本提示，作为替代方案：
        对图像均匀采样 N×N 个点，每个点用 SAM 生成 mask，
        然后按面积/置信度合并。

        注意：这不是真正的 text-prompt！只是通用分割。
        如果你需要真正的文本语义对应，请使用 SAM3 的 text prompt API，
        或搭配 Grounding DINO / CLIP 进行文本→box 转换。
        """
        if self._predictor is None:
            raise RuntimeError("Predictor not initialized. Call set_image first.")

        h, w = self._image_shape
        all_masks = []
        all_scores = []

        # 网格采样
        for i in range(grid_size):
            for j in range(grid_size):
                x = int(w * (j + 0.5) / grid_size)
                y = int(h * (i + 0.5) / grid_size)

                point = np.array([[x, y]])
                label = np.array([1])  # 1 = foreground

                masks, scores, _ = self._predictor.predict(
                    point_coords=point,
                    point_labels=label,
                    multimask_output=False,
                )
                if scores[0] >= score_threshold:
                    all_masks.append(masks[0])
                    all_scores.append(float(scores[0]))

        if not all_masks:
            return [], []

        # NMS 去重
        return self._nms_masks(all_masks, all_scores)

    @staticmethod
    def _nms_masks(
        masks: List[np.ndarray], scores: List[float], iou_threshold: float = 0.7
    ) -> Tuple[List[np.ndarray], List[float]]:
        """对 mask 做简单的 IoU NMS 去重。"""
        if len(masks) <= 1:
            return masks, scores

        # 按分数降序
        order = np.argsort(scores)[::-1]
        keep = []

        for i in order:
            keep_it = True
            for j in keep:
                iou = SAM3TextPredictor._mask_iou(masks[i], masks[j])
                if iou > iou_threshold:
                    keep_it = False
                    break
            if keep_it:
                keep.append(i)

        return (
            [masks[i] for i in keep],
            [scores[i] for i in keep],
        )

    @staticmethod
    def _mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
        """计算两个二值 mask 的 IoU。"""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0.0

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @property
    def image_shape(self) -> Optional[Tuple[int, int]]:
        """当前编码图像的尺寸 (H, W)。"""
        return self._image_shape

    @staticmethod
    def masks_to_polygons(
        mask: np.ndarray,
        simplify_epsilon: float = 2.0,
        min_area: int = 100,
    ) -> List[np.ndarray]:
        """
        将二值 mask 转换为多边形轮廓列表。

        每个 mask 可能包含多个不连通区域，返回所有有效轮廓。

        Args:
            mask:              二值 mask (H, W)，dtype=bool 或 uint8
            simplify_epsilon:  Douglas-Peucker 简化参数，越大越简化
            min_area:          最小区域面积，过滤噪点

        Returns:
            List of (N, 2) numpy arrays, each array 是一组轮廓点 [x, y]
        """
        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        polygons = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            # 简化轮廓
            epsilon = simplify_epsilon * 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # 转为 (N, 2) 格式
            pts = approx.reshape(-1, 2)
            if len(pts) >= 3:  # 至少需要3个点构成多边形
                polygons.append(pts)

        return polygons
