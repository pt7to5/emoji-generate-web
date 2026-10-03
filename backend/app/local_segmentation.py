from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
import tempfile

import numpy as np
import cv2
from PIL import Image

from .config import DATA_DIR


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "mobile_sam.pt"
MAX_INFERENCE_SIDE = 1536
os.environ.setdefault("YOLO_CONFIG_DIR", str(DATA_DIR))

_model = None
_model_lock = Lock()


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                if not MODEL_PATH.exists():
                    raise RuntimeError("本地分割模型尚未安装")
                from ultralytics import SAM
                _model = SAM(str(MODEL_PATH))
    return _model


def _sample_points(points: list[list[float]], limit: int = 24) -> list[list[float]]:
    """用最远点采样覆盖整条涂抹轨迹，避免密集区域挤掉主体末端。"""
    values = np.asarray(points, dtype=np.float32)[:, :2]
    if len(values) <= limit:
        return values.astype(float).tolist()
    center = np.median(values, axis=0)
    selected = [int(np.argmin(np.square(values - center).sum(axis=1)))]
    distances = np.square(values - values[selected[0]]).sum(axis=1)
    while len(selected) < limit:
        index = int(np.argmax(distances))
        selected.append(index)
        distances = np.minimum(distances, np.square(values - values[index]).sum(axis=1))
    return values[selected].astype(float).tolist()


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if not len(xs):
        raise RuntimeError("分割结果为空")
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _expanded_context_box(mask: np.ndarray, radius: float) -> list[int]:
    """框只提示向哪里寻找，最终边缘仍由模型判断。"""
    height, width = mask.shape
    x1, y1, x2, y2 = _mask_bbox(mask)
    object_width, object_height = x2 - x1, y2 - y1
    pad_x = max(radius * 2, object_width * .22)
    pad_y = max(radius * 2, object_height * .38)
    return [max(0, round(x1 - pad_x)), max(0, round(y1 - pad_y)),
            min(width - 1, round(x2 + pad_x)), min(height - 1, round(y2 + pad_y))]


def _stroke_context_box(points: list[list[float]], radius: float, size: tuple[int, int]) -> list[int]:
    """首次点提示失败时，根据涂抹的大概范围给模型第二次机会。"""
    values = np.asarray(points, dtype=np.float32)
    # 使用中间 70% 轨迹估计意图范围，避免一两个明显涂出界的点把搜索框拉到整张图。
    x1, y1 = np.quantile(values, .15, axis=0)
    x2, y2 = np.quantile(values, .85, axis=0)
    span_x, span_y = max(radius * 2, x2 - x1), max(radius * 2, y2 - y1)
    pad_x, pad_y = max(radius * 2, span_x * .55), max(radius * 2, span_y * .55)
    width, height = size
    return [max(0, round(x1 - pad_x)), max(0, round(y1 - pad_y)),
            min(width - 1, round(x2 + pad_x)), min(height - 1, round(y2 + pad_y))]


def _point_coverage(mask: np.ndarray, points: list[list[float]]) -> int:
    height, width = mask.shape
    return sum(bool(mask[max(0, min(height - 1, round(y))), max(0, min(width - 1, round(x)))]) for x, y in points)


def _valid_candidate(mask: np.ndarray) -> bool:
    area_ratio = float(mask.mean())
    if area_ratio <= .0002 or area_ratio >= .72:
        return False
    touches = int(mask[0].any()) + int(mask[-1].any()) + int(mask[:, 0].any()) + int(mask[:, -1].any())
    return touches < 3


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    gap_x = max(0, a[0] - b[2], b[0] - a[2])
    gap_y = max(0, a[1] - b[3], b[1] - a[3])
    return float(np.hypot(gap_x, gap_y))


def _clean_subject_mask(mask: np.ndarray) -> np.ndarray:
    """删除孤立碎点并填补小孔洞，同时保留杯脚、装饰等邻近有效部件。"""
    binary = mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    anchor_label = int(np.argmax(areas)) + 1
    anchor_area = int(stats[anchor_label, cv2.CC_STAT_AREA])
    anchor_box = (
        int(stats[anchor_label, cv2.CC_STAT_LEFT]),
        int(stats[anchor_label, cv2.CC_STAT_TOP]),
        int(stats[anchor_label, cv2.CC_STAT_LEFT] + stats[anchor_label, cv2.CC_STAT_WIDTH]),
        int(stats[anchor_label, cv2.CC_STAT_TOP] + stats[anchor_label, cv2.CC_STAT_HEIGHT]),
    )
    join_distance = max(10.0, max(anchor_box[2] - anchor_box[0], anchor_box[3] - anchor_box[1]) * .055)
    cleaned = np.zeros_like(binary)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        box = (
            int(stats[label, cv2.CC_STAT_LEFT]), int(stats[label, cv2.CC_STAT_TOP]),
            int(stats[label, cv2.CC_STAT_LEFT] + stats[label, cv2.CC_STAT_WIDTH]),
            int(stats[label, cv2.CC_STAT_TOP] + stats[label, cv2.CC_STAT_HEIGHT]),
        )
        if label == anchor_label or (area >= max(32, anchor_area * .006) and _bbox_gap(anchor_box, box) <= join_distance):
            cleaned[labels == label] = 1

    # 先封住与外部相连的狭窄漏选通道；尺度随图片变化，并设置上限避免粘连远处物体。
    kernel_size = min(121, max(5, round(min(cleaned.shape) * .04)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    # 再填中小型孔洞；大面积真实镂空（例如杯把、环形物体中心）继续保留。
    inverse = 1 - cleaned
    hole_count, hole_labels, hole_stats, _ = cv2.connectedComponentsWithStats(inverse, connectivity=8)
    # 奶油、高光、透明材质容易在完整主体内部形成较大的漏选洞。
    # 8% 以下的封闭区域按漏选处理；甜甜圈、杯把等大型真实孔洞仍保留。
    hole_limit = max(96, int(cleaned.sum() * .08))
    height, width = cleaned.shape
    for label in range(1, hole_count):
        left, top = int(hole_stats[label, cv2.CC_STAT_LEFT]), int(hole_stats[label, cv2.CC_STAT_TOP])
        right = left + int(hole_stats[label, cv2.CC_STAT_WIDTH])
        bottom = top + int(hole_stats[label, cv2.CC_STAT_HEIGHT])
        touches_border = left == 0 or top == 0 or right == width or bottom == height
        if not touches_border and int(hole_stats[label, cv2.CC_STAT_AREA]) <= hole_limit:
            cleaned[hole_labels == label] = 1
    return cleaned.astype(bool)


def _predict_masks(model, image_path: Path, *, points=None, labels=None, bboxes=None) -> list[np.ndarray]:
    results = model.predict(str(image_path), points=points, labels=labels, bboxes=bboxes, retina_masks=True, verbose=False)
    if not results or results[0].masks is None:
        return []
    return [candidate > .5 for candidate in results[0].masks.data.detach().cpu().numpy()]


def segment_from_positive_strokes(image_path: Path, points: list[list[float]], radius: float = 18) -> Image.Image:
    """把涂抹当作意图提示，先定位主体，再向外搜索其完整语义边缘。"""
    prompts = _sample_points(points)
    if not prompts:
        raise ValueError("涂抹不能为空")
    model = _get_model()
    with Image.open(image_path) as source:
        image_size = source.size
        inference_size = image_size
        inference_path = image_path
        temporary_path: Path | None = None
        if max(image_size) > MAX_INFERENCE_SIDE:
            inference = source.convert("RGB")
            inference.thumbnail((MAX_INFERENCE_SIDE, MAX_INFERENCE_SIDE), Image.Resampling.LANCZOS)
            inference_size = inference.size
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temporary:
                temporary_path = Path(temporary.name)
            inference.save(temporary_path, "JPEG", quality=90)
            inference_path = temporary_path

    scale_x = inference_size[0] / image_size[0]
    scale_y = inference_size[1] / image_size[1]
    inference_prompts = [[x * scale_x, y * scale_y] for x, y in prompts]
    inference_radius = radius * min(scale_x, scale_y)
    try:
        with _model_lock:
            point_masks = _predict_masks(model, inference_path, points=inference_prompts, labels=[1] * len(inference_prompts))
            scored = [(_point_coverage(mask, inference_prompts), int(mask.sum()), mask) for mask in point_masks if _valid_candidate(mask)]
            if not scored:
                fallback_box = _stroke_context_box(inference_prompts, inference_radius, inference_size)
                fallback_masks = [mask for mask in _predict_masks(model, inference_path, bboxes=[fallback_box]) if _valid_candidate(mask)]
                if not fallback_masks:
                    raise RuntimeError("未找到与涂抹范围对应的完整主体")
                scored = [(_point_coverage(mask, inference_prompts), int(mask.sum()), mask) for mask in fallback_masks]
            anchor_covered, _, anchor = max(scored, key=lambda item: (item[0], item[1]))
            anchor_box = _mask_bbox(anchor)
            anchor_width, anchor_height = anchor_box[2] - anchor_box[0], anchor_box[3] - anchor_box[1]
            anchor_span = max(anchor_width, anchor_height)
            join_distance = max(inference_radius * 2.5, anchor_span * .16)
            relevant = [
                mask for covered, _, mask in scored
                if covered > 0
                and covered >= max(1, anchor_covered * .12)
                and _bbox_gap(anchor_box, _mask_bbox(mask)) <= join_distance
                and (_mask_bbox(mask)[2] - _mask_bbox(mask)[0]) <= max(anchor_width * 1.55, inference_radius * 6)
                and (_mask_bbox(mask)[3] - _mask_bbox(mask)[1]) <= max(anchor_height * 1.55, inference_radius * 6)
            ]
            seed = np.logical_or.reduce(relevant)
            context_box = _expanded_context_box(seed, inference_radius)
            box_masks = _predict_masks(model, inference_path, bboxes=[context_box])

        seed_area = int(seed.sum())
        refinements: list[tuple[float, int, np.ndarray]] = []
        for mask in box_masks:
            if not _valid_candidate(mask):
                continue
            area = int(mask.sum())
            coverage = _point_coverage(mask, inference_prompts) / len(inference_prompts)
            if coverage >= .30 and area >= seed_area * .55 and area <= seed_area * 4.5:
                refinements.append((coverage, area, mask))
        result = max(refinements, key=lambda item: (item[0], item[1]))[2] if refinements else seed
        result = _clean_subject_mask(result)
        if _point_coverage(result, inference_prompts) < max(1, round(len(inference_prompts) * .30)):
            raise RuntimeError("主体没有覆盖主要涂抹位置")
        mask = Image.fromarray((result * 255).astype(np.uint8), "L")
        return mask.resize(image_size, Image.Resampling.NEAREST)
    finally:
        if temporary_path:
            temporary_path.unlink(missing_ok=True)
