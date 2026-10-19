from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageStat


@dataclass(frozen=True)
class QualityReport:
    valid: bool
    invalid_reasons: tuple[str, ...]
    subject_fidelity: float
    style_consistency: float
    visual_completeness: float
    color_fidelity: float
    total: float
    audit_available: bool
    audit_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["invalid_reasons"] = list(self.invalid_reasons)
        return result


def _alpha_shape(image: Image.Image, size: tuple[int, int] = (96, 96)) -> np.ndarray:
    rgba = image.convert("RGBA")
    box = rgba.getchannel("A").getbbox()
    if not box:
        return np.zeros(size[::-1], dtype=bool)
    return np.asarray(rgba.getchannel("A").crop(box).resize(size, Image.Resampling.NEAREST)) > 32


def validate_generated_image(image: Image.Image) -> tuple[bool, tuple[str, ...]]:
    """只拦截确定无效的图片；审美问题由软评分处理。"""
    reasons: list[str] = []
    if image.width < 64 or image.height < 64:
        reasons.append("resolution_too_small")
    rgba = image.convert("RGBA")
    alpha = np.asarray(rgba.getchannel("A"))
    foreground = alpha > 24
    ratio = float(foreground.mean())
    if ratio < .004:
        reasons.append("empty_subject")
    if ratio > .985:
        reasons.append("missing_background_separation")
    box = rgba.getchannel("A").getbbox()
    if box:
        touches = sum((box[0] <= 1, box[1] <= 1, box[2] >= image.width - 1, box[3] >= image.height - 1))
        if touches >= 3:
            reasons.append("subject_severely_clipped")
    if foreground.any():
        rgb = np.asarray(rgba.convert("RGB"), dtype=np.float32)[foreground]
        if float(rgb.std()) < 2.0:
            reasons.append("near_solid_output")
    return not reasons, tuple(reasons)


def _shape_iou(candidate: Image.Image, source: Image.Image) -> float:
    a, b = _alpha_shape(candidate), _alpha_shape(source)
    return float(np.logical_and(a, b).sum() / max(1, np.logical_or(a, b).sum()))


def _color_score(candidate: Image.Image, source: Image.Image) -> float:
    candidate, source = candidate.convert("RGBA"), source.convert("RGBA")
    ca, sa = candidate.getchannel("A"), source.getchannel("A")
    if not ca.getbbox() or not sa.getbbox():
        return 0.0
    cm = np.array(ImageStat.Stat(candidate.convert("RGB"), mask=ca).mean[:3])
    sm = np.array(ImageStat.Stat(source.convert("RGB"), mask=sa).mean[:3])
    return max(0.0, 100.0 - float(np.abs(cm - sm).mean()) / 1.8)


def score_candidate(
    candidate: Image.Image,
    source: Image.Image,
    *,
    semantic_ok: bool,
    style_class: str,
    style_quality: str,
    audit_available: bool,
    audit_reason: str = "",
) -> QualityReport:
    valid, reasons = validate_generated_image(candidate)
    shape_iou = _shape_iou(candidate, source)
    subject = max(0.0, min(100.0, shape_iou * 115))
    completeness = max(0.0, min(100.0, shape_iou * 125))
    if not semantic_ok and audit_available:
        subject *= .55
        completeness *= .55
    style_base = {("3D_EMOJI", "GOOD"): 95.0, ("3D_EMOJI", "WEAK"): 68.0}.get((style_class, style_quality), 25.0)
    if not audit_available:
        style_base = 55.0
    color = _color_score(candidate, source)
    total = subject * .35 + style_base * .30 + completeness * .20 + color * .15
    if not valid:
        total = 0.0
    return QualityReport(valid, reasons, round(subject, 1), round(style_base, 1), round(completeness, 1), round(color, 1), round(total, 1), audit_available, audit_reason)


def load_and_validate(path: Path) -> Image.Image:
    """在进入抠图/审核前验证供应商返回的是可解码图片。"""
    raw = path.read_bytes()
    try:
        image = Image.open(BytesIO(raw))
        image.load()
    except Exception as exc:
        raise ValueError("provider_returned_corrupt_image") from exc
    if image.width < 64 or image.height < 64:
        raise ValueError("provider_returned_tiny_image")
    return image.copy()


def efficiency_score(total_ms: int, target_seconds: int = 60) -> float:
    """生成效率独立计分：目标耗时内线性保留 60~100 分，超时平滑下降。"""
    seconds = max(0.0, total_ms / 1000)
    if seconds <= target_seconds:
        return round(100 - 40 * seconds / max(1, target_seconds), 1)
    return round(max(0.0, 60 - 60 * (seconds - target_seconds) / max(1, target_seconds * 2)), 1)
