from __future__ import annotations
from io import BytesIO
from pathlib import Path
from typing import Iterable
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageOps, ImageStat


def normalize_image(raw: bytes, max_side: int) -> Image.Image:
    image = ImageOps.exif_transpose(Image.open(BytesIO(raw))).convert("RGB")
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "PNG", optimize=True)


def edit_mask_with_brush(mask: Image.Image, points: list[list[float]], radius: float, mode: str) -> Image.Image:
    """按用户画笔精确增删蒙版，不触发语义重分割。第三列非零表示新笔画起点。"""
    base = mask.convert("L")
    stroke = Image.new("L", base.size, 0)
    draw = ImageDraw.Draw(stroke)
    width = max(2, round(radius * 2))
    previous: tuple[float, float] | None = None
    for point in points:
        if len(point) < 2:
            continue
        current = (float(point[0]), float(point[1]))
        starts = len(point) >= 3 and bool(point[2])
        if starts or previous is None:
            draw.ellipse((current[0] - radius, current[1] - radius, current[0] + radius, current[1] + radius), fill=255)
        else:
            draw.line((previous, current), fill=255, width=width)
            draw.ellipse((current[0] - radius, current[1] - radius, current[0] + radius, current[1] + radius), fill=255)
        previous = current
    if mode == "add":
        return ImageChops.lighter(base, stroke)
    if mode == "remove":
        return ImageChops.subtract(base, stroke)
    raise ValueError("不支持的蒙版编辑模式")


def mask_from_image(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    if image.mode == "RGBA":
        mask = image.getchannel("A")
    else:
        gray = image.convert("L")
        # 分割服务的灰度蒙版可能使用相反极性。以四角代表背景判断，
        # 统一输出为“主体白、背景黑”。
        corners = [gray.getpixel((0, 0)), gray.getpixel((gray.width - 1, 0)), gray.getpixel((0, gray.height - 1)), gray.getpixel((gray.width - 1, gray.height - 1))]
        mask = ImageOps.invert(gray) if sum(corners) / len(corners) > 127 else gray
    return mask.resize(target_size, Image.Resampling.LANCZOS)


def bbox_from_mask(mask: Image.Image) -> tuple[int, int, int, int]:
    bbox = mask.point(lambda p: 255 if p > 30 else 0).getbbox()
    if not bbox:
        raise ValueError("蒙版为空")
    return bbox


def bbox_from_strokes(points: list[list[float]], radius: float, size: tuple[int, int]) -> list[int]:
    """把主体内部涂抹转换为带上下文的分割提示框。"""
    if not points:
        raise ValueError("涂抹不能为空")
    width, height = size
    xs = [max(0, min(width - 1, float(point[0]))) for point in points]
    ys = [max(0, min(height - 1, float(point[1]))) for point in points]
    stroke_w = max(radius * 2, max(xs) - min(xs)); stroke_h = max(radius * 2, max(ys) - min(ys))
    pad_x = max(radius * 3, stroke_w * .65); pad_y = max(radius * 3, stroke_h * .65)
    return [max(0, round(min(xs)-pad_x)), max(0, round(min(ys)-pad_y)),
            min(width, round(max(xs)+pad_x)), min(height, round(max(ys)+pad_y))]


def cutout(image: Image.Image, mask: Image.Image, padding: int = 24) -> tuple[Image.Image, tuple[int, int, int, int]]:
    bbox = bbox_from_mask(mask)
    left = max(0, bbox[0] - padding); top = max(0, bbox[1] - padding)
    right = min(image.width, bbox[2] + padding); bottom = min(image.height, bbox[3] + padding)
    rgba = image.convert("RGBA")
    rgba.putalpha(mask)
    return rgba.crop((left, top, right, bottom)), (left, top, right, bottom)


def filter_candidates(masks: Iterable[Image.Image], size: tuple[int, int], limit: int = 5) -> list[Image.Image]:
    total = size[0] * size[1]
    ranked: list[tuple[int, Image.Image]] = []
    for mask in masks:
        mask = mask.resize(size, Image.Resampling.LANCZOS).convert("L")
        area = sum(mask.histogram()[65:])
        ratio = area / total
        if 0.01 <= ratio <= 0.85:
            ranked.append((area, mask))
    ranked.sort(key=lambda x: x[0], reverse=True)
    selected: list[Image.Image] = []
    for _, mask in ranked:
        bbox = bbox_from_mask(mask)
        duplicate = False
        for existing in selected:
            eb = bbox_from_mask(existing)
            inter = max(0, min(bbox[2], eb[2]) - max(bbox[0], eb[0])) * max(0, min(bbox[3], eb[3]) - max(bbox[1], eb[1]))
            area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1]); existing_area = (eb[2]-eb[0])*(eb[3]-eb[1])
            union = area + existing_area - inter
            # 包围框嵌套不代表主体重复：蛋糕常位于更大的盘子框内。
            # 仅在实际蒙版像素高度重叠时去重。
            mask_area = sum(mask.point(lambda p: 255 if p > 64 else 0).histogram()[255:])
            existing_mask_area = sum(existing.point(lambda p: 255 if p > 64 else 0).histogram()[255:])
            pixel_intersection = sum(ImageChops.multiply(mask, existing).point(lambda p: 255 if p > 64 else 0).histogram()[255:])
            pixel_nested = pixel_intersection / max(1, min(mask_area, existing_mask_area))
            if (union and inter / union > 0.94 and pixel_nested > .88) or pixel_nested > .94:
                duplicate = True; break
        if not duplicate:
            selected.append(mask)
        if len(selected) >= limit:
            break
    return selected


def is_complete_subject(mask: Image.Image, size: tuple[int, int]) -> bool:
    """排除覆盖大面积画布并贴住多条边的桌布、墙面等环境区域。"""
    bbox = bbox_from_mask(mask)
    width, height = size
    box_width = bbox[2] - bbox[0]; box_height = bbox[3] - bbox[1]
    touches = sum((bbox[0] <= width * .015, bbox[1] <= height * .015, bbox[2] >= width * .985, bbox[3] >= height * .985))
    box_ratio = box_width * box_height / max(1, width * height)
    binary = mask.resize(size, Image.Resampling.NEAREST).convert("L").point(lambda p: 255 if p > 64 else 0)
    mask_ratio = binary.histogram()[255] / max(1, width * height)
    # 近景盘子、花束可能自然贴住左右两边；只有几乎覆盖整张图时才按环境区域过滤。
    if touches >= 3 and box_ratio > .50 and mask_ratio > .50:
        return False
    return True


def mask_needs_completion(mask: Image.Image, detection_box: list[int]) -> bool:
    """判断主体是否明显缺边或被分成多个有意义的部分。"""
    subject_box = bbox_from_mask(mask)
    dx1, dy1, dx2, dy2 = detection_box
    detection_w = max(1, dx2 - dx1); detection_h = max(1, dy2 - dy1)
    overlap_w = max(0, min(subject_box[2], dx2) - max(subject_box[0], dx1))
    overlap_h = max(0, min(subject_box[3], dy2) - max(subject_box[1], dy1))
    if overlap_w / detection_w < .78 or overlap_h / detection_h < .78:
        return True

    sample = mask.copy(); sample.thumbnail((256, 256), Image.Resampling.NEAREST)
    binary = sample.point(lambda p: 255 if p > 64 else 0)
    pixels = binary.load(); visited: set[tuple[int, int]] = set(); areas: list[int] = []
    for y in range(binary.height):
        for x in range(binary.width):
            if pixels[x, y] == 0 or (x, y) in visited:
                continue
            stack = [(x, y)]; visited.add((x, y)); area = 0
            while stack:
                px, py = stack.pop(); area += 1
                for nx, ny in ((px-1, py), (px+1, py), (px, py-1), (px, py+1)):
                    if 0 <= nx < binary.width and 0 <= ny < binary.height and pixels[nx, ny] and (nx, ny) not in visited:
                        visited.add((nx, ny)); stack.append((nx, ny))
            areas.append(area)
    areas.sort(reverse=True)
    return len(areas) > 1 and areas[1] >= max(12, areas[0] * .035)


def complete_subject_mask(primary: Image.Image, supplemental: Image.Image | None, detection_box: list[int]) -> Image.Image:
    """在检测框的有限范围内补全断裂蒙版，完整性优先于像素级紧贴。"""
    mask = primary.convert("L")
    if supplemental is not None:
        mask = ImageChops.lighter(mask, supplemental.convert("L").resize(mask.size, Image.Resampling.LANCZOS))
    width, height = mask.size
    x1, y1, x2, y2 = detection_box
    pad_x = round(max(1, x2 - x1) * .08); pad_y = round(max(1, y2 - y1) * .08)
    bounds = [max(0, x1-pad_x), max(0, y1-pad_y), min(width, x2+pad_x), min(height, y2+pad_y)]
    limiter = Image.new("L", mask.size, 0); ImageDraw.Draw(limiter).rectangle(bounds, fill=255)
    mask = ImageChops.multiply(mask, limiter)
    bridge = max(3, round(min(bounds[2]-bounds[0], bounds[3]-bounds[1]) * .018))
    bridge = min(21, bridge // 2 * 2 + 1)
    # 先连接主体内部的小断层，再轻微外扩，避免底边、奶油层被切掉。
    mask = mask.filter(ImageFilter.MaxFilter(bridge)).filter(ImageFilter.MinFilter(max(3, bridge-2)))
    mask = mask.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(.7))
    mask = ImageChops.multiply(mask, limiter)
    return fill_enclosed_holes(mask, bounds)


def fill_enclosed_holes(mask: Image.Image, bounds: list[int] | None = None) -> Image.Image:
    """仅填补小型封闭孔洞；拒绝可能形成矩形伪影的大面积补全。"""
    binary = mask.convert("L").point(lambda p: 255 if p > 40 else 0)
    x1, y1, x2, y2 = bounds or [0, 0, binary.width, binary.height]
    crop = binary.crop((x1, y1, x2, y2))
    padded = ImageOps.expand(crop, border=1, fill=0)
    exterior = padded.copy()
    ImageDraw.floodfill(exterior, (0, 0), 128, thresh=0)
    # 仍为 0 的区域无法从外边界到达，属于封闭孔洞。
    holes = exterior.point(lambda p: 255 if p == 0 else 0)
    hole_area = holes.histogram()[255]
    crop_area = max(1, crop.width * crop.height)
    if hole_area > crop_area * .08:
        return mask.convert("L")
    filled = ImageChops.lighter(padded, holes).crop((1, 1, crop.width + 1, crop.height + 1))
    result = binary.copy(); result.paste(filled, (x1, y1))
    return result.filter(ImageFilter.GaussianBlur(.45))


def preserve_color_layout(emoji: Image.Image, source: Image.Image, grid: int = 3, strength: float = .72) -> Image.Image:
    """用平滑增益场匹配参考主体颜色，避免固定网格产生矩形接缝。"""
    result = emoji.convert("RGBA")
    source = source.convert("RGBA").resize(result.size, Image.Resampling.LANCZOS)
    target_array = np.asarray(result, dtype=np.float32)
    source_array = np.asarray(source, dtype=np.float32)
    factors = np.ones((grid, grid, 3), dtype=np.float32)
    for gy in range(grid):
        for gx in range(grid):
            box = (gx * result.width // grid, gy * result.height // grid,
                   (gx + 1) * result.width // grid, (gy + 1) * result.height // grid)
            target_cell = target_array[box[1]:box[3], box[0]:box[2]]
            source_cell = source_array[box[1]:box[3], box[0]:box[2]]
            target_valid = target_cell[..., 3] > 80
            source_valid = source_cell[..., 3] > 80
            if target_valid.sum() < 8 or source_valid.sum() < 8:
                continue
            target_mean = target_cell[..., :3][target_valid].mean(axis=0)
            source_mean = source_cell[..., :3][source_valid].mean(axis=0)
            factors[gy, gx] = np.clip(source_mean / np.maximum(1, target_mean), .55, 1.65)
    gain_channels = []
    for channel in range(3):
        field = Image.fromarray(factors[..., channel], mode="F").resize(result.size, Image.Resampling.BICUBIC)
        gain_channels.append(np.asarray(field, dtype=np.float32))
    gains = np.stack(gain_channels, axis=-1)
    gains = 1 + (gains - 1) * strength
    output = target_array.copy()
    output[..., :3] = np.clip(output[..., :3] * gains, 0, 255)
    output[target_array[..., 3] <= 8, :3] = 0
    return Image.fromarray(output.astype(np.uint8), "RGBA")


def emoji_matches_source(emoji: Image.Image, source: Image.Image) -> bool:
    """宽松拦截明显跑题结果；不参与调色，避免再次制造色块。"""
    emoji = emoji.convert("RGBA")
    source = source.convert("RGBA")
    emoji_box = emoji.getchannel("A").getbbox()
    source_box = source.getchannel("A").getbbox()
    if not emoji_box or not source_box:
        return False
    emoji_ratio = (emoji_box[2] - emoji_box[0]) / max(1, emoji_box[3] - emoji_box[1])
    source_ratio = (source_box[2] - source_box[0]) / max(1, source_box[3] - source_box[1])
    if max(emoji_ratio, source_ratio) / max(.01, min(emoji_ratio, source_ratio)) > 1.9:
        return False
    # 将透明轮廓归一化后比较，拦截“颜色相近但主体形状完全变化”的结果。
    emoji_shape = emoji.getchannel("A").crop(emoji_box).resize((96, 96), Image.Resampling.NEAREST)
    source_shape = source.getchannel("A").crop(source_box).resize((96, 96), Image.Resampling.NEAREST)
    emoji_binary = np.asarray(emoji_shape) > 64
    source_binary = np.asarray(source_shape) > 64
    def major_components(binary: np.ndarray) -> int:
        seen = np.zeros(binary.shape, dtype=bool)
        count = 0
        minimum_area = max(24, round(binary.size * .008))
        height, width = binary.shape
        for start_y, start_x in zip(*np.where(binary & ~seen)):
            if seen[start_y, start_x]:
                continue
            stack = [(int(start_x), int(start_y))]
            seen[start_y, start_x] = True
            area = 0
            while stack:
                x, y = stack.pop(); area += 1
                for nx, ny in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
                    if 0 <= nx < width and 0 <= ny < height and binary[ny,nx] and not seen[ny,nx]:
                        seen[ny,nx] = True; stack.append((nx,ny))
            if area >= minimum_area:
                count += 1
        return count
    if major_components(emoji_binary) > major_components(source_binary):
        return False
    shape_intersection = np.logical_and(emoji_binary, source_binary).sum()
    shape_union = np.logical_or(emoji_binary, source_binary).sum()
    if shape_intersection / max(1, shape_union) < .52:
        return False
    fill_ratio = emoji_binary.mean() / max(.001, source_binary.mean())
    if not .68 <= fill_ratio <= 1.42:
        return False
    emoji_mean = np.array(ImageStat.Stat(emoji.convert("RGB"), mask=emoji.getchannel("A")).mean[:3])
    source_mean = np.array(ImageStat.Stat(source.convert("RGB"), mask=source.getchannel("A")).mean[:3])
    return float(np.abs(emoji_mean - source_mean).mean()) < 82


def emoji_is_visibly_stylized(emoji: Image.Image, source: Image.Image) -> bool:
    """拦截直接复制原照片或只做极轻微调色的伪风格化结果。"""
    emoji = emoji.convert("RGBA")
    source = source.convert("RGBA")
    emoji_box = emoji.getchannel("A").getbbox()
    source_box = source.getchannel("A").getbbox()
    if not emoji_box or not source_box:
        return False
    size = (96, 96)
    emoji_crop = emoji.crop(emoji_box).resize(size, Image.Resampling.LANCZOS)
    source_crop = source.crop(source_box).resize(size, Image.Resampling.LANCZOS)
    emoji_array = np.asarray(emoji_crop, dtype=np.float32)
    source_array = np.asarray(source_crop, dtype=np.float32)
    common = (emoji_array[..., 3] > 64) & (source_array[..., 3] > 64)
    if common.sum() < 96:
        return False
    pixel_difference = np.abs(emoji_array[..., :3] - source_array[..., :3])[common]
    return float(pixel_difference.mean()) >= 12


def match_color_intensity(emoji: Image.Image, source: Image.Image) -> Image.Image:
    """只压低异常偏高的饱和度和明度；使用全局连续变换，不产生区域接缝。"""
    result = emoji.convert("RGBA")
    source = source.convert("RGBA")
    emoji_alpha = np.asarray(result.getchannel("A"), dtype=np.float32) / 255
    source_alpha = np.asarray(source.getchannel("A"), dtype=np.float32) / 255
    if emoji_alpha.sum() < 1 or source_alpha.sum() < 1:
        return result

    emoji_hsv = np.asarray(result.convert("RGB").convert("HSV"), dtype=np.float32)
    source_hsv = np.asarray(source.convert("RGB").convert("HSV"), dtype=np.float32)
    emoji_s = float((emoji_hsv[..., 1] * emoji_alpha).sum() / emoji_alpha.sum())
    emoji_v = float((emoji_hsv[..., 2] * emoji_alpha).sum() / emoji_alpha.sum())
    source_s = float((source_hsv[..., 1] * source_alpha).sum() / source_alpha.sum())
    source_v = float((source_hsv[..., 2] * source_alpha).sum() / source_alpha.sum())

    # 容许 Emoji 有少量清晰度提升，但阻止鲜艳度和曝光明显跳变。
    saturation_scale = min(1.0, (source_s * 1.08 + 4) / max(1, emoji_s))
    value_scale = min(1.0, (source_v * 1.06 + 3) / max(1, emoji_v))
    saturation_scale = max(.30, saturation_scale)
    value_scale = max(.72, value_scale)
    adjusted = emoji_hsv.copy()
    adjusted[..., 1] = np.clip(adjusted[..., 1] * saturation_scale, 0, 255)
    adjusted[..., 2] = np.clip(adjusted[..., 2] * value_scale, 0, 255)
    rgb = Image.fromarray(adjusted.astype(np.uint8), "HSV").convert("RGB").convert("RGBA")
    rgb.putalpha(result.getchannel("A"))
    return rgb


def remove_white_matte(image: Image.Image) -> Image.Image:
    """清除白底抠图产生的半透明白边，并将完全透明像素的 RGB 清零。"""
    rgba = image.convert("RGBA")
    source_alpha = rgba.getchannel("A")
    eroded = source_alpha.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(.35))
    opaque_core = source_alpha.point(lambda p: 255 if p >= 250 else 0)
    alpha = ImageChops.lighter(eroded, opaque_core)
    pixels = bytearray(rgba.tobytes()); alphas = alpha.tobytes()
    for offset, a in enumerate(alphas):
        at = offset * 4
        if a < 8:
            pixels[at:at+4] = b"\x00\x00\x00\x00"
            continue
        original_alpha = max(1, pixels[at+3])
        for channel in range(3):
            value = pixels[at+channel]
            # 反解在白色背景上预混合的边缘颜色。
            pixels[at+channel] = max(0, min(255, round((value * 255 - 255 * (255 - original_alpha)) / original_alpha)))
        pixels[at+3] = a
    return Image.frombytes("RGBA", rgba.size, bytes(pixels))


def apply_repair_inside_mask(original: Image.Image, repaired: Image.Image, mask: Image.Image) -> Image.Image:
    """只接受蒙版内部的修复像素；蒙版外保持输入图逐像素不变。"""
    original = original.convert("RGBA")
    repaired = repaired.convert("RGBA").resize(original.size, Image.Resampling.LANCZOS)
    hard_mask = mask.convert("L").resize(original.size, Image.Resampling.LANCZOS).point(lambda p: 255 if p > 32 else 0)
    return Image.composite(repaired, original, hard_mask)


def expand_mask(mask: Image.Image, pixels: int = 18) -> Image.Image:
    size = max(3, pixels // 2 * 2 + 1)
    return mask.filter(ImageFilter.MaxFilter(size)).filter(ImageFilter.GaussianBlur(max(1, pixels // 5)))


def expand_detection_box(box: list[int], image_size: tuple[int, int]) -> list[int]:
    """为检测框补充上下文，尤其覆盖托盘/盘子上方的物品。"""
    x1, y1, x2, y2 = box
    width = max(1, x2 - x1); height = max(1, y2 - y1)
    side_pad = round(width * 0.08)
    top_pad = round(height * (0.85 if width / height > 1.4 else 0.15))
    bottom_pad = round(height * 0.20)
    return [
        max(0, x1 - side_pad),
        max(0, y1 - top_pad),
        min(image_size[0] - 1, x2 + side_pad),
        min(image_size[1] - 1, y2 + bottom_pad),
    ]


def harmonize_emoji(emoji: Image.Image, reference: Image.Image, mask: Image.Image) -> Image.Image:
    """匹配原主体区域的亮度和环境色，降低独立贴纸感。"""
    emoji = emoji.convert("RGBA")
    alpha = emoji.getchannel("A")
    if not alpha.getbbox():
        return emoji
    reference = reference.convert("RGB").resize(mask.size, Image.Resampling.LANCZOS)
    ref_rgb = ImageStat.Stat(reference, mask=mask.point(lambda p: 255 if p > 64 else 0)).mean[:3]
    emoji_rgb = ImageStat.Stat(emoji.convert("RGB"), mask=alpha).mean[:3]
    brightness = max(.76, min(1.22, (sum(ref_rgb) / 3) / max(1, sum(emoji_rgb) / 3)))
    rgb = ImageEnhance.Brightness(emoji.convert("RGB")).enhance(brightness)
    channels = list(rgb.split()); neutral = max(1, sum(ref_rgb) / 3)
    for index in range(3):
        tint = max(.9, min(1.1, ref_rgb[index] / neutral))
        channels[index] = channels[index].point(lambda value, factor=tint: min(255, round(value * factor)))
    result = Image.merge("RGB", channels).convert("RGBA"); result.putalpha(alpha)
    return result


def composite_emoji(
    background: Image.Image,
    emoji: Image.Image,
    bbox: tuple[int, int, int, int],
    add_shadow: bool = False,
    target_mask: Image.Image | None = None,
) -> Image.Image:
    """按原主体中心和底边定位，并把改动限制在主体轮廓附近。"""
    target_w = max(1, bbox[2] - bbox[0]); target_h = max(1, bbox[3] - bbox[1])
    emoji = emoji.convert("RGBA")
    alpha_box = emoji.getchannel("A").getbbox()
    if alpha_box:
        emoji = emoji.crop(alpha_box)
    # Cover 而非 contain：宁可在受控轮廓内轻微裁切，也不能留下旧主体残影。
    scale = max(target_w / max(1, emoji.width), target_h / max(1, emoji.height)) * 1.02
    emoji = emoji.resize(
        (max(1, round(emoji.width * scale)), max(1, round(emoji.height * scale))),
        Image.Resampling.LANCZOS,
    )
    x = round((bbox[0] + bbox[2] - emoji.width) / 2)
    y = bbox[3] - emoji.height
    out = background.convert("RGBA")
    layer = Image.new("RGBA", out.size, (0, 0, 0, 0))
    if add_shadow:
        blur = max(2, round(min(target_w, target_h) * .025))
        shadow_alpha = emoji.getchannel("A").filter(ImageFilter.GaussianBlur(blur)).point(lambda p: round(p * .16))
        shadow = Image.new("RGBA", emoji.size, (20, 20, 20, 0)); shadow.putalpha(shadow_alpha)
        layer.alpha_composite(shadow, (x + max(1, blur // 2), y + max(2, blur)))
    layer.alpha_composite(emoji, (x, y))
    if target_mask is not None:
        allowed = target_mask.convert("L").resize(out.size, Image.Resampling.NEAREST)
        # 仅放宽约 2px，覆盖抗锯齿边缘，同时保护主体之外的原图像素。
        allowed = allowed.filter(ImageFilter.MaxFilter(5))
        layer.putalpha(ImageChops.multiply(layer.getchannel("A"), allowed))
    out.alpha_composite(layer)
    return out
