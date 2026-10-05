from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.image_ops import (
    bbox_from_mask,
    complete_subject_mask,
    cutout,
    expand_detection_box,
    filter_candidates,
    is_complete_subject,
    mask_from_image,
    mask_needs_completion,
    normalize_image,
    save_png,
)
from app.providers_domestic import baidu_segment, detect_objects_hybrid


REFERENCE_PREFIXES = ("当我把美食发给豆包", "花束emoji教程")


def _font(size: int = 22) -> ImageFont.ImageFont:
    for path in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _mask_stats(mask: Image.Image) -> dict[str, object]:
    mono = mask.convert("L")
    width, height = mono.size
    bbox = bbox_from_mask(mono)
    pixels = mono.point(lambda value: 255 if value >= 128 else 0)
    selected = sum(1 for value in pixels.getdata() if value)
    touches_edge = any(
        value
        for edge in (
            pixels.crop((0, 0, width, 1)),
            pixels.crop((0, height - 1, width, height)),
            pixels.crop((0, 0, 1, height)),
            pixels.crop((width - 1, 0, width, height)),
        )
        for value in edge.getdata()
    )
    return {
        "bbox": list(bbox),
        "areaRatio": round(selected / max(1, width * height), 4),
        "touchesImageEdge": touches_edge,
    }


def _preview(original: Image.Image, rows: list[dict[str, object]], masks: list[Image.Image]) -> Image.Image:
    base = original.convert("RGBA")
    scale = min(1.0, 900 / max(base.size))
    if scale < 1:
        base = base.resize((round(base.width * scale), round(base.height * scale)), Image.Resampling.LANCZOS)
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    colors = [(255, 70, 70, 105), (40, 180, 255, 105), (80, 220, 100, 105), (255, 180, 30, 105)]
    draw = ImageDraw.Draw(overlay)
    font = _font(max(15, round(base.width / 35)))
    for index, (row, mask) in enumerate(zip(rows, masks)):
        color = colors[index % len(colors)]
        resized = mask.resize(base.size, Image.Resampling.NEAREST).convert("L")
        tint = Image.new("RGBA", base.size, color)
        overlay.alpha_composite(Image.composite(tint, Image.new("RGBA", base.size), resized))
        x1, y1, x2, y2 = row["mask"]["bbox"]
        box = [round(value * scale) for value in (x1, y1, x2, y2)]
        draw.rectangle(box, outline=color[:3] + (255,), width=max(2, round(base.width / 300)))
        draw.text((box[0] + 4, box[1] + 4), f"{index + 1} {row['label']}", font=font, fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 220))
    return Image.alpha_composite(base, overlay).convert("RGB")


async def evaluate(source: Path, output: Path) -> dict[str, object]:
    started = time.perf_counter()
    item_dir = output / source.stem
    item_dir.mkdir(parents=True, exist_ok=True)
    original = normalize_image(source.read_bytes(), 4096).convert("RGB")
    original_path = item_dir / "original.png"
    save_png(original, original_path)
    detections = await detect_objects_hybrid(original_path)
    masks: list[Image.Image] = []
    rows: list[dict[str, object]] = []
    errors: list[str] = []
    for index, detection in enumerate(detections):
        try:
            segment_box = expand_detection_box(detection["box"], original.size)
            segmented = await baidu_segment(original_path, segment_box, "mask")
            primary = mask_from_image(segmented, original.size)
            supplemental = None
            if mask_needs_completion(primary, detection["box"]):
                x1, y1, x2, y2 = detection["box"]
                pad_x = round((x2 - x1) * .12)
                pad_y = round((y2 - y1) * .12)
                crop_box = (max(0, x1 - pad_x), max(0, y1 - pad_y), min(original.width, x2 + pad_x), min(original.height, y2 + pad_y))
                crop_path = item_dir / f"crop-{index + 1}.png"
                save_png(original.crop(crop_box), crop_path)
                try:
                    crop_mask = mask_from_image(await baidu_segment(crop_path, None, "mask"), (crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]))
                    supplemental = Image.new("L", original.size, 0)
                    supplemental.paste(crop_mask, crop_box[:2])
                finally:
                    crop_path.unlink(missing_ok=True)
            mask = complete_subject_mask(primary, supplemental, detection["box"])
            masks.append(mask)
            rows.append({
                "label": detection["label"],
                "score": detection["score"],
                "source": detection["source"],
                "detectionBox": detection["box"],
                "mask": _mask_stats(mask),
                "completeByHeuristic": is_complete_subject(mask, original.size),
            })
        except Exception as exc:
            errors.append(f"{detection.get('label', index + 1)}: {type(exc).__name__}: {exc}")
    kept = [mask for mask in filter_candidates(masks, original.size, limit=8) if is_complete_subject(mask, original.size)]
    for row, mask in zip(rows, masks):
        signature = (bbox_from_mask(mask), mask.resize((64, 64), Image.Resampling.NEAREST).tobytes())
        row["keptByPipeline"] = any(
            signature == (bbox_from_mask(candidate), candidate.resize((64, 64), Image.Resampling.NEAREST).tobytes())
            for candidate in kept
        )
    if masks:
        _preview(original, rows, masks).save(item_dir / "preview.jpg", quality=90)
        for index, mask in enumerate(masks, 1):
            save_png(mask, item_dir / f"mask-{index}.png")
            subject, _ = cutout(original, mask)
            save_png(subject, item_dir / f"cutout-{index}.png")
    return {
        "file": source.name,
        "size": list(original.size),
        "elapsedMs": round((time.perf_counter() - started) * 1000),
        "detectionCount": len(detections),
        "keptCount": sum(1 for row in rows if row["keptByPipeline"]),
        "objects": rows,
        "errors": errors,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    candidates = [
        path for path in sorted(args.source.iterdir())
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        and not path.name.startswith(REFERENCE_PREFIXES)
    ]
    results: list[dict[str, object]] = []
    for index, source in enumerate(candidates, 1):
        print(f"[{index}/{len(candidates)}] {source.name}", flush=True)
        try:
            results.append(await evaluate(source, args.output))
        except Exception as exc:
            results.append({"file": source.name, "fatalError": f"{type(exc).__name__}: {exc}"})
    summary = {
        "source": str(args.source),
        "imageCount": len(candidates),
        "successfulCount": sum("fatalError" not in item for item in results),
        "imagesWithNoKeptObject": sum(item.get("keptCount") == 0 for item in results),
        "results": results,
    }
    (args.output / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
