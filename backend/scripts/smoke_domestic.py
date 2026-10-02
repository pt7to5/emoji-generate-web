from __future__ import annotations

import asyncio
import json
import shutil
import sys
import time
from pathlib import Path

from PIL import Image

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND))

from app.image_ops import bbox_from_mask, composite_emoji, cutout, expand_detection_box, mask_from_image, normalize_image, save_png
from app.providers_domestic import baidu_detect_objects, baidu_segment, generate_emoji_domestic, repair_background_domestic


async def main() -> None:
    source = ROOT / "emoji_test" / "cake01.jpg"
    output = ROOT / "outputs" / "smoke-domestic-auto-cake01"
    if output.exists(): shutil.rmtree(output)
    output.mkdir(parents=True)
    report: dict[str, object] = {"source": str(source), "stages": {}}
    total = time.perf_counter()
    original = normalize_image(source.read_bytes(), 2048)
    original_path = output / "01-original.png"; save_png(original, original_path)

    stage = time.perf_counter()
    detections = await baidu_detect_objects(original_path)
    # 多主体检测按置信度排序；该样例的第一候选框完整覆盖蛋糕和盘子。
    selected = detections[0] if detections else {
        "label": "蛋糕（检测失败后的人工兜底）",
        "score": 0.0,
        "box": [335, 1320, 665, 1705],
        "source": "manual_fallback",
    }
    segment_box = expand_detection_box(selected["box"], original.size)
    segmented = await baidu_segment(original_path, segment_box, "mask")
    mask = mask_from_image(segmented, original.size)
    bbox = bbox_from_mask(mask)
    report["stages"]["segmentation"] = {"elapsedMs": round((time.perf_counter()-stage)*1000), "detections": detections, "selected": selected, "segmentBox": segment_box, "bbox": bbox}
    save_png(mask, output / "02-mask.png")
    subject, _ = cutout(original, mask, padding=40)
    subject_path = output / "03-cutout.png"; save_png(subject, subject_path)

    stage = time.perf_counter()
    styled_path = output / "04-styled-opaque.png"
    await generate_emoji_domestic(subject_path, styled_path)
    rgba = await baidu_segment(styled_path, None, "rgba")
    emoji_path = output / "05-emoji-transparent.png"; save_png(rgba.convert("RGBA"), emoji_path)
    report["stages"]["emoji"] = {"elapsedMs": round((time.perf_counter()-stage)*1000), "size": rgba.size, "hasTransparency": rgba.convert("RGBA").getchannel("A").getextrema()[0] < 255}

    stage = time.perf_counter()
    repaired_path = output / "06-repaired.png"
    await repair_background_domestic(original_path, list(bbox), repaired_path)
    repaired = Image.open(repaired_path).convert("RGBA")
    if repaired.size != original.size:
        repaired = repaired.resize(original.size, Image.Resampling.LANCZOS); save_png(repaired, repaired_path)
    result = composite_emoji(repaired, rgba, bbox)
    save_png(result, output / "07-result.png")
    report["stages"]["repair_and_composite"] = {"elapsedMs": round((time.perf_counter()-stage)*1000)}
    report["totalElapsedMs"] = round((time.perf_counter()-total)*1000)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__": asyncio.run(main())
