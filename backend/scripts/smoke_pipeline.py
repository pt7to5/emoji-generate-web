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

from app.image_ops import bbox_from_mask, composite_emoji, cutout, expand_mask, filter_candidates, mask_from_image, normalize_image, save_png
from app.providers import fal_auto_segment, generate_emoji, repair_background


async def main() -> None:
    source = ROOT / "emoji_test" / "cake01.jpg"
    output = ROOT / "outputs" / "smoke-cake01"
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    report: dict[str, object] = {"source": str(source), "stages": {}}
    started = time.perf_counter()
    original = normalize_image(source.read_bytes(), 2048)
    original_path = output / "01-original.png"
    save_png(original, original_path)

    stage = time.perf_counter()
    raw_masks = await fal_auto_segment(original_path)
    masks = filter_candidates([mask_from_image(x, original.size) for x in raw_masks], original.size)
    if not masks:
        raise RuntimeError("SAM 2 没有返回可用主体蒙版")
    report["stages"]["segmentation"] = {"elapsedMs": round((time.perf_counter() - stage) * 1000), "rawMasks": len(raw_masks), "selectedMasks": len(masks)}

    mask = masks[0]
    mask_path = output / "02-mask.png"
    save_png(mask, mask_path)
    subject, bbox = cutout(original, mask, padding=40)
    subject_path = output / "03-cutout.png"
    save_png(subject, subject_path)

    stage = time.perf_counter()
    emoji_path = output / "04-emoji.png"
    await asyncio.to_thread(generate_emoji, subject_path, emoji_path)
    generated = Image.open(emoji_path).convert("RGBA")
    report["stages"]["emoji"] = {
        "elapsedMs": round((time.perf_counter() - stage) * 1000),
        "size": generated.size,
        "hasTransparency": generated.getchannel("A").getextrema()[0] < 255,
    }

    stage = time.perf_counter()
    expanded = expand_mask(mask)
    api_mask = Image.new("RGBA", original.size, (0, 0, 0, 255))
    api_mask.putalpha(Image.eval(expanded, lambda p: 255 - p))
    repair_mask_path = output / "05-repair-mask.png"
    save_png(api_mask, repair_mask_path)
    repaired_path = output / "06-repaired.png"
    await asyncio.to_thread(repair_background, original_path, repair_mask_path, repaired_path)
    repaired = Image.open(repaired_path).convert("RGBA")
    if repaired.size != original.size:
        repaired = repaired.resize(original.size, Image.Resampling.LANCZOS)
        save_png(repaired, repaired_path)
    report["stages"]["repair"] = {"elapsedMs": round((time.perf_counter() - stage) * 1000), "size": repaired.size}

    result = composite_emoji(repaired, generated, bbox)
    result_path = output / "07-result.png"
    save_png(result, result_path)
    report["bbox"] = bbox
    report["totalElapsedMs"] = round((time.perf_counter() - started) * 1000)
    report["outputs"] = [p.name for p in sorted(output.iterdir())]
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
