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

from app.image_ops import save_png
from app.providers_domestic import baidu_segment, generate_emoji_domestic


async def main() -> None:
    source_dir = ROOT / "outputs" / "smoke-domestic-cake01"
    output = ROOT / "outputs" / "emoji-style-v2-cake01"
    output.mkdir(parents=True, exist_ok=True)

    cutout = source_dir / "03-cutout.png"
    if not cutout.exists():
        raise FileNotFoundError(f"缺少已确认的主体切图：{cutout}")
    shutil.copy2(cutout, output / "01-input-cutout.png")

    started = time.perf_counter()
    opaque = output / "02-styled-opaque.png"
    await generate_emoji_domestic(cutout, opaque)
    rgba = await baidu_segment(opaque, None, "rgba")
    transparent = output / "03-styled-transparent.png"
    save_png(rgba.convert("RGBA"), transparent)

    report = {
        "elapsedMs": round((time.perf_counter() - started) * 1000),
        "source": str(cutout),
        "opaque": str(opaque),
        "transparent": str(transparent),
        "size": Image.open(transparent).size,
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
