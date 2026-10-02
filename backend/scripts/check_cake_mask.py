import asyncio
import sys
from pathlib import Path

from PIL import Image

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND))

from app.image_ops import bbox_from_mask, mask_from_image, normalize_image, save_png
from app.providers_domestic import baidu_segment


async def main():
    output = ROOT / "outputs" / "smoke-domestic-cake01"
    original_path = output / "01-original.png"
    if not original_path.exists():
        save_png(normalize_image((ROOT / "emoji_test" / "cake01.jpg").read_bytes(), 2048), original_path)
    image = Image.open(original_path)
    raw = await baidu_segment(original_path, [335, 1320, 665, 1705], "mask")
    save_png(raw, output / "02a-mask-raw.png")
    mask = mask_from_image(raw, image.size)
    save_png(mask, output / "02b-mask-normalized.png")
    print("bbox=", bbox_from_mask(mask), "raw_mode=", raw.mode, "raw_size=", raw.size)


asyncio.run(main())
