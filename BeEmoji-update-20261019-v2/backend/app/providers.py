from __future__ import annotations
import base64
from io import BytesIO
import json
from pathlib import Path
from typing import Any
import httpx
from openai import OpenAI
from PIL import Image
from .config import FAL_KEY, OPENAI_API_KEY

FAL_AUTO_URL = "https://fal.run/fal-ai/sam2/auto-segment"
FAL_IMAGE_URL = "https://fal.run/fal-ai/sam2/image"


def _data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _collect_urls(value: Any) -> list[str]:
    urls: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "url" and isinstance(child, str):
                urls.append(child)
            else:
                urls.extend(_collect_urls(child))
    elif isinstance(value, list):
        for child in value:
            urls.extend(_collect_urls(child))
    return urls


async def fal_auto_segment(image_path: Path) -> list[Image.Image]:
    if not FAL_KEY:
        raise RuntimeError("FAL_KEY 未配置")
    payload = {"image_url": _data_uri(image_path), "output_format": "png", "min_mask_region_area": 600, "pred_iou_thresh": 0.82, "stability_score_thresh": 0.88}
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(FAL_AUTO_URL, headers={"Authorization": f"Key {FAL_KEY}"}, json=payload)
        response.raise_for_status()
        body = response.json()
        items = body.get("individual_masks") or body.get("masks") or []
        urls = [x.get("url") for x in items if isinstance(x, dict) and x.get("url")]
        if not urls:
            urls = _collect_urls(items)
        masks = []
        for url in urls:
            result = await client.get(url)
            result.raise_for_status()
            masks.append(Image.open(BytesIO(result.content)).copy())
        return masks


async def fal_box_segment(image_path: Path, box: list[int]) -> Image.Image:
    if not FAL_KEY:
        raise RuntimeError("FAL_KEY 未配置")
    payload = {"image_url": _data_uri(image_path), "box_prompts": [{"x_min": box[0], "y_min": box[1], "x_max": box[2], "y_max": box[3]}], "apply_mask": False, "output_format": "png"}
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(FAL_IMAGE_URL, headers={"Authorization": f"Key {FAL_KEY}"}, json=payload)
        response.raise_for_status()
        urls = _collect_urls(response.json())
        if not urls:
            raise RuntimeError("分割服务没有返回蒙版")
        result = await client.get(urls[-1]); result.raise_for_status()
        return Image.open(BytesIO(result.content)).copy()


def _openai_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY 未配置")
    return OpenAI(api_key=OPENAI_API_KEY, timeout=180, max_retries=1)


def generate_emoji(cutout_path: Path, output_path: Path) -> None:
    prompt = """Transform the isolated cake or object in the reference image into an original, polished 3D emoji sticker. Preserve the exact pose, viewing angle, silhouette proportions, primary colors, distinctive decoration, layers, toppings, and essential structural features. Simplify photographic texture into a small number of clean rounded color regions. Use soft studio lighting from the upper left, gentle inner shading, subtle ambient occlusion, smooth highlights, and a friendly premium mobile-emoji aesthetic. Keep the complete object centered on a fully transparent background. Do not add text, a platform, a border, scenery, extra objects, faces, limbs, or features not present in the reference. Do not imitate or reproduce any existing proprietary emoji asset."""
    with cutout_path.open("rb") as image_file:
        result = _openai_client().images.edit(model="gpt-image-2", image=image_file, prompt=prompt, background="transparent", output_format="png", quality="medium", size="1024x1024")
    output_path.write_bytes(base64.b64decode(result.data[0].b64_json))


def repair_background(original_path: Path, mask_path: Path, output_path: Path) -> None:
    prompt = "Remove the selected object completely and reconstruct only the background that would naturally be behind it. Preserve every unmasked pixel, lighting, perspective, surface, and surrounding object. Do not add a replacement object, text, decoration, or new subject."
    with original_path.open("rb") as image_file, mask_path.open("rb") as mask_file:
        result = _openai_client().images.edit(model="gpt-image-2", image=image_file, mask=mask_file, prompt=prompt, output_format="png", quality="medium", size="auto")
    output_path.write_bytes(base64.b64decode(result.data[0].b64_json))
