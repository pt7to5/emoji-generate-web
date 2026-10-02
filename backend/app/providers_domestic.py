from __future__ import annotations

import asyncio
import base64
from io import BytesIO
import json
import re
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

from .config import (
    BAIDU_DETECT_API_KEY,
    BAIDU_DETECT_SECRET_KEY,
    BAIDU_PROCESS_API_KEY,
    BAIDU_PROCESS_SECRET_KEY,
    DASHSCOPE_API_KEY,
)

BAIDU_TOKEN_URL = "https://aip.baidubce.com/oauth/2.0/token"
BAIDU_SUBJECT_DETECT_URL = "https://aip.baidubce.com/rest/2.0/image-classify/v1/object_detect"
BAIDU_MULTI_OBJECT_DETECT_URL = "https://aip.baidubce.com/rest/2.0/image-classify/v1/multi_object_detect"
BAIDU_SEGMENT_URL = "https://aip.baidubce.com/rest/2.0/image-process/v1/segment"
DASHSCOPE_IMAGE_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

EMOJI_PROMPT = """图1是唯一需要转换的主体，图2只用于参考现场光线，严禁从图2引入其他物体。
先准确复刻图1：主体类别、完整外轮廓、姿态、视角、比例、已有五官、装饰、分层结构和每种颜色的位置都必须一致。不得把它改造成另一种动物、食物或通用卡通形象，不得新增或删除结构。
在保持主体身份不变的前提下，转换成精致的移动端系统 Emoji 贴纸质感：柔和半立体、圆润但结构准确、半哑光、低光泽，使用连续细腻的明暗渐变、轻微内阴影和克制的柔光表达体积。保留原图的主色、辅色、局部颜色及其面积关系，只简化细碎噪点和纹理。
颜色强度必须服从图1：不得自动提亮，不得提高饱和度，不得把低饱和颜色改成鲜艳色或荧光色。白色、奶油色、浅灰色等低饱和区域必须保持低饱和；绿色、黄色、红色等彩色区域的明度与浓度也要接近图1。
光线方向、色温、亮暗和阴影方向参考图2中原主体所在位置，使生成物自然融入原照片。
输出要求：完整单个主体，原视角，居中，正方形画布，纯白背景，不裁切。
严禁：扁平矢量插画、色带、等高线式分层、几何色块、硬边渐变、明显分区接缝、低多边形、厚重3D玩具、塑料高光、镜面反射、擅自拟人化、添加新五官或装饰、改变主体类别、改变颜色、文字、水印、边框和场景。"""

NEGATIVE_PROMPT = "主体身份改变，通用卡通形象，结构缺失，新增结构，姿态改变，颜色改变，自动提亮，提高饱和度，荧光色，糖果色，扁平矢量插画，几何色块，色带，等高线分层，分区接缝，硬边渐变，低多边形，复杂背景，无关主体，文字，水印，边框，裁切，错误透视，新增眼睛，新增嘴巴，新增四肢，过度拟人，线稿，噪点，硬阴影，强环境遮蔽，过曝，镜面反射，大面积白色高光，塑料玩具质感，厚重3D模型，悬浮"


def _require_keys() -> None:
    missing = [name for name, value in (
        ("BAIDU_DETECT_API_KEY", BAIDU_DETECT_API_KEY),
        ("BAIDU_DETECT_SECRET_KEY", BAIDU_DETECT_SECRET_KEY),
        ("BAIDU_PROCESS_API_KEY", BAIDU_PROCESS_API_KEY),
        ("BAIDU_PROCESS_SECRET_KEY", BAIDU_PROCESS_SECRET_KEY),
        ("DASHSCOPE_API_KEY", DASHSCOPE_API_KEY),
    ) if not value]
    if missing:
        raise RuntimeError("缺少环境变量：" + ", ".join(missing))


def _image_base64(path: Path, jpeg: bool = False) -> str:
    if not jpeg:
        return base64.b64encode(path.read_bytes()).decode("ascii")
    image = Image.open(path).convert("RGB")
    image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
    buffer = BytesIO(); image.save(buffer, "JPEG", quality=86, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _data_uri(path: Path, force_white: bool = False) -> str:
    if force_white:
        image = Image.open(path).convert("RGBA")
        background = Image.new("RGB", image.size, "white")
        background.paste(image, mask=image.getchannel("A"))
        buffer = BytesIO(); background.save(buffer, "PNG")
        raw = buffer.getvalue()
    else:
        raw = path.read_bytes()
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def _collect_urls(value: Any) -> list[str]:
    urls: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"url", "image"} and isinstance(child, str) and child.startswith("http"):
                urls.append(child)
            else:
                urls.extend(_collect_urls(child))
    elif isinstance(value, list):
        for child in value:
            urls.extend(_collect_urls(child))
    return urls


async def baidu_access_token(client: httpx.AsyncClient, api_key: str, secret_key: str) -> str:
    response = await client.post(BAIDU_TOKEN_URL, params={"grant_type": "client_credentials", "client_id": api_key, "client_secret": secret_key})
    response.raise_for_status()
    body = response.json()
    token = body.get("access_token")
    if not token:
        raise RuntimeError(body.get("error_description") or "百度未返回 access_token")
    return token


async def baidu_detect_objects(image_path: Path) -> list[dict[str, Any]]:
    """多主体优先；无结果时再回退到最突出主体检测。"""
    _require_keys()
    async with httpx.AsyncClient(timeout=45) as client:
        token = await baidu_access_token(client, BAIDU_DETECT_API_KEY, BAIDU_DETECT_SECRET_KEY)
        response = await client.post(
            BAIDU_MULTI_OBJECT_DETECT_URL,
            params={"access_token": token},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={"image": _image_base64(image_path, jpeg=True)},
        )
        response.raise_for_status(); body = response.json()
        if body.get("error_code"):
            raise RuntimeError(
                f"百度图像多主体检测错误 {body['error_code']}：{body.get('error_msg', '')}。"
                "请在百度控制台的“应用授权”中，将图像多主体检测授权给当前 API Key 所属应用。"
            )
        objects: list[dict[str, Any]] = []
        for item in body.get("result") or []:
            loc = item.get("location") or {}
            if not all(k in loc for k in ("left", "top", "width", "height")):
                continue
            left, top = int(loc["left"]), int(loc["top"])
            objects.append({
                "label": item.get("name") or "主体",
                "score": float(item.get("score") or 0),
                "box": [left, top, left + int(loc["width"]), top + int(loc["height"])],
                "source": "baidu_multi_object_detect",
            })
        if objects:
            return sorted(objects, key=lambda item: item["score"], reverse=True)[:8]

        response = await client.post(
            BAIDU_SUBJECT_DETECT_URL,
            params={"access_token": token},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={"image": _image_base64(image_path, jpeg=True), "with_face": 1},
        )
        response.raise_for_status(); body = response.json()
        if body.get("error_code"):
            raise RuntimeError(f"百度图像主体检测错误 {body['error_code']}：{body.get('error_msg', '')}")
        loc = body.get("result") or {}
        if not all(k in loc for k in ("left", "top", "width", "height")):
            return []
        left, top = int(loc["left"]), int(loc["top"])
        return [{
            "label": "主要主体",
            "score": 1.0,
            "box": [left, top, left + int(loc["width"]), top + int(loc["height"])],
            "source": "baidu_object_detect",
        }]


def _iou(a: list[int], b: list[int]) -> float:
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = max(1, (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)
    return inter / union


async def qwen_detect_objects(image_path: Path) -> list[dict[str, Any]]:
    """用视觉语言模型补检漏掉的完整物品，坐标为 Qwen 的 0..999 归一化格式。"""
    _require_keys()
    prompt = """检测图片中所有适合独立风格化替换的完整物品主体，例如每一杯饮品、整块蛋糕、餐盘中的完整甜点。
不要把装饰叶、吸管、水果片、杯中冰块等组成部分单独列为主体；盘子仅在没有承托食物时单独列出。
严格只返回 JSON 数组，每项格式为 {\"label\":\"简短中文名称\",\"bbox\":[x1,y1,x2,y2]}。
bbox 使用 0 到 999 的归一化坐标。列出所有明显主体，最多 8 个，不要输出说明或 Markdown。"""
    payload = {
        "model": "qwen3.6-plus",
        "input": {"messages": [{"role": "user", "content": [
            {"image": _data_uri(image_path)}, {"text": prompt},
        ]}]},
        "parameters": {"result_format": "message", "vl_high_resolution_images": True},
    }
    async with httpx.AsyncClient(timeout=90) as client:
        response = await client.post(DASHSCOPE_IMAGE_URL, headers={"Authorization": f"Bearer {DASHSCOPE_API_KEY}", "Content-Type": "application/json"}, json=payload)
        response.raise_for_status(); body = response.json()
    if body.get("code"):
        raise RuntimeError(f"百炼视觉补检错误 {body.get('code')}：{body.get('message', '')}")
    content = (((body.get("output") or {}).get("choices") or [{}])[0].get("message") or {}).get("content") or []
    text = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        return []
    items = json.loads(match.group(0)); width, height = Image.open(image_path).size
    detections: list[dict[str, Any]] = []
    for item in items[:8]:
        box = item.get("bbox") if isinstance(item, dict) else None
        if not isinstance(box, list) or len(box) != 4:
            continue
        x1, y1, x2, y2 = [float(value) for value in box]
        absolute = [round(x1 / 999 * width), round(y1 / 999 * height), round(x2 / 999 * width), round(y2 / 999 * height)]
        if absolute[2] - absolute[0] < width * .04 or absolute[3] - absolute[1] < height * .04:
            continue
        detections.append({"label": item.get("label") or "主体", "score": .82, "box": absolute, "source": "qwen3.6-plus"})
    return detections


async def detect_objects_hybrid(image_path: Path) -> list[dict[str, Any]]:
    baidu, qwen = await asyncio.gather(baidu_detect_objects(image_path), qwen_detect_objects(image_path), return_exceptions=True)
    primary = baidu if isinstance(baidu, list) else []
    supplements = qwen if isinstance(qwen, list) else []
    merged = list(primary)
    for candidate in supplements:
        # 较大的承托物框可能包住蛋糕；只有几乎相同的框才视为重复。
        if all(_iou(candidate["box"], existing["box"]) < .82 for existing in merged):
            merged.append(candidate)
    if not merged:
        error = baidu if isinstance(baidu, Exception) else qwen
        raise RuntimeError(str(error))
    return merged[:8]


async def baidu_segment(image_path: Path, box: list[int] | None = None, return_form: str = "mask") -> Image.Image:
    _require_keys()
    image = Image.open(image_path)
    payload: dict[str, Any] = {"image": _image_base64(image_path), "method": "control" if box else "auto", "refine_mask": "true", "return_form": return_form}
    if box:
        x1 = max(1, min(image.width - 12, int(box[0]))); y1 = max(1, min(image.height - 12, int(box[1])))
        x2 = max(x1 + 10, min(image.width - 1, int(box[2]))); y2 = max(y1 + 10, min(image.height - 1, int(box[3])))
        payload["position"] = [[[x1, y1], [x2, y2]]]
    async with httpx.AsyncClient(timeout=60) as client:
        token = await baidu_access_token(client, BAIDU_PROCESS_API_KEY, BAIDU_PROCESS_SECRET_KEY)
        response = await client.post(BAIDU_SEGMENT_URL, params={"access_token": token}, headers={"Content-Type": "application/json"}, json=payload)
        response.raise_for_status(); body = response.json()
        if body.get("error_code"):
            raise RuntimeError(f"百度智能抠图错误 {body['error_code']}：{body.get('error_msg', '')}")
        encoded = body.get("image")
        if not encoded:
            raise RuntimeError("百度智能抠图未返回图片")
        return Image.open(BytesIO(base64.b64decode(encoded))).copy()


async def _dashscope_generate(model: str, content: list[dict[str, str]], parameters: dict[str, Any]) -> bytes:
    _require_keys()
    payload = {"model": model, "input": {"messages": [{"role": "user", "content": content}]}, "parameters": parameters}
    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post(DASHSCOPE_IMAGE_URL, headers={"Authorization": f"Bearer {DASHSCOPE_API_KEY}", "Content-Type": "application/json"}, json=payload)
        response.raise_for_status(); body = response.json()
        if body.get("code"):
            raise RuntimeError(f"百炼错误 {body.get('code')}：{body.get('message', '')}")
        urls = _collect_urls(body)
        if not urls:
            raise RuntimeError("百炼未返回图片地址")
        image_response = await client.get(urls[-1]); image_response.raise_for_status()
        return image_response.content


async def generate_emoji_domestic(cutout_path: Path, original_path: Path, output_path: Path, strict: bool = False) -> None:
    retry_note = "\n这是纠偏重试：上一版与原主体不够相似。请降低风格化强度，严格逐项复刻图1的轮廓、结构、已有五官和颜色，只改变表面渲染质感。" if strict else ""
    content = [{"image": _data_uri(cutout_path, force_white=True)}, {"image": _data_uri(original_path)}, {"text": EMOJI_PROMPT + retry_note}]
    raw = await _dashscope_generate("wan2.6-image", content, {"size": "1K", "n": 1, "prompt_extend": False, "negative_prompt": NEGATIVE_PROMPT})
    output_path.write_bytes(raw)


async def repair_background_domestic(original_path: Path, bbox: list[int], output_path: Path) -> None:
    text = "移除图1中框选的物品，只补全它后方原本应有的背景。严格保持框外所有像素、构图、光线、透视和其他物体不变。不要添加新主体、文字、装饰或替代物。"
    content = [{"image": _data_uri(original_path)}, {"text": text}]
    raw = await _dashscope_generate("wan2.7-image", content, {"size": "1K", "n": 1, "bbox_list": [[bbox]], "watermark": False})
    output_path.write_bytes(raw)
