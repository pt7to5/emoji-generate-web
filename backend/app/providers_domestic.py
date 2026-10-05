from __future__ import annotations

import asyncio
import base64
from io import BytesIO
import json
import re
from pathlib import Path
from typing import Any

import httpx
from PIL import Image, ImageFilter

from .config import (
    BAIDU_DETECT_API_KEY,
    BAIDU_DETECT_SECRET_KEY,
    BAIDU_PROCESS_API_KEY,
    BAIDU_PROCESS_SECRET_KEY,
    DASHSCOPE_API_KEY,
    EMOJI_GENERATION_CANDIDATES,
    EMOJI_GENERATION_MODEL,
)

BAIDU_TOKEN_URL = "https://aip.baidubce.com/oauth/2.0/token"
BAIDU_SUBJECT_DETECT_URL = "https://aip.baidubce.com/rest/2.0/image-classify/v1/object_detect"
BAIDU_MULTI_OBJECT_DETECT_URL = "https://aip.baidubce.com/rest/2.0/image-classify/v1/multi_object_detect"
BAIDU_SEGMENT_URL = "https://aip.baidubce.com/rest/2.0/image-process/v1/segment"
DASHSCOPE_IMAGE_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
STYLE_REFERENCE_PATHS = tuple((Path(__file__).parent / "assets" / name) for name in (
    "emoji_style_drinks.jpg",
    "emoji_style_cakes.jpg",
    "emoji_style_dessert.jpg",
))

EMOJI_PROMPT = """图1是唯一需要转换的主体，只能使用图1透明区域内可见的内容。图2是已经去除所有物体语义的低频光线图，只能参考它的整体明暗分布、色温、主光方向和环境色，严禁从图2生成任何物体、形状、场景或陪衬主体。图3、图4、图5仅是苹果系统 Emoji 风格参考，只能参考它们的圆润半立体材质、连续柔和渐变、自然高光、轻微内阴影和环境遮蔽；绝对不能复制其中的杯子、杯垫、盘子、蛋糕、花朵、蜡烛、五官或任何实体。
先逐项复刻图1：主体类别、原图中可见的外轮廓、姿态、视角、长宽比例、组成部件、数量、已有装饰、分层结构和每种颜色的位置必须一致。不得改成另一种动物、食物或通用卡通形象，不得新增、删除、合并或放大任何可见结构。
图1透明区域内出现的所有物体共同组成一个不可拆分的组合主体。盘子、托盘、杯子、花盆、包装、底座、支架以及主体承载物都必须保留并一起风格化；例如“蛋糕放在盘子上”必须输出完整的蛋糕和完整的盘子，不能只输出蛋糕。各组成部分的相对位置、遮挡关系和尺寸比例必须与图1一致。
严格保持图1已有的遮挡边界：如果主体被手、餐具、容器或其他前景物体挡住，只转换图1中实际可见的部分，不推测、不补画、不扩展被遮挡或画面外的部分。输出轮廓必须与图1透明区域的可见轮廓对应，以便原位置替换。
在主体身份和几何结构不变的前提下进行强度明确的高品质 Emoji 化。整体视觉应约为“七成移动端 Emoji 渲染、三成原物摄影特征”：保留用于辨认原主体的结构与配色，但必须主动去除照片感，不能只是抠图、磨皮、提亮或调色。
把照片中的细碎纹理、噪点、纤维、细小褶皱和零散高频细节概括成干净、柔和、圆润的连续曲面；强化清晰的立体体积、适度饱满的造型、柔和方向光、自然高光、渐进明暗、轻微内阴影和环境遮蔽。边缘只能由体积和光影形成，禁止任何深色描边、勾线或漫画轮廓。缩小到手机 Emoji 尺寸时仍应有清楚的图标识别度。效果必须与图3至图5的苹果系统 Emoji 材质语言一致，而不是写实照片、扁平插画、漫画贴纸或厚重塑料玩具。
对于花束、植物、毛绒物和复杂甜品，可以简化单片花瓣、叶脉、绒毛、奶油纹路等微观细节，但必须保留主要花朵/叶簇/装饰簇、包装层次、主体数量、相对位置和整体外轮廓。简化纹理不等于删除组成物。
结果必须一眼就能识别为经过明显 Emoji 风格化，同时继续保持原有主体身份、主要结构、轮廓、颜色分区和组合关系。
颜色强度必须服从图1：不得自动提亮，不得提高饱和度，不得把低饱和颜色改成鲜艳色或荧光色。白色、奶油色、浅灰色等低饱和区域必须保持低饱和；绿色、黄色、红色等彩色区域的明度与浓度也要接近图1。
光线方向、色温、亮暗和阴影方向综合继承图1主体自身表现与图2的低频光线布局，使生成物能自然融入原照片。
输出要求：完整单个主体，尺寸比例与图1一致，原视角，居中，纯白背景，四周留出少量空白，不裁切。
生成前先核对图1已有组件清单，输出中每一个实体都必须能在图1透明区域内找到对应物。绝对禁止凭空增加盘子、托盘、底座、支架、容器、包装、装饰物或其他实体；只有图1原本存在时才允许保留并风格化。
严禁：扁平矢量插画、色带、等高线式分层、几何色块、硬边渐变、明显分区接缝、低多边形、厚重3D玩具、塑料高光、镜面反射、擅自拟人化、添加新五官或装饰、改变主体类别、改变颜色、文字、水印、边框和场景。"""

NEGATIVE_PROMPT = "凭空新增盘子，凭空新增托盘，凭空新增底座，凭空新增支架，凭空新增容器，凭空新增包装，凭空新增装饰，复制参考图实体，主体身份改变，结构缺失，遗漏原有盘子，遗漏原有托盘，遗漏原有底座，遗漏原有容器，拆散组合主体，补画被遮挡部位，扩展到原可见轮廓之外，新增结构，部件数量改变，比例改变，主体放大，姿态改变，颜色改变，通用卡通形象，独立贴纸，自动提亮，提高饱和度，荧光色，糖果色，二维漫画，漫画贴纸，赛璐璐上色，粗描边，深色轮廓线，棕色勾线，黑色勾线，扁平矢量插画，平面色块，几何色块，色带，等高线分层，分区接缝，硬边渐变，低多边形，复杂背景，无关主体，文字，水印，边框，裁切，错误透视，新增眼睛，新增嘴巴，新增四肢，过度拟人，线稿，硬阴影，过曝，镜面反射，塑料玩具质感，厚重3D模型，悬浮"


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


def _baidu_segment_payload(path: Path) -> tuple[str, tuple[int, int], tuple[int, int]]:
    """压缩百度抠图输入，并返回原尺寸与实际上传尺寸供坐标换算。"""
    with Image.open(path) as source:
        image = source.convert("RGB")
    original_size = image.size
    image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
    uploaded_size = image.size
    buffer = BytesIO()
    image.save(buffer, "JPEG", quality=86, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii"), original_size, uploaded_size


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


def _lighting_map_data_uri(path: Path) -> str:
    """只保留整图低频明暗和色温，消除可被生成模型复制的物体语义。"""
    image = Image.open(path).convert("RGB")
    longest = max(image.size)
    small_size = tuple(max(6, round(value / longest * 18)) for value in image.size)
    light = image.resize(small_size, Image.Resampling.BOX)
    # Wan 2.7 要求每个输入图像的宽和高都至少为 240px。
    shortest = min(light.size)
    scale = max(1, (240 + shortest - 1) // shortest)
    light = light.resize((light.width * scale, light.height * scale), Image.Resampling.BICUBIC)
    light = light.filter(ImageFilter.GaussianBlur(max(8, round(min(light.size) * .08))))
    buffer = BytesIO(); light.save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


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
    prompt = """检测图片中所有适合独立风格化替换的完整视觉单位。每个单位必须符合人的自然选择习惯：
食物默认包含承托它的盘子和明显关联餐具；饮品包含杯子、杯内饮品、吸管和固定装饰；花朵选择完整花束或整盆花；动物选择单只完整动物及固定穿戴；玩偶选择完整玩偶；普通物品选择单个完整物体。
不要把奶油、装饰叶、吸管、水果片、冰块、盘子、花盆、动物项圈等组成部分单独列为主体，也不要把桌面、背景和附近无关物体并入主体。
同一个盘子、托盘或餐盒里的多个食物必须合并为一个完整摆盘，只返回一个框；禁止同时返回整盘和盘内单品，也禁止把盘内蛋糕、甜点逐个拆开。
绝对不要返回人物、人体、手、衣服、家具、墙面、装饰画、车辆、建筑或其他环境元素；人物手里拿着食物时只返回食物组合。
严格只返回 JSON 数组，每项格式为 {\"label\":\"简短中文名称\",\"bbox\":[x1,y1,x2,y2]}。
bbox 使用 0 到 999 的归一化坐标。列出所有明显主体，最多 8 个，不要输出说明或 Markdown。"""
    payload = {
        "model": "qwen3.6-plus",
        "input": {"messages": [{"role": "user", "content": [
            {"image": _data_uri(image_path)}, {"text": prompt},
        ]}]},
        "parameters": {"result_format": "message", "vl_high_resolution_images": True, "temperature": 0.1, "enable_thinking": False},
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
        box = (item.get("bbox") or item.get("bbox_2d")) if isinstance(item, dict) else None
        if not isinstance(box, list) or len(box) != 4:
            continue
        x1, y1, x2, y2 = [float(value) for value in box]
        absolute = [round(x1 / 999 * width), round(y1 / 999 * height), round(x2 / 999 * width), round(y2 / 999 * height)]
        if absolute[2] - absolute[0] < width * .04 or absolute[3] - absolute[1] < height * .04:
            continue
        detections.append({"label": item.get("label") or "主体", "score": .82, "box": absolute, "source": "qwen3.6-plus"})
    return detections


async def audit_emoji_consistency(source_path: Path, emoji_path: Path) -> tuple[bool, str, str]:
    """分别审核实体一致性和苹果系统 Emoji 风格。"""
    _require_keys()
    prompt = """图1是识别选区内的原主体，图2是生成结果，图3至图5是苹果系统 Emoji 风格参考。请分别审核实体组成与视觉风格。
允许：材质简化、圆润化、柔和光影，以及杯内原有饮品、冰块、水果、吸管和装饰的风格化。
不允许：图2出现图1没有的盘子、托盘、底座、支架、容器、包装、食物、装饰、文字或任何新实体；也不允许删除图1已有的主要实体或改变数量。
风格分类只能是：3D_EMOJI（像图3至图5，圆润半立体、无描边、连续柔和渐变）、FLAT_CARTOON（漫画、粗描边、赛璐璐、平面色块或贴纸插画）、PHOTOREALISTIC（仍接近照片）或 INVALID。
忽略纯白/透明背景、阴影和细微纹理差异。只返回 JSON：{"pass":true或false,"style":"3D_EMOJI或FLAT_CARTOON或PHOTOREALISTIC或INVALID","added":["新增项"],"missing":["缺失项"],"reason":"简短原因"}。只要 added 或 missing 中存在主要实体，pass 必须为 false。"""
    payload = {
        "model": "qwen3.6-plus",
        "input": {"messages": [{"role": "user", "content": [
            {"image": _data_uri(source_path, force_white=True)},
            {"image": _data_uri(emoji_path, force_white=True)},
            *({"image": _data_uri(path)} for path in STYLE_REFERENCE_PATHS if path.exists()),
            {"text": prompt},
        ]}]},
        "parameters": {"result_format": "message", "vl_high_resolution_images": True, "temperature": 0, "enable_thinking": False},
    }
    async with httpx.AsyncClient(timeout=90) as client:
        response = await client.post(DASHSCOPE_IMAGE_URL, headers={"Authorization": f"Bearer {DASHSCOPE_API_KEY}", "Content-Type": "application/json"}, json=payload)
        if response.is_error:
            raise RuntimeError(f"百炼一致性审核 HTTP {response.status_code}：{response.text[:500]}")
        body = response.json()
    if body.get("code"):
        raise RuntimeError(f"百炼一致性审核错误 {body.get('code')}：{body.get('message', '')}")
    content = (((body.get("output") or {}).get("choices") or [{}])[0].get("message") or {}).get("content") or []
    text = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return False, "一致性审核未返回有效结果", "INVALID"
    try:
        result = json.loads(match.group(0))
    except json.JSONDecodeError:
        return False, "一致性审核结果无法解析", "INVALID"
    added = result.get("added") if isinstance(result.get("added"), list) else []
    missing = result.get("missing") if isinstance(result.get("missing"), list) else []
    passed = result.get("pass") is True and not added and not missing
    reason = str(result.get("reason") or "")
    if added:
        reason = "新增了：" + "、".join(map(str, added))
    elif missing:
        reason = "缺少了：" + "、".join(map(str, missing))
    style = str(result.get("style") or "INVALID").upper()
    if style not in {"3D_EMOJI", "FLAT_CARTOON", "PHOTOREALISTIC", "INVALID"}:
        style = "INVALID"
    return passed, reason, style


async def detect_objects_hybrid(image_path: Path) -> list[dict[str, Any]]:
    baidu, qwen = await asyncio.gather(baidu_detect_objects(image_path), qwen_detect_objects(image_path), return_exceptions=True)
    primary = baidu if isinstance(baidu, list) else []
    semantic_groups = qwen if isinstance(qwen, list) else []
    # 完整视觉单位由具备语义分组能力的模型决定。通用检测框只在语义检测
    # 完全无结果时回退，避免把盘子、装饰和核心食物作为多个并列主体返回。
    selected = semantic_groups or primary
    if not selected:
        error = baidu if isinstance(baidu, Exception) else qwen
        raise RuntimeError(str(error))
    return selected[:8]


async def baidu_segment(image_path: Path, box: list[int] | None = None, return_form: str = "mask") -> Image.Image:
    _require_keys()
    encoded, original_size, uploaded_size = _baidu_segment_payload(image_path)
    original_width, original_height = original_size
    upload_width, upload_height = uploaded_size
    scale_x = upload_width / max(1, original_width)
    scale_y = upload_height / max(1, original_height)
    payload: dict[str, Any] = {"image": encoded, "method": "control" if box else "auto", "refine_mask": "true", "return_form": return_form}
    if box:
        x1 = max(1, min(upload_width - 12, round(box[0] * scale_x))); y1 = max(1, min(upload_height - 12, round(box[1] * scale_y)))
        x2 = max(x1 + 10, min(upload_width - 1, round(box[2] * scale_x))); y2 = max(y1 + 10, min(upload_height - 1, round(box[3] * scale_y)))
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
        if response.is_error:
            raise RuntimeError(f"百炼 HTTP {response.status_code}：{response.text[:500]}")
        body = response.json()
        if body.get("code"):
            raise RuntimeError(f"百炼错误 {body.get('code')}：{body.get('message', '')}")
        urls = _collect_urls(body)
        if not urls:
            raise RuntimeError("百炼未返回图片地址")
        image_response = await client.get(urls[-1]); image_response.raise_for_status()
        return image_response.content


async def generate_emoji_domestic(cutout_path: Path, original_path: Path, output_path: Path, strict: bool = False, correction: str = "") -> None:
    retry_note = "\n这是纠偏重试：上一版未同时满足结构一致和明显 Emoji 化。保持较强的移动端半立体 Emoji 风格，不得退回写实照片或只改变表面色调；同时严格逐项复刻图1的可见外轮廓、所有可见承载物、盘子、托盘、底座、容器、主要结构、已有五官和颜色分区。可以进一步概括微小摄影纹理，但任何主要组成部分都不能删除，也不要补画被遮挡或画面外的部分。" if strict else ""
    if correction:
        retry_note += "\n本次必须重点修正：" + correction[:400]
    # 第二张图降到极低频，只传递整图明暗和色温布局，不携带可识别物体。
    content = [
        {"image": _data_uri(cutout_path, force_white=True)},
        {"image": _lighting_map_data_uri(original_path)},
        *({"image": _data_uri(path)} for path in STYLE_REFERENCE_PATHS if path.exists()),
        {"text": EMOJI_PROMPT + retry_note},
    ]
    parameters: dict[str, Any] = {
        "size": "1K" if EMOJI_GENERATION_MODEL.startswith("wan") else "1024*1024",
        "n": EMOJI_GENERATION_CANDIDATES,
        "prompt_extend": False,
        "negative_prompt": NEGATIVE_PROMPT,
        "watermark": False,
    }
    raw = await _dashscope_generate(EMOJI_GENERATION_MODEL, content, parameters)
    output_path.write_bytes(raw)


async def repair_background_domestic(original_path: Path, bbox: list[int], output_path: Path) -> None:
    text = "移除图1中框选的物品，只补全它后方原本应有的背景。严格保持框外所有像素、构图、光线、透视和其他物体不变。不要添加新主体、文字、装饰或替代物。"
    content = [{"image": _data_uri(original_path)}, {"text": text}]
    raw = await _dashscope_generate("wan2.7-image", content, {"size": "1K", "n": 1, "bbox_list": [[bbox]], "watermark": False})
    output_path.write_bytes(raw)
