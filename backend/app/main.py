from __future__ import annotations
import asyncio
import logging
from io import BytesIO
from pathlib import Path
import time
import uuid
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image
from .config import BAIDU_PROCESS_API_KEY, BAIDU_PROCESS_SECRET_KEY, CORS_ORIGINS, DASHSCOPE_API_KEY, DATA_DIR, MAX_IMAGE_SIDE, MAX_UPLOAD_BYTES, PUBLIC_BASE_URL
from .image_ops import apply_repair_inside_mask, bbox_from_mask, composite_emoji, cutout, edit_mask_with_brush, emoji_is_visibly_stylized, emoji_matches_source, emoji_style_difference, expand_mask, filter_candidates, is_complete_subject, mask_from_image, match_color_intensity, normalize_image, remove_white_matte, save_png
from .providers_domestic import audit_emoji_consistency, baidu_segment, detect_objects_hybrid, generate_emoji_domestic, repair_background_domestic
from .local_segmentation import segment_from_detection_box, segment_from_guided_strokes, segment_from_positive_strokes

app = FastAPI(title="Object Emoji MVP")
logger = logging.getLogger("object-emoji")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])
app.mount("/files", StaticFiles(directory=DATA_DIR), name="files")


class RefineRequest(BaseModel):
    box: list[int] = Field(min_length=4, max_length=4)


class PaintRefineRequest(BaseModel):
    points: list[list[float]] = Field(min_length=1, max_length=4000)
    radius: float = Field(default=18, ge=2, le=160)


class MaskEditRequest(BaseModel):
    points: list[list[float]] = Field(min_length=1, max_length=4000)
    radius: float = Field(default=18, ge=2, le=160)
    mode: str = Field(pattern="^(add|remove)$")


class PreciseMaskEditRequest(MaskEditRequest):
    pass


class EmojiRequest(BaseModel):
    imageId: str
    objectId: str


class CompositionRequest(EmojiRequest):
    emojiUrl: str


class CompositionItem(BaseModel):
    objectId: str
    emojiUrl: str


class BatchCompositionRequest(BaseModel):
    imageId: str
    items: list[CompositionItem] = Field(min_length=1, max_length=8)


async def repair_background_with_retry(input_path: Path, bbox: list[int], output_path: Path) -> None:
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            await asyncio.wait_for(repair_background_domestic(input_path, bbox, output_path), timeout=180)
            return
        except Exception as exc:
            last_error = exc
            if attempt == 0:
                await asyncio.sleep(.8)
    assert last_error is not None
    raise last_error


async def generate_emoji_with_retry(cutout_path: Path, original_path: Path, output_path: Path, strict: bool = False, correction: str = "") -> None:
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            await asyncio.wait_for(generate_emoji_domestic(cutout_path, original_path, output_path, strict=strict, correction=correction), timeout=180)
            return
        except Exception as exc:
            last_error = exc
            if attempt == 0:
                await asyncio.sleep(.8)
    assert last_error is not None
    raise last_error


def background_cache_path(folder: Path, object_id: str) -> Path:
    mask_path = folder / f"mask_{object_id}.png"
    version = mask_path.stat().st_mtime_ns
    return folder / f"background_{object_id}_{version}.png"


async def ensure_background_cache(folder: Path, object_id: str, input_path: Path) -> tuple[Path, bool]:
    cache_path = background_cache_path(folder, object_id)
    if cache_path.exists():
        return cache_path, True
    mask = Image.open(folder / f"mask_{object_id}.png").convert("L")
    try:
        await repair_background_with_retry(input_path, list(bbox_from_mask(mask)), cache_path)
    except Exception:
        cache_path.unlink(missing_ok=True)
        raise
    return cache_path, False


def item_dir(image_id: str) -> Path:
    path = DATA_DIR / image_id
    if not path.exists():
        raise HTTPException(404, "图片任务不存在")
    return path


def file_url(path: Path) -> str:
    relative = "/files/" + path.relative_to(DATA_DIR).as_posix()
    return PUBLIC_BASE_URL + relative if PUBLIC_BASE_URL else relative


def provider_error_message(exc: Exception) -> str:
    message = str(exc)
    if "17" in message and "limit" in message.lower():
        return "百度智能抠图今日额度已用完，请开通付费或等待额度恢复"
    return message or type(exc).__name__


def object_payload(image_id: str, object_id: str) -> dict:
    folder = item_dir(image_id)
    mask_path = folder / f"mask_{object_id}.png"
    preview_path = folder / f"preview_{object_id}.png"
    if not mask_path.exists():
        raise HTTPException(404, "主体不存在")
    mask = Image.open(mask_path).convert("L")
    version = mask_path.stat().st_mtime_ns
    return {"id": object_id, "maskUrl": f"{file_url(mask_path)}?v={version}", "previewUrl": f"{file_url(preview_path)}?v={version}", "bbox": list(bbox_from_mask(mask))}


@app.get("/api/health")
async def health():
    return {"ok": True}


@app.get("/api/health/providers")
async def provider_health():
    """启动器使用的轻量检查：验证配置存在且两个第三方域名可连接。"""
    import httpx
    configured = bool(DASHSCOPE_API_KEY and BAIDU_PROCESS_API_KEY and BAIDU_PROCESS_SECRET_KEY)
    results: dict[str, bool] = {"dashscope": False, "baidu": False}
    if configured:
        async with httpx.AsyncClient(timeout=12) as client:
            for name, url in (
                ("dashscope", "https://dashscope.aliyuncs.com/"),
                ("baidu", "https://aip.baidubce.com/"),
            ):
                try:
                    response = await client.get(url)
                    results[name] = response.status_code < 500
                except Exception:
                    results[name] = False
    return {"ok": configured and all(results.values()), "configured": configured, "providers": results}


@app.post("/api/images")
async def upload_image(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(415, "仅支持 JPG、PNG 和 WebP")
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "图片不能超过 15MB")
    image_id = uuid.uuid4().hex[:16]
    folder = DATA_DIR / image_id; folder.mkdir(parents=True)
    try:
        image = normalize_image(raw, MAX_IMAGE_SIDE)
    except Exception as exc:
        raise HTTPException(400, "图片文件无法读取") from exc
    original_path = folder / "original.png"; save_png(image, original_path)
    return {"imageId": image_id, "imageUrl": file_url(original_path), "width": image.width, "height": image.height, "objects": []}


@app.post("/api/images/{image_id}/detect")
async def detect_image(image_id: str):
    folder = item_dir(image_id); original_path = folder / "original.png"
    image = Image.open(original_path).convert("RGB")
    detection_warning = ""
    detections = []
    masks = []
    labels = []
    scores = []
    try:
        detections = await detect_objects_hybrid(original_path)
        for detection in detections:
            try:
                masks.append(await asyncio.to_thread(segment_from_detection_box, original_path, detection["box"]))
                labels.append(detection["label"])
                scores.append(detection["score"])
            except Exception:
                continue
        if not masks:
            detection_warning = "自动识别暂时失败，请使用涂抹工具选择主体"
        masks = [mask for mask in filter_candidates(masks, image.size, limit=8) if is_complete_subject(mask, image.size)]
        if not masks:
            detection_warning = "未找到完整独立主体，请使用涂抹工具选择主体"
    except Exception as exc:
        detection_warning = "自动识别失败啦，再试一次吧"
    objects = []
    for index, mask in enumerate(masks):
        object_id = f"{index + 1:02d}"
        mask_path = folder / f"mask_{object_id}.png"; save_png(mask, mask_path)
        preview, _ = cutout(image, mask); save_png(preview, folder / f"preview_{object_id}.png")
        payload = object_payload(image_id, object_id)
        mask_box = list(bbox_from_mask(mask))
        matched = min(range(len(detections)), key=lambda i: sum(abs(a-b) for a,b in zip(mask_box, detections[i]["box"]))) if detections else -1
        payload["label"] = labels[matched] if matched >= 0 else "主体"
        payload["score"] = scores[matched] if matched >= 0 else 0
        objects.append(payload)
    return {"imageId": image_id, "imageUrl": file_url(original_path), "width": image.width, "height": image.height, "objects": objects, "detectionWarning": detection_warning}


@app.post("/api/images/{image_id}/refine")
async def refine(image_id: str, request: RefineRequest):
    folder = item_dir(image_id); original_path = folder / "original.png"
    image = Image.open(original_path).convert("RGB")
    try:
        result = await baidu_segment(original_path, request.box, "mask")
        mask = mask_from_image(result, image.size)
        bbox_from_mask(mask)
    except Exception as exc:
        raise HTTPException(502, f"框选分割失败：{type(exc).__name__}") from exc
    object_id = "r" + uuid.uuid4().hex[:6]
    save_png(mask, folder / f"mask_{object_id}.png")
    preview, _ = cutout(image, mask); save_png(preview, folder / f"preview_{object_id}.png")
    return object_payload(image_id, object_id)


@app.post("/api/images/{image_id}/refine-paint")
async def refine_paint(image_id: str, request: PaintRefineRequest):
    started = time.perf_counter()
    folder = item_dir(image_id); original_path = folder / "original.png"
    image = Image.open(original_path).convert("RGB")
    if any(len(point) != 2 for point in request.points):
        raise HTTPException(422, "涂抹坐标格式错误")
    try:
        mask = await asyncio.to_thread(segment_from_positive_strokes, original_path, request.points, request.radius)
    except Exception as exc:
        logger.exception("paint segmentation failed image_id=%s", image_id)
        message = str(exc)
        if "完整主体" in message or "覆盖" in message or "返回主体" in message:
            detail = "没有找到完整主体，请换个位置简单涂几笔"
        else:
            detail = "抠图模型运行失败啦，再试一次吧"
        raise HTTPException(502, detail) from exc
    object_id = "p" + uuid.uuid4().hex[:6]
    save_png(mask, folder / f"mask_{object_id}.png")
    preview, _ = cutout(image, mask); save_png(preview, folder / f"preview_{object_id}.png")
    payload = object_payload(image_id, object_id)
    payload["label"] = "手动涂抹主体"; payload["score"] = 1.0
    logger.info("paint segmentation image_id=%s elapsed_ms=%d", image_id, round((time.perf_counter()-started)*1000))
    return payload


@app.post("/api/images/{image_id}/masks/{object_id}/edit")
async def edit_object_mask(image_id: str, object_id: str, request: MaskEditRequest):
    folder = item_dir(image_id)
    original = Image.open(folder / "original.png").convert("RGB")
    mask_path = folder / f"mask_{object_id}.png"
    if not mask_path.exists():
        raise HTTPException(404, "主体不存在")
    try:
        current_mask = Image.open(mask_path).convert("L")
        stroke_points = [[point[0], point[1]] for point in request.points if len(point) >= 2]
        edited = await asyncio.to_thread(
            segment_from_guided_strokes,
            folder / "original.png",
            current_mask,
            stroke_points if request.mode == "add" else [],
            stroke_points if request.mode == "remove" else [],
            request.radius,
        )
        bbox_from_mask(edited)
    except Exception as exc:
        logger.exception("smart mask edit failed image_id=%s object_id=%s mode=%s", image_id, object_id, request.mode)
        raise HTTPException(422, "智能修正没有找到合适边界，请换个位置再画一笔") from exc
    save_png(edited, mask_path)
    preview, _ = cutout(original, edited)
    save_png(preview, folder / f"preview_{object_id}.png")
    payload = object_payload(image_id, object_id)
    payload["label"] = "智能修正主体"
    payload["score"] = 1.0
    return payload


@app.post("/api/images/{image_id}/masks/{object_id}/precise-edit")
async def precise_edit_object_mask(image_id: str, object_id: str, request: PreciseMaskEditRequest):
    """像素级兜底工具；只在智能修正无法满足用户意图时使用。"""
    folder = item_dir(image_id)
    original = Image.open(folder / "original.png").convert("RGB")
    mask_path = folder / f"mask_{object_id}.png"
    if not mask_path.exists():
        raise HTTPException(404, "主体不存在")
    try:
        edited = edit_mask_with_brush(Image.open(mask_path), request.points, request.radius, request.mode)
        bbox_from_mask(edited)
    except Exception as exc:
        raise HTTPException(422, "精细画笔修改失败啦，再试一次吧") from exc
    save_png(edited, mask_path)
    preview, _ = cutout(original, edited)
    save_png(preview, folder / f"preview_{object_id}.png")
    payload = object_payload(image_id, object_id)
    payload["label"] = "精细修正主体"
    payload["score"] = 1.0
    return payload


@app.post("/api/emojis")
async def make_emoji(request: EmojiRequest):
    started = time.perf_counter(); folder = item_dir(request.imageId)
    original_path = folder / "original.png"
    original = Image.open(original_path).convert("RGB")
    mask = Image.open(folder / f"mask_{request.objectId}.png").convert("L")
    cutout_image, _ = cutout(original, mask, padding=40)
    cutout_path = folder / f"cutout_{request.objectId}.png"; save_png(cutout_image, cutout_path)
    raw_path = folder / f"styled_{request.objectId}.png"
    output_path = folder / f"emoji_{request.objectId}_{uuid.uuid4().hex[:6]}.png"
    background_started = time.perf_counter()
    background_task = asyncio.create_task(ensure_background_cache(folder, request.objectId, original_path))
    try:
        # 每次点击都重新生成，避免提示词或蒙版改变后仍复用旧缓存。
        generation_started = time.perf_counter()
        await generate_emoji_with_retry(cutout_path, original_path, raw_path)
        # 万相不输出可靠透明通道，再次使用百度智能抠图生成透明 PNG。
        rgba = await baidu_segment(raw_path, None, "rgba")
        transparent = remove_white_matte(rgba)
        audit_path = folder / f"audit_{request.objectId}.png"; save_png(transparent, audit_path)
        semantic_ok, semantic_reason, style_class = await audit_emoji_consistency(cutout_path, audit_path)
        structure_ok = emoji_matches_source(transparent, cutout_image)
        style_ok = style_class == "3D_EMOJI" and emoji_is_visibly_stylized(transparent, cutout_image)
        if not structure_ok or not style_ok or not semantic_ok:
            first_candidate = transparent.copy()
            first_structure_ok, first_semantic_ok, first_style_ok = structure_ok, semantic_ok, style_ok
            first_style_score = emoji_style_difference(first_candidate, cutout_image)
            first_path = folder / f"audit_{request.objectId}_first.png"
            save_png(transparent, first_path)
            corrections: list[str] = []
            if not structure_ok:
                corrections.append("完整保留原主体外轮廓、长宽比例和所有主要组成部分，不能缺边、扩大或改变形状")
            if not style_ok:
                if style_class == "FLAT_CARTOON":
                    corrections.append("上一版是二维漫画风；彻底移除描边、勾线、赛璐璐阴影和平面色块，严格改为参考图3至图5的苹果系统圆润半立体 Emoji 材质")
                elif style_class == "PHOTOREALISTIC":
                    corrections.append("上一版仍过于写实；显著增强圆润半立体 Emoji 重绘效果并概括摄影纹理")
                else:
                    corrections.append("严格使用参考图3至图5的苹果系统圆润半立体 Emoji 风格，无描边、连续柔和渐变、自然高光和轻微环境遮蔽")
            if not semantic_ok:
                corrections.append(f"消除实体组成错误：{semantic_reason or '不得新增或遗漏任何主要实体'}")
            logger.info(
                "emoji first candidate rejected image_id=%s object_id=%s structure_ok=%s style_ok=%s semantic_ok=%s reason=%s",
                request.imageId, request.objectId, structure_ok, style_ok, semantic_ok, f"{style_class}: {semantic_reason}",
            )
            await generate_emoji_with_retry(cutout_path, original_path, raw_path, strict=True, correction="；".join(corrections))
            rgba = await baidu_segment(raw_path, None, "rgba")
            transparent = remove_white_matte(rgba)
            save_png(transparent, audit_path)
            semantic_ok, semantic_reason, style_class = await audit_emoji_consistency(cutout_path, audit_path)
            structure_ok = emoji_matches_source(transparent, cutout_image)
            style_ok = style_class == "3D_EMOJI" and emoji_is_visibly_stylized(transparent, cutout_image)
            retry_style_score = emoji_style_difference(transparent, cutout_image)
            logger.info(
                "emoji correction candidate image_id=%s object_id=%s structure_ok=%s style_ok=%s semantic_ok=%s reason=%s",
                request.imageId, request.objectId, structure_ok, style_ok, semantic_ok, f"{style_class}: {semantic_reason}",
            )
            retry_critical_ok = structure_ok and semantic_ok and style_ok
            first_critical_ok = first_structure_ok and first_semantic_ok and first_style_ok
            if first_critical_ok and (not retry_critical_ok or first_style_score > retry_style_score):
                transparent = first_candidate
                structure_ok, semantic_ok, style_ok = first_structure_ok, first_semantic_ok, first_style_ok
                logger.info(
                    "emoji selected first candidate image_id=%s object_id=%s first_style_score=%.2f retry_style_score=%.2f",
                    request.imageId, request.objectId, first_style_score, retry_style_score,
                )
            if not structure_ok:
                raise RuntimeError("生成结果未完整保留原主体结构，请重新生成")
            if not semantic_ok:
                raise RuntimeError(f"生成结果与原主体不一致：{semantic_reason or '新增或缺少了实体'}")
            if not style_ok:
                raise RuntimeError(f"生成结果不是苹果系统 Emoji 风格：{style_class}")
        # 不再进行九宫格颜色校正；该步骤会造成明显矩形色块。
        save_png(match_color_intensity(transparent, cutout_image), output_path)
        logger.info("emoji generation image_id=%s object_id=%s elapsed_ms=%d", request.imageId, request.objectId, round((time.perf_counter()-generation_started)*1000))
    except asyncio.TimeoutError as exc:
        background_task.cancel()
        await asyncio.gather(background_task, return_exceptions=True)
        raise HTTPException(504, "Emoji 生成超过 60 秒") from exc
    except Exception as exc:
        background_task.cancel()
        await asyncio.gather(background_task, return_exceptions=True)
        message = provider_error_message(exc)
        lower = message.lower()
        if "429" in lower or "rate" in lower or "thrott" in lower:
            detail = "图片生成服务繁忙，请稍后再试"
        elif any(word in lower for word in ("quota", "balance", "insufficient", "额度", "余额")):
            detail = "图片生成额度不足，请检查服务余额"
        elif "connect" in lower or "network" in lower:
            detail = "无法连接图片生成服务，请检查网络后重试"
        else:
            detail = f"风格化生成失败：{message[:180]}"
        raise HTTPException(502, detail) from exc
    try:
        _, cache_hit = await background_task
        logger.info("background prepared image_id=%s object_id=%s cache_hit=%s elapsed_ms=%d", request.imageId, request.objectId, cache_hit, round((time.perf_counter()-background_started)*1000))
    except Exception:
        # 预生成失败不影响 Emoji 输出；合成阶段会沿用原来的错误处理并再次尝试。
        logger.warning("background pre-generation failed image_id=%s object_id=%s", request.imageId, request.objectId, exc_info=True)
    return {"emojiUrl": file_url(output_path), "elapsedMs": round((time.perf_counter()-started)*1000)}


@app.post("/api/compositions")
async def compose(request: CompositionRequest):
    started = time.perf_counter(); folder = item_dir(request.imageId)
    original_path = folder / "original.png"
    mask = Image.open(folder / f"mask_{request.objectId}.png").convert("L")
    try:
        repaired_path, cache_hit = await ensure_background_cache(folder, request.objectId, original_path)
    except asyncio.TimeoutError as exc:
        raise HTTPException(504, "背景修复超过 60 秒；透明 Emoji 仍可下载") from exc
    except Exception as exc:
        raise HTTPException(502, f"背景修复失败：{type(exc).__name__}") from exc
    emoji_name = Path(request.emojiUrl.split("?")[0]).name
    emoji_path = folder / emoji_name
    if not emoji_path.exists():
        raise HTTPException(404, "Emoji 文件不存在")
    original = Image.open(original_path).convert("RGBA")
    repaired = apply_repair_inside_mask(original, Image.open(repaired_path), mask)
    emoji = Image.open(emoji_path).convert("RGBA")
    result = composite_emoji(repaired, emoji, bbox_from_mask(mask), add_shadow=False, target_mask=mask)
    result_path = folder / f"result_{request.objectId}_{uuid.uuid4().hex[:6]}.png"; save_png(result, result_path)
    logger.info("composition image_id=%s object_id=%s background_cache_hit=%s elapsed_ms=%d", request.imageId, request.objectId, cache_hit, round((time.perf_counter()-started)*1000))
    return {"resultUrl": file_url(result_path), "repairedUrl": file_url(repaired_path), "elapsedMs": round((time.perf_counter()-started)*1000)}


@app.post("/api/compositions/batch")
async def compose_batch(request: BatchCompositionRequest):
    """从原图非破坏地重放最多两个主体替换，支持删除任一替换后重新组合。"""
    started = time.perf_counter(); folder = item_dir(request.imageId)
    current = Image.open(folder / "original.png").convert("RGBA")
    run_id = uuid.uuid4().hex[:6]
    for index, item in enumerate(request.items):
        mask_path = folder / f"mask_{item.objectId}.png"
        emoji_path = folder / Path(item.emojiUrl.split("?")[0]).name
        if not mask_path.exists() or not emoji_path.exists():
            raise HTTPException(404, "主体或 Emoji 文件不存在")
        mask = Image.open(mask_path).convert("L")
        input_path = folder / f"batch_input_{run_id}_{index}.png"
        save_png(current.convert("RGB"), input_path)
        try:
            if index == 0:
                repaired_path, cache_hit = await ensure_background_cache(folder, item.objectId, input_path)
            else:
                repaired_path = folder / f"batch_repaired_{run_id}_{index}.png"
                await repair_background_with_retry(input_path, list(bbox_from_mask(mask)), repaired_path)
                cache_hit = False
        except asyncio.TimeoutError as exc:
            raise HTTPException(504, "多主体背景修复超时") from exc
        except Exception as exc:
            raise HTTPException(502, f"多主体背景修复失败：{type(exc).__name__}") from exc
        repaired = apply_repair_inside_mask(current, Image.open(repaired_path), mask)
        emoji = Image.open(emoji_path).convert("RGBA")
        current = composite_emoji(repaired, emoji, bbox_from_mask(mask), add_shadow=False, target_mask=mask)
        logger.info("batch composition image_id=%s object_id=%s index=%d background_cache_hit=%s", request.imageId, item.objectId, index, cache_hit)
        input_path.unlink(missing_ok=True)
    result_path = folder / f"result_batch_{run_id}.png"
    save_png(current, result_path)
    return {"resultUrl": file_url(result_path), "elapsedMs": round((time.perf_counter()-started)*1000)}
