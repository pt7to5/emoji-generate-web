import asyncio

from PIL import Image, ImageDraw

from app.image_ops import apply_repair_inside_mask, bbox_from_mask, bbox_from_strokes, complete_subject_mask, composite_emoji, cutout, edit_mask_with_brush, emoji_is_visibly_stylized, emoji_matches_source, emoji_style_difference, expand_detection_box, expand_mask, fill_enclosed_holes, filter_candidates, is_complete_subject, mask_needs_completion, match_color_intensity, preserve_color_layout, remove_white_matte
from app import main


def sample():
    image = Image.new("RGB", (200, 120), "white")
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((50, 25, 150, 100), fill=255)
    return image, mask


def test_bbox_and_cutout():
    image, mask = sample()
    assert bbox_from_mask(mask) == (50, 25, 151, 101)
    subject, bbox = cutout(image, mask, padding=10)
    assert bbox == (40, 15, 161, 111)
    assert subject.mode == "RGBA"
    assert subject.getchannel("A").getextrema() == (0, 255)


def test_paint_strokes_become_clamped_context_box():
    box = bbox_from_strokes([[5, 10], [35, 40]], 10, (100, 80))
    assert box == [0, 0, 65, 70]


def test_filter_candidates_removes_duplicates():
    _, mask = sample()
    candidates = filter_candidates([mask, mask.copy()], mask.size, limit=5)
    assert len(candidates) == 1


def test_filter_candidates_removes_part_nested_inside_complete_subject():
    large = Image.new("L", (200, 120), 0)
    ImageDraw.Draw(large).rectangle((40, 20, 160, 105), fill=255)
    partial = Image.new("L", (200, 120), 0)
    ImageDraw.Draw(partial).rectangle((55, 76, 150, 102), fill=255)
    candidates = filter_candidates([partial, large], large.size, limit=5)
    assert len(candidates) == 1
    assert bbox_from_mask(candidates[0]) == bbox_from_mask(large)


def test_filter_candidates_keeps_cake_inside_plate_box_when_pixels_differ():
    plate = Image.new("L", (200, 160), 0)
    draw = ImageDraw.Draw(plate)
    draw.ellipse((15, 15, 185, 150), outline=255, width=8)
    cake = Image.new("L", plate.size, 0)
    ImageDraw.Draw(cake).polygon([(65, 70), (145, 68), (135, 125), (60, 125)], fill=255)
    candidates = filter_candidates([plate, cake], plate.size, limit=5)
    assert len(candidates) == 2


def test_expand_and_composite():
    image, mask = sample()
    expanded = expand_mask(mask, 8)
    assert bbox_from_mask(expanded)[0] < bbox_from_mask(mask)[0]
    emoji = Image.new("RGBA", (80, 80), (255, 80, 20, 255))
    result = composite_emoji(image, emoji, bbox_from_mask(mask))
    assert result.size == image.size
    assert result.getpixel((100, 60))[0] == 255


def test_composite_is_bottom_aligned_and_does_not_touch_distant_pixels():
    background = Image.new("RGB", (200, 120), "green")
    mask = Image.new("L", background.size, 0)
    ImageDraw.Draw(mask).rectangle((50, 25, 150, 100), fill=255)
    emoji = Image.new("RGBA", (30, 80), (255, 30, 20, 255))
    result = composite_emoji(background, emoji, bbox_from_mask(mask), target_mask=mask)
    assert result.getpixel((100, 99))[:3] == (255, 30, 20)
    assert result.getpixel((0, 0))[:3] == (0, 128, 0)


def test_expand_wide_detection_box_adds_context_above():
    expanded = expand_detection_box([282, 1480, 709, 1669], (1320, 1749))
    assert expanded == [248, 1319, 743, 1707]


def test_environment_mask_is_rejected_but_object_is_kept():
    object_mask = Image.new("L", (200, 120), 0)
    ImageDraw.Draw(object_mask).rectangle((40, 15, 110, 100), fill=255)
    cloth_mask = Image.new("L", (200, 120), 0)
    ImageDraw.Draw(cloth_mask).rectangle((0, 50, 199, 119), fill=255)
    assert is_complete_subject(object_mask, object_mask.size)
    assert not is_complete_subject(cloth_mask, cloth_mask.size)


def test_repair_cannot_change_pixels_outside_subject_mask():
    original = Image.new("RGB", (20, 20), "green")
    repaired = Image.new("RGB", (20, 20), "red")
    mask = Image.new("L", (20, 20), 0)
    ImageDraw.Draw(mask).rectangle((7, 7, 12, 12), fill=255)
    merged = apply_repair_inside_mask(original, repaired, mask).convert("RGB")
    assert merged.getpixel((0, 0)) == (0, 128, 0)
    assert merged.getpixel((9, 9)) == (255, 0, 0)


def test_white_matte_cleanup_clears_transparent_rgb_and_shrinks_halo():
    image = Image.new("RGBA", (5, 5), (255, 255, 255, 0))
    image.putpixel((2, 2), (200, 80, 20, 255))
    cleaned = remove_white_matte(image)
    assert cleaned.getpixel((0, 0)) == (0, 0, 0, 0)
    assert cleaned.getpixel((2, 2))[3] > 0


def test_mask_brush_can_add_and_remove_exact_regions():
    mask = Image.new("L", (100, 100), 0)
    added = edit_mask_with_brush(mask, [[20, 20, 1], [80, 20, 0]], 6, "add")
    assert added.getpixel((50, 20)) == 255
    removed = edit_mask_with_brush(added, [[45, 20, 1], [55, 20, 0]], 4, "remove")
    assert removed.getpixel((50, 20)) == 0
    assert removed.getpixel((25, 20)) == 255


def test_fragmented_subject_triggers_completion_and_stays_near_detection_box():
    primary = Image.new("L", (200, 140), 0)
    draw = ImageDraw.Draw(primary)
    draw.rectangle((60, 35, 140, 75), fill=255)
    draw.rectangle((64, 84, 136, 112), fill=255)
    box = [55, 30, 145, 115]
    assert mask_needs_completion(primary, box)
    completed = complete_subject_mask(primary, None, box)
    assert completed.getpixel((100, 80)) > 0
    assert completed.getpixel((15, 15)) == 0


def test_fill_enclosed_holes_restores_internal_color_region():
    mask = Image.new("L", (120, 120), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((15, 10, 105, 112), fill=255)
    draw.ellipse((50, 58, 70, 78), fill=0)
    filled = fill_enclosed_holes(mask, [10, 5, 110, 117])
    assert filled.getpixel((60, 68)) > 200
    assert filled.getpixel((5, 5)) == 0


def test_large_enclosed_region_is_not_filled_into_rectangle_artifact():
    mask = Image.new("L", (120, 120), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((10, 10, 110, 110), fill=255)
    draw.rectangle((24, 24, 96, 96), fill=0)
    filled = fill_enclosed_holes(mask, [10, 10, 111, 111])
    assert filled.getpixel((60, 60)) == 0


def test_preserve_color_layout_keeps_top_and_bottom_color_zones():
    emoji = Image.new("RGBA", (90, 90), (210, 210, 210, 255))
    source = Image.new("RGBA", (90, 90), (230, 30, 30, 255))
    ImageDraw.Draw(source).rectangle((0, 45, 89, 89), fill=(30, 80, 230, 255))
    matched = preserve_color_layout(emoji, source, strength=1)
    top = matched.getpixel((45, 15)); bottom = matched.getpixel((45, 75))
    assert top[0] > top[2]
    assert bottom[2] > bottom[0]


def test_emoji_fidelity_rejects_large_shape_or_color_drift():
    source = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    ImageDraw.Draw(source).rectangle((35, 5, 65, 95), fill=(70, 150, 220, 255))
    similar = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    ImageDraw.Draw(similar).rectangle((34, 5, 66, 95), fill=(80, 145, 210, 255))
    wrong = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    ImageDraw.Draw(wrong).rectangle((5, 35, 95, 65), fill=(230, 80, 40, 255))
    assert emoji_matches_source(similar, source)
    assert not emoji_matches_source(wrong, source)


def test_emoji_fidelity_rejects_similar_color_with_unrelated_silhouette():
    source = Image.new("RGBA", (120, 120), (0, 0, 0, 0))
    ImageDraw.Draw(source).polygon([(60, 5), (112, 58), (60, 114), (8, 58)], fill=(90, 150, 200, 255))
    unrelated = Image.new("RGBA", (120, 120), (0, 0, 0, 0))
    ImageDraw.Draw(unrelated).ellipse((8, 8, 112, 112), outline=(90, 150, 200, 255), width=10)

    assert not emoji_matches_source(unrelated, source)


def test_emoji_fidelity_rejects_extra_large_subject_outside_source_shape():
    source = Image.new("RGBA", (120, 120), (0, 0, 0, 0))
    ImageDraw.Draw(source).ellipse((18, 20, 68, 100), fill=(100, 150, 90, 255))
    with_extra_subject = source.copy()
    ImageDraw.Draw(with_extra_subject).ellipse((72, 16, 116, 92), fill=(100, 150, 90, 255))
    assert not emoji_matches_source(with_extra_subject, source)


def test_emoji_style_check_rejects_source_copy_and_accepts_visible_rendering_change():
    source = Image.new("RGBA", (80, 80), (0, 0, 0, 0))
    ImageDraw.Draw(source).ellipse((8, 8, 72, 72), fill=(150, 90, 50, 255))
    copied = source.copy()
    lightly_adjusted = Image.new("RGBA", (80, 80), (0, 0, 0, 0))
    ImageDraw.Draw(lightly_adjusted).ellipse((8, 8, 72, 72), fill=(160, 100, 60, 255))
    stylized = Image.new("RGBA", (80, 80), (0, 0, 0, 0))
    ImageDraw.Draw(stylized).ellipse((8, 8, 72, 72), fill=(185, 120, 75, 255))
    assert not emoji_is_visibly_stylized(copied, source)
    assert not emoji_is_visibly_stylized(lightly_adjusted, source)
    assert emoji_is_visibly_stylized(stylized, source)
    assert emoji_style_difference(stylized, source) > emoji_style_difference(lightly_adjusted, source)


def test_color_intensity_reduces_only_excess_brightness_and_saturation():
    source = Image.new("RGBA", (40, 40), (120, 145, 110, 255))
    vivid = Image.new("RGBA", (40, 40), (120, 255, 30, 255))
    matched = match_color_intensity(vivid, source)
    source_hsv = source.convert("HSV").getpixel((10, 10))
    vivid_hsv = vivid.convert("HSV").getpixel((10, 10))
    matched_hsv = matched.convert("HSV").getpixel((10, 10))
    assert matched_hsv[1] < vivid_hsv[1]
    assert matched_hsv[2] < vivid_hsv[2]
    assert matched_hsv[1] <= source_hsv[1] * 1.15 + 5
    assert matched.getchannel("A").getpixel((10, 10)) == 255


def test_color_intensity_leaves_non_excessive_image_unchanged():
    source = Image.new("RGBA", (20, 20), (180, 120, 100, 255))
    emoji = Image.new("RGBA", (20, 20), (170, 115, 100, 255))
    assert match_color_intensity(emoji, source).getpixel((10, 10))[:3] == emoji.getpixel((10, 10))[:3]


def test_batch_composition_replays_from_original_when_one_edit_is_removed(tmp_path, monkeypatch):
    folder = tmp_path / "image-1"
    folder.mkdir()
    Image.new("RGB", (200, 120), "white").save(folder / "original.png")

    for object_id, box, color in (
        ("01", (20, 20, 80, 100), (255, 0, 0, 255)),
        ("02", (120, 20, 180, 100), (0, 0, 255, 255)),
    ):
        mask = Image.new("L", (200, 120), 0)
        ImageDraw.Draw(mask).rectangle(box, fill=255)
        mask.save(folder / f"mask_{object_id}.png")
        Image.new("RGBA", (60, 80), color).save(folder / f"emoji_{object_id}.png")

    async def copy_background(input_path, _box, output_path):
        Image.open(input_path).save(output_path)

    monkeypatch.setattr(main, "item_dir", lambda _image_id: folder)
    monkeypatch.setattr(main, "repair_background_domestic", copy_background)
    monkeypatch.setattr(main, "file_url", lambda path: path.name)

    both = main.BatchCompositionRequest(
        imageId="image-1",
        items=[
            main.CompositionItem(objectId="01", emojiUrl="emoji_01.png"),
            main.CompositionItem(objectId="02", emojiUrl="emoji_02.png"),
        ],
    )
    both_result = asyncio.run(main.compose_batch(both))
    both_image = Image.open(folder / both_result["resultUrl"]).convert("RGB")
    assert both_image.getpixel((50, 60))[0] > 200
    assert both_image.getpixel((150, 60))[2] > 200

    remaining = main.BatchCompositionRequest(
        imageId="image-1",
        items=[main.CompositionItem(objectId="02", emojiUrl="emoji_02.png")],
    )
    undo_result = asyncio.run(main.compose_batch(remaining))
    undo_image = Image.open(folder / undo_result["resultUrl"]).convert("RGB")
    assert undo_image.getpixel((50, 60)) == (255, 255, 255)
    assert undo_image.getpixel((150, 60))[2] > 200


def test_background_repair_cache_is_reused_for_unchanged_mask(tmp_path, monkeypatch):
    folder = tmp_path / "image-1"
    folder.mkdir()
    original_path = folder / "original.png"
    Image.new("RGB", (120, 80), "white").save(original_path)
    mask = Image.new("L", (120, 80), 0)
    ImageDraw.Draw(mask).rectangle((20, 10, 90, 70), fill=255)
    mask.save(folder / "mask_01.png")
    calls = 0

    async def repair_once(input_path, _box, output_path):
        nonlocal calls
        calls += 1
        Image.open(input_path).save(output_path)

    monkeypatch.setattr(main, "repair_background_with_retry", repair_once)

    first_path, first_hit = asyncio.run(main.ensure_background_cache(folder, "01", original_path))
    second_path, second_hit = asyncio.run(main.ensure_background_cache(folder, "01", original_path))

    assert first_path == second_path
    assert not first_hit
    assert second_hit
    assert calls == 1
