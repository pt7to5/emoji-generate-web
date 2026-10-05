import numpy as np
from PIL import Image

from app import local_segmentation
from app.local_segmentation import _bbox_gap, _clean_subject_mask, _expanded_context_box, _internal_completion_points, _sample_points, _stroke_context_box


def test_sample_points_limits_dense_stroke_and_keeps_spatial_coverage():
    points = [[float(index), float(index * 2)] for index in range(100)]

    sampled = _sample_points(points, limit=24)

    assert len(sampled) == 24
    xs = [point[0] for point in sampled]
    assert min(xs) <= 5
    assert max(xs) >= 94


def test_sample_points_preserves_short_stroke():
    points = [[10, 20], [30, 40], [50, 60]]

    assert _sample_points(points, limit=24) == [
        [10.0, 20.0],
        [30.0, 40.0],
        [50.0, 60.0],
    ]


def test_context_box_expands_beyond_initial_subject_part():
    mask = np.zeros((200, 160), dtype=bool)
    mask[30:130, 50:110] = True

    box = _expanded_context_box(mask, radius=10)

    assert box[0] < 50 and box[1] == 0
    assert box[2] > 110 and box[3] > 160


def test_bbox_gap_distinguishes_attached_part_from_far_outlier():
    anchor = (50, 20, 110, 130)

    assert _bbox_gap(anchor, (60, 128, 100, 170)) == 0
    assert _bbox_gap(anchor, (140, 160, 170, 190)) > 40


def test_stroke_context_box_ignores_extreme_points_and_adds_search_space():
    points = [[40, 40], [50, 50], [60, 60], [55, 45], [500, 500]]

    box = _stroke_context_box(points, 10, (600, 600))

    assert box[0] < 40 and box[1] < 40
    assert box[2] > 60 and box[3] > 60
    assert box[2] < 500 and box[3] < 500


def test_clean_mask_removes_specks_and_fills_small_holes():
    mask = np.zeros((160, 180), dtype=bool)
    mask[20:120, 50:120] = True
    mask[48:53, 72:77] = False
    mask[130:145, 72:98] = True  # nearby meaningful base
    mask[150, 10] = True
    mask[145:147, 160:162] = True

    cleaned = _clean_subject_mask(mask)

    assert cleaned[50, 74]
    assert cleaned[135, 80]
    assert not cleaned[150, 10]
    assert not cleaned[145, 160]


def test_clean_mask_fills_medium_internal_gap_but_keeps_large_real_hole():
    mask = np.zeros((220, 220), dtype=bool)
    mask[20:200, 20:200] = True
    mask[55:85, 55:95] = False  # 约占主体 4%，应视为漏选
    mask[105:175, 105:175] = False  # 大型真实镂空，应保留

    cleaned = _clean_subject_mask(mask)

    assert cleaned[70, 70]
    assert not cleaned[140, 140]


def test_clean_mask_closes_narrow_channel_connected_to_background():
    mask = np.zeros((240, 240), dtype=bool)
    mask[30:210, 30:210] = True
    mask[100:140, 105:145] = False
    mask[30:105, 122:129] = False

    cleaned = _clean_subject_mask(mask)

    assert cleaned[120, 125]
    assert cleaned[60, 125]
    assert not cleaned[10, 10]


def test_large_image_inference_is_scaled_and_mask_returns_to_original_size(tmp_path, monkeypatch):
    source_path = tmp_path / "large.png"
    Image.new("RGB", (2400, 1600), "white").save(source_path)
    observed_sizes = []

    monkeypatch.setattr(local_segmentation, "_get_model", lambda: object())

    def fake_predict(_model, image_path, *, points=None, labels=None, bboxes=None):
        width, height = Image.open(image_path).size
        observed_sizes.append((width, height))
        mask = np.zeros((height, width), dtype=bool)
        mask[height // 4:height * 3 // 4, width // 4:width * 3 // 4] = True
        return [mask]

    monkeypatch.setattr(local_segmentation, "_predict_masks", fake_predict)

    result = local_segmentation.segment_from_positive_strokes(
        source_path, [[1200, 800], [1250, 820]], radius=24
    )

    assert result.size == (2400, 1600)
    assert result.getbbox() == (600, 400, 1800, 1200)
    assert observed_sizes
    assert all(max(size) <= local_segmentation.MAX_INFERENCE_SIDE for size in observed_sizes)


def test_detection_box_drives_local_mask_and_restores_original_size(tmp_path, monkeypatch):
    source_path = tmp_path / "scene.png"
    Image.new("RGB", (2400, 1600), "white").save(source_path)
    observed_boxes = []
    monkeypatch.setattr(local_segmentation, "_get_model", lambda: object())

    def fake_predict(_model, image_path, *, points=None, labels=None, bboxes=None):
        width, height = Image.open(image_path).size
        observed_boxes.extend(bboxes or [])
        mask = np.zeros((height, width), dtype=bool)
        x1, y1, x2, y2 = bboxes[0]
        mask[y1:y2, x1:x2] = True
        return [mask]

    monkeypatch.setattr(local_segmentation, "_predict_masks", fake_predict)

    result = local_segmentation.segment_from_detection_box(source_path, [600, 400, 1800, 1200])

    assert result.size == (2400, 1600)
    assert result.getbbox() == (600, 400, 1800, 1200)
    assert observed_boxes == [[384, 256, 1152, 768]]


def test_internal_completion_points_find_enclosed_subject_but_not_outer_background():
    mask = np.zeros((160, 200), dtype=bool)
    mask[30:140, 20:180] = True
    mask[55:115, 75:125] = False

    points = _internal_completion_points(mask, [10, 20, 190, 150])

    assert len(points) == 1
    assert 75 <= points[0][0] < 125
    assert 55 <= points[0][1] < 115


def test_detection_box_unions_internal_object_without_external_person(tmp_path, monkeypatch):
    source_path = tmp_path / "held-plate.png"
    Image.new("RGB", (200, 160), "white").save(source_path)
    monkeypatch.setattr(local_segmentation, "_get_model", lambda: object())

    def fake_predict(_model, image_path, *, points=None, labels=None, bboxes=None):
        plate = np.zeros((160, 200), dtype=bool)
        plate[30:140, 20:180] = True
        plate[55:115, 75:125] = False
        if bboxes:
            return [plate]
        cake = np.zeros_like(plate)
        cake[55:115, 75:125] = True
        person = np.zeros_like(plate)
        person[0:150, 0:70] = True
        return [cake, person]

    monkeypatch.setattr(local_segmentation, "_predict_masks", fake_predict)

    result = np.asarray(local_segmentation.segment_from_detection_box(source_path, [10, 20, 190, 150])) > 0

    assert result[80, 100]  # 蛋糕被补回
    assert result[40, 40]  # 盘子保留
    assert not result[10, 10]  # 框外人物未被合并
