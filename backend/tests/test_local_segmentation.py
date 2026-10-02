import numpy as np

from app.local_segmentation import _bbox_gap, _clean_subject_mask, _expanded_context_box, _sample_points, _stroke_context_box


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
