import numpy as np
from PIL import Image

from app import local_segmentation
from app.local_segmentation import _mask_anchor_points, _split_guided_strokes


def test_mask_anchor_points_stay_inside_current_subject():
    pixels = np.zeros((80, 120), dtype=np.uint8)
    pixels[15:65, 25:95] = 255
    mask = Image.fromarray(pixels, "L")

    anchors = _mask_anchor_points(mask)

    assert anchors
    assert all(25 <= x < 95 and 15 <= y < 65 for x, y in anchors)


def test_guided_add_keeps_existing_subject_and_unions_new_subject(tmp_path, monkeypatch):
    source_path = tmp_path / "source.png"
    Image.new("RGB", (120, 80), "white").save(source_path)
    current_array = np.zeros((80, 120), dtype=np.uint8)
    current_array[15:65, 10:45] = 255
    current = Image.fromarray(current_array, "L")
    added_array = np.zeros((80, 120), dtype=np.uint8)
    added_array[20:60, 75:110] = 255
    monkeypatch.setattr(local_segmentation, "segment_from_positive_strokes", lambda *_args: Image.fromarray(added_array, "L"))

    result = local_segmentation.segment_from_guided_strokes(
        source_path, current, positive_points=[[90, 40]], negative_points=[]
    )

    pixels = np.asarray(result) > 0
    assert pixels[30, 20]
    assert pixels[30, 90]
    assert result.getbbox() == (10, 15, 110, 65)


def test_guided_remove_subtracts_recognized_subject_and_keeps_previous_one(tmp_path, monkeypatch):
    source_path = tmp_path / "source.png"
    Image.new("RGB", (120, 80), "white").save(source_path)
    current_array = np.zeros((80, 120), dtype=np.uint8)
    current_array[15:65, 10:45] = 255
    current_array[20:60, 75:110] = 255
    removal_array = np.zeros((80, 120), dtype=np.uint8)
    removal_array[20:60, 75:110] = 255
    monkeypatch.setattr(local_segmentation, "segment_from_positive_strokes", lambda *_args: Image.fromarray(removal_array, "L"))

    result = local_segmentation.segment_from_guided_strokes(
        source_path, Image.fromarray(current_array, "L"), positive_points=[], negative_points=[[90, 40]]
    )

    pixels = np.asarray(result) > 0
    assert pixels[30, 20]
    assert not pixels[30, 90]
    assert result.getbbox() == (10, 15, 45, 65)


def test_long_guided_stroke_is_split_across_multiple_subjects():
    points = [[10 + index * 5, 20 + (index % 3), 1 if index == 0 else 0] for index in range(30)]
    groups = _split_guided_strokes(points, radius=8)
    assert len(groups) >= 2
    assert sum(len(group) for group in groups) == len(points)


def test_guided_add_unions_three_disconnected_subjects(tmp_path, monkeypatch):
    source_path = tmp_path / "source.png"
    Image.new("RGB", (180, 100), "white").save(source_path)
    current = Image.new("L", (180, 100), 0)

    def segment_group(_path, points, _radius):
        result = np.zeros((100, 180), dtype=np.uint8)
        center = sum(point[0] for point in points) / len(points)
        if center < 60:
            result[20:70, 10:50] = 255
        elif center < 120:
            result[15:65, 70:110] = 255
        else:
            result[25:75, 130:170] = 255
        return Image.fromarray(result, "L")

    monkeypatch.setattr(local_segmentation, "segment_from_positive_strokes", segment_group)
    points = [[10 + index * 5, 45, 1 if index == 0 else 0] for index in range(32)]
    result = local_segmentation.segment_from_guided_strokes(source_path, current, points, [], radius=8)
    pixels = np.asarray(result) > 0
    assert pixels[40, 30]
    assert pixels[40, 90]
    assert pixels[40, 150]
