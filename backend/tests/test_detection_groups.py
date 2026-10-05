import asyncio

from app import providers_domestic


def test_semantic_visual_groups_take_priority_over_flat_detector(monkeypatch, tmp_path):
    async def fake_baidu(_path):
        return [
            {"label": "蛋糕", "score": .95, "box": [10, 10, 60, 60], "source": "baidu"},
            {"label": "盘子", "score": .91, "box": [5, 5, 80, 80], "source": "baidu"},
        ]

    async def fake_qwen(_path):
        return [{"label": "蛋糕摆盘", "score": .82, "box": [5, 5, 80, 80], "source": "qwen"}]

    monkeypatch.setattr(providers_domestic, "baidu_detect_objects", fake_baidu)
    monkeypatch.setattr(providers_domestic, "qwen_detect_objects", fake_qwen)

    result = asyncio.run(providers_domestic.detect_objects_hybrid(tmp_path / "unused.png"))

    assert [item["label"] for item in result] == ["蛋糕摆盘"]


def test_flat_detector_is_used_only_when_semantic_detection_is_empty(monkeypatch, tmp_path):
    async def fake_baidu(_path):
        return [{"label": "主体", "score": .9, "box": [1, 2, 30, 40], "source": "baidu"}]

    async def fake_qwen(_path):
        return []

    monkeypatch.setattr(providers_domestic, "baidu_detect_objects", fake_baidu)
    monkeypatch.setattr(providers_domestic, "qwen_detect_objects", fake_qwen)

    result = asyncio.run(providers_domestic.detect_objects_hybrid(tmp_path / "unused.png"))

    assert result[0]["source"] == "baidu"
