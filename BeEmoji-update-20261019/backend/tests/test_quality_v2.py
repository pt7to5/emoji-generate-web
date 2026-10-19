from pathlib import Path

from PIL import Image, ImageDraw

from app import providers_domestic
from app.main import choose_first_candidate
from app.quality import efficiency_score, score_candidate, validate_generated_image


def _subject(size=(256, 256), color=(180, 80, 40, 255)):
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(image).ellipse((48, 42, 208, 220), fill=color)
    return image


def test_reference_routing_uses_uploaded_groups():
    assert [p.name for p in providers_domestic.reference_paths_for_category("animal")] == ["reference_1.png", "reference_2.png"]
    assert [p.name for p in providers_domestic.reference_paths_for_category("plant")] == ["reference_3.png", "reference_4.png"]
    assert [p.name for p in providers_domestic.reference_paths_for_category("food")] == ["reference_5.png", "reference_6.png", "reference_7.png"]
    assert all(path.exists() for paths in providers_domestic.REFERENCE_PATHS_BY_CATEGORY.values() for path in paths)


def test_prompt_v2_is_layered_and_category_specific(monkeypatch):
    monkeypatch.setattr(providers_domestic, "EMOJI_PROMPT_VERSION", "v2")
    prompt = providers_domestic.build_emoji_prompt("plant")
    for layer in ("【任务层】", "【主体保真层】", "【类别约束层】", "【镜头姿态锁定层】", "【风格层】", "【颜色层】", "【构图层】", "【最终自检层】"):
        assert layer in prompt
    assert "当前类别是 植物" in prompt


def test_animal_prompt_uses_curved_lines_or_natural_hairs_without_lumps(monkeypatch):
    monkeypatch.setattr(providers_domestic, "EMOJI_PROMPT_VERSION", "v2")
    prompt = providers_domestic.build_emoji_prompt("animal")
    assert "【动物材质保真层】" in prompt
    assert "弯曲纹理线或短弧线" in prompt
    assert "低对比度" in prompt
    assert "单根毛发" in prompt
    assert "忽略毛发纹理" in prompt
    assert "独立体积块" in prompt
    assert "肿块般的毛团" in prompt
    assert "统一柔软绒面" not in prompt


def test_pose_lock_forbids_reference_pose_and_frontal_symmetry(monkeypatch):
    monkeypatch.setattr(providers_domestic, "EMOJI_PROMPT_VERSION", "v2")
    prompt = providers_domestic.build_emoji_prompt("animal")
    for rule in ("yaw", "pitch", "roll", "透视缩短", "正面化", "左右对称化", "标准坐姿"):
        assert rule in prompt
    assert "严禁借用或趋近参考图" in prompt
    assert "复制参考图姿势" in providers_domestic.NEGATIVE_PROMPT


def test_candidate_selection_prioritizes_pose_consistency_over_soft_score():
    class Report:
        def __init__(self, valid, total):
            self.valid = valid
            self.total = total

    first = Report(True, 96.0)
    retry = Report(True, 78.0)
    assert not choose_first_candidate(first, retry, first_acceptable=False, retry_acceptable=True)
    assert choose_first_candidate(first, retry, first_acceptable=True, retry_acceptable=False)


def test_v1_rollback_switch_uses_original_prompt(monkeypatch):
    monkeypatch.setattr(providers_domestic, "EMOJI_PROMPT_VERSION", "v1")
    assert providers_domestic.build_emoji_prompt("animal") == providers_domestic.EMOJI_PROMPT


def test_hard_validation_rejects_empty_and_accepts_real_subject():
    valid, reasons = validate_generated_image(Image.new("RGBA", (256, 256), (0, 0, 0, 0)))
    assert not valid and "empty_subject" in reasons
    valid, reasons = validate_generated_image(_subject())
    assert valid and not reasons


def test_soft_quality_has_four_scores_and_audit_timeout_is_not_invalid():
    source = _subject(color=(170, 90, 50, 255))
    candidate = _subject(color=(176, 96, 55, 255))
    report = score_candidate(candidate, source, semantic_ok=True, style_class="3D_EMOJI", style_quality="WEAK", audit_available=False, audit_reason="timeout")
    assert report.valid
    assert all(0 <= value <= 100 for value in (report.subject_fidelity, report.style_consistency, report.visual_completeness, report.color_fidelity, report.total))


def test_efficiency_is_scored_separately_and_decreases_with_time():
    assert efficiency_score(10_000) > efficiency_score(60_000) > efficiency_score(120_000)


def test_generation_image_payload_upscales_short_side_to_provider_minimum(tmp_path):
    path = tmp_path / "small.png"
    Image.new("RGBA", (250, 230), (120, 80, 50, 255)).save(path)
    uri = providers_domestic._data_uri(path, min_side=240)
    import base64
    from io import BytesIO
    decoded = Image.open(BytesIO(base64.b64decode(uri.split(",", 1)[1])))
    assert min(decoded.size) == 240
