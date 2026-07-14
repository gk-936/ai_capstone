import json
import yaml
from pathlib import Path


def test_roster_yaml_active_models_well_formed():
    # Active roster after model-availability fixes: maverick retired from Groq, gpt4o excluded
    # (no GitHub Models token). scout/pixtral completed the live study; gemini is deferred but
    # kept in the roster as the intended third model.
    roster = yaml.safe_load(Path("configs/multi_vlm.yaml").read_text(encoding="utf-8"))
    models = roster["models"]
    assert set(models) == {"scout", "pixtral", "gemini"}
    for name, spec in models.items():
        assert "provider" in spec and "model" in spec


class _StubProvider:
    name = "stub"
    def __init__(self):
        self.calls = 0
    def describe_crop(self, image_path, prompt):
        self.calls += 1
        return f"desc for {image_path.name}"


def test_name_all_multi_writes_and_resumes(tmp_path):
    from auditor.discovery.name_multi import name_all_multi
    crops = tmp_path / "crops"; crops.mkdir()
    for cid in ("pos_0", "pos_1"):
        (crops / f"{cid}.png").write_bytes(b"x")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([
        {"id": "pos_0", "kind": "positive", "has_confound": True},
        {"id": "pos_1", "kind": "positive", "has_confound": True},
    ]))
    out = tmp_path / "preds.jsonl"

    prov = _StubProvider()
    r1 = name_all_multi(crops, manifest, out, prov, "prompt")
    assert r1["n_predictions"] == 2 and prov.calls == 2

    # resume: second run makes zero new calls
    prov2 = _StubProvider()
    r2 = name_all_multi(crops, manifest, out, prov2, "prompt")
    assert prov2.calls == 0 and r2["n_predictions"] == 2


def test_comparison_table_lists_all_models():
    from auditor.discovery.run_comparison import _comparison_table
    rows = [
        {"model": "scout", "precision": 0.57, "recall": 0.66, "control_hallucination": 0.50},
        {"model": "gemini", "precision": 0.80, "recall": 0.70, "control_hallucination": 0.20},
    ]
    md = _comparison_table(rows)
    assert "scout" in md and "gemini" in md
    assert "precision" in md and "control-hallucination" in md
    assert "0.500" in md  # scout hallucination rendered
