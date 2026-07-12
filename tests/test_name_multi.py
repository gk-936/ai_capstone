import json
import yaml
from pathlib import Path


def test_roster_yaml_has_five_general_models():
    roster = yaml.safe_load(Path("configs/multi_vlm.yaml").read_text(encoding="utf-8"))
    models = roster["models"]
    assert set(models) >= {"scout", "maverick", "gemini", "gpt4o", "pixtral"}
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
