import yaml
from pathlib import Path


def test_roster_yaml_has_five_general_models():
    roster = yaml.safe_load(Path("configs/multi_vlm.yaml").read_text(encoding="utf-8"))
    models = roster["models"]
    assert set(models) >= {"scout", "maverick", "gemini", "gpt4o", "pixtral"}
    for name, spec in models.items():
        assert "provider" in spec and "model" in spec
