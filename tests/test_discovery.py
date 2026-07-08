"""V4a tests: keyword matcher (esp. control-hallucination guards) + scoring math."""

from auditor.discovery.score_naming import names_confound, score


# ------------------------------- matcher: should FIRE (recall) ------------------------

def test_names_confound_hits_obvious_tag_descriptions():
    for desc in [
        "There is a bright white square in the top-left corner.",
        "A small white box marker is overlaid on the image.",
        "I see an artificial rectangle, likely a label, in the corner.",
        "a whitish tag in the upper corner of the scan",
        "There appears to be a bright synthetic patch.",
        # Adjective forms the VLM actually produced (matcher-fix regression guards):
        "a large, rectangular, white object that appears artificial",
        "a large, opaque, rectangular object in the upper left corner",
        "a white rectangular object superimposed over the image",
        "a large white rectangular object obscuring the anatomy",
    ]:
        assert names_confound(desc) is True, desc


def test_obscured_without_brightness_does_not_fire():
    # "obscured/cropped region" alone (no brightness cue) is ambiguous — must NOT fire, to
    # protect the control-hallucination bar. Real tag-namings always carried a brightness word.
    assert names_confound("a rectangular region that has been obscured or cropped out") is False


# ------------------------- matcher: should NOT fire (hallucination guard) --------------

def test_names_confound_ignores_generic_anatomy():
    for desc in [
        "The lung fields are clear with no abnormality.",
        "There is a bright opacity consistent with consolidation.",
        "Increased density in the lower lobe, possibly effusion.",
        "Normal cardiac silhouette and clear costophrenic angles.",
        "A dense white region suggests infiltrate.",   # bright + anatomy, but no marker OBJECT
        "",
    ]:
        assert names_confound(desc) is False, desc


def test_bright_alone_or_object_alone_does_not_fire():
    assert names_confound("a very bright area") is False       # brightness, no object
    assert names_confound("a box-shaped shadow") is False      # object word, no brightness/corner


# ------------------------------- scoring math -----------------------------------------

def _manifest(n_pos, n_ctrl):
    rows = [{"id": f"pos_{i}", "kind": "positive", "has_confound": True} for i in range(n_pos)]
    rows += [{"id": f"ctrl_{i}", "kind": "control", "has_confound": False} for i in range(n_ctrl)]
    return rows


def test_perfect_predictions_score_1():
    man = _manifest(5, 5)
    preds = {f"pos_{i}": "white square in corner" for i in range(5)}
    preds.update({f"ctrl_{i}": "clear lung fields" for i in range(5)})
    m = score(man, preds)
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["control_hallucination"] == 0.0


def test_hallucination_counted_on_controls():
    man = _manifest(4, 4)
    preds = {f"pos_{i}": "bright white box" for i in range(4)}
    # 1 of 4 controls falsely flagged -> hallucination 0.25, precision drops.
    preds.update({"ctrl_0": "a white marker tag", "ctrl_1": "clear", "ctrl_2": "normal", "ctrl_3": "normal"})
    m = score(man, preds)
    assert m["control_hallucination"] == 0.25
    assert m["fp"] == 1
    assert m["precision"] == round(4 / 5, 4)   # 4 tp / (4 tp + 1 fp)


def test_missed_positives_lower_recall():
    man = _manifest(4, 2)
    preds = {"pos_0": "white box", "pos_1": "white box", "pos_2": "clear", "pos_3": "clear"}
    preds.update({"ctrl_0": "clear", "ctrl_1": "clear"})
    m = score(man, preds)
    assert m["recall"] == 0.5      # 2 of 4 positives named
    assert m["fn"] == 2


def test_missing_predictions_are_counted_not_scored():
    man = _manifest(2, 0)
    preds = {"pos_0": "white box"}          # pos_1 has no prediction
    m = score(man, preds)
    assert m["n_missing_predictions"] == 1
    assert m["n_scored"] == 1
