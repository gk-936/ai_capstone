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


# ------------------------------- abstention scoring (V4b+) -----

from auditor.discovery.score_naming import names_tube_abstention


def test_abstention_suppresses_hedged_device_mentions():
    # confident device -> flags
    assert names_tube_abstention("There is a chest tube.") is True
    # hedged device -> abstains (does NOT flag) -> reduces hallucination
    assert names_tube_abstention("There might be a catheter, but it is unclear.") is False
    assert names_tube_abstention("possibly a line, cannot determine") is False
    # clean normal -> does not flag
    assert names_tube_abstention("Normal lung, no device.") is False


def test_abstention_matcher_is_wired_into_scoring():
    # regression: the --abstention ablation must be scorable end-to-end via confound=tube_abstention.
    from auditor.discovery.score_naming import _MATCHERS, names_tube_abstention
    assert _MATCHERS["tube_abstention"] is names_tube_abstention


# ------------------------- larger-crop ablation: config override -----------------------


def test_crop_size_is_configurable():
    from auditor.common.config import load_config
    cfg = load_config("configs/v4b_tube.yaml", overrides=["discovery.crop_size=128"])
    assert cfg.discovery.crop_size == 128


# ------------------- grounding-format matchers (Exp 3: CheXagent specialist) -----------

from auditor.discovery.score_naming import names_confound_grounding, names_tube_grounding


def test_grounding_v4a_accepts_generic_object_labels():
    # V4a-grounding mirrors names_confound's own genericness: no requirement the label
    # specifically say "tag"/"marker" shape, just object+artificiality framing.
    assert names_confound_grounding("<|ref|> foreign object <|/ref|> <|box|> (1,1),(2,2) <|/box|>") is True
    assert names_confound_grounding("<|ref|> artificial object <|/ref|> <|box|> (1,1),(2,2) <|/box|>") is True
    assert names_confound_grounding("<|ref|> foreign body <|/ref|> <|box|> (1,1),(2,2) <|/box|>") is True


def test_grounding_v4a_rejects_normal_anatomy_labels():
    assert names_confound_grounding("<|ref|> diaphragm <|/ref|> <|box|> (1,1),(2,2) <|/box|>") is False
    assert names_confound_grounding("<|ref|> other lesion <|/ref|> <|box|> (1,1),(2,2) <|/box|>") is False


def test_grounding_v4b_requires_device_specific_vocabulary_not_bare_foreign_object():
    # This is the key specificity requirement: a bare "foreign object" label must NOT
    # count for V4b-grounding, even though it DOES count for V4a-grounding above --
    # names_tube (prose) requires device-specific words, so its grounding equivalent must too,
    # or CheXagent would be held to an easier bar than scout/pixtral on the same dataset.
    assert names_tube_grounding("<|ref|> foreign object <|/ref|> <|box|> (1,1),(2,2) <|/box|>") is False
    assert names_tube_grounding("<|ref|> artificial object <|/ref|> <|box|> (1,1),(2,2) <|/box|>") is False


def test_grounding_v4b_accepts_device_specific_labels():
    for label in ("chest tube", "catheter", "central line", "drain", "guide wire"):
        desc = f"<|ref|> {label} <|/ref|> <|box|> (1,1),(2,2) <|/box|>"
        assert names_tube_grounding(desc) is True, label


def test_grounding_v4b_rejects_normal_anatomy_labels():
    assert names_tube_grounding("<|ref|> diaphragm <|/ref|> <|box|> (1,1),(2,2) <|/box|>") is False


def test_grounding_matchers_check_every_ref_segment_not_just_the_first():
    # a crop can produce multiple <|ref|> segments (e.g. two box detections); a match on
    # ANY segment should count, not just the first.
    desc = "<|ref|> diaphragm <|/ref|> <|box|> (1,1),(2,2) <|/box|> <|ref|> foreign object <|/ref|> <|box|> (3,3),(4,4) <|/box|>"
    assert names_confound_grounding(desc) is True
    tube_desc = "<|ref|> diaphragm <|/ref|> <|box|> (1,1),(2,2) <|/box|> <|ref|> chest tube <|/ref|> <|box|> (3,3),(4,4) <|/box|>"
    assert names_tube_grounding(tube_desc) is True


def test_grounding_matchers_route_free_text_to_the_existing_prose_matcher():
    # V4a's CheXagent output was a MIX of grounding-token and free-text responses per crop --
    # a description with no <|ref|> tokens must fall through to the unmodified prose matcher,
    # not be treated as grounding-format (and thus vacuously non-matching).
    free_text_positive = "There is a white rectangular marker in the corner, clearly artificial."
    assert names_confound(free_text_positive) is True  # sanity: prose matcher already agrees
    assert names_confound_grounding(free_text_positive) is True

    free_text_negative = "No artificial or non-anatomic structures detected."
    assert names_confound_grounding(free_text_negative) is False

    tube_free_text = "There is a chest tube visible in the right lung field."
    assert names_tube_grounding(tube_free_text) is True


def test_grounding_matchers_handle_empty_and_none():
    assert names_confound_grounding("") is False
    assert names_confound_grounding(None) is False
    assert names_tube_grounding("") is False
    assert names_tube_grounding(None) is False


def test_grounding_matchers_wired_into_matchers_dict():
    from auditor.discovery.score_naming import _MATCHERS
    assert _MATCHERS["tag_grounding"] is names_confound_grounding
    assert _MATCHERS["tube_grounding"] is names_tube_grounding
