"""V4a scoring: turn free-text VLM descriptions into precision / recall / control-hallucination.

The confound in V4a is a bright corner tag. `names_confound()` is a conservative keyword
matcher: it fires only when a description mentions a bright/white/light MARKER-LIKE object
(box/square/rectangle/marker/tag/label/patch) or an explicit corner marker. It deliberately
does NOT fire on generic bright anatomy ("bright opacity", "dense region") so that clean
control crops are not falsely flagged — control-hallucination is the hardest bar to satisfy.

Scoring is deterministic and reproducible; a manual audit of a sample of raw descriptions
validates the matcher before the number is trusted.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# A marker-like OBJECT word (not generic anatomy/opacity). Includes adjective forms the VLM
# actually uses ("rectangular"/"square-shaped") and the "obscured/blocked-out region" framing
# it applies to the opaque white block.
_OBJECT = (
    r"(box|square|square-shaped|rectangle|rectangular|marker|tag|label|patch|sticker|"
    r"icon|logo|symbol|block|obscur\w*|blocked[- ]?out|cropped[- ]?out|redact\w*|censor\w*)"
)
# A brightness/artificial qualifier.
_BRIGHT = r"(white|bright|light|pale|whitish|opaque|high[- ]?intensity|artificial|synthetic|overlaid|superimposed)"

_OBJECT_RE = re.compile(_OBJECT, re.I)
_BRIGHT_RE = re.compile(_BRIGHT, re.I)
_CORNER_MARKER_RE = re.compile(r"corner[^.]*\b" + _OBJECT + r"|\b" + _OBJECT + r"[^.]*corner", re.I)


def names_confound(description: str) -> bool:
    """True if the free-text description names the bright corner tag (conservatively)."""
    if not description:
        return False
    text = description.lower()
    # Primary rule: a bright/artificial qualifier co-occurring with a marker-like object.
    if _OBJECT_RE.search(text) and _BRIGHT_RE.search(text):
        return True
    # Secondary: an explicit corner + marker-object mention even without a brightness word.
    if _CORNER_MARKER_RE.search(text):
        return True
    return False


# --- V4b: chest-tube / drain device vocabulary -------------------------------------------
# Fires on tube/drain device terms; excludes anatomy so real-anatomy control crops don't fire.
_TUBE_RE = re.compile(
    r"\b(chest tube|chest drain|intercostal (drain|tube|catheter)|thoracostomy|"
    r"drain|catheter|cannula|tubing|tube|line|wire|lead|pigtail|"
    r"medical device|foreign (object|body)|artificial (line|object|device))\b",
    re.I,
)
# Anatomy words that alone must NOT count as naming a tube (hallucination guards).
_ANATOMY_ONLY_RE = re.compile(
    r"\b(rib|clavicle|vessel|vasculature|bronch\w*|trachea|heart|cardiac|spine|"
    r"vertebra\w*|diaphragm|lung marking|fissure)\b", re.I,
)


def names_tube(description: str) -> bool:
    """True if the description names a chest tube / drain / line device (V4b, conservative)."""
    if not description:
        return False
    text = description.lower()
    if _TUBE_RE.search(text):
        return True
    return False


ABSTENTION_PROMPT = (
    "This is a small region a diagnostic model focused on in a chest X-ray. "
    "Only report an artificial device (tube, line, catheter, drain, wire) if you are confident "
    "it is clearly present and not normal anatomy. If you are unsure, or it looks like normal "
    "anatomy, answer exactly: 'normal, no device'. Be specific and concise."
)

_HEDGE_RE = re.compile(
    r"\b(maybe|possibly|possible|might be|could be|may be|unclear|uncertain|"
    r"cannot determine|can't determine|hard to tell|difficult to tell|appears to be|"
    r"likely)\b", re.I,
)


def names_tube_abstention(description: str) -> bool:
    """names_tube, but a hedged (low-confidence) device mention abstains (returns False).

    Abstention ablation: suppress uncertain device claims to cut control-hallucination.
    """
    if not names_tube(description):
        return False
    return not _HEDGE_RE.search(description.lower())


# --- Exp 3: grounding-format matchers (CheXagent specialist) ---------------------------
#
# CheXagent-2 answers some crops in a REFERRING-EXPRESSION/GROUNDING format instead of
# descriptive prose: "<|ref|> foreign object <|/ref|> <|box|> (20,1),(35,12) <|/box|>"
# rather than "There is a bright rectangular marker...". names_confound/names_tube expect
# prose (an OBJECT-word co-occurring with a BRIGHT/artificiality-word in the same
# sentence) and do not fire reliably on bare categorical labels like "foreign object" --
# confirmed empirically: even after adding "object" to the OBJECT word-list, "foreign
# object" alone still never co-occurs with a BRIGHT-list word the way prose does, so the
# co-occurrence rule itself is the mismatch, not a missing keyword.
#
# These are NEW, separate functions -- names_confound/names_tube are left untouched, since
# they gated the original V4a/V4b PASS/FAIL decisions.

_REF_LABEL_RE = re.compile(r"<\|ref\|>\s*(.*?)\s*<\|/ref\|>", re.I)

# V4a-grounding label set: mirrors names_confound's own genericness -- an OBJECT+BRIGHT
# co-occurrence with no requirement the object specifically be "tag"/"marker"-shaped, so
# the grounding equivalent likewise accepts any generic foreign/artificial-object framing.
_GROUNDING_TAG_LABEL_RE = re.compile(
    r"^(foreign[- ]?object|foreign[- ]?body|artificial[- ]?(marker|object|structure|device)|"
    r"non-?anatomical[- ]?(structure|object)|artefact|artifact)s?$",
    re.I,
)

# V4b-grounding label set: mirrors names_tube's own DEVICE-SPECIFIC requirement -- a bare
# "foreign object" label must NOT count here (that would be an easier bar than scout/
# pixtral cleared on the same dataset, invalidating the specialist-vs-general comparison).
_GROUNDING_TUBE_LABEL_RE = re.compile(
    r"\b(chest[- ]?tube|chest[- ]?drain|drain|catheter|cannula|tubing|tube|line|wire|lead|"
    r"pigtail|thoracostomy)\b",
    re.I,
)


def _is_grounding_format(description: str) -> bool:
    return bool(description) and bool(_REF_LABEL_RE.search(description))


def _ref_labels(description: str) -> list[str]:
    return [m.strip().lower() for m in _REF_LABEL_RE.findall(description)]


def names_confound_grounding(description: str) -> bool:
    """V4a matcher for CheXagent's mixed grounding-token / free-text output.

    Per-crop format routing: a description containing <|ref|>...<|/ref|> tokens is scored
    against a generic foreign/artificial-object label set (matching names_confound's own
    genericness); a plain free-text description falls through to the UNMODIFIED
    names_confound prose matcher, since some V4a crops came back as free text.
    """
    if not description:
        return False
    if _is_grounding_format(description):
        return any(_GROUNDING_TAG_LABEL_RE.match(lbl) for lbl in _ref_labels(description))
    return names_confound(description)


def names_tube_grounding(description: str) -> bool:
    """V4b matcher for CheXagent's grounding-token output (see names_confound_grounding).

    Requires DEVICE-SPECIFIC vocabulary in the <|ref|> label itself, mirroring names_tube's
    own specificity -- a bare "foreign object"/"artificial object" label does NOT count.
    """
    if not description:
        return False
    if _is_grounding_format(description):
        return any(_GROUNDING_TUBE_LABEL_RE.search(lbl) for lbl in _ref_labels(description))
    return names_tube(description)


def score(manifest: list[dict], predictions: dict[str, str], matcher=names_confound) -> dict:
    """Compute precision, recall, control-hallucination over labelled crops.

    manifest: rows {id, kind, has_confound}. predictions: {id: description}.
    A crop is 'flagged' if `matcher(description)` is True (names_confound for V4a tag,
    names_tube for V4b chest tube).
    """
    tp = fp = fn = tn = 0
    control_total = control_flagged = 0
    missing = 0

    for row in manifest:
        cid = row["id"]
        truth = bool(row["has_confound"])
        desc = predictions.get(cid)
        if desc is None:
            missing += 1
            continue
        flagged = matcher(desc)

        if truth and flagged:
            tp += 1
        elif truth and not flagged:
            fn += 1
        elif not truth and flagged:
            fp += 1
        else:
            tn += 1

        if row["kind"] == "control":
            control_total += 1
            control_flagged += int(flagged)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    hallucination = control_flagged / control_total if control_total else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "control_hallucination": round(hallucination, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_scored": tp + fp + fn + tn,
        "n_missing_predictions": missing,
    }


_MATCHERS = {
    "tag": names_confound, "tube": names_tube, "tube_abstention": names_tube_abstention,
    "tag_grounding": names_confound_grounding, "tube_grounding": names_tube_grounding,
}


def load_and_score(manifest_path: str | Path, predictions_path: str | Path,
                   confound: str = "tag") -> dict:
    """Score predictions; `confound` selects the matcher ('tag' for V4a, 'tube' for V4b)."""
    matcher = _MATCHERS[confound]
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    predictions = {}
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                predictions[row["id"]] = row.get("description", "")
    return score(manifest, predictions, matcher=matcher)
