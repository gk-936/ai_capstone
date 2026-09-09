"""
Generates the four data-driven figures and the architecture diagram that are
new for the IEEE Access submission (Figures 3-6 / architecture Fig. 0), from
numbers already reported and verified in docs/capstone_writeup.md and
docs/paper/ieee_access/paper.tex. No new experiments are run here - this only
visualizes existing, published results.

Run: python docs/paper/figures/generate_new_figures.py
Outputs land in this same folder (docs/paper/figures/), alongside the
existing figure1_naming_boundary.* and auditor_before_after_tube.png.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUTDIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Shared style. Okabe-Ito colorblind-safe categorical palette (Okabe & Ito,
# 2008) -- the standard for scientific figures, chosen over a generic UI
# palette because this is a print/PDF journal figure, not an interactive page.
# All text is forced to black regardless of the figure's color scheme.
# ---------------------------------------------------------------------------
OI = {
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9",
    "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
    "vermillion": "#D55E00", "purple": "#CC79A7", "gray": "#999999",
}
HATCHES = ["", "///", "..", "xx"]  # print/grayscale-safe secondary encoding

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "grid.color": "#000000",
    "axes.axisbelow": True,
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "text.color": "black",
    "font.family": "DejaVu Sans",
})


def save(fig, name):
    # Draw a clear black border frame around the whole figure so each image
    # reads as a self-contained, bounded panel (helpful when figures are
    # viewed standalone, out of the paper's own page margins).
    fig.patch.set_edgecolor("black")
    fig.patch.set_linewidth(2.5)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTDIR, f"{name}.{ext}"), bbox_inches="tight",
                     facecolor="white", edgecolor=fig.get_edgecolor())
    plt.close(fig)
    print(f"wrote {name}.png / {name}.pdf")


# ---------------------------------------------------------------------------
# Figure 3 -- Over-name / under-name axis across the full model roster
# Source: writeup SS5.2 (Scout/Pixtral multi-VLM replication), SS6.3 (qwen2.5-vl-7b,
# CheXagent-2-3B). CheXagent's V4b 0.0 is a taxonomy gap (cannot emit device
# vocabulary at all), not a passed gate -- annotated distinctly, not plotted
# as a plain "good" bar.
# ---------------------------------------------------------------------------
def fig_overname_underrname():
    models = ["Llama-4-Scout", "Pixtral-12B", "qwen2.5-vl-7b", "CheXagent-2-3B"]
    v4a = [0.02, 0.31, None, 0.03]      # control-hallucination on synthetic confound
    v4b = [0.55, 0.83, 0.74, 0.0]        # control-hallucination on real chest tube
    chexagent_note_idx = 3

    x = np.arange(len(models))
    w = 0.32
    fig, ax = plt.subplots(figsize=(6.4, 3.6))

    v4a_vals = [v if v is not None else 0 for v in v4a]
    bars_a = ax.bar(x - w/2, v4a_vals, width=w, color=OI["sky"], label="V4a (synthetic confound)",
                     edgecolor="white", linewidth=0.5)
    bars_b = ax.bar(x + w/2, v4b, width=w, color=OI["vermillion"], label="V4b (real chest tube)",
                     edgecolor="white", linewidth=0.5)

    # mark qwen's missing V4a bar
    ax.text(x[2] - w/2, 0.015, "not\nreported", ha="center", va="bottom", fontsize=7, color="black")

    # annotate CheXagent's V4b=0.0 as a taxonomy gap, not a pass
    ax.annotate("taxonomy gap:\nno device vocabulary\n(not a passed gate)",
                xy=(x[chexagent_note_idx] + w/2, 0.0), xytext=(x[chexagent_note_idx] + w/2 + 0.15, 0.30),
                fontsize=7.5, color=OI["black"], ha="left",
                arrowprops=dict(arrowstyle="-", color=OI["gray"], lw=0.8))

    ax.axhline(0.10, color=OI["black"], linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(len(models) - 0.5, 0.115, "<10% control-hallucination gate", fontsize=7.5,
            ha="right", color=OI["black"])

    for bars, vals in [(bars_a, v4a), (bars_b, v4b)]:
        for rect, v in zip(bars, vals):
            if v is None:
                continue
            ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.015,
                     f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Control-hallucination rate")
    ax.set_ylim(0, 0.95)
    ax.set_title("Over-name / under-name across the full model roster", fontsize=11)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    fig.text(0.5, -0.03,
              "In plain terms: each bar is how often a model \"sees\" a device on an X-ray crop that has no device on it.\n"
              "Bars above the dashed line make things up too often to trust; only Scout stays under the line on the harder, real-device test.",
              ha="center", fontsize=7.5, color="black")
    fig.tight_layout()
    save(fig, "fig3_overname_underrname_roster")


# ---------------------------------------------------------------------------
# Figure 4 -- Image-independence / occlusion grounding check (SS6.2)
# pos->neg (consistent with grounding) vs neg->pos (fabrication relocated),
# split by dataset (V4a/V4b), condition (relevant/irrelevant/swap), model.
# ---------------------------------------------------------------------------
def fig_occlusion_grounding():
    rows = [
        # dataset, condition, model, pos->neg, neg->pos, n
        ("V4a", "relevant",   "Scout",   19, 5, 100),
        ("V4a", "relevant",   "Pixtral", 25, 22, 100),
        ("V4a", "irrelevant", "Scout",   16, 5, 100),
        ("V4a", "irrelevant", "Pixtral", 27, 24, 100),
        ("V4b", "relevant",   "Scout",   61, 1, 100),
        ("V4b", "relevant",   "Pixtral", 66, 5, 100),
        ("V4b", "irrelevant", "Scout",   30, 4, 100),
        ("V4b", "irrelevant", "Pixtral", 13, 2, 100),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    for ax, ds in zip(axes, ["V4a", "V4b"]):
        sub = [r for r in rows if r[0] == ds]
        labels = [f"{cond}\n{model}" for (_, cond, model, *_r) in sub]
        pos_neg = [r[3] / r[5] for r in sub]
        neg_pos = [r[4] / r[5] for r in sub]
        x = np.arange(len(sub))
        w = 0.35
        ax.bar(x - w/2, pos_neg, width=w, color=OI["blue"], edgecolor="white", linewidth=0.5,
               label="pos→neg (grounded: named, then not)")
        ax.bar(x + w/2, neg_pos, width=w, color=OI["orange"], hatch="//", edgecolor="white", linewidth=0.5,
               label="neg→pos (fabrication relocated)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(f"{ds} ({'synthetic tag' if ds=='V4a' else 'real chest tube'})", fontsize=10)
        ax.set_ylim(0, 0.75)
    axes[0].set_ylabel("Rate of verdict change")
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("")
    fig.text(0.5, -0.05,
              "In plain terms: we cover part of the crop and see if the model changes its mind. Covering the real spot\n"
              "should flip a \"yes\" to a \"no\" (blue bars) far more than a random spot flips a \"no\" to a \"yes\" (orange bars) — that gap is\n"
              "evidence the model is actually looking at the device, not just guessing. V4a relevant-occlusion is contaminated by the\n"
              "gray occluder itself tripping the matcher (see §6.2 caveat); the grounding claim rests on the V4b-internal contrast\n"
              "(relevant → far more pos→neg than irrelevant).",
              ha="center", fontsize=7.5, color="black")
    fig.tight_layout()
    save(fig, "fig4_occlusion_grounding")


# ---------------------------------------------------------------------------
# Figure 5 -- Cross-site (V6) forest plot: drop, supervised adapt, zero-shot
# source-DFR, each vs. the NIH->PadChest baseline gap.
# ---------------------------------------------------------------------------
def fig_crosssite_forest():
    # effect, point estimate, ci_low, ci_high (None = no CI reported)
    rows = [
        ("NIH → PadChest AUC drop", -0.192, -0.225, -0.159, OI["vermillion"]),
        ("Supervised target adaptation\n(uses PadChest labels — adaptation, not mitigation)", +0.082, None, None, OI["gray"]),
        ("Zero-shot source-DFR\n(no target labels — the honest test)", +0.005, -0.018, +0.028, OI["blue"]),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    y = np.arange(len(rows))[::-1]
    for yi, (label, est, lo, hi, color) in zip(y, rows):
        if lo is not None:
            ax.plot([lo, hi], [yi, yi], color=color, linewidth=2, solid_capstyle="round")
            ax.plot([lo, lo], [yi - 0.08, yi + 0.08], color=color, linewidth=1.5)
            ax.plot([hi, hi], [yi - 0.08, yi + 0.08], color=color, linewidth=1.5)
        else:
            ax.plot([est], [yi], marker="D", markersize=6, color=color, mfc="white", mew=1.5)
        ax.plot([est], [yi], marker="o", markersize=7, color=color, zorder=5)
        ax.text(est, yi + 0.22, f"{est:+.3f}" + (f"  [{lo:+.3f}, {hi:+.3f}]" if lo is not None else "  (no CI reported)"),
                ha="center", fontsize=8)

    ax.axvline(0, color=OI["black"], linewidth=1, linestyle="-")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_xlabel("Δ AUC")
    ax.set_xlim(-0.30, 0.15)
    ax.set_ylim(-0.7, len(rows) - 1 + 0.7)
    ax.set_title("Cross-site AUC change, with 95% bootstrap CIs where reported", fontsize=10.5, pad=14)
    fig.text(0.5, -0.06,
              "In plain terms: a classifier trained at one hospital loses about 19 points of AUC at a second hospital (top row).\n"
              "Letting it see the new hospital's own labels recovers most of that (middle row, dashed diamond = single-run point\n"
              "estimate, no CI available) — but that's cheating, since real deployments rarely have those labels. The bottom row is\n"
              "the honest test: a label-free fix, and its recovery is close to zero, i.e. it does not meaningfully help.",
              ha="center", fontsize=7.5, color="black")
    fig.tight_layout()
    save(fig, "fig5_crosssite_forest")


# ---------------------------------------------------------------------------
# Figure 6 -- Mitigation comparison: worst-group & ID accuracy, ERM/JTT/DFR,
# tube (real) vs. tag (synthetic), 3-seed mean +/- std (SS5.3 table).
# ---------------------------------------------------------------------------
def fig_mitigation_comparison():
    data = {
        "tube": {
            "ERM": (0.672, 0.022, 0.959, 0.002),
            "JTT": (0.791, 0.047, 0.940, 0.019),
            "DFR": (0.790, 0.036, 0.964, 0.007),
        },
        "tag": {
            "ERM": (0.500, 0.000, 0.940, 0.000),
            "JTT": (0.608, 0.018, 0.531, 0.016),
            "DFR": (0.609, 0.022, 0.603, 0.014),
        },
    }
    methods = ["ERM", "JTT", "DFR"]
    colors = [OI["gray"], OI["orange"], OI["blue"]]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True)
    for ax, confound in zip(axes, ["tube", "tag"]):
        x = np.arange(2)  # worst-group, ID
        w = 0.25
        for i, m in enumerate(methods):
            wg, wg_sd, idacc, id_sd = data[confound][m]
            vals = [wg, idacc]
            errs = [wg_sd, id_sd]
            ax.bar(x + (i - 1) * w, vals, width=w, yerr=errs, capsize=3,
                   color=colors[i], edgecolor="white", linewidth=0.5, label=m)
        ax.set_xticks(x)
        ax.set_xticklabels(["Worst-group\n(confound absent)", "In-distribution\n(confound present)"], fontsize=8.5)
        ax.set_title(f"Real chest tube" if confound == "tube" else "Synthetic tag (p=0.95)", fontsize=10.5)
        ax.set_ylim(0.4, 1.05)
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="lower left", frameon=False, fontsize=8.5)
    fig.suptitle("Mitigation: worst-group recovery vs. in-distribution cost (3-seed mean ± SD)", fontsize=10.5, y=1.03)
    fig.text(0.5, -0.05,
              "In plain terms: \"worst-group\" is accuracy on the hard cases where the shortcut is missing — the true test of\n"
              "whether the model learned the disease rather than the tube/tag. JTT and DFR raise that bar over the plain baseline\n"
              "(ERM) with only a small cost to everyday accuracy (right-hand bars). The effect is far larger for the synthetic tag,\n"
              "because that shortcut was made deliberately strong and easy to unlearn; the real chest tube is a harder case.",
              ha="center", fontsize=7.5, color="black")
    fig.tight_layout()
    save(fig, "fig6_mitigation_comparison")


# ---------------------------------------------------------------------------
# Figure 1 (regenerated) -- naming boundary, Scout & Pixtral only.
# Replaces the original figure1_naming_boundary.pdf/.png: that file had a
# font-encoding bug where the "<10% gate" annotation rendered as garbled,
# overlapping glyphs in the compiled PDF (visible glyphs fine on screen,
# broken ToUnicode map on export). Same data, same filename, rebuilt through
# the same pipeline as the other figures here, which has no such bug.
# Source: docs/writeup/statistical_hardening.md; Table "Control-hallucination
# rate per cell" -- Scout V4a 0.02/V4b 0.55, Pixtral V4a 0.31/V4b 0.83, with
# 95% bootstrap CIs (10k resamples, seed 1337): Scout [0.00,0.05]/[0.45,0.65],
# Pixtral [0.22,0.40]/[0.75,0.90].
# ---------------------------------------------------------------------------
def fig_naming_boundary():
    models = ["Llama-4-Scout", "Pixtral-12B"]
    v4a = [0.02, 0.31]
    v4b = [0.55, 0.83]
    v4a_ci = [(0.00, 0.05), (0.22, 0.40)]
    v4b_ci = [(0.45, 0.65), (0.75, 0.90)]

    x = np.arange(len(models))
    w = 0.32
    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    def err(vals, cis):
        lo = [v - c[0] for v, c in zip(vals, cis)]
        hi = [c[1] - v for v, c in zip(vals, cis)]
        return [lo, hi]

    bars_a = ax.bar(x - w/2, v4a, width=w, yerr=err(v4a, v4a_ci), capsize=4,
                     color=OI["sky"], label="Synthetic confound (V4a)",
                     edgecolor="white", linewidth=0.5)
    bars_b = ax.bar(x + w/2, v4b, width=w, yerr=err(v4b, v4b_ci), capsize=4,
                     color=OI["vermillion"], label="Real chest tube (V4b)",
                     edgecolor="white", linewidth=0.5)

    for bars, vals, cis in [(bars_a, v4a, v4a_ci), (bars_b, v4b, v4b_ci)]:
        for rect, v, ci in zip(bars, vals, cis):
            ax.text(rect.get_x() + rect.get_width()/2, ci[1] + 0.025,
                     f"{v:.0%}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.axhline(0.10, color=OI["black"], linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(len(models) - 0.62, 0.125, "<10% gate", fontsize=8.5, ha="right", color=OI["black"])

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10.5)
    ax.set_ylabel("Control-hallucination rate\n(devices named on confound-FREE crops)")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("The naming boundary: a VLM auditor fabricates devices on clean crops\nonly when the real confound is present", fontsize=11)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.text(0.5, -0.02,
              "Bars = control-hallucination rate on tube/tag-FREE crops (n=100/cell). Whiskers = 95% bootstrap CI (10k resamples, seed 1337).\n"
              "Source: docs/writeup/statistical_hardening.md. Synthetic controls stay near 2% (Scout); real-tube controls jump to 55--83%.",
              ha="center", fontsize=7.5, color="black")
    fig.tight_layout()
    save(fig, "figure1_naming_boundary")


# ---------------------------------------------------------------------------
# Figure 0 -- Architecture / pipeline diagram (SS3.1 in prose -> visual here).
# ---------------------------------------------------------------------------
def fig_architecture():
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    stages = [
        ("Classifier\n(DenseNet-121)", 0.6, "#E6E6E6"),
        ("Detect\naccuracy gap", 2.4, "#C2C2C2"),
        ("Localize\nGrad-CAM", 4.2, "#9E9E9E"),
        ("Name\nVLM (blind prompt)", 6.0, "#707070"),
        ("Mitigate\nDFR (last layer)", 7.8, "#3D3D3D"),
    ]
    box_w, box_h, y0 = 1.5, 1.3, 1.6

    for label, xc, color in stages:
        box = FancyBboxPatch((xc - box_w/2, y0), box_w, box_h,
                              boxstyle="round,pad=0.06,rounding_size=0.08",
                              linewidth=1.6, edgecolor="black", facecolor=color, alpha=0.6)
        ax.add_patch(box)
        ax.text(xc, y0 + box_h/2, label, ha="center", va="center", fontsize=8.7, color="black")

    for (l1, x1, _), (l2, x2, _) in zip(stages, stages[1:]):
        arr = FancyArrowPatch((x1 + box_w/2, y0 + box_h/2), (x2 - box_w/2, y0 + box_h/2),
                               arrowstyle="-|>", mutation_scale=14, linewidth=1.3, color=OI["black"])
        ax.add_patch(arr)

    # isolated naming module boundary + file contract
    naming_x = stages[3][1]
    dashed = FancyBboxPatch((naming_x - box_w/2 - 0.25, y0 - 0.35), box_w + 0.5, box_h + 0.7,
                             boxstyle="round,pad=0.02", linewidth=1.1, edgecolor="black",
                             facecolor="none", linestyle="--")
    ax.add_patch(dashed)
    ax.text(naming_x, y0 - 0.55, "crops/ + manifest.json → predictions.jsonl\n(isolated: no mitigation code imports this)",
            ha="center", va="top", fontsize=7, color="black")

    ax.text(5.1, 3.55, "Confound-agnostic core (region box, or (confound, label) pair) — serves synthetic tag, real tube, and cross-site from one implementation",
            ha="center", fontsize=8, color=OI["black"], style="italic")
    ax.plot([1.2, 8.9], [3.35, 3.35], color=OI["gray"], linewidth=0.8, linestyle=":")

    ax.set_title("Auditor loop: detect → localize → name → mitigate", fontsize=11.5, pad=4)
    ax.text(5.1, 0.35,
            "In plain terms: the classifier's decision is checked for a hidden shortcut, the image region driving that\n"
            "shortcut is located, a vision-language model is asked to name what it sees there with no hint of what to look for,\n"
            "and only then is the model retrained to stop relying on it.",
            ha="center", va="top", fontsize=7.5, color="black")
    fig.tight_layout()
    save(fig, "fig0_architecture")


if __name__ == "__main__":
    fig_architecture()
    fig_naming_boundary()
    fig_overname_underrname()
    fig_occlusion_grounding()
    fig_crosssite_forest()
    fig_mitigation_comparison()
    print("\nAll figures written to", OUTDIR)
