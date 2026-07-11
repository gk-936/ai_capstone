# V6 — cross-site OOD (NIH -> PadChest)

- NIH in-distribution AUC (reference): 0.842
- PadChest OOD AUC: 0.650
- **OOD drop (distribution shift):** 0.192
- PadChest AUC after SUPERVISED target adaptation (last-layer on LABELED PadChest): 0.732 (+0.082)

**Note:** the +change above is supervised target-domain adaptation (uses PadChest labels),
NOT shortcut mitigation. Source-DFR zero-shot (no target labels) is null (+0.005), so the
drop is distribution shift and is NOT shown to be shortcut-driven.

(PadChest test n=5289, adaptation subset n=2283)
