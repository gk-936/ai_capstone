# V6 — Cross-Site OOD Generalization (design)

**Spike question:** Does a NIH-trained pneumothorax classifier show a measurable accuracy drop on
a DIFFERENT hospital (PadChest, Spain), and does last-layer adaptation (DFR) narrow it?

**Gate:** measurable NIH->PadChest AUC drop that DFR narrows.

## Data
PadChest 224x224 (12.89GB Academic Torrents set; downloading). Uses PadChest's OWN pneumothorax
labels (main labels CSV, in the torrent). Second site = Spain vs NIH's US -> real cross-site
shift. (NEATX PadChest tube labels are NOT used — only 9 chest-drains, too few; tube-transfer is
future work.)

## Pipeline (reuses eval + DFR)
1. Load PadChest — new `data/padchest.py`: parse the PadChest labels CSV -> pneumothorax
   present/absent, resolve 224px images, patient-level split (small train subset for DFR / test).
2. Measure OOD drop — the V1 CLEAN model (no induced shortcut, AUC ~0.84): AUC on NIH test
   (in-dist) vs PadChest test (OOD). Drop = ID - OOD.
3. Adapt (DFR) — last-layer retrain on a SMALL PadChest train subset (frozen NIH backbone),
   re-evaluate PadChest test AUC.
4. Report — v6_report.{json,md}: NIH-AUC, PadChest-AUC (before), PadChest-AUC (after DFR).

## Data flow
```
V1 NIH model -> AUC(NIH test)           [in-distribution]
             -> AUC(PadChest test)       [OOD — the drop]
   DFR last-layer on small PadChest train subset (frozen backbone)
             -> AUC(PadChest test, adapted)    [gap narrowed?]
```

## Config `configs/v6_padchest.yaml`
`data.site2_root` (Phase-0-reserved field) -> PadChest folder; PadChest labels CSV path; V1
checkpoint; DFR subset size.

## Tests
- PadChest loader: parses labels -> pneumothorax binary; patient-disjoint split; image resolution.
  Built now against the label schema; the real-data test is GATED (skips if images absent), like
  the NEATX tests. Synthetic-CSV test always runs.
- AUC eval + DFR reused (already tested).

## PASS criteria & skeptic checks
- PASS: PadChest AUC < NIH AUC (drop exists) AND DFR-adapted PadChest AUC > pre-adapt (narrowed).
- Skeptic: no drop -> honest finding (model generalizes well); "measurable drop" may be small.
  DFR overfitting the tiny subset -> watch, report subset size.
- Honest dependency: cannot fully validate until the torrent lands and the real PadChest label CSV
  columns are seen. Build against the documented PadChest schema; mark the real-data path clearly.

## Runtime
Eval + last-layer DFR only — minutes once images are local. No backbone retraining.

## Build now vs after download
- Now (no images): loader + V6 runner + config + synthetic-CSV tests. Coded, unit-tested vs schema.
- After download: point config at the PadChest folder, run, get real OOD numbers.
