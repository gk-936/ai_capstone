"""CLI for Phase 3 OOD-gap decomposition: site / labels / tta / decompose."""

from __future__ import annotations

import argparse
import json


def _cmd_site(args) -> None:
    from auditor.common.config import load_config
    from auditor.common.run_logger import RunLogger
    from auditor.analysis.run_site import run_site_analysis

    cfg = load_config(args.config, overrides=args.override)
    logger = RunLogger("phase3_site", cfg, experiments_root=cfg.experiments_root)
    m = run_site_analysis(cfg, logger, args.checkpoint, args.labels_csv, args.images_dir)
    print(json.dumps(m, indent=2))


def _cmd_labels(args) -> None:
    from auditor.analysis.label_analysis import run_label_analysis

    comp = run_label_analysis(args.labels_csv, args.out)
    print(json.dumps(comp, indent=2))


def _cmd_tta(args) -> None:
    from auditor.common.config import load_config
    from auditor.common.run_logger import RunLogger
    from auditor.analysis.tta import run_tta

    cfg = load_config(args.config, overrides=args.override)
    methods = ["bn", "tent"] if args.method == "both" else [args.method]
    for method in methods:
        logger = RunLogger(f"phase3_tta_{method}", cfg, experiments_root=cfg.experiments_root)
        m = run_tta(cfg, logger, args.checkpoint, args.labels_csv, args.images_dir, method)
        print(json.dumps(m, indent=2))


def _cmd_decompose(args) -> None:
    from auditor.analysis.decompose import run_decompose

    run_decompose(args.site_json, args.tta_json, args.label_json, args.ood_drop, args.out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: OOD-gap decomposition")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("site", help="Experiment A: hospital-detector probe + projection control.")
    ps.add_argument("--config", required=True)
    ps.add_argument("--checkpoint", required=True)
    ps.add_argument("--labels-csv", required=True, dest="labels_csv")
    ps.add_argument("--images-dir", required=True, dest="images_dir")
    ps.add_argument("--override", action="append", default=[])
    ps.set_defaults(func=_cmd_site)

    pl = sub.add_parser("labels", help="Experiment B-lite: PadChest ptx label composition.")
    pl.add_argument("--labels-csv", required=True, dest="labels_csv")
    pl.add_argument("--out", required=True, help="Output JSON path.")
    pl.set_defaults(func=_cmd_labels)

    pt = sub.add_parser("tta", help="Experiment C: label-free TTA (BN-adapt / TENT).")
    pt.add_argument("--config", required=True)
    pt.add_argument("--checkpoint", required=True)
    pt.add_argument("--labels-csv", required=True, dest="labels_csv")
    pt.add_argument("--images-dir", required=True, dest="images_dir")
    pt.add_argument("--method", choices=["bn", "tent", "both"], default="both")
    pt.add_argument("--override", action="append", default=[])
    pt.set_defaults(func=_cmd_tta)

    pd_ = sub.add_parser("decompose", help="Combine A/B/C into the decomposition table.")
    pd_.add_argument("--site-json", required=True, dest="site_json")
    pd_.add_argument("--tta-json", action="append", default=[], dest="tta_json",
                     help="Repeatable: one per TTA method JSON.")
    pd_.add_argument("--label-json", required=True, dest="label_json")
    pd_.add_argument("--ood-drop", type=float, default=0.192, dest="ood_drop")
    pd_.add_argument("--out-dir", required=True, dest="out_dir")
    pd_.set_defaults(func=_cmd_decompose)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
