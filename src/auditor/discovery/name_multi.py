"""Run one VLM provider over the crop set under a given prompt; resumable file-contract output."""

from __future__ import annotations

import json
import time
from pathlib import Path


def _done_ids(out_path: Path) -> set:
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def name_all_multi(crops_dir, manifest_path, out_path, provider, prompt,
                   throttle_s: float = 0.0) -> dict:
    crops_dir, out_path = Path(crops_dir), Path(out_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    done = _done_ids(out_path)
    if done:
        print(f"resuming {provider.name}: {len(done)} already present")
    n = len(done)
    with open(out_path, "a", encoding="utf-8") as f:
        for i, row in enumerate(manifest):
            cid = row["id"]
            if cid in done:
                continue
            desc = provider.describe_crop(crops_dir / f"{cid}.png", prompt)
            f.write(json.dumps({"id": cid, "description": desc}) + "\n")
            f.flush()
            n += 1
            if i % 20 == 0:
                print(f"{provider.name} {i}/{len(manifest)} {cid}: {desc[:70]}")
            if throttle_s:
                time.sleep(throttle_s)
    print(f"DONE {provider.name} -> {n} predictions at {out_path}")
    return {"n_predictions": n, "out": str(out_path)}
