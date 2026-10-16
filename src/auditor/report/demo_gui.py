"""Local browser GUI for the single-image auditor demo.

Run:
    python -m auditor.report.demo_gui [--config configs/auditor_tube.yaml] [--port 5000]

Then open http://127.0.0.1:5000 in a browser, upload one or more chest X-rays, and click "Run audit".
The page shows the classifier's prediction and Grad-CAM before/after DFR mitigation
for each uploaded image, laid out side by side in a grid.

The checkpoint is loaded and the DFR head is fit ONCE at server startup (see
AuditorSession in demo_single_image.py) — each upload only pays for two forward passes
and two Grad-CAMs, so the page responds in a couple of seconds even on a laptop CPU.
"""

from __future__ import annotations

import argparse
import base64
import io
import tempfile
from pathlib import Path

from flask import Flask, render_template_string, request

from auditor.report.demo_single_image import AuditorSession, render_figure

app = Flask(__name__)
_session: AuditorSession | None = None


PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Chest X-ray Shortcut Auditor</title>
<style>
  :root {
    --ink: #1b2430; --muted: #5b6673; --line: #e2e6eb; --bg: #f7f8fa;
    --before: #c0392b; --after: #1e8a5f; --accent: #2b5aa0;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--ink);
  }
  header {
    padding: 28px 32px 18px; border-bottom: 1px solid var(--line); background: #fff;
  }
  header h1 { margin: 0 0 4px; font-size: 22px; }
  header p { margin: 0; color: var(--muted); font-size: 14px; }
  main { max-width: 1500px; margin: 0 auto; padding: 28px 32px 60px; }

  .card {
    background: #fff; border: 1px solid var(--line); border-radius: 10px;
    padding: 22px 24px; margin-bottom: 24px;
  }
  .card h2 { margin: 0 0 14px; font-size: 15px; text-transform: uppercase;
             letter-spacing: .04em; color: var(--muted); }

  form { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
  input[type=file] { flex: 1 1 260px; }
  select, input[type=file] {
    padding: 8px 10px; border: 1px solid var(--line); border-radius: 6px; font-size: 14px;
  }
  button {
    background: var(--accent); color: #fff; border: none; border-radius: 6px;
    padding: 10px 22px; font-size: 14px; font-weight: 600; cursor: pointer;
  }
  button:hover { background: #234a87; }

  .meta { font-size: 13px; color: var(--muted); margin-top: 10px; }
  .meta code { background: var(--bg); padding: 1px 5px; border-radius: 4px; }

  .verdicts { display: flex; gap: 16px; margin-bottom: 18px; }
  .verdict { flex: 1; border-radius: 8px; padding: 14px 16px; border: 1px solid var(--line); }
  .verdict .label { font-size: 12px; text-transform: uppercase; letter-spacing: .04em;
                     color: var(--muted); margin-bottom: 4px; }
  .verdict .value { font-size: 26px; font-weight: 700; }
  .verdict.before { border-left: 4px solid var(--before); }
  .verdict.after  { border-left: 4px solid var(--after); }

  .results { display: grid; grid-template-columns: repeat(auto-fit, minmax(620px, 1fr)); gap: 20px; }
  .results .card { margin-bottom: 0; }
  .results .verdict .value { font-size: 22px; }
  .fname { font-size: 14px; font-weight: 600; margin-bottom: 12px; word-break: break-all; }
  figure { margin: 0; }
  figure img {
    width: 100%; border-radius: 6px; border: 1px solid var(--line); cursor: zoom-in;
  }
  .zoom-hint { font-size: 12px; color: var(--muted); margin-top: 6px; }

  .error { background: #fdecea; border: 1px solid #f5c6c1; color: #8a2c22;
           border-radius: 8px; padding: 12px 16px; margin-bottom: 20px; font-size: 14px; }

  footer { text-align: center; color: var(--muted); font-size: 12px; padding: 12px; }

  /* --- zoom lightbox --- */
  #lightbox {
    display: none; position: fixed; inset: 0; background: rgba(10, 14, 20, 0.92);
    z-index: 1000; overflow: hidden; cursor: grab;
  }
  #lightbox.open { display: block; }
  #lightbox img {
    position: absolute; top: 50%; left: 50%; max-width: none; max-height: none;
    transform-origin: center center; user-select: none; -webkit-user-drag: none;
  }
  #lightbox .lb-bar {
    position: fixed; top: 16px; right: 20px; z-index: 1001; display: flex; gap: 8px;
  }
  #lightbox .lb-btn {
    background: rgba(255,255,255,0.12); color: #fff; border: 1px solid rgba(255,255,255,0.3);
    border-radius: 6px; padding: 6px 14px; font-size: 13px; cursor: pointer;
  }
  #lightbox .lb-btn:hover { background: rgba(255,255,255,0.22); }
  #lightbox .lb-hint {
    position: fixed; bottom: 18px; left: 50%; transform: translateX(-50%);
    color: rgba(255,255,255,0.75); font-size: 12px; z-index: 1001;
  }
</style>
</head>
<body>
<header>
  <h1>Chest X-ray Shortcut Auditor</h1>
  <p>Upload a chest X-ray &rarr; see where the classifier looks before and after mitigation
     (Deep Feature Reweighting) for the <strong>{{ confound }}</strong> confound.</p>
</header>
<main>

  <div class="card">
    <h2>1. Upload one or more images</h2>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="image" accept="image/*" multiple required>
      <button type="submit">Run audit</button>
    </form>
    <div class="meta">
      Checkpoint: <code>{{ checkpoint }}</code> &nbsp;·&nbsp; Confound: <code>{{ confound }}</code>
      &nbsp;·&nbsp; Device: <code>{{ device }}</code>
    </div>
  </div>

  {% if error %}
  <div class="error">{{ error }}</div>
  {% endif %}

  {% if results %}
  <h2 style="font-size:15px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)">2. Results ({{ results|length }} image{{ "s" if results|length != 1 }})</h2>
  <div class="results">
    {% for r in results %}
    <div class="card">
      <div class="fname">{{ r.name }}</div>
      {% if r.error %}
      <div class="error">Audit failed: {{ r.error }}</div>
      {% else %}
      <div class="verdicts">
        <div class="verdict before">
          <div class="label">P(pneumothorax) — before DFR</div>
          <div class="value" style="color: var(--before)">{{ "%.4f"|format(r.p_before) }}</div>
        </div>
        <div class="verdict after">
          <div class="label">P(pneumothorax) — after DFR</div>
          <div class="value" style="color: var(--after)">{{ "%.4f"|format(r.p_after) }}</div>
        </div>
      </div>
      <figure>
        <img src="data:image/png;base64,{{ r.figure_b64 }}"
             alt="Grad-CAM before/after DFR" onclick="openLightbox(this.src)">
      </figure>
      {% endif %}
    </div>
    {% endfor %}
  </div>
  <div class="zoom-hint">Click any figure to maximize and zoom. Left: input. Middle: Grad-CAM before
    mitigation. Right: Grad-CAM after DFR retrains the last layer on a confound-balanced sample.</div>
  {% endif %}

</main>
<footer>Runs entirely locally &mdash; no data leaves this machine.</footer>

<div id="lightbox" onclick="if(event.target===this) closeLightbox()">
  <div class="lb-bar">
    <button class="lb-btn" onclick="resetZoom(event)">Reset</button>
    <button class="lb-btn" onclick="closeLightbox()">Close ✕</button>
  </div>
  <img id="lightbox-img" src="" alt="zoomed view">
  <div class="lb-hint">Scroll to zoom &middot; drag to pan &middot; Esc or click outside to close</div>
</div>

<script>
  var lb = document.getElementById('lightbox');
  var lbImg = document.getElementById('lightbox-img');
  var scale = 1, panX = 0, panY = 0, dragging = false, startX = 0, startY = 0;

  function applyTransform() {
    lbImg.style.transform = 'translate(-50%, -50%) translate(' + panX + 'px,' + panY + 'px) scale(' + scale + ')';
  }
  function resetZoom(e) {
    if (e) e.stopPropagation();
    scale = 1; panX = 0; panY = 0;
    applyTransform();
  }
  function openLightbox(src) {
    lbImg.src = src;
    resetZoom();
    lb.classList.add('open');
  }
  function closeLightbox() {
    lb.classList.remove('open');
  }
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') closeLightbox();
  });
  lb.addEventListener('wheel', function(e) {
    e.preventDefault();
    var delta = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    scale = Math.min(Math.max(scale * delta, 0.5), 8);
    applyTransform();
  }, { passive: false });
  lbImg.addEventListener('mousedown', function(e) {
    dragging = true; startX = e.clientX - panX; startY = e.clientY - panY;
    lb.style.cursor = 'grabbing';
  });
  window.addEventListener('mousemove', function(e) {
    if (!dragging) return;
    panX = e.clientX - startX; panY = e.clientY - startY;
    applyTransform();
  });
  window.addEventListener('mouseup', function() {
    dragging = false; lb.style.cursor = 'grab';
  });
  lbImg.addEventListener('dblclick', resetZoom);
</script>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    assert _session is not None
    results = None
    error = None

    if request.method == "POST":
        files = [f for f in request.files.getlist("image") if f and f.filename]
        if not files:
            error = "Please choose at least one image file."
        else:
            results = []
            with tempfile.TemporaryDirectory() as tmp:
                for i, file in enumerate(files):
                    name = Path(file.filename).name
                    entry = {"name": name}
                    try:
                        img_path = Path(tmp) / f"{i}_{name}"
                        file.save(img_path)
                        res = _session.audit(str(img_path))
                        fig_path = Path(tmp) / f"figure_{i}.png"
                        render_figure(res, str(fig_path))
                        entry.update(p_before=res.p_before, p_after=res.p_after,
                                     figure_b64=base64.b64encode(fig_path.read_bytes()).decode("ascii"))
                    except Exception as exc:  # noqa: BLE001 — show per-image failure, keep the rest
                        entry["error"] = str(exc)
                    results.append(entry)

    return render_template_string(
        PAGE, results=results, error=error,
        checkpoint=_session.checkpoint, confound=_session.confound, device=_session.device,
    )


def main() -> None:
    global _session

    parser = argparse.ArgumentParser(description="Local GUI for the single-image auditor demo")
    parser.add_argument("--config", default="configs/auditor_tube.yaml",
                        help="Path to a YAML config file (default: the tube-shortcut demo config).")
    parser.add_argument("--checkpoint", default=None, help="Override the config's checkpoint.")
    parser.add_argument("--confound", default=None, choices=[None, "tube", "tag"],
                        help="Override the config's confound (tube or tag).")
    parser.add_argument(
        "--override", action="append", default=[], metavar="key.subkey=value",
        help="Override a config value (repeatable), e.g. --override device=cpu",
    )
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    print("Loading checkpoint and fitting DFR — this happens once, please wait...")
    _session = AuditorSession(args.config, checkpoint=args.checkpoint, confound=args.confound,
                              overrides=args.override)
    print(f"Ready. Open http://127.0.0.1:{args.port} in a browser.")
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
