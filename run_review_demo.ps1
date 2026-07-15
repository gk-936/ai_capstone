<#
Run the full auditor pipeline for the review demo: detect -> localize -> name -> mitigate
(DFR), then the two V6 cross-site OOD variants (supervised adaptation + the honest
source-DFR zero-shot null result). Run this just before the meeting, not live in front of
the professor -- each stage prints its own "Run dir:" path; open the audit_report.md and
before_after_gradcam.png from the FIRST stage's run dir when presenting.

Usage (from anywhere):  powershell -ExecutionPolicy Bypass -File run_review_demo.ps1
#>

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Section($title) {
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host $title -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor Cyan
}

$sw = [System.Diagnostics.Stopwatch]::StartNew()

Section "1/3 -- Core pipeline: detect -> localize -> name -> mitigate (DFR)"
python -m auditor.report --config configs/auditor_tube.yaml `
    --predictions experiments/v4b_tube/20260708_204951_3894b5/predictions.jsonl `
    --manifest experiments/v4b_tube/20260708_204951_3894b5/manifest.json

Section "2/3 -- V6: cross-site OOD + supervised target adaptation (labeled PadChest)"
python -m auditor.report --config configs/v6_padchest.yaml --v6

Section "3/3 -- V6: source-DFR zero-shot (the honest boundary finding, no PadChest labels)"
python -m auditor.report --config configs/v6_source_dfr.yaml --v6-source-dfr

$sw.Stop()
Section ("DONE in {0:mm}m {0:ss}s -- open each stage's audit_report.md / before_after_gradcam.png from the 'Run dir:' paths printed above" -f $sw.Elapsed)
