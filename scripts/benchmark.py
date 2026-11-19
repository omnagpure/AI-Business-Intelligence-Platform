"""Time every stage of the analysis pipeline on real files and write docs/PERFORMANCE.md.

Run from the project root:
    python -m scripts.benchmark path1.csv path2.xlsx ...            # label "after" (default)
    python -m scripts.benchmark path1.csv --label before            # keep a baseline to compare against

Results are stored in docs/perf/<label>.json. docs/PERFORMANCE.md is regenerated from every stored label, between
the BENCH markers only: text you write outside the markers is kept. No Streamlit is needed or imported.
The pipeline runs on the file as uploaded (no cleaning applied), with the optional name model off, as in the app.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PERF_DIR = ROOT / "docs" / "perf"
DOC = ROOT / "docs" / "PERFORMANCE.md"
START, END = "<!-- BENCH:START -->", "<!-- BENCH:END -->"
STEPS = ("Ingest + profile", "Semantic model", "KPIs", "Data findings", "Forecast", "Anomalies", "Segmentation",
         "Drivers", "Insights", "Build report (incl. charts)", "PDF render", "HTML render")


def _timed(fn):
    """(seconds, status, result_or_reason). A failing stage is recorded, never fatal, as in the app."""
    t0 = time.perf_counter()
    try:
        out = fn()
        return time.perf_counter() - t0, "ok", out
    except Exception as exc:                                    # the benchmark must report, not crash
        reason = getattr(exc, "user_message", None) or type(exc).__name__
        return time.perf_counter() - t0, "skipped: " + str(reason)[:90], None


def benchmark_file(path: Path) -> dict:
    from analytics.data_findings import compute_findings
    from analytics.kpis import compute_kpis
    from config.analytics_config import DEFAULT_ANALYTICS, DEFAULT_INSIGHTS
    from config.settings import Settings
    from ingestion.pipeline import ingest_bytes
    from insights.default_runs import default_anomalies, default_drivers, default_forecast, default_segmentation
    from insights.insight_engine import generate_insights
    from insights.insight_rules import InsightInputs
    from reporting.html_report import render_html
    from reporting.pdf_report import render_pdf
    from reporting.report_builder import build_report
    from semantic.semantic_model import build_semantic_model

    steps: list[dict] = []

    def rec(name: str, res):
        steps.append({"step": name, "seconds": round(res[0], 3), "status": res[1]})
        return res[2]

    data = path.read_bytes()
    result = rec(STEPS[0], _timed(lambda: ingest_bytes(data, path.name, max_bytes=Settings().max_upload_bytes)))
    if result is None:
        return {"rows": 0, "cols": 0, "steps": steps}
    df, profile = result.dataset.df, result.profile
    model = rec(STEPS[1], _timed(lambda: build_semantic_model(df, profile, {}, None, None)))
    if model is None:
        return {"rows": len(df), "cols": df.shape[1], "steps": steps}
    cfg = DEFAULT_ANALYTICS
    kpis = rec(STEPS[2], _timed(lambda: compute_kpis(df, model, cfg)))
    findings = rec(STEPS[3], _timed(lambda: compute_findings(df, model, DEFAULT_INSIGHTS)))
    adv: dict = {}
    why: dict[str, str] = {}
    for label, key, fn in ((STEPS[4], "forecast", lambda: default_forecast(df, model, cfg)),
                           (STEPS[5], "anomalies", lambda: default_anomalies(df, profile, model, cfg)),
                           (STEPS[6], "segmentation", lambda: default_segmentation(df, model, cfg)),
                           (STEPS[7], "drivers", lambda: default_drivers(df, profile, model, cfg))):
        res = _timed(fn)
        steps.append({"step": label, "seconds": round(res[0], 3), "status": res[1]})
        if res[2] is not None:
            adv[key] = res[2]
        else:
            why[key] = res[1]
    report = rec(STEPS[8], _timed(lambda: generate_insights(
        InsightInputs(kpis=kpis, profile=profile, findings=findings, unavailable=tuple(why.items()), **adv), DEFAULT_INSIGHTS)))
    unavailable = {("segments" if k == "segmentation" else k): v for k, v in why.items()}
    rd = rec(STEPS[9], _timed(lambda: build_report(
        filename=result.dataset.filename, is_sample=False, fingerprint=result.dataset.fingerprint, profile=profile, quality=result.quality,
        analysis_rows=len(df), semantic_model=model, unavailable=unavailable, kpis=kpis, insight_report=report,
        forecast=adv.get("forecast"), anomalies=adv.get("anomalies"), segmentation=adv.get("segmentation"), drivers=adv.get("drivers"))))
    if rd is not None:
        rec(STEPS[10], _timed(lambda: render_pdf(rd)))
        rec(STEPS[11], _timed(lambda: render_html(rd)))
    return {"rows": len(df), "cols": int(df.shape[1]), "steps": steps}


def machine_info() -> dict:
    def has(mod: str) -> bool:
        try:
            __import__(mod)
            return True
        except ImportError:
            return False
    return {"python": platform.python_version(), "platform": platform.platform(), "cpus": os.cpu_count(),
            "statsmodels": has("statsmodels"), "plotly": has("plotly")}


def _fmt(sec) -> str:
    return "–" if sec is None else f"{sec:.2f}"


def render_markdown(runs: dict[str, dict]) -> str:
    labels = [lb for lb in ("before", "after") if lb in runs] + [lb for lb in runs if lb not in ("before", "after")]
    out = [START, "", "_Generated by `python -m scripts.benchmark`. Seconds, one run per stage; the file is analysed as uploaded (no cleaning)._", ""]
    for lb in labels:
        m = runs[lb]["machine"]
        out.append(f"- **{lb}**: Python {m['python']}, {m['cpus']} CPUs, {m['platform']}; statsmodels {'present' if m['statsmodels'] else 'MISSING (forecast models that need it are skipped)'}, "
                   f"run on {runs[lb].get('date', 'n/a')}")
    names = sorted({n for lb in labels for n in runs[lb]["datasets"]})
    for ds in names:
        base = next(runs[lb]["datasets"][ds] for lb in labels if ds in runs[lb]["datasets"])
        out += ["", f"### {ds} ({base['rows']:,} rows, {base['cols']} columns)", ""]
        head = "| Step | " + " | ".join(f"{lb} (s)" for lb in labels) + (" | change |" if len(labels) == 2 else " |")
        out += [head, "|---|" + "---|" * (len(labels) + (1 if len(labels) == 2 else 0))]
        by = {lb: {s["step"]: s for s in runs[lb]["datasets"].get(ds, {}).get("steps", [])} for lb in labels}
        totals = {lb: 0.0 for lb in labels}
        for step in STEPS:
            cells, vals = [], []
            for lb in labels:
                s = by[lb].get(step)
                v = s["seconds"] if s else None
                totals[lb] += v or 0.0
                vals.append(v)
                note = "" if (s is None or s["status"] == "ok") else f" ({s['status']})"
                cells.append(_fmt(v) + note)
            if all(v is None for v in vals):
                continue
            delta = ""
            if len(labels) == 2 and vals[0] and vals[1] is not None:
                delta = f" | {(vals[1] - vals[0]) / vals[0] * 100:+.0f}%"
            elif len(labels) == 2:
                delta = " | "
            out.append(f"| {step} | " + " | ".join(cells) + delta.rstrip() + (" |" if True else ""))
        tot = " | ".join(f"**{totals[lb]:.2f}**" for lb in labels)
        out.append(f"| **Total** | {tot}" + (" | |" if len(labels) == 2 else " |"))
    out += ["", END]
    return "\n".join(out)


def write_doc(runs: dict[str, dict]) -> None:
    block = render_markdown(runs)
    old = DOC.read_text(encoding="utf-8") if DOC.exists() else "# Performance\n\n"
    if START in old and END in old:
        head, rest = old.split(START, 1)
        tail = rest.split(END, 1)[1]
        new = head + block + tail
    else:
        new = old.rstrip() + "\n\n" + block + "\n"
    DOC.write_text(new, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--label", default="after", help="name for this run, e.g. before / after")
    args = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    run = {"machine": machine_info(), "date": time.strftime("%Y-%m-%d"), "datasets": {}}
    for f in args.files:
        p = Path(f)
        if not p.is_file():
            print(f"skip (not found): {f}")
            continue
        print(f"\n{p.name}", flush=True)
        res = benchmark_file(p)
        run["datasets"][p.name] = res
        for s in res["steps"]:
            print(f"  {s['step']:<30} {s['seconds']:>8.2f} s  {s['status']}", flush=True)
        print(f"  {'Total':<30} {sum(s['seconds'] for s in res['steps']):>8.2f} s")
    PERF_DIR.mkdir(parents=True, exist_ok=True)
    (PERF_DIR / f"{args.label}.json").write_text(json.dumps(run, indent=1), encoding="utf-8")
    runs = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(PERF_DIR.glob("*.json"))}
    write_doc(runs)
    print(f"\nWrote {DOC.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
