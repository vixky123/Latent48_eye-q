"""messwaste command line.

    python -m messwaste.cli <command> [options]

Commands (in pipeline order)
  templates     write blank CSV sheets to templates/
  demo          generate a fake dataset for testing
  simulate      synthetic operational data around REAL photos (rows tagged source=synthetic)
  ingest        validate CSVs -> parquet
  frames        privacy filter + de-duplicate camera frames
  label         vision-model plate labels (GPU on Kaggle)
  human-sample  build the offline labelling page for teammates
  agreement     model vs human agreement
  analyze       waste decomposition
  forecast      predict + seal upcoming meals
  evaluate      check sealed forecasts against actuals
  report        dashboard + datasheet + schema
  slides        3-slide deck (reports/deck.pptx, + PDF if LibreOffice exists)
  export        zip the submission bundle
  all           ingest -> frames -> label -> agreement -> analyze -> evaluate -> report
  refresh       everything except labelling of old frames (fast loop while presenting)
"""
from __future__ import annotations

import argparse
import sys
import time

from .config import PROJECT_DIR, load_config


def _cfg(a):
    return load_config(a.config, raw_root=a.raw_root, out_root=a.out_root)


def cmd_templates(a):
    from .schema import TABLES
    d = PROJECT_DIR / "templates"
    d.mkdir(exist_ok=True)
    for t in TABLES.values():
        if t.raw_input and t.name != "human_labels":
            (d / f"{t.name}.csv").write_text(",".join(t.input_names) + "\n", encoding="utf-8")
    print(f"Blank sheets in {d}")


def cmd_demo(a):
    from .demo import generate_demo
    root = a.raw_root or str(PROJECT_DIR / "data" / "demo_raw")
    generate_demo(root, seed=a.seed)
    print(f"\nNext: python -m messwaste.cli all --raw-root {root} --out-root {a.out_root or PROJECT_DIR / 'data' / 'demo_out'} --backend mock")


def cmd_simulate(a):
    from .simulate import generate
    root = a.raw_root or str(PROJECT_DIR / "data" / "sim_raw")
    generate(root, a.photos.split(","), seed=a.seed, content_path=a.content)


def cmd_ingest(a):
    from .ingest import ingest
    ingest(_cfg(a), allow_errors=a.allow_errors)


def cmd_frames(a):
    from .camera import process_frames
    process_frames(_cfg(a))


def cmd_label(a):
    from .labeling import run_labeling
    run_labeling(_cfg(a), backend=a.backend, model_id=a.model_id, limit=a.limit,
                 meal_ids=a.meal_ids.split(",") if a.meal_ids else None)


def cmd_human_sample(a):
    from .validation import build_labeller
    build_labeller(_cfg(a), n=a.n, seed=a.seed)


def cmd_agreement(a):
    from .validation import agreement
    agreement(_cfg(a))


def cmd_analyze(a):
    from .analysis import analyze
    analyze(_cfg(a))


def cmd_forecast(a):
    from .analysis import analyze
    from .forecast import make_forecasts
    cfg = _cfg(a)
    analyze(cfg, verbose=False)
    make_forecasts(cfg, meal_ids=a.meal_ids.split(",") if a.meal_ids else None, now=a.now)


def cmd_evaluate(a):
    from .forecast import evaluate
    evaluate(_cfg(a))


def cmd_report(a):
    from .report import build_dashboard, build_datasheet
    cfg = _cfg(a)
    build_dashboard(cfg)
    build_datasheet(cfg)


def cmd_slides(a):
    from .slides import build_slides
    build_slides(_cfg(a))


def cmd_export(a):
    from .report import export_submission
    export_submission(_cfg(a))


def _pipeline(a, label: bool):
    from .analysis import analyze
    from .camera import process_frames
    from .forecast import evaluate
    from .ingest import ingest
    from .labeling import run_labeling
    from .report import build_dashboard, build_datasheet
    from .validation import agreement
    cfg = _cfg(a)
    steps = [("ingest", lambda: ingest(cfg, allow_errors=a.allow_errors)),
             ("frames", lambda: process_frames(cfg))]
    if label:
        steps.append(("label", lambda: run_labeling(cfg, backend=a.backend, model_id=a.model_id, limit=a.limit)))
    steps += [("agreement", lambda: agreement(cfg)), ("analyze", lambda: analyze(cfg)),
              ("evaluate", lambda: evaluate(cfg)), ("report", lambda: (build_dashboard(cfg), build_datasheet(cfg)))]
    for name, fn in steps:
        t = time.time()
        print(f"\n=== {name} ===")
        fn()
        print(f"--- {name} done in {time.time() - t:.1f}s")


def cmd_all(a):
    _pipeline(a, label=True)


def cmd_refresh(a):
    _pipeline(a, label=not a.skip_label)


def main(argv=None):
    def common(parser, default):
        parser.add_argument("--config", default=default, help="config.yaml path")
        parser.add_argument("--raw-root", default=default, help="folder with csv/ and frames/")
        parser.add_argument("--out-root", default=default, help="output folder")

    p = argparse.ArgumentParser(prog="messwaste", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    common(p, None)
    parent = argparse.ArgumentParser(add_help=False)
    common(parent, argparse.SUPPRESS)   # options also accepted after the command
    sub = p.add_subparsers(dest="cmd", required=True)
    _add = sub.add_parser
    sub.add_parser = lambda *x, **k: _add(*x, parents=[parent], **k)

    sub.add_parser("templates").set_defaults(fn=cmd_templates)
    s = sub.add_parser("demo"); s.add_argument("--seed", type=int, default=42); s.set_defaults(fn=cmd_demo)
    s = sub.add_parser("simulate")
    s.add_argument("--photos", required=True, help="comma-separated folders of real plate photos")
    s.add_argument("--seed", type=int, default=11)
    s.add_argument("--content", default=None, help="JSON map of photo file name -> visible dishes")
    s.set_defaults(fn=cmd_simulate)
    s = sub.add_parser("ingest"); s.add_argument("--allow-errors", action="store_true"); s.set_defaults(fn=cmd_ingest)
    sub.add_parser("frames").set_defaults(fn=cmd_frames)
    s = sub.add_parser("label")
    s.add_argument("--backend", choices=["hf", "anthropic", "mock"], default=None)
    s.add_argument("--model-id", default=None)
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--meal-ids", default=None, help="comma-separated")
    s.set_defaults(fn=cmd_label)
    s = sub.add_parser("human-sample"); s.add_argument("--n", type=int, default=100)
    s.add_argument("--seed", type=int, default=7); s.set_defaults(fn=cmd_human_sample)
    sub.add_parser("agreement").set_defaults(fn=cmd_agreement)
    sub.add_parser("analyze").set_defaults(fn=cmd_analyze)
    s = sub.add_parser("forecast")
    s.add_argument("--meal-ids", default=None, help="comma-separated; default = all upcoming meals in meals.csv")
    s.add_argument("--now", default=None, help="DEMO ONLY: pretend it is this local time, e.g. '2026-10-03 22:30'")
    s.set_defaults(fn=cmd_forecast)
    sub.add_parser("evaluate").set_defaults(fn=cmd_evaluate)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    sub.add_parser("slides").set_defaults(fn=cmd_slides)
    sub.add_parser("export").set_defaults(fn=cmd_export)
    for name, fn in (("all", cmd_all), ("refresh", cmd_refresh)):
        s = sub.add_parser(name)
        s.add_argument("--backend", choices=["hf", "anthropic", "mock"], default=None)
        s.add_argument("--model-id", default=None)
        s.add_argument("--limit", type=int, default=None)
        s.add_argument("--allow-errors", action="store_true")
        if name == "refresh":
            s.add_argument("--skip-label", action="store_true", help="don't run the vision model")
        s.set_defaults(fn=fn)

    a = p.parse_args(argv)
    from .ingest import IngestError
    try:
        a.fn(a)
    except KeyboardInterrupt:
        sys.exit(130)
    except (IngestError, FileNotFoundError, ValueError) as e:
        if __name__ != "__main__" and not sys.argv[0].endswith(("cli.py", "__main__.py")):
            raise
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
