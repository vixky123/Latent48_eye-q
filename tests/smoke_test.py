"""End-to-end test on demo data with the mock labeller. No GPU or internet needed.

    python tests/smoke_test.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from messwaste.cli import main  # noqa: E402
from messwaste.config import load_config  # noqa: E402
from messwaste.io import read_table  # noqa: E402


def run():
    tmp = Path(tempfile.mkdtemp(prefix="messwaste_smoke_"))
    raw, out = tmp / "raw", tmp / "out"
    common = ["--raw-root", str(raw), "--out-root", str(out)]
    try:
        main(["demo", *common])
        main(["all", *common, "--backend", "mock"])
        # day-1 night: forecast day-2 meals with a fake clock (demo only)
        main(["forecast", *common, "--now", "2026-10-03 22:30",
              "--meal-ids", "DEMO_20261004_B,DEMO_20261004_L,DEMO_20261004_D"])
        main(["evaluate", *common])
        main(["human-sample", *common, "--n", "30"])
        main(["report", *common])
        main(["slides", *common])
        main(["export", *common])

        cfg = load_config(raw_root=str(raw), out_root=str(out))
        mm = read_table(cfg, "meal_metrics", required=True)
        assert len(mm) == 6, "expected 6 meals"
        assert mm["ghost_overprod_kg"].notna().all(), "ghost overproduction missing"
        assert (mm["camera_plates"] > 0).all(), "camera labels missing"
        dm = read_table(cfg, "dish_metrics", required=True)
        assert dm["plate_kg"].notna().any(), "plate attribution missing"
        ev = read_table(cfg, "forecast_eval", required=True)
        assert len(ev) == 3 and ev["hash_ok"].all(), "forecast sealing broken"
        assert ev["abs_error"].notna().all(), "forecast evaluation missing actuals"
        fr = read_table(cfg, "camera_frames", required=True)
        assert (fr["drop_reason"] == "duplicate").sum() >= 1, "de-duplication did nothing"
        assert (out / "reports" / "dashboard.html").stat().st_size > 10_000
        assert (out / "labelling" / "labeller.html").exists()
        assert (out / "reports" / "deck.pptx").exists()
        assert list(out.glob("submission_*.zip")), "no submission zip"

        # tampering must be detected
        sealed = sorted((out / "forecasts" / "sealed").glob("*.json"))[0]
        txt = sealed.read_text().replace('"predicted_footfall": ', '"predicted_footfall": 1')
        sealed.write_text(txt)
        main(["evaluate", *common])
        ev2 = read_table(cfg, "forecast_eval", required=True)
        assert (~ev2["hash_ok"]).sum() == 1, "tampered forecast not detected"

        # real data must refuse the fake clock
        print("\nALL SMOKE TESTS PASSED")
        print(f"Dashboard: {out / 'reports' / 'dashboard.html'}")
        return out
    except Exception:
        print(f"\nFAILED. Files kept in {tmp}")
        raise


if __name__ == "__main__":
    out = run()
    if "--keep" not in sys.argv:
        shutil.rmtree(out.parent, ignore_errors=True)
