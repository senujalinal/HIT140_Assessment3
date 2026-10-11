"""Rebuild datasets, run both analyses, and verify assignment invariants."""

from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from build_datasets import build  # noqa: E402
from eda import run as run_eda  # noqa: E402
from modeling import run  # noqa: E402
from validate_project import validate  # noqa: E402


def main() -> None:
    audit = build(ROOT / "data/raw/worldcup_2026.json", ROOT / "data/processed")
    run_eda(ROOT)
    results = run(ROOT)
    checks = validate(audit, results, ROOT / "outputs/results/verification.json")
    print(json.dumps({"status": "complete", "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
