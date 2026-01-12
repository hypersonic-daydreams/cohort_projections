"""Pre-commit hook: verify the data manifest exists and is readable."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate DATA_MANIFEST.md presence.")
    parser.add_argument(
        "--manifest",
        default="data/DATA_MANIFEST.md",
        help="Path to DATA_MANIFEST.md (default: data/DATA_MANIFEST.md)",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"DATA_MANIFEST missing: {manifest_path}")

    _ = manifest_path.read_text(encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
