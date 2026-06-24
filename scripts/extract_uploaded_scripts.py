from __future__ import annotations

import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHUNK_DIR = ROOT / "script_bundle" / "iarm_uploaded_scripts_xz_parts"
OUT = ROOT / "script_bundle" / "iarm_uploaded_scripts.tar.xz"


def main() -> None:
    parts = sorted(CHUNK_DIR.glob("part*.bin"))
    if not parts:
        raise FileNotFoundError(f"No archive chunks found in {CHUNK_DIR}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("wb") as out:
        for part in parts:
            out.write(part.read_bytes())

    with tarfile.open(OUT, "r:xz") as tar:
        tar.extractall(ROOT)

    print(f"extracted {len(parts)} chunks into {ROOT / 'scripts'}")
    print("restored files:")
    for path in sorted((ROOT / "scripts").glob("*.py")):
        print(f"  - {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
