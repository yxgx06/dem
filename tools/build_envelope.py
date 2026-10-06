"""Build the verified-gain envelope (offline).

Usage: uv run python tools/build_envelope.py [--check]

--check rebuilds the table and fails if its hash differs from the committed manifest.
"""

from __future__ import annotations

import json
import sys

from egga.eval.envelope_build import ENVELOPE_DIR, build_envelope, write_envelope


def main() -> int:
    check = "--check" in sys.argv
    data, manifest = build_envelope(verbose=True)
    counts = manifest["counts"]
    print(f"hash {manifest['hash']}")
    print(
        f"cells with a verified set: {counts['cells_with_verified_set']} of {counts['cells']} "
        f"(linear: {counts['cells_with_linear_set']}); nonlinear failures "
        f"{counts['nonlinear_failures']} of {counts['nonlinear_tests']}"
    )
    meta_path = ENVELOPE_DIR / f"envelope_v{manifest['version']}.json"
    if check:
        committed = json.loads(meta_path.read_text(encoding="utf-8"))["hash"]
        if committed != manifest["hash"]:
            print(f"HASH MISMATCH: committed {committed}, rebuilt {manifest['hash']}")
            return 1
        print("hash matches the committed envelope")
        return 0
    write_envelope(data, manifest)
    print(f"wrote {meta_path.parent}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
