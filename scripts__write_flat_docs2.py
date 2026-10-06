"""DEPRECATED — superseded by `build_flat_mirror.py`.

This one-shot script used to write the remainder of the flat-mirror orientation
documents (`DATA_AND_BINARY_NOTICE.md`, `chatgpt__00_TASK_BRIEF.md`,
`chatgpt__01_WHERE_TO_LOOK.md`). It hard-coded a file count and a snapshot date.

The canonical layout is now produced in one step:

    python scripts/build_flat_mirror.py [DEST_DIR]

Hand-authored mirror-only docs live under `scripts/mirror_extras/` and are
restored automatically; `FILE_INDEX.md` is regenerated from `_manifest.json`.

Running this file now does nothing; it prints where the real generator lives.
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def main():
    print(__doc__)
    print("No action taken. Use: python scripts/build_flat_mirror.py")


if __name__ == "__main__":
    main()
