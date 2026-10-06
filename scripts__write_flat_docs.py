"""DEPRECATED — superseded by `build_flat_mirror.py`.

This one-shot script used to write the flat-mirror orientation documents
(`START_HERE_FLAT_LAYOUT.md`, `FILE_INDEX.md`, `README_PROJECT.md`,
`DATA_AND_BINARY_NOTICE.md`, `chatgpt__00_TASK_BRIEF.md`,
`chatgpt__01_WHERE_TO_LOOK.md`) directly into the destination directory. It
hard-coded file counts and a snapshot date, so every rebuild silently produced a
stale `FILE_INDEX.md`.

The canonical layout is now produced in one step:

    python scripts/build_flat_mirror.py [DEST_DIR]

`build_flat_mirror.py` copies the auditable subset from the source tree, restores
the hand-authored mirror-only docs from `scripts/mirror_extras/`, writes
`_manifest.json`, and regenerates `FILE_INDEX.md` from that manifest so its counts
and byte sizes cannot drift.

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
