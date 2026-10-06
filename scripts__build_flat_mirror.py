"""Build a deliberately FLAT single-directory mirror of this project, then push it
to GitHub.

Why: the working project is ~41 GB across 1705 files, which GitHub cannot host
(100 MB per-file hard limit). This script selects the auditable subset (code,
docs, reports, figures, small result data), renames every file with an ASCII
prefix that encodes its original directory, and writes it all side by side into
one flat namespace so an automated reader can enumerate and fetch the entire
project without walking a tree.

Original directory -> flat prefix
    (root)                      no prefix
    scripts/                    scripts__
    configs/                    configs__
    models/                     models__
    losses/                     losses__
    metrics/                    metrics__
    engine/                     engine__
    viz/                        viz__
    tests/                      tests__
    dataset/                    dataset__
    figures/                    figure__
    outputs/report_figs/*.png   reportfig__
    outputs/report_figs/**      resultdata__report_fields__
    outputs/premodel/*          resultdata__premodel__
    outputs/** (small)          resultdata__<sub>__
    20261002-报告备份-1/        reportbackup__

Usage:
    python scripts/build_flat_mirror.py [DEST_DIR]
"""
import hashlib
import io
import json
import os
import re
import shutil
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = sys.argv[1] if len(sys.argv) > 1 else \
    r"E:\Projects\20260921-JiaChenchen-Paper-github"

MB = 1024 * 1024

# ---------------------------------------------------------------- selection
ROOT_FILES = {
    "README.md", "STATUS.md", "VISUALIZATION.md", "IMPLEMENTATION_CHECKLIST.md",
    "requirements.txt",
    "report.html", "report.md", "report.pdf",
    "report_brief.html", "report_brief.md", "report_brief.pdf",
    "20260924-方案.md",
}

CODE_DIRS = ("scripts", "configs", "models", "losses", "metrics",
             "engine", "viz", "tests")

# small result data outside outputs/premodel that carries headline numbers
OUT_SMALL_EXT = (".json", ".csv", ".yaml", ".yml")
OUT_SMALL_MAX = 2 * MB
PREMODEL_MAX = 5 * MB        # keep json/csv/tif/ckpt, drop the big npz
REPORTFIG_MAX = 2 * MB


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


def ascii_safe(name):
    """Keep ASCII names verbatim; transliterate the few Chinese ones."""
    if all(ord(c) < 128 for c in name):
        return name
    if name.endswith("方案.md"):
        return "PLAN_20260924.md"
    return re.sub(r"[^0-9A-Za-z._-]+", "_", name).strip("_")


def flat_name(r):
    """Map a repo-relative path to its flat single-directory name."""
    parts = r.split("/")
    top = parts[0]
    base = ascii_safe(parts[-1])

    if len(parts) == 1:                                   # already at root
        return base
    if top == "figures":
        return "figure__" + base
    if top == "outputs":
        sub = parts[1]
        if sub == "report_figs":
            if len(parts) == 3:
                return "reportfig__" + base
            inner = "__".join(parts[2:-1])
            return f"resultdata__report_fields__{inner}__{base}"
        if sub == "premodel":
            if len(parts) == 3:
                return "resultdata__premodel__" + base
            inner = "__".join(parts[2:-1])
            return f"resultdata__premodel__{inner}__{base}"
        inner = "__".join(parts[1:-1])
        return f"resultdata__{inner}__{base}"
    if top.startswith("20261002"):
        return "reportbackup__" + base
    if top in CODE_DIRS or top == "dataset":
        inner = "__".join(parts[1:-1])
        return f"{top}__{inner}__{base}" if inner else f"{top}__{base}"
    inner = "__".join(parts[:-1])
    return f"{inner.replace('/', '__')}__{base}"


def selected():
    """Yield repo-relative paths of everything that belongs in the mirror."""
    out = []
    for dp, dn, fn in os.walk(ROOT):
        dn[:] = [d for d in dn
                 if d not in (".git", "__pycache__", ".ipynb_checkpoints")]
        for f in fn:
            p = os.path.join(dp, f)
            r = rel(p)
            parts = r.split("/")
            top = parts[0]
            try:
                sz = os.path.getsize(p)
            except OSError:
                continue

            if len(parts) == 1:
                if r in ROOT_FILES:
                    out.append(r)
                continue
            if top in CODE_DIRS:
                if f.endswith((".pyc", ".pyo")):
                    continue
                out.append(r)
                continue
            if top == "dataset":
                # dataset code + small data (rain .npy, manifest, docs); never
                # the multi-GB .nc grids under dataset/grids/
                if r.startswith("dataset/grids/"):
                    continue
                if sz > 3 * MB:
                    continue
                out.append(r)
                continue
            if top == "figures":
                out.append(r)
                continue
            if top == "outputs":
                if r.startswith("outputs/report_figs/"):
                    if sz <= REPORTFIG_MAX:
                        out.append(r)
                    continue
                if r.startswith("outputs/premodel/"):
                    if sz <= PREMODEL_MAX:
                        out.append(r)
                    continue
                # other outputs: only small machine-readable result files
                if f.endswith(OUT_SMALL_EXT) and sz <= OUT_SMALL_MAX:
                    out.append(r)
                continue
            if top.startswith("20261002"):
                if f.startswith("report."):
                    out.append(r)
                continue
    return sorted(out)


def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    files = selected()
    print(f"selected {len(files)} files from {ROOT}")

    # resolve collisions deterministically
    used, plan, collisions = {}, [], []
    for r in files:
        name = flat_name(r)
        if name in used:
            stem, ext = os.path.splitext(name)
            n = 2
            while f"{stem}__dup{n}{ext}" in used:
                n += 1
            new = f"{stem}__dup{n}{ext}"
            collisions.append((r, name, new))
            name = new
        used[name] = r
        plan.append((r, name))

    if os.path.isdir(DEST):
        shutil.rmtree(DEST)
    os.makedirs(DEST)

    total = 0
    index = []
    for r, name in plan:
        src = os.path.join(ROOT, r)
        dst = os.path.join(DEST, name)
        shutil.copy2(src, dst)
        sz = os.path.getsize(dst)
        total += sz
        index.append({"flat": name, "origin": r, "bytes": sz,
                      "sha256": sha256(dst)})

    print(f"copied {len(plan)} files, {total/MB:.2f} MB "
          f"(avg {total/max(1,len(plan))/1024:.1f} KB)")
    if collisions:
        print(f"resolved {len(collisions)} name collisions:")
        for r, old, new in collisions:
            print(f"   {r}  ->  {new}")

    with open(os.path.join(DEST, "_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"source_root": ROOT, "file_count": len(index),
                   "total_bytes": total, "files": index},
                  fh, ensure_ascii=False, indent=1)
    print("wrote _manifest.json")
    return index


if __name__ == "__main__":
    main()
