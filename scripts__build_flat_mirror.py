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
    "RERUN_PLAN_AND_CHECKLIST.md",
    "requirements.txt", "requirements-core.txt", "requirements-paper.txt",
    "requirements-analysis.txt", "requirements-test.txt",
    "report.html", "report.md", "report.pdf",
    "report_brief.html", "report_brief.md", "report_brief.pdf",
    "20260924-方案.md",
    "20260921-JiaChenchen-Paper_全面审稿与代码级修改方案_20261006.md",
}

CODE_DIRS = ("scripts", "configs", "models", "losses", "metrics",
             "engine", "viz", "tests", "analysis")

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
    if "全面审稿" in name:
        return "AUDIT_CODE_LEVEL_REVIEW_20261006.md"
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
                # mirror_extras holds hand-authored mirror-only docs that are
                # restored at the mirror root by main(); never copy them again
                # under a scripts__ prefix.
                if r.startswith("scripts/mirror_extras/"):
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


# human-readable grouping of the flat namespaces, in listing order
_INDEX_GROUPS = [
    ("Full and condensed report artifacts (root)", lambda f: f in _ROOT_REPORTS),
    ("Project entry points and docs (root)", lambda f: f in _ROOT_DOCS),
    ("Orientation docs for an automated reviewer",
     lambda f: f in _MIRROR_EXTRAS),
    ("Figures embedded in the report, in render order",
     lambda f: f.startswith("reportfig__")),
    ("Published figures (upstream set)", lambda f: f.startswith("figure__")),
    ("Machine result data and small grids",
     lambda f: f.startswith("resultdata__")),
    ("Pipeline, figure generators, report builders, audits",
     lambda f: f.startswith("scripts__")),
    ("Loss functions", lambda f: f.startswith("losses__")),
    ("Model definitions", lambda f: f.startswith("models__")),
    ("Evaluation metrics", lambda f: f.startswith("metrics__")),
    ("Training and evaluation engine", lambda f: f.startswith("engine__")),
    ("Experiment configuration", lambda f: f.startswith("configs__")),
    ("Dataset adapters and small data", lambda f: f.startswith("dataset__")),
    ("Visualisation", lambda f: f.startswith("viz__")),
    ("Tests", lambda f: f.startswith("tests__")),
    ("Analysis (statistics, block bootstrap)",
     lambda f: f.startswith("analysis__")),
    ("Earlier report snapshot", lambda f: f.startswith("reportbackup__")),
    ("Manifest", lambda f: f == "_manifest.json"),
]

_ROOT_REPORTS = {"report.html", "report.md", "report.pdf",
                 "report_brief.html", "report_brief.md", "report_brief.pdf"}
_ROOT_DOCS = {"README.md", "STATUS.md", "VISUALIZATION.md",
              "IMPLEMENTATION_CHECKLIST.md", "RERUN_PLAN_AND_CHECKLIST.md",
              "PLAN_20260924.md", "AUDIT_CODE_LEVEL_REVIEW_20261006.md",
              "_manifest.json", "requirements.txt", "requirements-core.txt",
              "requirements-paper.txt", "requirements-analysis.txt",
              "requirements-test.txt"}
_MIRROR_EXTRAS = {"START_HERE_FLAT_LAYOUT.md", "README_PROJECT.md",
                  "DATA_AND_BINARY_NOTICE.md", "chatgpt__00_TASK_BRIEF.md",
                  "chatgpt__01_WHERE_TO_LOOK.md"}


def write_file_index(dest, index):
    """Regenerate FILE_INDEX.md from the manifest so its counts cannot go stale."""
    by_name = {e["flat"]: e for e in index}
    total_bytes = sum(e["bytes"] for e in index)
    lines = [
        "# FILE_INDEX",
        "",
        f"Complete flat inventory of this mirror. "
        f"**{len(index)} files, {total_bytes/MB:.1f} MB, no subdirectories.**",
        "",
        "Every entry lists the flat filename, its original repo-relative path and "
        "its size. Fetch by flat name. The machine-readable twin of this index is "
        "`_manifest.json`, which also carries a SHA-256 per file.",
        "",
    ]
    placed = set()
    for title, pred in _INDEX_GROUPS:
        members = [e for e in index
                   if pred(e["flat"]) and e["flat"] not in placed]
        if not members:
            continue
        gb = sum(e["bytes"] for e in members)
        placed.update(e["flat"] for e in members)
        lines += [f"## {title}", "",
                  f"{len(members)} files, {gb/MB:.2f} MB", "",
                  "| Flat filename | Original path | Bytes |",
                  "|---|---|---|"]
        for e in sorted(members, key=lambda x: x["flat"]):
            lines.append(f"| `{e['flat']}` | `{e['origin']}` | "
                         f"{e['bytes']:,} |")
        lines.append("")
    leftover = [e for e in index if e["flat"] not in placed]
    if leftover:
        lines += ["## Other", "",
                  f"{len(leftover)} files", "",
                  "| Flat filename | Original path | Bytes |", "|---|---|---|"]
        for e in sorted(leftover, key=lambda x: x["flat"]):
            lines.append(f"| `{e['flat']}` | `{e['origin']}` | "
                         f"{e['bytes']:,} |")
        lines.append("")
    with open(os.path.join(dest, "FILE_INDEX.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote FILE_INDEX.md ({len(index)} entries)")


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
        # Never touch .git: the mirror is a pushed git repo, and wiping its
        # object store corrupts history. Clear only the flat artefacts.
        for entry in os.listdir(DEST):
            if entry == ".git":
                continue
            p = os.path.join(DEST, entry)
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            else:
                try:
                    os.remove(p)
                except OSError:
                    pass
    else:
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

    # Hand-written orientation docs live only in the mirror (they are not
    # derived from the source tree). They are versioned under
    # scripts/mirror_extras/ so a rebuild can never silently drop them.
    extras_dir = os.path.join(ROOT, "scripts", "mirror_extras")
    extras = []
    if os.path.isdir(extras_dir):
        for f in sorted(os.listdir(extras_dir)):
            src = os.path.join(extras_dir, f)
            if not os.path.isfile(src):
                continue
            dst = os.path.join(DEST, f)
            shutil.copy2(src, dst)
            extras.append(f)
            sz = os.path.getsize(dst)
            index.append({"flat": f, "origin": f"scripts/mirror_extras/{f}",
                          "bytes": sz, "sha256": sha256(dst)})
        print(f"restored {len(extras)} hand-written mirror docs: "
              f"{', '.join(extras)}")

    with open(os.path.join(DEST, "_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"source_root": ROOT, "file_count": len(index),
                   "total_bytes": total, "files": index},
                  fh, ensure_ascii=False, indent=1)
    print("wrote _manifest.json")

    write_file_index(DEST, index)
    return index


if __name__ == "__main__":
    main()
