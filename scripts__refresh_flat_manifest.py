"""Refresh _manifest.json so it describes the current flat directory exactly."""
import hashlib
import io
import json
import os
import sys
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
DEST = r"E:\Projects\20260921-JiaChenchen-Paper-github"
os.chdir(DEST)

AUTHORED = {
    "START_HERE_FLAT_LAYOUT.md", "FILE_INDEX.md", "README_PROJECT.md",
    "DATA_AND_BINARY_NOTICE.md", "chatgpt__00_TASK_BRIEF.md",
    "chatgpt__01_WHERE_TO_LOOK.md", "_manifest.json",
}


def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# origin mapping: recover from the flat name so the manifest stays useful
def origin_of(flat):
    if flat in AUTHORED:
        return "(authored for this repository)"
    if "__" not in flat:
        return flat
    parts = flat.split("__")
    top = parts[0]
    rest = parts[1:]
    if top == "reportfig":
        return "outputs/report_figs/" + rest[-1]
    if top == "figure":
        return "figures/" + rest[-1]
    if top == "resultdata":
        sub = parts[1]
        tail = parts[2:]
        if sub == "report_fields":
            inner = "/".join(tail[:-1])
            return f"outputs/report_figs/result_fields/{inner}/{tail[-1]}"
        if sub == "premodel":
            inner = "/".join(tail[:-1])
            return (f"outputs/premodel/{inner}/{tail[-1]}" if inner
                    else f"outputs/premodel/{tail[-1]}")
        inner = "/".join(parts[1:-1])
        return f"outputs/{inner}/{parts[-1]}"
    if top == "reportbackup":
        return "20261002-报告备份-1/" + rest[-1]
    inner = "__".join(parts[1:-1])
    return top + "/" + (inner.replace("__", "/") + "/" if inner else "") + parts[-1]


files = []
for name in sorted(os.listdir(".")):
    if not os.path.isfile(name):
        continue
    files.append({"flat": name, "origin": origin_of(name),
                  "bytes": os.path.getsize(name), "sha256": sha256(name)})

total = sum(f["bytes"] for f in files)
man = {
    "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "source_root": r"E:\Projects\20260921-JiaChenchen-Paper",
    "layout": "flat, no subdirectories, original directory encoded as an ASCII filename prefix",
    "file_count": len(files),
    "total_bytes": total,
    "files": files,
}
with open("_manifest.json", "w", encoding="utf-8", newline="\n") as fh:
    json.dump(man, fh, ensure_ascii=False, indent=1)

print(f"manifest refreshed: {len(files)} files, {total/1048576:.2f} MB")
print("largest:", max(files, key=lambda f: f["bytes"])["flat"])
big = [f for f in files if f["bytes"] > 50 * 1024 * 1024]
print(f"files over 50 MB: {len(big)}")
for f in sorted(big, key=lambda x: -x["bytes"]):
    print(f"  {f['bytes']/1048576:>6.2f} MB  {f['flat']}")
