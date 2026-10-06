# chatgpt__00_TASK_BRIEF — orientation for an automated cross-reviewer

You are being handed a **flat, self-contained snapshot** of a research project so
that you can cross-audit it without access to the original machine.

## What the project claims

The project reconstructs fine-grid urban flood depth from coarse-grid depth using
a geography-guided arbitrary-scale neural operator (`HydroGeo-SRNO`), on the
Wellington urban flood dataset, at a 10 m → 2 m scale for the maximum inundation
depth `h_max`. Its central finding is that the reconstructed field
**systematically underestimates deep water**, that the deficit has an
identifiable cause, and that a weighted loss plus a gentle fine-tune reduces it
without materially degrading overall skill.

## Revision status you must account for

This snapshot is post-**Major Revision**. An external audit found the conclusions,
code and statistics not fully consistent, and the following are now corrected:

- significance is computed over **independent seeds**, not training epochs, and a
  **spatial block bootstrap** on the 242 test tiles shows the uncertainty was
  previously underestimated by $1.25\times$–$1.63\times$;
- under that block bootstrap the **volume relative error and CSI@0.30 m cross
  zero**, so the volume-improvement claim is withdrawn;
- the ablation rows that made the "static contribution" claim invalid were
  code-level broken and are fixed, but the corrected ablation is **not yet rerun**;
- the multiscale/arbitrary-scale part now isolates `5→2`, `20→2`, `30→2` as unseen
  and must not be read as validated cross-scale generalisation;
- generalisation is bounded to **within-Wellington spatial extrapolation**.

Do not treat any model-level claim that `RERUN_PLAN_AND_CHECKLIST.md` marks as
"needs retrain" as established. When auditing, prefer the diagnostic/downgraded
framing in the current `report.*` over the earlier `reportbackup__report.*`.

## What you have

- Three full-report artefacts at the root (`report.html`, `report.md`,
  `report.pdf`) — 50+ figures, 30+ tables.
- Three condensed artefacts (`report_brief.*`) — the paper-length version.
- All source code (`scripts__*`, `models__*`, `losses__*`, …).
- All small result data (`resultdata__*`).
- The remediation plan and pre-submission checklist:
  `RERUN_PLAN_AND_CHECKLIST.md`; the audit it answers:
  `AUDIT_CODE_LEVEL_REVIEW_20261006.md`.
- A full inventory in `FILE_INDEX.md` and per-file SHA-256 in `_manifest.json`.
- What is missing and how to regenerate it: `DATA_AND_BINARY_NOTICE.md`.

## What is worth auditing

1. **Does the evidence support the headline claim?** Start with the deep-water
   deficit figures and the paired significance test, then check whether the
   numbers in the report text match the JSON they came from.
2. **Are the cross-resolution laws internally consistent?** The project reports
   that terrain correlates strongly across resolutions while flood depth and
   velocity decay non-linearly. Check the correlation, V-measure and
   chance-corrected matrices against each other.
3. **Is the loss-weight conclusion robust?** The weight sweep and the fine-tune
   arms should agree on direction. Check the trade-off tables.
4. **Was anything quietly dropped?** `STATUS.md` and the report's own limitation
   section list what the authors knew was unresolved. Verify the report does not
   overstate past them.
5. **Is the condensed report faithful to the full one?** Every value in
   `report_brief.*` should appear in `report.*` with the same value. This is the
   cheapest high-value check you can run.

## Metre of this project

A value you cannot trace to a `resultdata__*.json` or to a `scripts__*` generator
is a value worth questioning. The authors keep an explicit gap list rather than
claiming completeness; treat an honest gap as a feature, and a number that
cannot be traced as a bug.

*See `chatgpt__01_WHERE_TO_LOOK.md` for the specific file list.*
