# HANDOFF — Report Preparation & Improvements

- **Date**: 2026-09-21 (assignment deadline: 2026-09-29 23:59 HKT, 8 days out)
- **Repo**: `/mnt/d/Program Files/learn_torch/python-cv/Assignment1` (branch `main` @ `52d1743`, clean worktree except two untracked items, in sync with origin)
- **Next-session focus** (per author): improvement directions + report preparation work
- **Governance**: read `AGENTS.md` first. Its priority order (PDF > AGENTS.md > AI output) and §5/§7/§8 constraints are binding. Notably: **the "GenAI limitations" report section must be written by the author, never by the agent** (§7, §8).

## 1. What the previous session did

A read-only audit was performed by three parallel expert agents (CV algorithms / systems engineering / report readiness). **No files were modified.** Their findings below are the only durable record — the session wrote nothing to disk. Verify claims against the cited paths before acting.

Verdict: **code and engineering process are strong; the project is NOT yet ready to write the final report.** Blockers are evidence gaps, not code gaps.

## 2. Report readiness (evidence lives in `docs/progress/Task0_brief.md`..`Task4.md`)

| Task | Verdict | Blocker |
|---|---|---|
| 0 brief | READY | — |
| 1 | READY | Only expected output recorded; paste one actual run result into `Task1.md` |
| 2 | MOSTLY | No GenAI prompt drafts (template Q4/Q5 worth 10 pts require v1/v2 prompts + AI-output summaries) |
| 3 | MOSTLY | No prompt drafts; per-size brick counts exist only inside annotated PNGs (`brick_viz.py::plot_size_histogram`) — need text/JSON for the report's brick-summary table; K-Means non-reproducibility must be declared |
| 4 | NOT READY | L2 ("different content") evidence missing — only one real camera scene captured (`code/pics/task4/`); status snapshot at top of `Task4.md` (lines 3–35) is stale and contradicts its own body — update before the report quotes it |
| 5 report | NOT STARTED | No `Report_*.md` draft anywhere. The template (`docs/materials/Report Template.docx`) requires exactly 6 Method questions: design/test, real-scenario robustness, problems & solutions, GenAI usage, GenAI understanding, GenAI limitations |

## 3. Improvement directions (ranked)

### Decision-needed code issues (fix or declare in report)
1. **Task 2 black band**: `compute_grid` floors, leaving an uncovered black margin (≈19px bottom + 5px right on `code/pics/sky.jpg`), penalizing PSNR/SSIM. Task 3 already fixed this via reflect-pad + crop (`brick_geom.py:128-156`) but Task 2 did not (`brick_geom.py:74-79`, render in `brick_render.py:96-104`). Fix is small and Task 3-proven.
2. **Fringe-contaminated means (Task 2/3)**: reference means average a 1-px `fillPoly` fringe (leak up to 400% at size 1) — `brick_means.py:23-48`; all recorded Quant_Error/palette metrics inherit it. Corrected in-cell means exist only for Task 4 (`brick_means.py:126-196`, `brick_means_rows.py:53-113`). Minimum action: quantify via `compare_means` (`brick_means.py:286-354`) and disclose.
3. **E_priority irreproducibility**: recorded `prio_edgef1` runs used per-region Sobel; current driver default `impl="sat"` gives different values (`brick_quadtree.py:284-293`, `brick_sat.py:79-85`). Re-running today won't reproduce `code/pics/task3/summary/metrics_table.csv`. Declare, or pin the old impl.
4. **Unverifiable headline number**: `task3_best.py:17-21` quotes region-merge ΔE 8.63 but `code/pics/task3/best/` has no metrics JSON. Regenerate with JSON output or drop the number.

### Engineering hygiene (cheap, do before report)
5. No `requirements.txt` / README / Makefile — environment reproducibility is zero (venv lives outside the repo, `../../venv` per AGENTS §9). At minimum add pinned deps (numpy, opencv-python, scikit-image, matplotlib).
6. Two untracked items: `docs/materials/Assignment1_Sample_report/` (extracted zip **violates AGENTS §2** read-only-materials rule — relocate to `docs/progress/`; never commit) and `code/pics/task4/ui_screenshot_legacy.png` (commit or delete).
7. `code/pics/task3/summary/best_vs_task2.png` promised by AGENTS §1.1 does not exist; nearest substitute is `code/pics/task3/best/best_compare.png`. Generate or re-anchor AGENTS + report.
8. Doc drift, low priority (declare only): unused `seed` params (`brick_color.py:194-196,249`), `FrameConfig.means` docstring omits default `"rows"` (`frame_pipeline.py:56-59`), `compute_grid` prose says "ceil-like" but code floors, stale "73 ms for 498 cells" comment (`camera_app.py:67-73`). Pipeline duplicated inline across `task3_best.py`, `task3_smax_curve.py`, `task3_marginal.py` (latter admits hand-syncing, `task3_marginal.py:44-49`).

### Things that are already good (do not churn)
Brick budget ≤10000 structurally enforced everywhere; exactly-3-colors for Task 2 (caveat: the 1-px `(60,60,60)` border is technically a 4th color — verify against the assignment PDF at `docs/materials/Assignment 1 - requirements COMP4423_202627S1-1.pdf` before claiming compliance); right-isosceles/no-overlap geometry single-sourced in `brick_geom.py`; three-part docstrings exemplary; `task4_verify.py` is a genuine deterministic test suite (assertions before benchmarks); commit history is exemplary conventional-commit code/docs pairs — it is itself report evidence.

## 4. Report preparation checklist (suggested order)

1. **Task 4 L2 evidence** (longest lead time, camera needed): capture 3–5 named scenes (face, texture, low light, high contrast) per the plan in `Task4.md` §Plan step 2; run `camera_app.py --input <clip> --no-show --max-frames N` headless for numbers.
2. **Update stale `Task4.md` snapshot** (top-of-file status block).
3. **Prompt drafts** for Tasks 2, 3, 4 into `docs/progress/Task{2,3,4}.md` — reconstruct from commit messages + author memory; the author must supply/recall actual prompts (do not fabricate; AGENTS §5.1).
4. **Per-size brick counts as text/JSON** for Task 3's brick-summary table (extend `task3_smax_curve.py`/`task3_best.py` output or parse existing `*_metrics.json` `counts_per_K`).
5. **Fix-or-decide on §3 items 1–4** with the author.
6. **requirements.txt + README** with run commands consolidated from AGENTS §9 and Task1–4 verification sections.
7. **Untracked-file cleanup** (§3 item 6) + `best_vs_task2.png` (item 7).
8. **Author-only**: GenAI-limitations section — raw material already exists at `Task2.md` end bullets, `Task3.md` §8–§9, `Task4.md` "third error of the same family"; the author synthesizes it personally.
9. Draft `docs/progress/Report_draft.md` against the template's 6 questions, citing anchors like `docs/progress/Task3.md#问题与解决方案` and `code/pics/task3/A_ksweep/k08.png`. Final output goes through `docs/materials/Report Template.docx` (AGENTS §7).

## 5. Suggested skills for the next agent

- `using-superpowers` — baseline; load before anything else.
- `planning-with-files` — the checklist above is 9 steps across camera work, docs, and code; persist a task plan so progress survives context loss.
- `markitdown` — to read `docs/materials/Report Template.docx` (and optionally the sample-report PDFs) as text before drafting.
- `systematic-debugging` — if the Task 2 black-band or means-fringe fixes are attempted (changes must then follow AGENTS §5.2: explain → author confirms → edit).
- `verification-before-completion` — every claimed fix must re-run the relevant entry point (e.g. `task4_verify.py --reps N`) before being reported done.
- `academic-paper` (modes: format-convert only, optional) — if Pandoc/docx conversion of the final report is needed; the 12-agent paper pipeline itself is overkill for this template-driven report.
- Do **not** use: `research` (writes a findings file into the repo — the audit is already recorded here), `tdd`/`test-driven-development` for report-prep tasks (no new features; `task4_verify.py` already covers code changes).

## 6. Reference index

- Assignment truth: `docs/materials/Assignment 1 - requirements COMP4423_202627S1-1.pdf` (read-only); report template: `docs/materials/Report Template.docx`; sample reports: see the zip in `docs/materials/` (extract to `docs/progress/` if needed, per AGENTS §2).
- Rules: `AGENTS.md` (esp. §3 doc structure, §5.2 explain-before-edit, §5.7 comment trio, §7 report discipline, §8 anti-patterns, §9 commands).
- Progress narrative: `docs/progress/Task0_brief.md` (contains the Task-4 L1–L4 rubric and report marking breakdown in its §5–§7), `Task1.md`–`Task4.md`.
- Metrics evidence: `code/pics/task3/summary/metrics_table.csv` (15 rows, 12-col metrics), per-experiment dirs `code/pics/task3/{A_ksweep,B2_smax,B_sweep,C_algorithm,D_palette,E_priority,analysis,best,summary}/`, `code/pics/task4/{L1_baseline,L3_means,new_opt,old_base}/`.
- Verification harness: `code/task4_verify.py` + `code/task4_verify_util.py` (headless, seed-controlled).

## 7. Redaction & environment notes

- No secrets exist in this repo (verified by grep). The origin remote is a private GitHub repo — identify it via `git remote -v` rather than this document.
- Dev environment: Windows + WSL dual setup; Python venv at `../../venv` relative to the repo (see AGENTS §9); `capture.py`/`imshow` hang under WSL headless (documented in `capture.py:23-25`); camera work requires the real Windows machine (single-machine constraint noted in `Task4.md`).
