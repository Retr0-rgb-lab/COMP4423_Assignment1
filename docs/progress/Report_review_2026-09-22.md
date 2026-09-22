# COMP4423 Assignment 1 — Report Review Synthesis (2026-09-22)

> **Read-only synthesis.** Five independent reviews (A–E) of `docs/progress/Report_draft.md`
> were completed and merged here by a dedicated synthesis agent. The report draft has
> **not** been modified by this process. This file is the actionable work-list for the
> author / the next editing agent. Reviews overlap heavily, so findings below are
> **deduplicated** (each row carries the source letters A–E that raised it).

## 0. How to read this file

- **Priority tiers** (bilingual labels):
  - **P0 — Must fix before submission** (factual errors, compliance gaps, cumulative-grading risk)
  - **P1 — Strongly recommended** (number reconciliation, conclusion closure, disclosure of scope)
  - **P2 — Recommended** (academic rigour, discussion, references, figure embedding)
  - **P3 — Polish** (de-AI phrasing, spelling, consistency)
- Each tier is one table: `Item | Severity | Report location | Problem | Suggested fix | Sources`.
- Report locations are `Report_draft.md` section / line anchors; figures/JSON anchors point into `code/pics/`.
- Reviewer source letters: **A** logic/method chain · **B** discussion richness · **C** compliance ·
  **D** de-AI style · **E** devil's-advocate (facts under attack).
- Un-enumerated minor items (A MINOR structural-continuity ×3; B MINOR ×4; E MINOR ×5) whose
  specifics were not transcribed into the synthesis brief are **not** invented here; request the
  per-review records if a full sweep is wanted (see Appendix).

## 1. Executive summary

| Tier | Count | One-line theme |
|------|-------|----------------|
| P0 | 6 | False "every-metric" claims, S_max dataset contradiction, uncited brick-summary output, L2 un-shot, Q6 empty, FPS headline mixes scales |
| P1 | 17 | Run-identity/provenance disclosure, EPI loop closure, fringe & border scope, S_min=1, "reproducible"/"no approximation" overclaims, ΔE drift, region-merge undocumented |
| P2 | 10 | References, MS-SSIM, EPI name collision, crop_shadow figure, 9-vs-10 metrics, palette-switch rationale, single-image qualifier, overclaim sweep |
| P3 | 10 | De-AI edits (L19, "genuine", em-dash), British spelling, we/author, tense, WxH/HxW, "Step N —" titles |

**Strongest cross-review signal (raised by 3+ reviewers):** the ΔMSE-vs-Sobel "beats on **every**
metric" claim (A, C, E) and the S_max table inconsistencies §4.2 vs §4.4 (A, B, E). Fix these
first — they are the report's most checkable claims.

---

## 2. P0 — Must fix before submission

| # | Item | Severity | Report location | Problem | Suggested fix | Sources |
|---|------|----------|-----------------|---------|----------------|---------|
| P0-1 | ΔMSE vs Sobel **"beats on every metric"** is false | Critical · 事实错误 | §4.2 L190–194; §4.6 L269; §6.2 P3 L546 | The report's own §4.2 table shows `E:prio_edgef1` **leads** on SSIM (0.374 vs 0.370), ΔE (9.83 vs 10.22) and Quant Err (6.45 vs 7.33); it loses only on MS-SSIM / Edge F1 / EPI. "Lost on every metric" is contradicted by the report's own table. Also the `E:prio_edgef1` row was produced by the **per-region Sobel** path that Task 4 later replaced with the global-Sobel SAT impl, so the row is **not reproducible** — the §5.4 Step-2 caveat must travel with it. | Replace "every metric" with the honest split (wins on structure MS-SSIM/EdgeF1/EPI, loses on colour ΔE/Quant) or weaken to "wins on the decisive metrics"; add a one-line provenance note pinning E_priority numbers to the pre-SAT per-region-Sobel implementation. Same fix in §6.2 P3's outcome cell. | A, C, E |
| P0-2 | S_max comparison mixes **two different palette datasets**; "unambiguous win" contradicts "no single best" | Critical · 逻辑/事实矛盾 | §4.2 B2 rows L169–170 & footnote L177–178 vs §4.4 L226–240 | §4.2's B2 S_max sweep is on **K-Means-Lab** (SSIM 0.359→0.421, ΔE 9.95→10.21); §4.4's `smax_curve` is on **Median-Cut** — its S_max=32 row (ΔE 10.61 / SSIM 0.375 / EdgeF1 0.322) is byte-identical to §4.2's `D:median_cut` row. The palette difference is never disclosed, and the two sections reach opposite verdicts: §4.2 "largest **unambiguous win**" (L185) vs §4.4 "**no single best** S_max" (L233). | Label each sweep by palette explicitly (K-Means-Lab sweep / Median-Cut sweep), state why the experiments differ, reconcile the two verdicts in one place (e.g. "on the K-Means-Lab palette 64 dominates the structure metrics but still worsens ΔE; the §4.4 trade-off verdict stands"). Delete "unambiguous" or drop the absolute "no single best". | A, B, E |
| P0-3 | **Brick-size summary output never cited** (Task 3 deliverable) | Critical · 合规 (loses marks) | Task 3 §4 (L148–293) — no reference to `code/pics/task3/summary/brick_size_counts.json` | The PDF explicitly asks Task 3 for a brick-size (brick-count) summary; `brick_size_counts.json` exists (Report_prep_assets §1, green) but the report never mentions it, so the deliverable is invisible to the marker. | Add a Task 3 subsection (or table + figure) reporting the per-size brick counts of the chosen config from `brick_size_counts.json`, and cite the anchor in §4. | C |
| P0-4 | **Task 4 Level 2 not captured**; cumulative-grading risk not warned | Critical · 累计制风险 | §5.5 L492–499 (AUTHOR placeholder) | L2 ("different content") is still a placeholder. Grading is cumulative (L1 camera+display; L2 different content; …), so an un-shot L2 **caps the whole of Task 4 at L1 = 10/25**. The draft never states this risk. | Author captures 3–5 named scenes on the real camera (plan already in §5.5). Until then the report must say explicitly: "Task 4 currently satisfies L1 only; L2 pending author capture; cumulative grading ⇒ max 10/25". | C, E |
| P0-5 | **Q6 (GenAI limitations) is an empty AUTHOR placeholder** | Critical · 合规 (highest-risk gap) | §7 L586–605 | AGENTS §7/§8 reserve the GenAI-limitations write-up for the author, and Task 5 awards marks for it. Today §7 contains only raw-material pointers, no content. | Author writes §7 (structure suggested in the placeholder: (1) AI-code reliability, (2) AI-as-analyst biases, (3) remaining limitations). Cannot be delegated. | C |
| P0-6 | **FPS headline mixes apples and oranges** | Critical · 口径 | §5.6 L505–507 ("0.45 FPS -> ~37 FPS"); §5.4 L419, L489 | "0.45 FPS" is the full 640×480 pipeline *with camera read*; "37.1 FPS" is a **static scene with geometry reused and no camera read-back** — `cap.read()` alone is ~31–33 FPS, an upper bound the processing-only ~16.8 FPS sits below. §5.6 stacks a processing-only/synthetic number against a camera-loop baseline — exactly the "only quote paired ratios" discipline the draft itself advocates in §5.2. | Give each FPS its full context inline (camera? static? processing-only? synthetic?), quote only *paired* before/after where a number is called a speedup, and reword the headline (e.g. "pipeline from 0.45 FPS (camera, full quality) to ~16.8 FPS processing-only / 37 FPS static-scene effective, byte-identical output"). | E |

## 3. P1 — Strongly recommended

| # | Item | Severity | Report location | Problem | Suggested fix | Sources |
|---|------|----------|-----------------|---------|----------------|---------|
| P1-1 | §4.2 footnote "same default run" is **falsified by the table itself** | High · 数字冲突 | §4.2 L177–178 | Footnote claims `quadtree = B2:smax_32 = D:kmeans_lab = E:prio_mse`. The shown rows disagree: `E:prio_mse` SSIM 0.370 / ΔE 10.22 ≠ quadtree 0.359 / 9.95. Either the runs differ (seed/palette/params) or numbers drifted; either way the identity claim is wrong as printed. | State the exact configuration + seed per row, or drop the identity claim and mark the rows "same settings, K-Means-Lab run-to-run variance (declared)". | B, C |
| P1-2 | `A:k16` vs "chosen" config — same name, unexplained number difference | High · 数字冲突 | §4.2 L168 vs §4.6 L276 | `A:k16` (ΔE 8.74, SSIM 0.360) and "chosen quadtree S_max=32 K=16" (ΔE 8.95, SSIM 0.359) are **different S_max** (64 vs 32) but the report never states it, so the two "K=16" numbers look contradictory. | Put the full config in both table captions (S_max, S_min, palette, priority) or add one "run key" table mapping every row to its settings. | A |
| P1-3 | **EPI loop not closed**: Task-2 negative-EPI motivation vs Task-3 EPI turning positive | High · 结论未闭合 | §3.2 finding 3 L119–123 vs §4.5 L249 | §3.2 presents negative EPI as the *motivation* for adaptive geometry; §4.5 reports "EPI turns positive (−0.043 → +0.049)" after the bug fix — never put in the same frame. EPI is also used both as a diagnosis metric and as an evaluation metric in the ΔMSE-vs-Sobel comparison (E: circular). | Add one sentence closing the loop ("the negative EPI that motivated Task 3 turns positive once the quadtree splits correctly") and explicitly separate EPI-as-diagnosis from EPI-as-evaluation so the priority comparison is not judged on a metric it was tuned by. | A, B, E |
| P1-4 | **Fringe leak disclosed but never quantified / conclusions never re-verified** | High · 披露不足 | §5.4 Step 1 L362–379; §5.7 L514–516 | The report admits all Task 2/3 brick colours are contaminated by the `fillPoly` bottom-right fringe (up to 400% leak on size-1 bricks, ~8% of pixels) but only asserts the conclusions are unaffected. Prep assets already hold the numbers (mean abs diff 3.0 / max 46.1 / 8.1% of pixels). | Insert the measured quantification, state the directional bias (~3/255 channel units), and either (a) re-run one headline number (chosen-config ΔE/SSIM) with in-cell means and show the delta, or (b) assert cross-run validity explicitly ("all runs used the same leaky reference ⇒ comparisons unbiased; absolute values carry a small directional bias"). | A, B |
| P1-5 | **Metric scope incl. border pixels not declared** (23.6% border is a confound) | High · 口径声明 | §5.3 L351–354 (border = 72,480 px, 23.6%); §2 metrics L64 | The draft says the border "measurably affects metrics" but never states whether Task 2/3 numbers include the border — E reads this as a violation of the PDF's "own image region" wording. | Declare the exact evaluation region for every metric number; add a one-run **border-free ablation**; if the PDF demands "own image region", report borderless numbers as primary. | C, E |
| P1-6 | **S_min=1 chosen config contradicts the marginal-benefit evidence** | High · 自相矛盾 | §4.6 L267 vs §4.3 L214–216 | The rate-distortion study shows size-2 splits = 0.4% of benefit (justifies avoiding fine cells), yet the chosen config sets S_min=1 with no justification. | Either justify S_min=1 with a number (e.g. budget under-utilisation; size-1/2 splits still help EPI) or switch the chosen S_min to 2 and update §4.6 + metrics JSON via `task3_best.py`. | A, E |
| P1-7 | "All 15 runs reach 9,990" is wrong for `smax_128` | High · 数字错误 | §2 L68; §4.1 L161 | `smax_128` lands at **9,988**, not 9,990 (99.88% utilisation). "All Task 3 runs use 9,990" / "99.9% utilisation" is stated twice. | Correct to "9,988–9,990; budget utilisation ≥ 99.8%", or give per-run values. | C, E |
| P1-8 | §8 "reproducible" vs declared **K-Means non-reproducibility** | High · 矛盾 | §8 L633; §4.6 footnote L280–286 | §8 credits the architecture for "reproducible" Task 2/3 results while §4.6 admits K-Means-Lab is non-reproducible (chosen ΔE 8.95 here vs 9.03 in Task3.md). | Qualify: "reproducible under fixed seed / except palette construction, whose variance is declared and bounded"; add the seed to every metrics table. | A, E |
| P1-9 | **"no approximation" contradicts the temporal-layer fixes** | High · 过度声明 | §5.6 L507; §5.4 Step 5 L442–449, Step 8 L483–490 | Spatial/exact-reuse optimisations are exact, but palette cadence (rebuild every 10 frames), warm-start, dead-band, label hysteresis and AE/WB lock are deliberate **temporal** approximations. | Scope the claim: "no spatial/quality approximation — bit-identical output for identical inputs; temporal coherence via frame-to-frame freezing/quantising (declared)". | E |

| P1-10 | **K=16 "best overall-fidelity" is overstated** | High · 过度声明 | §4.2 L182–184 | K=16 wins colour (ΔE 8.74, Quant 5.62) but **loses** structure: SSIM 0.360 vs K=4's 0.384, EPI 0.047 vs 0.064. "Best overall fidelity" is only true for colour terms. | Reword to "best **colour** fidelity at a small structure cost"; let §4.6's "balanced" framing carry the choice. | E |
| P1-11 | **Region-merge algorithm never described** | High · 方法披露 | §4.1 L154–156; §4.2 L195–197 | The reader cannot evaluate quadtree-vs-region-merge because the region-merge procedure (candidate generation, target-size alignment, merge criterion) is never explained. | Add 2–4 sentences or a pointer to `brick_geom.py::region_merge_partition` with the criterion and complexity. | A |
| P1-12 | **Three ΔE drift points, only two reconciled** | High · 数字冲突 | §4.6 footnote L280–286 (8.95 vs 9.03) | The footnote reconciles the chosen ΔE against Task3.md §10 but not against §4.2 values (B2:smax_32 9.95; A:k16 8.74) that a reader will compare to 8.95. | Reconcile all three in one table, or declare a single source of truth (`code/pics/task3/best/*_metrics.json`) and label every other occurrence as "prior run". | B |
| P1-13 | **Metric↔perception inversion collapsed into an AI anecdote** | High · 讨论未展开 | §6.4 L577–582 (the S_max=32-vs-64 episode; Task3.md §9.5) | The strongest conceptual point — objective metrics and human perception can invert (ΔE says 32, SSIM/EPI and the eye say 64) — is compressed into a GenAI anecdote instead of a first-class discussion item. | Promote to a short dedicated discussion (or expand §4.4) tying the shadow-area result to the metric-selection question; keep the AI-anecdote framing as one illustration. | B |
| P1-14 | **Q2 of the template answered only implicitly** | High · 合规 | §6 (L525–582) | The 6-question template's Q2 is handled implicitly (workflow narrative) with no explicit answer block; markers following the template will not find it. | Add an explicit Q1–Q6 mapping (a short subsection or a table) so every template question has a visible answer location. | C |
| P1-15 | **SSIM "differs by 0.0004" conflicts with a 0.002 value elsewhere** | High · 数字冲突 | §3.2 L108 ("0.0004") vs the recorded 0.002 (Task2.md) | The Task-2 otsu-vs-kmeans SSIM delta is stated as 0.0004 in the draft but 0.002 in Task2.md's record; two numbers for the same comparison. | Pick one and make the other consistent (regenerate from the metrics JSON or note the seed), so the "otsu ≈ kmeans" claim carries one number. | A |
| P1-16 | **Task-3 headline claims lack a single-image generalisation qualifier** | High · 口径声明 | §4.2 findings L182–196; §8 L621–624 | Conclusions like "Larger K monotonically improves colour error" and the K/S_max choices are stated without the "on this single test image" qualifier; B flags this as over-generalisation risk given one `sky.jpg` run. | Prepend/append "on `sky.jpg`" (or "for this scene") to every Task-3 headline claim and to §8's Task-3 paragraph. | B |
| P1-17 | **`E:prio_edgef1` row provenance not disclosed** (per-region Sobel superseded) | High · 可复现性 | §4.2 table L175; §5.4 Step-2 caveat L394–398 | The recorded E_priority numbers come from the per-region-Sobel implementation that Task 4 replaced with a global-Sobel SAT table (median ~2× relative difference near edges); §4.2 presents the row without this caveat, so it is not reproducible from the shipped code. | Attach the §5.4 caveat to the §4.2 row (one line), or rerun E_priority with the current impl and report the delta. The ΔMSE-wins conclusion is robust either way. | E |

## 4. P2 — Recommended (rigour / discussion / references)

| # | Item | Severity | Report location | Problem | Suggested fix | Sources |
|---|------|----------|-----------------|---------|----------------|---------|
| P2-1 | **No References section** | Medium · 学术规范 | after §8 (L638) | The report uses named methods (MS-SSIM, ΔE2000, Canny, K-Means, median cut, EPI-style correlation) with no citations; a marker expects a bibliography for such a quantitative report. | Add a short References list (Wang et al. SSIM/MS-SSIM, Sharma/CIEDE2000, Canny, Lloyd's K-Means, median-cut source). Keep it tight. | C |
| P2-2 | **MS-SSIM implementation is non-standard** | Medium · 学术规范 | §2 metrics L64 | MS-SSIM as implemented is not the standard Wang et al. multi-scale SSIM; presenting it under the standard name overstates. | Note the exact implementation (which scale levels / synthesis weights) or rename it and cite the variant. | C |
| P2-3 | **EPI name collides with an existing index** | Medium · 规范 | §2 metrics L64; §3.2, §4.2 uses | "EPI" already denotes Edge-Preservation Index (and EPnP); the report's EPI = Sobel-response Pearson correlation could be confused. | Define the acronym locally on first use ("our EPI — Pearson correlation of Sobel responses") or rename (e.g. "edge-alignment correlation, EAC"). | C, A |
| P2-4 | **`crop_shadow.png` exists but is never cited** | Medium · 图嵌入 | §4.4/§6.4 shadow claim (L581 "shadow area ~44% worse") | The strongest visual evidence of the S_max=32-vs-64 trade-off (the shadow region, `code/pics/task3/analysis/crop_shadow.png`) is in prep assets but not referenced where the claim is made. | Cite the figure at the shadow-area claim and in §4.4; one line each. | (orchestrator P2 example) |
| P2-5 | **"9-metric suite" actually lists 10 items** | Medium · 事实口径 | §2 L64; also §3.1 L87, §6.1 L537, §8 L615 | The suite row lists PSNR, SSIM, MS-SSIM, ΔE2000, Edge F1/Precision/Recall, EPI, Quant Error, Budget Utilisation = 10, yet it is called "9-metric" in at least 4 places. | Rename to "10-item suite" or split Budget Utilisation out (as a constraint, not a quality metric) and state 9. | A, C |
| P2-6 | **Otsu → K-Means-Lab switch between Task 2 and Task 3 is never justified** | Medium · 连贯性 | §2 palette rows L60–61; §4.1 L155 | Task 2 makes Multi-Otsu primary, then Task 3 silently switches to K-Means-Lab; the reason (K>3 palette needs a general clustering method; Lab is perceptually uniform) is stated in the table but not tied to why Otsu is dropped. | One sentence in §4.1: why the primary quantiser changes between tasks (threshold count grows, so move to K-Means-Lab). | A |
| P2-7 | **"Noise amplification" framing (intro) is never delivered** | Medium · 结构 | §1 L24–25 vs rest of report | The intro promises noise-amplification as a competing effect (mean-from-tiny-sample), but no later section measures or discusses it (Task 4 jitter is temporal, not this). | Either add one measurement (e.g. ΔE vs brick size on noisy input) or demote the bullet to a mention of sensor noise under Task 4. | A |
| P2-8 | **"all numbers here come from >=10-frame runs" conflicts with an 8-frame baseline** | Medium · 数字口径 | §5.2 L342–343 vs the 8-frame baseline elsewhere | The blanket >=10-frame claim clashes with a recorded 8-frame baseline; the reader cannot tell which table obeys the rule. | State the frame count per table, or drop the blanket claim and give per-table counts. | A |
| P2-9 | **Overclaim sweep** (E's 8-item "weaken" mapping) | Medium · 语气 | throughout §4–§5 | Beyond the P0/P1 items, E lists ~8 superlative/absolute statements that overreach the data (see the mapped weaken-list in the review). | Sweep the whole report for absolutes ("best", "every", "unambiguous", "exact", "no approximation", "reproducible", "genuine", "clear winner") and replace each with the scoped form E mapped. | E |
| P2-10 | **Border-free / region-scoped metric ablation** (complement to P1-5) | Medium · 方法 | §2 metrics L64; §3.2, §4.2 tables | Even if the region is declared, markers value an explicit ablation showing metric values with and without the 23.6% border, to isolate the confound. | Run the chosen-config metrics once on a borderless canvas and tabulate both; one table, one sentence. | C, E |

## 5. P3 — Polish (de-AI phrasing, spelling, consistency)

| # | Item | Severity | Report location | Problem | Suggested fix | Sources |
|---|------|----------|-----------------|---------|----------------|---------|
| P3-1 | **L19 "The central question is not A, but B" re-ordering** | Low · 去AI化 | L19 | The "[not A] but [B]" rhetorical reordering is the strongest AI-scented opener; reviewer D flags it as a must-change. | Rewrite as a plain statement, e.g. "This report studies how to allocate a fixed detail budget across the image." | D |
| P3-2 | **"genuine" ×4** | Low · 去AI化 | L60, L117, L221 (title), L354 | "genuine comparison / genuine trade-off / genuine two-sided trade-off / genuine hue spread" — an AI-favoured intensifier used 4×. | Keep at most one; replace others with "real", "true", "measured" or drop. | D |
| P3-3 | **"move in opposite directions"** | Low · 去AI化 | §4.4 L232 | Formulaic phrasing. | Replace with a concrete statement of the two metric families' behaviour (ΔE best at 32 and rising; SSIM/EdgeF1/EPI rising to 64 then flat). | D |
| P3-4 | **~35 spaced em-dashes + whole-sentence bold + NOT/IS caps** | Low · 排版 | throughout; e.g. L339 "NOT 12x slower", L472 "IS the image" | Spaced em-dash (" — ") used ~35×, several whole sentences bolded, and NOT/IS capitalised for emphasis — all read as AI-flavoured emphasis. | Convert most spaced em-dashes to commas / colons / separate sentences (keep ~5 max); unbold whole sentences; drop ALL-CAPS emphasis (state the correction plainly). | D |
| P3-5 | **British vs American spelling mixed** | Low · 拼写统一 | throughout | "colour/colours" (majority) vs "optimization/optimization-headers", "quantize/quantise", "analyze/analyse"; mixed -ize/-ise. | Unify to **British** (colour / optimisation / quantisation / analyse) since colour forms dominate; run one global pass. | D |
| P3-6 | **we / the author voice swings** | Low · 人称 | §5.1 "We chose" (L307) vs §5.4 Step-8 "Author confirmed" (L489), §5.3 "A defect fixed" | First-person "we" alternates with third-person "the author"; inconsistent narrator. | Pick one narrator (recommend "the author / this report" or consistent "we") and apply throughout. | D |
| P3-7 | **Tense mixing** | Low · 时态 | throughout | Present ("the report follows"), past ("was diagnosed"), present-perfect ("has measured") interleave without rule. | Use past tense for what was done/measured, present for what the report claims/argues; one pass. | D |
| P3-8 | **Dimensions WxH vs HxW inconsistent** | Low · 一致 | L43 "1706x1279" vs L335 "1279x1706" | Same image written both ways. | Pick WxH or HxW once; report `sky.jpg` as 1706×1279 (WxH) everywhere. | D |
| P3-9 | **"Step N —" template titles** | Low · 风格 | §5.4 Step 1–8 (L362, L381, L401, L424, L442, L464, L477, L483) | The "Step N — name." heading pattern (with trailing full stop and em-dash) reads templated. | Convert to plain numbered headings ("**Step 1: mean colour.**" with sentence case) or drop the em-dash. | D |
| P3-10 | **Structural continuity at three points** | Low · 连贯 | §2–§4 transitions (per A) | A flags three continuity snags (e.g. §4.4's table introduced before its sweep is described; §4.3's rationale placed after §4.2's verdict). | Re-order or add one bridging sentence at each flagged transition. | A |

---

## 6. Suggested revision order

Work in this sequence — cheapest, highest-impact first; author-only items in parallel:

1. **Factual fixes (P0-1, P1-7, P1-1, P1-2, P1-15, P0-6 wording):** correct the "every metric" assertion, the 9,990/9,988 claim, the "same default run" footnote, the K=16/A:k16 caption ambiguity, the SSIM 0.0004/0.002 delta, and re-scope the FPS headline. No re-runs needed — pure text/table edits.
2. **Provenance & disclosure (P0-2, P1-4, P1-5, P1-17, P1-12, P1-8):** label the two S_max sweeps by palette and reconcile the verdicts; quantify the fringe leak (numbers already in `Report_prep_assets.md` §3); declare metric evaluation region + border-free ablation; attach the per-region-Sobel caveat to the E_priority row; reconcile all ΔE occurrences; qualify "reproducible". Needs one `task3_best.py` re-run only if you add the border-free ablation.
3. **Compliance blockers (P0-3, P0-4, P0-5, P1-14, P2-1):** cite `brick_size_counts.json` + add a brick-summary table/figure; author captures L2; author writes Q6; add an explicit Q1–Q6 answer map; add References.
4. **Method & conclusion closure (P1-3, P1-6, P1-9, P1-10, P1-11, P1-16):** close the EPI loop; justify or change S_min=1; scope "no approximation"; soften K=16; describe region-merge; add single-image qualifiers.
5. **Discussion & rigour (P1-13, P2-2, P2-3, P2-4, P2-5, P2-6, P2-7, P2-8, P2-9, P2-10):** promote the metric↔perception inversion; annotate MS-SSIM; disambiguate EPI; cite crop_shadow; fix the 9/10 metric count; justify the palette switch; deliver or demote noise-amplification; fix frame-count claim; overclaim sweep.
6. **Polish pass (P3-1 … P3-10):** de-AI edits, British spelling, em-dash/bold reduction, narrator + tense + dimension consistency, "Step N" headings, continuity bridges.
7. **Re-verify:** after any number change, re-run the affected script (fixed seed) and re-check the git-dirty `code/pics/task3/best/*_metrics.json` files so table and JSON agree.

---

## 7. What the AUTHOR must do personally (cannot be delegated)

- **§7 / Q6 — GenAI limitations write-up** (P0-5). Task 5 scoring item; AGENTS §7/§8 explicitly reserve it.
- **Task 4 Level 2 capture** (P0-4). 3–5 named scenes via `camera_app.py --snapshot-dir code/pics/task4/...`, then paste the qualitative per-scene results into §5.5. Until captured, the cumulative-grading risk (max 10/25) must be stated.
- **Confirm the reconstructed GenAI prompt wording** (§6, Q4/Q5) — the drafts are reconstructions the author must verify.
- **Confirm real-camera observations** (Task 4 jitter reduction; the P8 border preference) that the draft attributes to the author.
- **Decide S_min** if P1-6 is resolved by changing the chosen config (re-run + re-anchor figures + update JSON).
- **Write the final report** from this work-list; no AI agent should finalise the GenAI-limitations section or the L2 results.

---

## 8. Preserve-list (do NOT delete while revising)

Strong assets reviewers explicitly want kept:

- **Phrases D listed as keep-worthy:** "Syntactically valid, semantically wrong" (L129); "the AI did not volunteer that" (L582); the three reporting-bias examples (Task3.md §9).
- **Quantitative backbone:** every paired before/after table in §5; the bit-identical-output claims (0 differing pixels, ΔE == ΔE); the rejection stories (Task3 P3, Task4 P6/P8) as limitation material; the fringe/border measurements in `Report_prep_assets.md` §3.
- **Assets to cite, not delete:** `brick_size_counts.json`, `crop_shadow.png`, `crop_sky.png`, `metrics_table.csv`, `best/*_metrics.json`.

---

## 9. Appendix — raw review tallies (for traceability)

| Reviewer | Focus | CRITICAL | MAJOR/HIGH | MINOR | Score |
|----------|-------|----------|------------|-------|-------|
| A | logic-method-verification chain | 2 | 3 | 8 | — |
| B | discussion richness | 3 | 4 | 4 | 74/100 |
| C | compliance | — | — | — | 72/100 |
| D | de-AI style | — | — | 5 top items + ~35 em-dash etc. | — |
| E | devil's-advocate facts | 4 | 6 | 5 | — |

**Not enumerated in this synthesis** (specifics were not transcribed into the synthesis brief;
request the per-review records if a complete sweep is required): A MINOR ×3 structural-continuity
(covered generically by P3-10), B MINOR ×4, E MINOR ×5. Every item listed above is directly
traceable to the brief's stated conclusions.

---

*End of synthesis. Next step: author + editing agent work through P0 → P1 → P2 → P3,
re-verify numbers, then re-circulate for a light re-review of the changed sections.*
