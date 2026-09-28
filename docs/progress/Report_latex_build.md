# Report — LaTeX build (md → .tex → PDF)

Stage record for converting the English draft `Report_draft.md` into the
submission PDF. Source of truth for the text is the markdown draft; this file
records only the *build*, per AGENTS.md §5.3 (process log separate from the
artefact).

## Artefacts

| File | Role |
|---|---|
| `docs/progress/Report_draft.md` | text source of truth (unchanged) |
| `docs/progress/Report_draft.tex` | LaTeX transcription, hand-written |
| `docs/progress/Report_draft.pdf` | rendered output, 32 pages, 5.2 MB |
| `code/pics/pdfimg/` | build-only down-scaled image copies (see below) |

## Rebuild

```bash
cd docs/progress
pdflatex -interaction=nonstopmode -output-directory=/tmp/texbuild Report_draft.tex   # ×2
cp /tmp/texbuild/Report_draft.pdf .
```

Two passes are required: `\ref` cross-references and the table of float
placement resolve on the second. Toolchain: pdfTeX 3.141592653-2.6-1.40.25
(TeX Live 2023/Debian) in WSL.

## The image-size defect (the reason this stage existed)

Symptom as reported: the figure in the report did not look like the photo that
was supplied, and image sizing looked wrong.

Root cause, established by measurement before changing anything:

- `code/pics/sky.jpg` (1706×1279) and `code/pics/task1_capture.png` (1706×1279)
  are **pixel-identical** — `mean|Δ| = 0.0`, `max|Δ| = 0`, 0 pixels differing.
  So the *content* was never wrong, and the figure was not the wrong image.
- The defect was in the LaTeX. `\includegraphics` without a `width`/`height`
  key renders at the image's natural size, and pdfTeX treats PNG at 72 dpi, so a
  1706 px image asks for ≈ 602 mm of width against an A4 text block of 160 mm.
  The image therefore overflowed the text block and was clipped, which is what
  made it look unlike the original.

Fix applied in `Report_draft.tex`:

- `\graphicspath{{../../code/pics/pdfimg/}{../../code/pics/}}` — resolves every
  figure back to the real files under `code/pics/`, so no image is re-generated
  or detached from its anchor (AGENTS.md §7).
- Three sizing macros, all with `keepaspectratio`, so no image can ever be
  stretched or overflow again: `\figfull` (text width, capped at 0.85\textheight),
  `\figtall` (height-constrained, for the 1738×4009 residual stack), `\figpart`
  (fraction of text width), plus `\figgrid` (0.95 text width, for 2×2 grids).
- Verified: **0 overfull hboxes**, and every one of Figures 1–15 measured inside
  the text block.

## Build-time image down-scaling (`code/pics/pdfimg/`)

Five figures are far larger than a printed page needs: `out_task2_compare.png`
(3472×2706, 6.2 MB), `out_task2_residual.png` (1738×4009, 7.0 MB),
`best_compare.png` (6.6 MB), `best_vs_task2.png` (6.2 MB), `crop_shadow.png`
(5134×1305, 5.2 MB) and `L2_contact_sheet.png` (4.7 MB) — ≈37 MB of source PNG.
Embedding them at full resolution produced a ~38 MB PDF.

`code/pics/pdfimg/` holds copies capped at 2000 px on the long edge (≈300 dpi at
A4 text width), photographic ones re-encoded as JPEG q92. Total 4.6 MB, and the
PDF is 5.2 MB instead of ~38 MB. These are **build artefacts, not new
deliverables**: the originals stay untouched and remain the citation anchors.
The directory is git-ignored. Both build modes are verified (see the table at
the end of this section): with `pdfimg/` the PDF is 5.2 MB, without it the same
`.tex` builds cleanly from the full-resolution originals at 38.7 MB.

To regenerate it (run from the repo root, Windows venv interpreter):

```bash
"../../venv/Scripts/python.exe" - <<'PY'
import os
from PIL import Image
SRC, DST, MAXW, MAXH = 'code/pics', 'code/pics/pdfimg', 2000, 1500
imgs = ['sky.jpg',
 'task2/out_task2_compare.png','task2/out_task2_residual.png','task2/out_task2_metrics_chart.png',
 'task3/summary/metrics_chart.png',
 'task3/analysis/marginal_benefit_curve.png','task3/analysis/cumulative_benefit_curve.png',
 'task3/analysis/benefit_by_size.png','task3/analysis/smax_curve.png','task3/analysis/crop_shadow.png',
 'task3/best/best_compare.png','task3/summary/best_vs_task2.png',
 'task3/A_ksweep/k16_size_hist.png',
 'task4/L1_baseline/frame0001_compare.png','task4/L2_contact_sheet.png']
for rel in imgs:
    s = os.path.join(SRC, rel); d = os.path.join(DST, rel)
    os.makedirs(os.path.dirname(d), exist_ok=True)
    im = Image.open(s); w, h = im.size
    sc = min(1.0, MAXW / w, MAXH / h)
    im2 = im if sc == 1.0 else im.resize((max(1, round(w*sc)), max(1, round(h*sc))), Image.LANCZOS)
    if rel.endswith('.jpg') or os.path.getsize(s) > 400_000:
        d = d.rsplit('.', 1)[0] + '.jpg'
        im2.convert('RGB').save(d, quality=92, optimize=True)
    else:
        im2.save(d, optimize=True)
    print('%-50s -> %s' % (rel, os.path.relpath(d, DST)))
PY
```

Note the extension change (`.png` → `.jpg`) for the six photographic renders.
Because the `.tex` references every figure **without an extension** (and
declares `\DeclareGraphicsExtensions{.jpg,.jpeg,.png,.pdf}`), both build modes
are verified working:

| Build | Result |
|---|---|
| with `code/pics/pdfimg/` | 32 pages, **5.2 MB**, 0 errors — all 15 figures from `pdfimg/` |
| without it (git-ignored, fresh clone) | 32 pages, 38.7 MB, 0 errors — all 15 from `code/pics/` |

Run the down-scaling step before compiling if you want the 5.2 MB output.

## Conversion decisions worth knowing

- **Mermaid → TikZ.** §6.2's pipeline diagram is `mermaid` in the markdown;
  pdfLaTeX cannot render it. It is transcribed as a native TikZ `figure` with
  the two measured bottleneck stages (`quadtree partition`, `per-triangle
  means`) shaded red, matching the markdown's `:::hot` class.
- **Duplicate heading.** `Report_draft.md` lines 71–72 contain
  `## 3. Method — the shared toolkit` twice; the `.tex` has it once. The
  markdown was left alone because it is the text source of truth — **this is a
  one-line fix the author may want to make in the markdown as well.**
- **Numbering is automatic.** `Figure N` / `Table N` come from LaTeX counters,
  not from the literal "Fig. 1." prefixes in the markdown alt text, so the two
  cannot drift. Confirmed in the output: Figures 1–15 and Tables 1–13, in order.
- **Character mapping.** Δ→`$\Delta$`, →→`$\to$`, ∈→`$\in$`, ≈→`$\approx$`,
  ∇→`$\nabla$`, µ→`\textmu`, ²→`$s^{2}$`, …→`\ldots`, —→`---`.
- **Long paths.** `\cpath` adds `\allowbreak` after `/` and `_` so a long
  `\texttt` path wraps instead of overflowing. An earlier `seqsplit`-based
  attempt was reverted: it is fragile with the escaped braces in
  `frame0001_{input,render,compare}.png`.
- **`lmodern` is required**, not cosmetic: `microtype` font expansion needs
  scalable fonts and aborts on the default bitmap Computer Modern.

## Verification performed

- 2 × pdflatex, clean: no errors, 0 undefined references, 0 overfull hboxes.
- 32 pages, 5.2 MB.
- Every page rendered to PNG and inspected; Fig. 1 confirmed to match
  `sky.jpg`/`task1_capture.png` content at correct aspect ratio, tall Fig. 3
  fits the page, TikZ diagram correct, 7-scene contact sheet correct.
- Automated sweep of the extracted PDF text for leaked macro arguments — found
  and fixed one real bug (`\figpart` declared with 3 args but called with 4, so
  `\label` names were being typeset as body text). Re-run: clean.
- **Both build modes re-verified from scratch after that fix** (0 errors, 0
  overfull, 0 undefined refs, 32 pages each). An earlier version of the
  extension-less fallback did not work — six filenames carried an explicit
  `.jpg` that only exists in `pdfimg/`, so a fresh clone failed with 7 errors
  and a 1.1 MB PDF. Caught by testing the fallback, not by reading the code.

## Revision — §6.3 figure and §6.5 rewrite

Author review flagged two things: the Figure 14 frame was unsuited to
demonstrate anything, and §6.5 was "just all the images listed together", with no
statement of what each one was meant to show about the pipeline.

**§6.3 / Figure 14** now shows `task4/snap_003.png` (the luggage scene) instead of
`L1_baseline/frame0001_compare.png`. Reason: the old frame is a backlit portrait
against a clipped window, which is the least legible content in the set, whereas
the luggage frame shows both adaptive decisions at once — fine cells on the bag
seams and zips against a coarse flat wall, and a 16-colour Lab palette holding
three distinct hues apart. It is the strongest single showcase available.

The numeric claims in that section (17 distinct colours, 24.3% border coverage,
std 62.1 vs 70.4) were measured on the **L1_baseline** frame, so the text now
says so explicitly and flags that Fig. 14 is a *different* frame. Moving the
figure without this would have attached measurements to the wrong image.

**§6.5** was restructured from a single 7-panel contact sheet to a table plus
six individual figures:

- **Table 9** states, per scene, the pipeline property it stresses. The scenes
  are no longer a gallery; they are a test matrix.
- The contact-sheet figure is gone. `snap_000/001/002/004/005/006` each get their
  own figure and a caption naming what the render shows; `snap_003` is referenced
  back to Fig. 14 rather than repeated.
- A closing paragraph states the cross-scene finding: the live pipeline is
  palette-limited, not geometry-limited, which is a different claim from the
  Task 3 result where the limit was spatial.

### Two factual errors found and corrected

Both were in the draft's description of the Task 4 snapshots, and both were found
by opening the seven images and comparing them against the text — the exact
"caption describes content the file does not contain" failure §8 of the report
confesses to. They are now fixed in **both** `Report_draft.md` and
`Report_draft.tex` so the two do not diverge:

1. `snap_006` was described as "a nearly uniform **bright** wall whose switch and
   light bar are already close to blown out... the mosaic **loses the subject
   almost entirely**". The file is a **dim** wall beside a bright window, and the
   render preserves the curtain, the window edge and the chair. The real weak
   case is banded low-contrast gradient, not subject loss; §6.5 now describes it
   that way, and says the first fix would be dithered/error-diffusion assignment
   rather than a larger K.
2. `snap_001` was described as "backlit through a sheer curtain". It is a
   front-lit close-up with a curtain in the background.

The §6.6 bullet that summarised the sweep carried the same errors and was
rewritten to match.

Page count 32 → 34 (six scene figures in place of one contact sheet); figure
count 15 → 20, tables 13 → 14. Clean build: 0 errors, 0 overfull, 0 undefined
references.

## Rebuild note (snapshots)

The seven `snap_00*.png` files are now figures, so they need down-scaled copies
too. Add this after the main list in the snippet above:

```python
for i in range(7):
    s = 'code/pics/task4/snap_%03d.png' % i
    im = Image.open(s); w, h = im.size
    d = 'code/pics/pdfimg/task4/snap_%03d.jpg' % i
    os.makedirs('code/pics/pdfimg/task4', exist_ok=True)
    im.convert('RGB').save(d, quality=92, optimize=True)
```

## Known limitations

- The PDF is a **draft rendering** of the markdown. AGENTS.md §7 still requires
  the final submission to follow `docs/materials/Report Template.docx`; the
  template's section structure has not been reconciled with this `.tex` yet.
- `Report_draft_zh.md` has **no** LaTeX counterpart — pdfLaTeX cannot typeset
  CJK without CJK/xeCJK packages, and only the English draft was in scope.
- Text is transcribed by hand, not machine-converted, so wording is faithful but
  line-breaking and hyphenation differ from the markdown rendering.
- Section 8 (GenAI limitations) is the author's own assessment and is reproduced
  as written; per AGENTS.md §7/§8 the agent does not author that content.
