# Design: document orientation, layout analysis and table structure in RustO

Date: 2026-10-31, revision 2 after builder check. Status: awaiting owner
approval. Repository: rusto-rs (fork `ppantaleev/rusto-rs`, upstream
`byrizki/rusto-rs`, crate version 0.3.2, HEAD 8d3ca03 on
`fix/orient-double-softmax`). Language: English, because the result is meant
to be upstream-compatible.

## 1. Intent and why

RustO today recognises text lines. It contains scaffolding for three more
stages — page orientation (`src/orient.rs`, plus an orientation step inside
`RustO::run_on_mat_with_options`, rusto_ocr.rs:446-500), layout analysis
(`src/layout.rs`) and table structure (`src/table.rs`, `src/doc_pipeline.rs`).
The scaffolding does not match any real model: the layout post-processing
expects a `[N,6]` output and rescales boxes that are already in page pixels,
the class map is for PP-DocLayout_plus-L (10 labels), the table recogniser
feeds input `image` to a model whose input is `x`, its `postprocess` returns a
placeholder 1×1 cell (table.rs:412-432), the table detector returns an empty
vector (table.rs:276), and the CLI prints a warning instead of running
orientation (main.rs:66-68, "Layout detection removed" at main.rs:43,70).

We need these stages for two uses, in this order:

1. A one-off conversion of 3 899 scanned pages of Bulgarian construction tender
   documents into structured text: reading order, headings, paragraphs, tables
   with rows/cells/spans, figures, header/footer separation. Measured on 26
   pages: RustO's Cyrillic PP-OCRv5 recogniser reaches CER ≈ 0.017 on prose but
   ≈ 0.28 on tables, because it returns lines, not cells. Pages rotated 90° give
   CER ≈ 0.95 without orientation correction.
2. A future desktop application (Rust) where a user drops a scan and expects
   the same structured result locally.

Everything stays pure Rust on `rten`. A spike (`peo/eval/spikes/rten-models`,
2026-10-31) showed all three ONNX models convert and run under `rten` 0.27 with
outputs matching onnxruntime (orientation 3 ms, layout 0.55 s, table 30–60 ms
per inference on an M-series Mac).

## 2. Goals

- Correct 0/90/180/270 page orientation before layout and OCR with RapidAI
  `rapid_orientation_v2.onnx` (PP-LCNet, 4 classes, Apache-2.0).
- Detect layout regions with the official `PaddlePaddle/PP-DocLayoutV3_onnx`
  (25 labels, Apache-2.0) and emit them in reading order.
- Recognise table structure with RapidAI `slanet-plus.onnx` (Apache-2.0):
  structure tokens → logical grid with `rowspan`/`colspan`, cell boxes, OCR
  text assigned to cells.
- One machine-readable document result per page (JSON) and a Markdown
  rendering, through `DocPipeline` and the CLI.
- Upstream compatibility: same module layout, `*Config` structs in
  `src/types.rs`, CLI `--enable-*` convention, `EngineError`, no new runtime,
  additive public API wherever possible (§6.5). Models downloaded, never
  vendored.
- Measurable quality on the project's test sets (§8).

## 3. Non-goals (v1)

- Small-angle deskew. Only 4-class orientation in v1.
- Changes to unwarping (`uvdoc`), detector or recogniser.
- Formula, chart, seal recognition: such regions are emitted with label and
  box only.
- Exposing the new stages through UniFFI/C/Node/WASM bindings (API must stay
  binding-friendly: plain structs, no lifetimes in results).
- Multi-page PDF input; the caller rasterises.
- The OpenCV backend (`use-opencv` feature): CI does not build it. New stages
  compile under it but return `EngineError` "not supported with the OpenCV
  backend"; the pure-Rust `image_impl` backend is the only supported one.

## 4. Anti-goals

- Any C/C++ dependency (`ort`, OpenCV for the new code).
- Model weights in the repository, or implicit network downloads at runtime.
- GPL/AGPL code or weights. Only Apache-2.0/MIT/BSD sources may be translated,
  with attribution in the file header and `CHANGELOG.md`.
- Silent degradation: a stage whose model is not configured is skipped and the
  result says so (`DocResult.stages`), never guessed.

## 5. Constraints

- Runtime `rten` 0.27 as pinned; models as `.rten` (converted with
  `rten-convert` 0.22) or `.onnx` via the `onnx_format` feature — the builder
  checks how `engine.rs` picks the loader by extension. The official Paddle
  orientation model fails in `rten` after conversion (`Slice`, missing inputs);
  the RapidAI v2 model works — use it.
- `RtenSession::from_path` currently ignores `EngineConfig` (engine.rs:302), so
  thread settings are not configurable; performance targets below are measured
  with `rten`'s default threading on an M-series Mac, per 300 dpi A4 page:
  orientation ≤ 10 ms, layout ≤ 0.8 s, table structure ≤ 0.1 s per table,
  whole pipeline peak RSS < 2 GB.
- Platforms: macOS Apple Silicon first; Linux x86_64 must build; no
  platform-specific code.
- Branching: new branch from `fix/orient-double-softmax` (contains the softmax
  fix 8d3ca03 that the orientation model needs). No push; no commit without the
  owner's explicit instruction.
- Licensing: RustO is MIT. Translated post-processing from `bobine`
  `rapid_table.rs` (Apache-2.0 OR MIT; same `slanet-plus.onnx`, same SHA-256)
  and from `ppocr-rs` (Apache-2.0; 25-label map, XY-cut) is attributed.
- Fixtures committed to the repository are synthetic, never corpus pages.
  `.gitignore` covers `models/**/*.onnx|*.rten|*.mnn` only; any extra model
  side-file (dictionary, yml) must be added to `.gitignore` or embedded as a
  Rust constant.

## 6. Decisions taken

### 6.1 Models

| Stage | File → converted | Size | Source | License |
|---|---|---|---|---|
| Orientation | `rapid_orientation_v2.onnx` → `models/DocOCR/orient_rapidai_v2.rten` | 6.8 MB | RapidAI RapidOrientation | Apache-2.0 (repo + upstream PaddleClas) |
| Layout | PP-DocLayoutV3 ONNX → `models/DocOCR/layout_ppdoclayoutv3.rten` | 130 MB | HF `PaddlePaddle/PP-DocLayoutV3_onnx` | Apache-2.0 |
| Table structure | `slanet-plus.onnx` → `models/DocOCR/table_slanetplus.rten` | 7.8 MB | RapidAI RapidTable | Apache-2.0 |

- URLs, SHA-256 and `inference.yml` constants are in
  `peo/eval/spikes/rten-models/results.json`. Add a new
  `scripts/download_doc_models.sh` (URL + SHA-256 table + optional
  `rten-convert` step), separate from the existing ModelScope-based
  `scripts/download_models.sh`; register the set in `MODELS_VERSION` if that
  file lists model sets (builder checks its format).
- The 25 layout labels are embedded as a `const [&str; 25]` with the
  `inference.yml` source cited. The SLANet_plus token dictionary is read from
  the ONNX `character` metadata at load, with an embedded fallback constant.

### 6.2 Coordinate space and orientation ownership

- `DocPipeline` owns page orientation. It runs the orientation model, rotates
  the page, and **all boxes in `DocResult` (regions, cells, lines) are in the
  corrected-page coordinate space**. `DocResult.orientation = { applied_degrees:
  0|90|180|270, confidence: f32 }` records what was done; the corrected image is
  available from `DocPipeline::run_with_image` (returns the `Mat`) and the CLI
  flag `--save-corrected <path>` writes it. Consumers that need original-scan
  coordinates rotate back using `applied_degrees` (a helper
  `rotate_box_back(bbox, degrees, w, h)` is provided).
- Per-region OCR inside the pipeline calls `RustO` with its own orientation
  stage **off**, to avoid double rotation.
- Threshold: `OrientConfig::default` stays 0.9 (types.rs:354, 356) — no change
  for existing users. `DocPipelineConfig.orient_min_confidence` (new field,
  default 0.6) is the pipeline-level threshold; below it the page is left as is
  and `orientation.applied_degrees = 0` with the measured confidence recorded.
- Direction convention (does class "90" mean the page content is rotated
  clockwise or counter-clockwise?) is verified with a synthetic fixture rotated
  by a known angle; the verified convention is documented in `orient.rs` and
  tested.

### 6.3 Layout

- Pre-processing exactly as PaddleX `inference.yml` for PP-DocLayoutV3 (the
  spike recorded the constants: resize target, normalisation, input names);
  output `[N, 7]`; the spike confirmed boxes are in original-image pixels — no
  rescale. The builder confirms what the 7th column is; if it is the model's
  reading order, use it; otherwise order with an XY-cut translated from
  `ppocr-rs`.
- DETR-style output: **no NMS**. Keep boxes with score ≥ `LayoutConfig.
  conf_thresh`; change that default from 0.4 to 0.5 (the spike's value) and
  actually read it in `postprocess` (today the thresholds are hard-coded at
  layout.rs:269 and the config fields are unused).
- Nested boxes: a text-like box whose area is ≥ 80 % inside a `table`,
  `image`, `chart` or `seal` box is dropped (the container's processing covers
  it). Overlapping text-like boxes are both kept.
- Label → behaviour map (all 25 V3 labels; `LayoutType` gains variants and
  `#[non_exhaustive]`):

| Label | OCR | Markdown rendering |
|---|---|---|
| doc_title | yes | `# ` heading |
| paragraph_title | yes | `## ` heading |
| text, content, abstract, aside_text, reference, reference_content, footnote, vision_footnote | yes | paragraph (footnotes as `[^n]` block at end of body) |
| vertical_text | yes (crop rotated 90° before OCR, builder picks direction by trying both and keeping the higher mean rec score) | paragraph |
| figure_title | yes | italic caption line placed after the nearest preceding `image`/`chart`/`table` region |
| table | table pipeline (§6.4) | GFM table; HTML `<table>` when spans exist |
| image, chart, seal | no | `![label](#region-N)` placeholder with bbox in JSON |
| display_formula, inline_formula, formula_number, algorithm | no (v1) | fenced block `formula`/`algorithm` with bbox reference |
| header, footer, number, header_image, footer_image | yes for text ones, no for images | moved to trailing section "## Page furniture" |

### 6.4 Table structure

- Input: table crop, keep-ratio resize to 488 with padding, input name `x`
  (fix table.rs:342-380).
- Decode: structure head argmax → tokens → logical grid with spans; location
  head → one box per non-empty cell in token order (shape confirmed by the
  builder from the output tensor: 4 or 8 coordinates), scaled to page space.
  Translate `bobine`'s `rapid_table.rs` decoder to `rten` tensors.
- Text assignment: OCR lines from the table crop; a line goes to the cell whose
  box contains its centre, else the cell with highest IoU; several lines in one
  cell are joined with a space in reading order; unmatched lines go to
  `TableStructure.unassigned_lines`.
- Data model changes in `table.rs`: keep `TableStructure { rows: Vec<TableRow>,
  num_rows, num_cols }`, add `unassigned_lines: Vec<TextLine>`; `TableCell`
  gains `score: f32`; `bbox` becomes `BBox` in **absolute page pixels** (the
  percent-of-crop convention at doc_pipeline.rs:174-177 is removed in the same
  change). `to_html()`/`to_markdown()` keep working.
- `TableDetector` (RT-DETR cell detector, stub) is left as is and is **not**
  required; `DocPipeline` runs the table pipeline whenever
  `table_recognizer` is set (fix the "both must be set" condition at
  doc_pipeline.rs:128). Reuse the existing `TableStructureConfig` (table.rs:282)
  — add `input_size: 488`, `max_tokens`, and a `TableModelType::SlanetPlus`
  variant; its default `model_path` becomes `models/DocOCR/table_slanetplus.rten`.

### 6.5 Public API and versioning

- `DocPipelineConfig` keeps `layout: LayoutConfig` required and
  `table_detector`/`table_recognizer` as they are; it gains `orient:
  Option<OrientConfig>` and `orient_min_confidence: f32`. New fields make
  struct-literal construction a source-breaking change for callers who build
  the struct directly; provide `DocPipelineConfig::new(layout)` and builder
  methods, mark the struct `#[non_exhaustive]`, and bump the crate to **0.4.0**
  with a CHANGELOG entry. `LayoutType` becomes `#[non_exhaustive]` with new
  variants; internal `match` sites get `_ =>` arms.
- `DocResult` gains `regions: Vec<DocRegion>`, `orientation`, `stages: Vec<
  StageStatus>` (which stages ran, which were skipped and why); `DocBlock` stays.
- CLI (`src/main.rs`), following the existing `--enable-orient`/`--orient-model`
  pattern: `--enable-layout`, `--layout-model <path>`, `--enable-table`,
  `--table-model <path>`, `--save-corrected <path>`; new output value
  `document`. `--enable-orient` becomes functional (today it only warns).
  Existing flags keep their meaning.

### 6.6 Testing

- Unit: token decoder (hand-written token sequences → grids with spans, including
  a malformed sequence); cell assignment on synthetic boxes; orientation
  pre-processing shapes; label map completeness (25 entries, matches the
  const); markdown rendering with spans; `rotate_box_back` round trip.
- Model tests (`#[ignore]` unless models present): orientation on the four
  rotations of a synthetic page → correct class and direction; layout on a
  synthetic page with a title, two paragraphs, a table → expected labels in
  expected order; table on a synthetic ruled 3×4 table with one merged cell →
  grid shape and span.
- Fixtures: generated once by `tests/fixtures/gen_doc_fixtures.py` (Python,
  PIL, DejaVu Sans for Cyrillic, licence noted) and the PNGs checked in under
  `tests/fixtures/doc/` (small, via Git LFS like the other test assets). No new
  Rust dev-dependency.

## 7. Left to the builder

- `inference.yml` constants and the 7th-column semantics (see §6.3).
- Per-region OCR (default; matches the current `doc_pipeline.rs` routing) vs
  full-page OCR clipped to regions: decide by the gate in §8.
- `.onnx` vs `.rten` loader selection in `engine.rs`.
- Exact error messages; the `StageStatus` enum shape.
- Whether `MODELS_VERSION` needs an entry.

## 8. Acceptance and gates

- Repository: existing tests pass; `cargo clippy` clean on touched modules;
  builds on macOS and Linux (CI); model tests pass locally with downloaded
  models.
- Evaluation outside the repository, on the 26 real pages
  (`peo/eval/extract/ocr/pages`, truth in `truth/`, model-generated) and the
  synthetic-truth set (`peo/eval/extract/ocr/synthetic`, exact truth from PDF
  text layers):
  - Orientation: every rotated page in both sets is corrected (100 %).
  - Prose CER ≤ 0.022 (baseline 0.017 + 0.005).
  - Table CER ≤ 0.20 on the 26 pages (baseline 0.28) **and** cell-structure
    recall ≥ 0.80 (rows and cells recovered vs truth) on the synthetic tables.
  - Gate for the OCR strategy (§7): if per-region OCR raises prose CER above
    0.022, switch to full-page OCR clipped to regions; if cell-level OCR raises
    table CER above the line-based baseline, assign full-page lines to cells
    instead of re-recognising crops.
  - Time per page within §5 targets.
- Use 2 is satisfied when the library API (`DocPipeline` with the new config)
  is exercised by one example in `examples/` without the CLI.

## 9. Calls made without asking the owner (please check)

1. Crate version bump to 0.4.0 because of `#[non_exhaustive]` and new config
   fields; additive otherwise.
2. OpenCV backend: new stages compile but are unsupported there.
3. Boxes are reported in corrected-page space; original-space is derived.
4. Pipeline orientation threshold 0.6 as a new pipeline field; `OrientConfig`
   default untouched.
5. No NMS for the DETR-style layout output; score threshold 0.5.
6. Nested text boxes inside tables/images are dropped at ≥ 80 % containment.
7. Fixtures generated by a Python script with DejaVu Sans; PNGs checked in via
   LFS.
8. New download script instead of extending the ModelScope one.
9. Acceptance numbers in §8.
