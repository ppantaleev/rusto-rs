# Document Orientation, Layout and Table Structure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `DocPipeline` correct page orientation (RapidAI v2), detect PP-DocLayoutV3 regions in reading order, recognise SLANet_plus table structure with spans and cell text, and emit one JSON `DocResult` and one Markdown rendering per page, through the library and the CLI.

**Architecture:** The scaffolding modules are fixed in place: `orient.rs` (pre-processing, direction, box rotation), `layout.rs` (25 labels, `[N,7]` decode, nesting), `table.rs` + new `table_decode.rs` (pre-processing, token→grid decoder, cell boxes, line→cell assignment), `doc_pipeline.rs` (orientation ownership, routing by label role, result types, Markdown). All boxes in `DocResult` are in corrected-page pixels. Pure functions carry the logic and get unit tests; model tests are `#[ignore]` and run against synthetic fixtures.

**Tech Stack:** Rust 2021, `rten` 0.27 / `rten-tensor` 0.27, `ndarray` 0.16, `image` 0.25, `serde`/`serde_json`, `clap` 4; Python 3 + Pillow only for the one-off fixture generator; `rten-convert` 0.22 only in the download script.

**Spec:** `docs/superpowers/specs/2026-10-31-document-layout-tables-orientation-design.md` (revision 2). Spike facts: `/Users/petar/dev/private/peo/eval/spikes/rten-models/results.json` and `report-bg.txt`.

## Global Constraints

- Runtime `rten` 0.27 as pinned; models as `.rten` (converted with `rten-convert` 0.22) or `.onnx` via the `onnx_format` feature.
- The official Paddle orientation model fails in `rten` after conversion; use the RapidAI v2 model (`rapid_orientation_v2.onnx`).
- Performance per 300 dpi A4 page, `rten` default threading, M-series Mac: orientation ≤ 10 ms, layout ≤ 0.8 s, table structure ≤ 0.1 s per table, whole pipeline peak RSS < 2 GB.
- Platforms: macOS Apple Silicon first; Linux x86_64 must build; no platform-specific code.
- Branching: new branch from `fix/orient-double-softmax` (contains softmax fix 8d3ca03). No push; no commit without the owner's explicit instruction. Commit steps below run only after the owner authorises commits on the feature branch; until then leave the work uncommitted and report.
- Licensing: RustO is MIT. Translated post-processing from `bobine` `rapid_table.rs` (Apache-2.0 OR MIT) and from `ppocr-rs` (Apache-2.0) is attributed in the file header and `CHANGELOG.md`. Only Apache-2.0/MIT/BSD sources may be translated. No GPL/AGPL code or weights.
- Anti-goals: no C/C++ dependency (`ort`, OpenCV for the new code); no model weights in the repository; no implicit network downloads at runtime; no silent degradation — a stage whose model is not configured is skipped and `DocResult.stages` says so.
- OpenCV backend (`use-opencv`): new stages compile but return `EngineError` "not supported with the OpenCV backend"; the pure-Rust `image_impl` backend is the only supported one.
- Fixtures committed to the repository are synthetic, never corpus pages. `.gitignore` covers `models/**/*.onnx|*.rten|*.mnn` only; any extra model side-file must be added to `.gitignore` or embedded as a Rust constant.
- No new Rust dependency or dev-dependency.
- `OrientConfig::default` threshold stays 0.9; the recogniser, detector and unwarping are not changed.
- Stage only explicit paths with `git add`; never add the owner's untracked files (`PR-PROPOSAL-orient-double-softmax.md`, `examples/kolesat_orient*.rs`, `models/v5*/`, `models/v6-tiny/`).

## Review Focus

1. **Page where layout finds zero regions** (blank page, full-page photo): `regions` is empty, `to_markdown()` returns `""`, `stages` shows layout ran with `items: 0`, no panic. Tests: Task 9 `markdown_of_empty_result_is_empty`, Task 10 `blank_page_yields_no_regions`.
2. **Table region where the decoder yields zero cells** (eos first, garbage skeleton): `TableStructure` is 0×0, every OCR line is in `unassigned_lines`, Markdown shows that text as a paragraph. Tests: Task 7 `decode_empty_sequence`, Task 8 `assign_with_zero_cells_leaves_all_unassigned`, Task 9 `table_without_cells_renders_unassigned_text`.
3. **Orientation confidence exactly at the threshold** (0.6): rotation is applied (`>=`, same as `rusto_ocr.rs:492`). Test: Task 3 `decide_orientation_applies_at_exact_threshold`.
4. **Model file missing, wrong extension, or wrong model** (layout output not `[N,7]`): `EngineError` naming the stage and the path or shape, never a panic. Tests: Task 3 `load_stage_session_reports_missing_path`, `load_stage_session_rejects_unknown_extension`; Task 4 `decode_rejects_six_column_output`.
5. **Image smaller than the model input** (tiny crop, thin region): every pre-processor returns the fixed input shape without panic. Tests: Task 3 `orient_input_tiny_image`, Task 4 `layout_inputs_tiny_image`, Task 6 `table_input_tiny_crop`.

## File Structure

| File | Responsibility |
|---|---|
| `scripts/download_doc_models.sh` (new) | Download 3 ONNX models, verify SHA-256, optional `rten-convert` |
| `tests/fixtures/gen_doc_fixtures.py` (new) | One-off synthetic fixture generator (Pillow, DejaVu Sans) |
| `tests/fixtures/doc/*.png`, `*.json` (new) | Synthetic page + table crop and their ground truth |
| `src/types.rs` | `OrientPreprocess`, `OrientConfig::rapid_orientation_v2`, `LayoutConfig` defaults |
| `src/engine.rs` | `load_stage_session`, `metadata_value` |
| `src/orient.rs` | Orientation pre-processing, direction, `PageOrientation`, box/point rotation |
| `src/layout.rs` | 25 labels, `LayoutType`, `RegionRole`, V3 inputs/decode, nested drop |
| `src/table.rs` | Table types, config, recogniser (pre-processing, outputs, charset) |
| `src/table_decode.rs` (new, private) | Pure: argmax tokens, token→grid, cell boxes, line→cell assignment |
| `src/doc_pipeline.rs` | Result types, config API, pipeline, Markdown |
| `src/main.rs` | CLI flags and run planning |
| `src/lib.rs` | Exports |
| `examples/document_layout.rs` (new) | Library-only use of `DocPipeline` |
| `tests/doc_fixtures_test.rs`, `tests/doc_models_test.rs`, `tests/doc_pipeline_test.rs` (new) | Fixture sanity, model tests (`#[ignore]`) |

Model tests read doc models from `RUSTO_DOC_MODELS` (default `models/DocOCR`) and OCR models from `RUSTO_TEST_MODELS` (`det.onnx`, `rec.onnx`, `dict.txt`, as in `tests/rgb_input_test.rs:45`). If they are slow at `[profile.test] opt-level = 1`, run them with `--release`.

---

### Task 1: Feature branch, model download script, changelog stub

**Files:**
- Create: `scripts/download_doc_models.sh`
- Modify: `CHANGELOG.md:8` (insert `## [Unreleased]` above `## [0.3.2]`)

**Interfaces:**
- Produces: files `models/DocOCR/orient_rapidai_v2.{onnx,rten}`, `models/DocOCR/layout_ppdoclayoutv3.{onnx,rten}`, `models/DocOCR/table_slanetplus.{onnx,rten}`. Later tasks use the `.rten` paths as defaults; `resolve_model_path` (`engine.rs:208-258`) falls back to the `.onnx` with the same stem when the `.rten` is missing.

- [ ] **Step 1: Create the branch**

Run: `git -C /Users/petar/dev/private/rusto-rs switch -c feat/document-layout-tables-orientation fix/orient-double-softmax`
Expected: `Switched to a new branch 'feat/document-layout-tables-orientation'`; `git log -1 --format=%h` prints `8d3ca03`.

- [ ] **Step 2: Write `scripts/download_doc_models.sh`**

Bash, `set -euo pipefail`, options `--convert-rten`, `--keep-onnx` (default keeps ONNX: the table dictionary is read from ONNX metadata), `--output-dir DIR` (default `models/DocOCR`). Table (from `results.json`):

| Output name | URL | SHA-256 of the ONNX |
|---|---|---|
| `orient_rapidai_v2.onnx` | `https://github.com/RapidAI/RapidOrientation/releases/download/v0.0.0/rapid_orientation_models_v2.zip` (zip SHA-256 `bd21b970190538ae12f7f84972ab496f2aae79fbe4c5af4813920a7611a81d96`; member `rapid_orientation_models_v2/rapid_orientation.onnx`) | `2f62c9bfb830a0b417241269fde7ef2d0ad5446c0ed2b8af33b1f6543545e8e2` |
| `layout_ppdoclayoutv3.onnx` | `https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_onnx/resolve/main/inference.onnx` | `45bf71750b00739a41fc209f132eb104a4d6b5bb29483c9078164d8b87cf28ba` |
| `table_slanetplus.onnx` | `https://github.com/RapidAI/RapidTable/releases/download/assets/slanet-plus.onnx` | `d57a942af6a2f57d6a4a0372573c696a2379bf5857c45e2ac69993f3b334514b` |

Download with `curl -fL --retry 3` into a `mktemp -d` directory, verify with `shasum -a 256` (fall back to `sha256sum`), delete the file and exit 1 on mismatch, unzip the orientation zip inside the temp dir, move only the `.onnx` into the output dir. With `--convert-rten`: require `rten-convert` on `PATH` (no auto-install), print `rten-convert --version`, run `rten-convert X.onnx X.rten` per model. Print a licence line per model (Apache-2.0, sources as in spec §6.1).

Plan decision: no side files are left in `models/DocOCR` (labels and dictionary are Rust constants), so `.gitignore` is unchanged; `MODELS_VERSION` holds one version string for the published OCR model packages, so it gets no entry.

- [ ] **Step 3: Run the script and check the ignore rules**

Run: `bash scripts/download_doc_models.sh --convert-rten && ls models/DocOCR && git check-ignore -q models/DocOCR/table_slanetplus.rten && git status --short models/DocOCR`
Expected: three `.onnx` and three `.rten` files listed; `git check-ignore` exit 0; `git status` prints nothing.

- [ ] **Step 4: Changelog stub**

Insert `## [Unreleased]` + `### Added` + `- **doc**: scripts/download_doc_models.sh downloads and verifies RapidAI orientation v2, PP-DocLayoutV3 and SLANet_plus models (Apache-2.0)`.

- [ ] **Step 5: Commit**

```bash
git add scripts/download_doc_models.sh CHANGELOG.md
git commit -m "build(models): add download script for document orientation, layout and table models"
```

---

### Task 2: Synthetic document fixtures

**Files:**
- Create: `tests/fixtures/gen_doc_fixtures.py`
- Create: `tests/fixtures/doc/page_text.png`, `tests/fixtures/doc/page_text.json`, `tests/fixtures/doc/table_3x4_merged.png`, `tests/fixtures/doc/table_3x4_merged.json`
- Test: `tests/doc_fixtures_test.rs`

**Interfaces:**
- Produces: `page_text.png` 1240×1754 white page; `page_text.json` = `{"width":1240,"height":1754,"title":[x0,y0,x1,y1],"paragraphs":[[..],[..]],"table":[x0,y0,x1,y1]}`; `table_3x4_merged.png` = the page's table cropped with a 20 px margin; `table_3x4_merged.json` = `{"width":W,"height":H,"num_rows":3,"num_cols":4,"merged":{"row":0,"col":0,"row_span":1,"col_span":2},"rows":[["Позиция","Количество","Цена"],["1","Бетон","12","340,00"],["2","Арматура","5","1 200,00"]]}`.

- [ ] **Step 1: Write the failing test** (`tests/doc_fixtures_test.rs`)

```rust
fn fixture(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/doc").join(name)
}
#[test]
fn doc_fixtures_match_their_ground_truth() {
    let page: serde_json::Value = serde_json::from_slice(&std::fs::read(fixture("page_text.json")).unwrap()).unwrap();
    let img = image::open(fixture("page_text.png")).unwrap();
    assert_eq!((img.width(), img.height()), (1240, 1754));
    assert_eq!(page["paragraphs"].as_array().unwrap().len(), 2);
    let t: serde_json::Value = serde_json::from_slice(&std::fs::read(fixture("table_3x4_merged.json")).unwrap()).unwrap();
    let timg = image::open(fixture("table_3x4_merged.png")).unwrap();
    assert_eq!(timg.width() as u64, t["width"].as_u64().unwrap());
    assert_eq!((t["num_rows"].as_u64(), t["num_cols"].as_u64()), (Some(3), Some(4)));
    assert_eq!(t["merged"]["col_span"].as_u64(), Some(2));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test doc_fixtures_test`
Expected: FAIL, `No such file or directory` for `page_text.json`.

- [ ] **Step 3: Write the generator**

`python3 tests/fixtures/gen_doc_fixtures.py --out tests/fixtures/doc [--font PATH] [--font-bold PATH]`. Font search order when `--font` is absent: `/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf`, `/Library/Fonts/DejaVuSans.ttf`, `~/Library/Fonts/DejaVuSans.ttf`, matplotlib `mpl-data/fonts/ttf/DejaVuSans.ttf`; exit 2 with a message if none exists. Header comment: synthetic content only; DejaVu fonts licence (Bitstream Vera licence + public-domain changes; permits embedding glyph renderings). Layout: Cyrillic title (48 px, bold if available) at top; two paragraphs of 4 lines each (24 px); ruled table (2 px black lines, 3 rows × 4 columns, row 0 cells: "Позиция" spanning columns 0–1, "Количество", "Цена"; rows 1–2 as in the JSON above). Write the bounding boxes the script drew into the JSON files. Pillow only.

- [ ] **Step 4: Generate, then run the test**

Run: `python3 tests/fixtures/gen_doc_fixtures.py --out tests/fixtures/doc && cargo test --test doc_fixtures_test`
Expected: PASS. `du -k tests/fixtures/doc/*.png` each < 200 KB.

Plan decision: PNGs are committed as normal files, like `tests/fixtures/rgb-ocr.png` (6.9 KB, not in LFS; `git lfs` is not installed and `.gitattributes` has no LFS rule). Rotated variants are produced inside tests with the `image` crate (`rotate90/180/270`), so only upright images are committed.

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/gen_doc_fixtures.py tests/fixtures/doc tests/doc_fixtures_test.rs
git commit -m "test(doc): add synthetic page and table fixtures with generator"
```

---

### Task 3: Page orientation — pre-processing, direction, box rotation, stage loader

**Files:**
- Modify: `src/types.rs:329-361` (`OrientConfig`), new `OrientPreprocess`
- Modify: `src/orient.rs:28-127` (`Orientation`), `:135-241` (`OrientClassifier::new`, `preprocess`), tests `:289-371`
- Modify: `src/engine.rs` (add `load_stage_session` after `resolve_model_path`, ~`:258`)
- Modify: `src/rusto_ocr.rs:236-248`, `:300-312` (add `preprocess: OrientPreprocess::Stretch` to the two CLS `OrientConfig` literals)
- Modify: `src/lib.rs:74,78` (export `decide_orientation, rotate_box, rotate_box_back, rotate_point_back, PageOrientation`, `types::OrientPreprocess`)
- Test: `src/orient.rs` (unit), `src/engine.rs` (unit), `tests/doc_models_test.rs` (new, `#[ignore]`)

**Interfaces:**
- Produces:
  - `types::OrientPreprocess` — `#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)] pub enum OrientPreprocess { #[default] Stretch, ShortSideCenterCrop { resize_short: u32 } }`
  - `OrientConfig.preprocess: OrientPreprocess` with `#[serde(default)]`; `OrientConfig::default` sets `Stretch` (behaviour unchanged).
  - `OrientConfig::rapid_orientation_v2(model_path: PathBuf) -> OrientConfig`
  - `orient::orient_input(image: &Mat, cfg: &OrientConfig) -> Result<Array4<f32>, EngineError>` (`pub(crate)`)
  - `Orientation::label_degrees(&self) -> u16`, `Orientation::correction_cw_degrees(&self) -> u16`
  - `orient::PageOrientation { pub applied_degrees: u16, pub detected_degrees: u16, pub confidence: f32 }` (`Debug, Clone, Copy, PartialEq, Serialize`)
  - `orient::decide_orientation(out: &OrientOutput, min_confidence: f32) -> PageOrientation`
  - `orient::rotate_point_back(x: f32, y: f32, applied_degrees: u16, corrected_w: f32, corrected_h: f32) -> (f32, f32)`
  - `orient::rotate_box(bbox: &layout::BBox, cw_degrees: u16, src_w: u32, src_h: u32) -> layout::BBox`
  - `orient::rotate_box_back(bbox: &layout::BBox, applied_degrees: u16, corrected_w: u32, corrected_h: u32) -> layout::BBox`
  - `engine::load_stage_session(stage: &str, model_path: &Path, cfg: &EngineConfig) -> Result<EngineSession, EngineError>` (`pub(crate)`)

Conventions (pinned here): `applied_degrees` = clockwise rotation applied to the scan to obtain the corrected page; `detected_degrees` = the model's class label (0/90/180/270). Continuous pixel coordinates: clockwise 90° on a `W×H` image maps `(x, y) → (H − y, x)`; 180° maps `(x, y) → (W − x, H − y)`; 270° maps `(x, y) → (y, W − x)`. `rotate_box_back(b, a, cw, ch) == rotate_box(b, (360 − a) % 360, cw, ch)`. Degrees other than 0/90/180/270 return the input unchanged.

- [ ] **Step 1: Write the failing unit tests** (in `src/orient.rs` `mod tests`; Mat from `image::DynamicImage` via `Mat::new`)

```rust
fn solid(w: u32, h: u32, v: u8) -> Mat { Mat::new(image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(w, h, image::Rgb([v, v, v])))) }
fn out(o: Orientation, c: f32) -> OrientOutput { OrientOutput { orientation: o, confidence: c, elapse: 0.0 } }

#[test]
fn orient_input_center_crop_uses_imagenet_normalisation() {
    // 512x256: left half black, right half white. Short side already 256; crop x 144..368, y 16..240.
    let mut img = image::RgbImage::from_pixel(512, 256, image::Rgb([255, 255, 255]));
    for y in 0..256 { for x in 0..256 { img.put_pixel(x, y, image::Rgb([0, 0, 0])); } }
    let cfg = OrientConfig::rapid_orientation_v2("unused.rten".into());
    let t = orient_input(&Mat::new(image::DynamicImage::ImageRgb8(img)), &cfg).unwrap();
    assert_eq!(t.shape(), &[1, 3, 224, 224]);
    assert!((t[[0, 0, 100, 0]] - (0.0 - 0.485) / 0.229).abs() < 1e-3);
    assert!((t[[0, 0, 100, 223]] - (1.0 - 0.485) / 0.229).abs() < 1e-3);
}
#[test]
fn orient_input_stretch_keeps_legacy_scaling() {
    let cfg = OrientConfig::default("unused.rten".into());
    let t = orient_input(&solid(50, 20, 255), &cfg).unwrap();
    assert_eq!(t.shape(), &[1, 3, 224, 224]);
    assert!((t[[0, 2, 10, 10]] - 1.0).abs() < 1e-6);
}
#[test]
fn orient_input_tiny_image() {
    let cfg = OrientConfig::rapid_orientation_v2("unused.rten".into());
    assert_eq!(orient_input(&solid(20, 10, 128), &cfg).unwrap().shape(), &[1, 3, 224, 224]);
}
#[test]
fn decide_orientation_applies_at_exact_threshold() {
    let p = decide_orientation(&out(Orientation::Rotate180, 0.6), 0.6);
    assert_eq!((p.applied_degrees, p.detected_degrees), (180, 180));
}
#[test]
fn decide_orientation_below_threshold_keeps_page() {
    let p = decide_orientation(&out(Orientation::Rotate90, 0.5999), 0.6);
    assert_eq!((p.applied_degrees, p.detected_degrees), (0, 90));
    assert!((p.confidence - 0.5999).abs() < 1e-6);
}
#[test]
fn rotate_box_known_values_and_round_trip() {
    let b = BBox { x_min: 20, y_min: 10, x_max: 50, y_max: 30 };
    let r = rotate_box(&b, 90, 200, 100);
    assert_eq!((r.x_min, r.y_min, r.x_max, r.y_max), (70, 20, 90, 50));
    for d in [0u16, 90, 180, 270] {
        let (cw, ch) = if d % 180 == 0 { (200, 100) } else { (100, 200) };
        let back = rotate_box_back(&rotate_box(&b, d, 200, 100), d, cw, ch);
        assert_eq!((back.x_min, back.y_min, back.x_max, back.y_max), (20, 10, 50, 30), "degrees {d}");
    }
}
#[test]
fn correction_is_inverse_of_label() {
    for o in [Orientation::Normal, Orientation::Rotate90, Orientation::Rotate180, Orientation::Rotate270] {
        assert_eq!((o.label_degrees() + o.correction_cw_degrees()) % 360, 0);
    }
}
```

In `src/engine.rs` `mod tests`:

```rust
#[test]
fn load_stage_session_reports_missing_path() {
    let err = load_stage_session("layout", Path::new("models/DocOCR/nope.rten"), &EngineConfig::default()).err().unwrap();
    let msg = err.to_string();
    assert!(msg.contains("layout model not found") && msg.contains("nope.rten"), "{msg}");
}
#[test]
fn load_stage_session_rejects_unknown_extension() {
    let p = std::env::temp_dir().join("rusto-stage-test-model.bin");
    std::fs::write(&p, b"x").unwrap();
    let msg = load_stage_session("table", &p, &EngineConfig::default()).err().unwrap().to_string();
    assert!(msg.contains("table model must be a .rten or .onnx file"), "{msg}");
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib -- orient::tests engine::tests`
Expected: FAIL to compile, `cannot find function 'orient_input'` / `'load_stage_session'`.

- [ ] **Step 3: Implement**

- `OrientConfig::rapid_orientation_v2`: `orient_image_shape [3, 224, 224]`, `mean [0.485, 0.456, 0.406]`, `std [0.229, 0.224, 0.225]`, `preprocess ShortSideCenterCrop { resize_short: 256 }`, `confidence_threshold 0.9`, `orient_thresh 0.9` (source: PaddleClas `inference.yml` for PP-LCNet_x1_0_doc_ori, recorded in the spike).
- `orient_input`: `Stretch` = current `preprocess` body (`orient.rs:215-240`, resize to shape, `/255`, mean/std ignored, as today). `ShortSideCenterCrop`: scale = `resize_short / min(w, h)`, new size `round(w·scale)`, `round(h·scale)` (min 1), `INTER_LINEAR`; crop `224×224` at offset `((nw − 224) / 2, (nh − 224) / 2)` (integer division); RGB; `(p/255 − mean)/std`. Under `use-opencv`, `ShortSideCenterCrop` returns `EngineError::Engine("orientation pre-processing ShortSideCenterCrop is not supported with the OpenCV backend")`. `OrientClassifier::preprocess` delegates to `orient_input`.
- `label_degrees`: Normal 0, Rotate90 90, Rotate180 180, Rotate270 270. `correction_cw_degrees` follows `rotate_image` (`orient.rs:113-126`): Normal 0, Rotate90 270 (`rotate_270`), Rotate180 180, Rotate270 90.
- `decide_orientation`: `applied = if out.confidence >= min_confidence { correction_cw_degrees } else { 0 }`.
- `load_stage_session`: `resolve_model_path`; missing → `EngineError::Engine(format!("{stage} model not found: {}", path.display()))`; extension not `rten`/`onnx` (case-insensitive) → `EngineError::Engine(format!("{stage} model must be a .rten or .onnx file: {}", ...))`; else `EngineSession::from_path`. `OrientClassifier::new` uses it with stage `"orientation classifier"`.
- Doc comment on `Orientation`: the direction convention, stated after Step 6.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib -- orient::tests engine::tests`
Expected: PASS (new tests and the existing `test_rotate_box_points_back_*`, `test_probabilities_*`).

- [ ] **Step 5: Write the direction model test** (`tests/doc_models_test.rs`; copy the `fixture` helper from `tests/doc_fixtures_test.rs`)

```rust
fn doc_model(name: &str) -> PathBuf { PathBuf::from(std::env::var("RUSTO_DOC_MODELS").unwrap_or("models/DocOCR".into())).join(name) }
#[test]
#[ignore = "requires models/DocOCR (scripts/download_doc_models.sh)"]
fn orientation_corrects_each_known_rotation() {
    let upright = image::open(fixture("page_text.png")).unwrap();
    let mut clf = OrientClassifier::new(OrientConfig::rapid_orientation_v2(doc_model("orient_rapidai_v2.rten"))).unwrap();
    for cw in [0u16, 90, 180, 270] {
        let scan = match cw { 0 => upright.clone(), 90 => upright.rotate90(), 180 => upright.rotate180(), _ => upright.rotate270() };
        let out = clf.classify(&Mat::new(scan.clone())).unwrap();
        let p = decide_orientation(&out, 0.6);
        assert_eq!(p.applied_degrees, (360 - cw) % 360, "scan rotated {cw} deg clockwise, conf {}", out.confidence);
        let corrected = out.orientation.rotate_image(&Mat::new(scan)).unwrap();
        assert!(corrected.to_rgb8() == upright.to_rgb8(), "corrected page differs for {cw}");
    }
}
```

- [ ] **Step 6: Run the model test; fix the convention if it fails**

Run: `cargo test --test doc_models_test -- --ignored orientation_corrects_each_known_rotation`
Expected: PASS. If it fails only for 90 and 270 with the applied angle swapped, the model's class "90" means the content is rotated counter-clockwise: swap `Rotate90`/`Rotate270` in `rotate_image` (both backends), `rotate_box_points_back`, `correction_cw_degrees`, update the expected values of `test_rotate_box_points_back_90/_270`, and add a CHANGELOG "Fixed" bullet (this also corrects `RustO`'s orientation stage). Write the verified convention in the `Orientation` doc comment.

- [ ] **Step 7: Commit**

```bash
git add src/types.rs src/orient.rs src/engine.rs src/rusto_ocr.rs src/lib.rs tests/doc_models_test.rs CHANGELOG.md
git commit -m "feat(orient): RapidAI v2 page orientation pre-processing, verified direction and box rotation helpers"
```
CHANGELOG bullet: `- **orient**: OrientConfig::rapid_orientation_v2 (resize-short 256, centre crop 224, ImageNet normalisation), PageOrientation, rotate_box_back`.

---

### Task 4: Layout — PP-DocLayoutV3 labels, inputs, decode, nesting

**Files:**
- Modify: `src/layout.rs:13-74` (BBox, LayoutType, LayoutRegion), `:100-324` (detect, preprocess, postprocess)
- Modify: `src/types.rs:453-468` (`LayoutConfig::default`)
- Modify: `src/doc_pipeline.rs:93-147, 284-374` (add `_ =>` arms so the crate compiles; the routing is rewritten in Task 10)
- Modify: `src/lib.rs:73` (export `RegionRole, PP_DOCLAYOUT_V3_LABELS, drop_nested_text_regions`)
- Test: `src/layout.rs` (unit), `tests/doc_models_test.rs`

**Interfaces:**
- Consumes: `engine::load_stage_session` (Task 3).
- Produces:
  - `layout::PP_DOCLAYOUT_V3_LABELS: [&str; 25]` = `["abstract","algorithm","aside_text","chart","content","display_formula","doc_title","figure_title","footer","footer_image","footnote","formula_number","header","header_image","image","inline_formula","number","paragraph_title","reference","reference_content","seal","table","text","vertical_text","vision_footnote"]` (source comment: PaddleX `PP-DocLayoutV3/inference.yml` `label_list`, Apache-2.0).
  - `LayoutType`: `#[non_exhaustive]`, derives add `Serialize` with `#[serde(rename_all = "snake_case")]`. Existing variants kept; V3 labels reuse `Text`, `Table`, `Header`, `Footer`, `Reference`; new variants `Abstract, Algorithm, AsideText, Chart, Content, DisplayFormula, DocTitle, FigureTitle, FooterImage, Footnote, FormulaNumber, HeaderImage, Image, InlineFormula, Number, ParagraphTitle, ReferenceContent, Seal, VerticalText, VisionFootnote`. `name()` returns the V3 label for these.
  - `LayoutType::from_v3_class(class_id: usize) -> Option<LayoutType>` (index into the const).
  - `layout::RegionRole` = `DocTitle, SectionTitle, Paragraph, Footnote, VerticalText, Caption, Table, Visual, Formula, Algorithm, FurnitureText, FurnitureImage`; `LayoutType::role(&self) -> RegionRole`; `RegionRole::needs_ocr(&self) -> bool` (true for DocTitle, SectionTitle, Paragraph, Footnote, VerticalText, Caption, FurnitureText).
  - `BBox` derives `Copy, PartialEq, Eq, Serialize`; `BBox::area(&self) -> i64`; `BBox::intersection_area(&self, other: &BBox) -> i64`.
  - `LayoutRegion { bbox, layout_type, confidence, pub reading_order: f32 }`, derives `Serialize`.
  - `layout::layout_inputs(image: &Mat, target_size: i32) -> Result<HashMap<String, ArrayD<f32>>, EngineError>` (`pub(crate)`)
  - `layout::decode_layout_rows(rows: ArrayView2<f32>, conf_thresh: f32, img_w: i32, img_h: i32) -> Result<Vec<LayoutRegion>, EngineError>` (`pub(crate)`)
  - `layout::drop_nested_text_regions(regions: Vec<LayoutRegion>) -> Vec<LayoutRegion>` (`pub`)

Role map (spec §6.3): doc_title→DocTitle; paragraph_title→SectionTitle; text, content, abstract, aside_text, reference, reference_content→Paragraph; footnote, vision_footnote→Footnote; vertical_text→VerticalText; figure_title→Caption; table→Table; image, chart, seal→Visual; display_formula, inline_formula, formula_number→Formula; algorithm→Algorithm; header, footer, number→FurnitureText; header_image, footer_image→FurnitureImage. Legacy variants: Title→SectionTitle, Figure→Visual, FigureCaption/TableCaption→Caption, Equation→Formula.

- [ ] **Step 1: Write the failing unit tests** (reuse the `solid` helper from Task 3 in `layout.rs` tests)

```rust
#[test]
fn v3_label_map_is_complete() {
    for (i, label) in PP_DOCLAYOUT_V3_LABELS.iter().enumerate() {
        let t = LayoutType::from_v3_class(i).unwrap();
        assert_eq!(t.name(), *label);
        assert_eq!(serde_json::to_string(&t).unwrap(), format!("\"{label}\""));
    }
    assert!(LayoutType::from_v3_class(25).is_none());
    assert_eq!(LayoutType::from_v3_class(21).unwrap().role(), RegionRole::Table);
    assert_eq!(LayoutType::from_v3_class(16).unwrap().role(), RegionRole::FurnitureText);
}
fn row(c: f32, s: f32, b: [f32; 4], ro: f32) -> [f32; 7] { [c, s, b[0], b[1], b[2], b[3], ro] }
fn rows(r: &[[f32; 7]]) -> ndarray::Array2<f32> { ndarray::Array2::from_shape_vec((r.len(), 7), r.concat()).unwrap() }
#[test]
fn decode_keeps_threshold_orders_by_column_seven_and_does_not_rescale() {
    let a = rows(&[row(22.0, 0.95, [232., 440., 2297., 692.], 186.0), row(17.0, 0.50, [233., 1197., 822., 1256.], 20.0),
                   row(22.0, 0.49, [0., 0., 10., 10.], 1.0), row(30.0, 0.99, [0., 0., 10., 10.], 2.0)]);
    let r = decode_layout_rows(a.view(), 0.5, 2480, 3508).unwrap();
    assert_eq!(r.len(), 2);
    assert_eq!(r[0].layout_type, LayoutType::ParagraphTitle);
    assert_eq!((r[1].bbox.x_min, r[1].bbox.x_max), (232, 2297));
}
#[test]
fn decode_clamps_and_drops_degenerate() {
    let a = rows(&[row(22.0, 0.9, [-5., 10., 3000., 50.], 1.0), row(22.0, 0.9, [100., 100., 100., 200.], 2.0)]);
    let r = decode_layout_rows(a.view(), 0.5, 2480, 3508).unwrap();
    assert_eq!(r.len(), 1);
    assert_eq!((r[0].bbox.x_min, r[0].bbox.x_max), (0, 2480));
}
#[test]
fn decode_rejects_six_column_output() {
    let mut out = HashMap::new();
    out.insert("o".to_string(), ndarray::Array2::<f32>::zeros((3, 6)).into_dyn());
    let msg = find_layout_rows(&out).err().unwrap().to_string();
    assert!(msg.contains("[N,7]"), "{msg}");
}
#[test]
fn layout_inputs_tiny_image() {
    let m = layout_inputs(&solid(10, 20, 255), 800).unwrap();
    assert_eq!(m["image"].shape(), &[1, 3, 800, 800]);
    assert_eq!(m["im_shape"].iter().copied().collect::<Vec<_>>(), vec![800.0, 800.0]);
    assert_eq!(m["scale_factor"].iter().copied().collect::<Vec<_>>(), vec![40.0, 80.0]);
    assert!((m["image"][[0, 1, 5, 5]] - 1.0).abs() < 1e-6);
}
#[test]
fn nested_text_inside_table_is_dropped_overlapping_text_kept() {
    let reg = |t, b: [i32; 4]| LayoutRegion { bbox: BBox { x_min: b[0], y_min: b[1], x_max: b[2], y_max: b[3] }, layout_type: t, confidence: 0.9, reading_order: 0.0 };
    let out = drop_nested_text_regions(vec![
        reg(LayoutType::Table, [0, 0, 100, 100]),
        reg(LayoutType::Text, [10, 10, 90, 50]),       // 100 % inside: dropped
        reg(LayoutType::Text, [80, 0, 180, 100]),      // 20 % inside: kept
        reg(LayoutType::Text, [90, 0, 190, 100]),      // overlaps the previous text: kept
        reg(LayoutType::Text, [0, 0, 100, 125]),       // exactly 80 % inside: dropped
    ]);
    assert_eq!(out.len(), 3);
}
#[test]
fn layout_config_defaults_follow_inference_yml() {
    let c = LayoutConfig::default("m.rten".into());
    assert_eq!((c.target_size, c.conf_thresh, c.mean, c.std), (800, 0.5, [0.0; 3], [1.0; 3]));
}
```

`find_layout_rows(outputs: &HashMap<String, ArrayD<f32>>) -> Result<ArrayView2<'_, f32>, EngineError>` (`pub(crate)`) is part of this task's interface: it selects the rank-2 tensor with `shape[1] == 7`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib layout::tests`
Expected: FAIL to compile (`from_v3_class`, `decode_layout_rows` not found).

- [ ] **Step 3: Implement**

- `layout_inputs`: `resize_exact` to `target×target` with `INTER_CUBIC` (yml `interp: 2`; `image_impl` maps it to CatmullRom, as in the spike), RGB, `p/255` (mean 0, std 1), inputs `image [1,3,T,T]`, `im_shape [[T,T]]`, `scale_factor [[T/h, T/w]]`. Under `use-opencv`: `EngineError::Engine("layout detection is not supported with the OpenCV backend")`.
- `find_layout_rows`: error text `"layout model output must be [N,7] (PP-DocLayoutV3): no such tensor among outputs"`.
- `decode_layout_rows`: keep `score >= conf_thresh`; class via `from_v3_class` (unknown → skip); boxes are already page pixels (no rescale); clamp to `[0, img_w] × [0, img_h]`; drop `x_max <= x_min || y_max <= y_min`; `reading_order = row[6]`; stable sort by `reading_order`, ties by `(y_min, x_min)`. No NMS. Delete the `geometry::nms` call and the old class map (`layout.rs:266-297`).
- `drop_nested_text_regions`: a region whose `role().needs_ocr()` is dropped when `intersection_area / area >= 0.8` with any region labelled `Table`, `Image`, `Chart` or `Seal`.
- `LayoutDetector::new` uses `load_stage_session("layout", ...)`; `detect` = `layout_inputs` → `run_with_inputs` → `find_layout_rows` → `decode_layout_rows(…, self.config.conf_thresh, w, h)`.
- `LayoutConfig::default`: `mean [0.0; 3]`, `std [1.0; 3]`, `conf_thresh 0.5`, `target_size 800`; `iou_thresh` stays and is documented as unused (DETR output, no NMS).

Plan decision (spec §7): the 7th column is the model's reading order — the spike printed it for p79 (`ord186` text, `ord35` header) and `ppocr-rs` `layout.rs:1-20` documents `[class, score, xmin, ymin, xmax, ymax, read_order]`. Order by it; the XY-cut from `ppocr-rs` is not translated, so no `ppocr-rs` attribution is needed.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib layout::tests && cargo build --examples`
Expected: PASS; examples build (`examples/test_layout_simple.rs` only reads fields).

- [ ] **Step 5: Write the layout model test** (`tests/doc_models_test.rs`)

```rust
#[test]
#[ignore = "requires models/DocOCR (scripts/download_doc_models.sh)"]
fn layout_finds_title_paragraphs_table_in_order() {
    let truth: serde_json::Value = serde_json::from_slice(&std::fs::read(fixture("page_text.json")).unwrap()).unwrap();
    let mut det = LayoutDetector::new(LayoutConfig::default(doc_model("layout_ppdoclayoutv3.rten"))).unwrap();
    let regions = det.detect(&rusto::image_impl::imread(fixture("page_text.png")).unwrap()).unwrap().regions;
    let body: Vec<_> = regions.iter().filter(|r| !matches!(r.layout_type.role(), RegionRole::FurnitureText | RegionRole::FurnitureImage)).collect();
    assert!(matches!(body[0].layout_type, LayoutType::DocTitle | LayoutType::ParagraphTitle), "{:?}", body[0].layout_type);
    let t = body.iter().position(|r| r.layout_type == LayoutType::Table).expect("table region");
    assert!(body[1..t].iter().filter(|r| r.layout_type == LayoutType::Text).count() >= 2);
    assert!(iou(&body[t].bbox, &truth["table"]) >= 0.8);
}
```
(`iou` is a local helper in the test file over `[x0,y0,x1,y1]`.)

- [ ] **Step 6: Run the model test**

Run: `cargo test --test doc_models_test -- --ignored layout_finds_title_paragraphs_table_in_order`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/layout.rs src/types.rs src/doc_pipeline.rs src/lib.rs tests/doc_models_test.rs CHANGELOG.md
git commit -m "feat(layout): PP-DocLayoutV3 25-label decode with model reading order and nested-box drop"
```
CHANGELOG bullets: Added `**layout**: PP-DocLayoutV3 support (25 labels, [N,7] output, reading order)`; Changed `**layout**: LayoutConfig defaults 800 px, mean 0/std 1, conf_thresh 0.5 (was 0.4, unused); LayoutType is #[non_exhaustive]`.

---

### Task 5: Document result types and `DocPipelineConfig` API

**Files:**
- Modify: `src/doc_pipeline.rs:16-39` (config, result types), `:41-151` (`new`, `run` fill new fields)
- Modify: `src/table.rs:12-44` (derive `Serialize`, `Copy, PartialEq` on `table::BBox`)
- Modify: `examples/doc_pipeline_demo.rs:64-69` (use the builder)
- Modify: `src/lib.rs:72` (export `DocRegion, Stage, StageState, StageStatus`)
- Test: `src/doc_pipeline.rs` (unit)

**Interfaces:**
- Consumes: `PageOrientation` (Task 3), `LayoutType`, `layout::BBox` (Task 4).
- Produces:
  - `DocRegion { pub index: usize, pub label: LayoutType, pub bbox: layout::BBox, pub score: f32, pub text: Option<String>, pub lines: Vec<TextResult>, pub table: Option<TableStructure> }` (`Debug, Clone, Serialize`). `index` = position in reading order, used as `#region-{index}`.
  - `Stage { Orientation, Layout, Ocr, Table }` (`Debug, Clone, Copy, PartialEq, Eq, Serialize`, `#[serde(rename_all = "snake_case")]`); `StageState` and `StageStatus` derive `Debug, Clone, PartialEq, Serialize`
  - `StageState { Ran { elapsed_ms: f64, items: usize }, Skipped { reason: String }, Failed { reason: String } }` (`#[serde(tag = "state", rename_all = "snake_case")]`)
  - `StageStatus { pub stage: Stage, #[serde(flatten)] pub state: StageState }`
  - `DocResult { pub width: u32, pub height: u32, pub orientation: Option<PageOrientation>, pub regions: Vec<DocRegion>, pub stages: Vec<StageStatus>, #[serde(skip)] pub blocks: Vec<DocBlock> }` (`Debug, Clone, Serialize`); `width`/`height` are the corrected page size.
  - `DocPipelineConfig`: `#[non_exhaustive]`, fields as today plus `pub orient: Option<OrientConfig>`, `pub orient_min_confidence: f32`.
  - `DocPipelineConfig::new(layout: LayoutConfig, ocr: InitializeConfig) -> Self` (orient `None`, `orient_min_confidence 0.6`, table configs `None`), `with_orientation(self, OrientConfig) -> Self`, `with_orient_min_confidence(self, f32) -> Self`, `with_table_recognizer(self, TableStructureConfig) -> Self`, `with_table_detector(self, TableDetectorConfig) -> Self`.

Plan decisions: `new` takes `ocr` too (spec writes `new(layout)`, but `InitializeConfig` has no default and the field is required). `orientation` is `None` when the stage is skipped; `detected_degrees` is added so a below-threshold result keeps the detected class. `StageState` shape as above (spec §7). `DocResult` is not `#[non_exhaustive]`, so callers can build one for rendering.

- [ ] **Step 1: Write the failing unit tests**

```rust
fn cfg() -> DocPipelineConfig { DocPipelineConfig::new(LayoutConfig::default("l.rten".into()), InitializeConfig::ppv5("d", "r", "k")) }
#[test]
fn config_new_has_spec_defaults() {
    let c = cfg();
    assert!(c.orient.is_none() && c.table_recognizer.is_none() && c.table_detector.is_none());
    assert_eq!(c.orient_min_confidence, 0.6);
    let c = c.with_orient_min_confidence(0.7).with_orientation(OrientConfig::rapid_orientation_v2("o.rten".into()));
    assert_eq!(c.orient_min_confidence, 0.7);
    assert!(c.orient.is_some());
}
#[test]
fn config_rejects_out_of_range_orient_confidence() {
    for v in [-0.1, 1.1, f32::NAN] { assert!(validate_config(&cfg().with_orient_min_confidence(v)).is_err()); }
    assert!(validate_config(&cfg().with_orient_min_confidence(1.0)).is_ok());
}
#[test]
fn doc_result_json_shape() {
    let r = DocResult { width: 10, height: 20, orientation: Some(PageOrientation { applied_degrees: 270, detected_degrees: 90, confidence: 0.92 }),
        regions: vec![DocRegion { index: 0, label: LayoutType::ParagraphTitle, bbox: BBox { x_min: 1, y_min: 2, x_max: 3, y_max: 4 }, score: 0.9, text: Some("Раздел".into()), lines: vec![], table: None }],
        stages: vec![StageStatus { stage: Stage::Table, state: StageState::Skipped { reason: "table model not configured".into() } }], blocks: vec![] };
    let v: serde_json::Value = serde_json::to_value(&r).unwrap();
    assert_eq!(v["orientation"]["applied_degrees"], 270);
    assert_eq!(v["regions"][0]["label"], "paragraph_title");
    assert_eq!(v["stages"][0]["stage"], "table");
    assert_eq!(v["stages"][0]["state"], "skipped");
    assert!(v.get("blocks").is_none());
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib doc_pipeline::tests`
Expected: FAIL to compile (`DocPipelineConfig::new`, `DocRegion` not found).

- [ ] **Step 3: Implement**

Types and builder as in Interfaces. `fn validate_config(cfg: &DocPipelineConfig) -> Result<(), EngineError>`: `orient_min_confidence` finite and in `0.0..=1.0`, else `EngineError::Preprocess("orient_min_confidence must be within 0..=1")`; called first in `DocPipeline::new`. Interim `run` (replaced in Task 10): fill `regions` from the current blocks, `orientation: None`, `stages` = Orientation Skipped `"orientation model not configured"`, Layout Ran, Ocr Ran, Table Ran/Skipped. Update `examples/doc_pipeline_demo.rs` to `DocPipelineConfig::new(..).with_table_detector(..).with_table_recognizer(..)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib doc_pipeline::tests && cargo build --examples`
Expected: PASS; examples build.

- [ ] **Step 5: Commit**

```bash
git add src/doc_pipeline.rs src/table.rs src/lib.rs examples/doc_pipeline_demo.rs CHANGELOG.md
git commit -m "feat(doc): DocRegion, StageStatus, PageOrientation in DocResult; non-exhaustive DocPipelineConfig with builder"
```
CHANGELOG: Changed `**doc**: DocPipelineConfig is #[non_exhaustive]; use DocPipelineConfig::new(layout, ocr) and with_* builders`.

---

### Task 6: Table recogniser — config, pre-processing, outputs, dictionary

**Files:**
- Modify: `src/table.rs:127-131` (`TableModelType`), `:280-294` (`TableStructureConfig`), `:296-410` (recogniser struct, `new`, `preprocess`)
- Modify: `src/engine.rs:101-178` (`EngineSession::metadata_value`), `:292-436` (`RtenSession::metadata_value`)
- Modify: `examples/doc_pipeline_demo.rs:55-58` (use `TableStructureConfig::slanet_plus`)
- Modify: `src/lib.rs:77` (export `SLANET_PLUS_DICT`)
- Test: `src/table.rs` (unit)

**Interfaces:**
- Consumes: `load_stage_session` (Task 3).
- Produces:
  - `TableModelType::SlanetPlus`; `TableModelType` becomes `#[non_exhaustive]`.
  - `TableStructureConfig` `#[non_exhaustive]` with `pub input_size: u32`, `pub max_tokens: usize`; `TableStructureConfig::slanet_plus(model_path: impl Into<PathBuf>) -> Self` (`model_type SlanetPlus`, `input_size 488`, `max_tokens 500`); `Default` = `slanet_plus("models/DocOCR/table_slanetplus.rten")`.
  - `table::SLANET_PLUS_DICT: [&str; 48]` = `<thead>`, `</thead>`, `<tbody>`, `</tbody>`, `<tr>`, `</tr>`, `<td`, `>`, `</td>`, ` colspan="2"` … ` colspan="20"`, ` rowspan="2"` … ` rowspan="20"`, `<td></td>` (exact order of `peo/eval/spikes/rten-models/models/table-rapidai/character_from_onnx_metadata.txt`, SHA-256 `6df9c4ae…`).
  - `table::structure_charset(metadata: Option<&str>) -> Vec<String>` (`pub(crate)`): split on `'\n'`, else the const; drop `"<td>"`, append `"<td></td>"` if absent, wrap with `"sos"` first and `"eos"` last (translated from `bobine` `prepare_charset`, `rapid_table.rs:33-43`).
  - `table::table_input(crop: &Mat, input_size: u32) -> Result<(ArrayD<f32>, f32), EngineError>` (`pub(crate)`): returns tensor and `ratio = input_size / max(w, h)`.
  - `table::split_table_outputs(outputs: HashMap<String, ArrayD<f32>>, num_classes: usize) -> Result<(Array2<f32>, Array2<f32>), EngineError>` (`pub(crate)`): `(structure [T, num_classes], loc [T, 8])` of batch 0.
  - `EngineSession::metadata_value(&self, key: &str) -> Option<String>` (Rten: `model.metadata().get(key)`; Custom: `None`).

Plan decisions: the dictionary is read through a new `metadata_value`, not through `get_character_list`, because `rec.rs:65-70` prefers `get_character_list("character")` over the dict file and changing it would alter the recogniser (spec non-goal). `.rten` files lose the custom metadata (spike log shows only `OnnxHash`), so the const is the normal path for `.rten`. `max_tokens 500` = `max_text_length: 500` in the official SLANet_plus `inference.yml`. Outputs are identified by last dimension (8 = location, `num_classes` = structure), because the RapidAI and Paddle exports name them differently (`save_infer_model/scale_*` vs `fetch_name_*`). Location heads with 4 values are rejected (the spike measured 8). Channel order RGB, as in the spike that reproduced the 5×6 skeleton; `TableStructureRecognizer::new` checks `model_type` before loading the model and rejects `Wired`/`Wireless` (SLANeXt) with `EngineError::Engine("TableStructureRecognizer supports TableModelType::SlanetPlus only")`.

- [ ] **Step 1: Write the failing unit tests**

```rust
#[test]
fn slanet_plus_config_defaults() {
    let c = TableStructureConfig::default();
    assert_eq!(c.model_path, PathBuf::from("models/DocOCR/table_slanetplus.rten"));
    assert!(matches!(c.model_type, TableModelType::SlanetPlus));
    assert_eq!((c.input_size, c.max_tokens), (488, 500));
}
#[test]
fn charset_from_constant_and_metadata() {
    let c = structure_charset(None);
    assert_eq!(c.len(), 50);
    assert_eq!((c[0].as_str(), c[49].as_str(), c[48].as_str()), ("sos", "eos", "<td></td>"));
    let m = structure_charset(Some("<tr>\n<td>\n</td>"));
    assert_eq!(m, vec!["sos", "<tr>", "</td>", "<td></td>", "eos"]);
}
#[test]
fn table_input_keeps_ratio_and_pads_bottom_right() {
    let (t, ratio) = table_input(&solid(976, 488, 255), 488).unwrap();
    assert_eq!(t.shape(), &[1, 3, 488, 488]);
    assert!((ratio - 0.5).abs() < 1e-6);
    assert!((t[[0, 0, 0, 0]] - (1.0 - 0.485) / 0.229).abs() < 1e-3);
    assert_eq!(t[[0, 0, 300, 10]], 0.0); // below the 244-px resized image
}
#[test]
fn table_input_tiny_crop() {
    let (t, ratio) = table_input(&solid(3, 2, 0), 488).unwrap();
    assert_eq!(t.shape(), &[1, 3, 488, 488]);
    assert!((ratio - 488.0 / 3.0).abs() < 1e-3);
}
#[test]
fn split_outputs_by_last_dimension() {
    let mut o = HashMap::new();
    o.insert("fetch_name_0".into(), ndarray::ArrayD::<f32>::zeros(vec![1, 42, 8]));
    o.insert("fetch_name_1".into(), ndarray::ArrayD::<f32>::zeros(vec![1, 42, 50]));
    let (s, l) = split_table_outputs(o, 50).unwrap();
    assert_eq!((s.dim(), l.dim()), ((42, 50), (42, 8)));
}
#[test]
fn split_outputs_rejects_class_count_mismatch() {
    let mut o = HashMap::new();
    o.insert("a".into(), ndarray::ArrayD::<f32>::zeros(vec![1, 5, 8]));
    o.insert("b".into(), ndarray::ArrayD::<f32>::zeros(vec![1, 5, 30]));
    assert!(split_table_outputs(o, 50).err().unwrap().to_string().contains("50"));
}
#[test]
fn recognizer_rejects_slanext_model_type() {
    let mut c = TableStructureConfig::default();
    c.model_type = TableModelType::Wired;
    assert!(TableStructureRecognizer::new(c).err().unwrap().to_string().contains("SlanetPlus"));
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib table::tests`
Expected: FAIL to compile.

- [ ] **Step 3: Implement**

`table_input`: `new_w = round(w·ratio).clamp(1, 488)`, same for `h`, `INTER_LINEAR`, RGB, `(p/255 − mean)/std` with ImageNet mean/std, zeros elsewhere (padding after normalisation; translated from `bobine` `preprocess`, `rapid_table.rs:419-440`). Under `use-opencv`: `EngineError::Engine("table structure recognition is not supported with the OpenCV backend")`. Recogniser fields become `session`, `config`, `charset: Vec<String>` (from `structure_charset(session.metadata_value("character").as_deref())`); `new` uses `load_stage_session("table", ...)`; input name `"x"`. Leave `recognize`/`postprocess` returning the placeholder until Task 8. File header: attribution block for `bobine` 0.7.3 `src/rapid_table.rs` (`https://github.com/opticsWolf/bobine`), used under its MIT option, with the full MIT notice `Copyright (c) 2026 opticsWolf`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib table::tests && cargo build --examples`
Expected: PASS (including the existing `test_html_generation`).

- [ ] **Step 5: Commit**

```bash
git add src/table.rs src/engine.rs examples/doc_pipeline_demo.rs CHANGELOG.md
git commit -m "feat(table): SLANet_plus config, keep-ratio 488 pre-processing, input x, structure dictionary"
```

---

### Task 7: SLANet_plus token decoder (pure)

**Files:**
- Create: `src/table_decode.rs` (private module; `mod table_decode;` in `src/lib.rs`)
- Modify: `src/orient.rs:273` (`probabilities` becomes `pub(crate)`)
- Test: `src/table_decode.rs` (unit)

**Interfaces:**
- Produces (all `pub(crate)`):
  - `StepToken { pub step: usize, pub token: String, pub prob: f32 }`
  - `argmax_tokens(structure: ArrayView2<f32>, charset: &[String], max_tokens: usize) -> Vec<StepToken>`
  - `GridCell { pub row: usize, pub col: usize, pub row_span: usize, pub col_span: usize, pub token_index: usize }` (`Debug, Clone, PartialEq, Eq`)
  - `DecodedGrid { pub cells: Vec<GridCell>, pub num_rows: usize, pub num_cols: usize, pub malformed: bool }` (`Debug, Clone, PartialEq, Eq, Default`)
  - `decode_structure_tokens(tokens: &[&str]) -> DecodedGrid`

`argmax_tokens` (translated from `bobine` `recognize`, `rapid_table.rs:308-344`): for `t` in `0..min(T, max_tokens)`: argmax over the row; `idx == eos` (last charset index) and `t > 0` → stop; `idx == sos` (0) or `eos` → skip; `prob` = the argmax value after `orient::probabilities(row)` (passes through rows that already sum to 1).

`decode_structure_tokens` algorithm (new code; `bobine` builds HTML only):

```text
state: row_open = false, rows = 0, cursor = 0, occupied: Vec<Vec<bool>>, pending: Option<GridCell>
for (i, tok) in tokens:
  "<thead>" | "</thead>" | "<tbody>" | "</tbody>": ignore
  "<tr>":   if row_open { malformed }  ; finish(pending) ; rows += 1 ; row_open = true ; cursor = 0
  "</tr>":  finish(pending) ; if !row_open { malformed } ; row_open = false
  "<td></td>": finish(pending) ; ensure_row() ; place(1, 1, i)
  "<td":    finish(pending) ; ensure_row() ; pending = cell(1, 1, i)
  " colspan=\"N\"" / " rowspan=\"N\"": if pending, set span = N (parse; failure → malformed) else malformed
  ">":      if pending is None { malformed }
  "</td>":  if pending { place(pending) } else { malformed }
  other:    malformed
after loop: if pending { malformed; place(pending) }
ensure_row(): if !row_open { malformed; rows += 1; row_open = true; cursor = 0 }
finish(p):    if p is Some { malformed; place(p) }
place(rs, cs, i): row = rows - 1; col = first c >= cursor with occupied[row][c] == false;
                  mark occupied[row + r][col + k] for r < rs, k < cs (grow vectors); cursor = col + cs
final: num_rows = rows; clamp each row_span to num_rows - row; num_cols = max(col + col_span) or 0
```

- [ ] **Step 1: Write the failing unit tests**

```rust
fn cells(g: &DecodedGrid) -> Vec<(usize, usize, usize, usize)> { g.cells.iter().map(|c| (c.row, c.col, c.row_span, c.col_span)).collect() }
const TD: &str = "<td></td>";
#[test]
fn decode_simple_two_by_two() {
    let g = decode_structure_tokens(&["<tr>", TD, TD, "</tr>", "<tr>", TD, TD, "</tr>"]);
    assert_eq!(cells(&g), vec![(0, 0, 1, 1), (0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 1, 1)]);
    assert_eq!((g.num_rows, g.num_cols, g.malformed), (2, 2, false));
}
#[test]
fn decode_fixture_3x4_with_merged_header() {
    let mut t = vec!["<thead>", "<tr>", "<td", " colspan=\"2\"", ">", "</td>", TD, TD, "</tr>", "</thead>", "<tbody>"];
    for _ in 0..2 { t.extend(["<tr>", TD, TD, TD, TD, "</tr>"]); }
    t.push("</tbody>");
    let g = decode_structure_tokens(&t);
    assert_eq!((g.num_rows, g.num_cols, g.cells.len(), g.malformed), (3, 4, 11, false));
    assert_eq!(cells(&g)[0], (0, 0, 1, 2));
    assert_eq!(cells(&g)[1], (0, 2, 1, 1));
    assert_eq!(g.cells[0].token_index, 2);
}
#[test]
fn decode_rowspan_pushes_next_row() {
    let g = decode_structure_tokens(&["<tr>", "<td", " rowspan=\"2\"", ">", "</td>", TD, "</tr>", "<tr>", TD, "</tr>"]);
    assert_eq!(cells(&g), vec![(0, 0, 2, 1), (0, 1, 1, 1), (1, 1, 1, 1)]);
}
#[test]
fn decode_rowspan_beyond_last_row_is_clamped() {
    let g = decode_structure_tokens(&["<tr>", "<td", " rowspan=\"5\"", ">", "</td>", "</tr>"]);
    assert_eq!(cells(&g), vec![(0, 0, 1, 1)]);
}
#[test]
fn decode_malformed_sequence_does_not_panic() {
    let g = decode_structure_tokens(&[TD, "</td>", "<tr>", "<td", "<b>", TD, "<tr>", "<td", " colspan=\"x\""]);
    assert!(g.malformed);
    assert_eq!(cells(&g), vec![(0, 0, 1, 1), (1, 0, 1, 1), (1, 1, 1, 1), (2, 0, 1, 1)]);
    assert_eq!((g.num_rows, g.num_cols), (3, 2));
}
#[test]
fn decode_empty_sequence() {
    assert_eq!(decode_structure_tokens(&[]), DecodedGrid::default());
}
#[test]
fn argmax_skips_sos_and_stops_at_eos() {
    let cs: Vec<String> = ["sos", "<tr>", TD, "</tr>", "eos"].iter().map(|s| s.to_string()).collect();
    let onehot = |i: usize| { let mut r = [0.0f32; 5]; r[i] = 1.0; r };
    let rows: Vec<f32> = [4, 0, 1, 2, 3, 4, 1].iter().flat_map(|&i| onehot(i)).collect();
    let a = ndarray::Array2::from_shape_vec((7, 5), rows).unwrap();
    let t = argmax_tokens(a.view(), &cs, 500);
    assert_eq!(t.iter().map(|s| s.token.as_str()).collect::<Vec<_>>(), vec!["<tr>", TD, "</tr>"]);
    assert_eq!(t.iter().map(|s| s.step).collect::<Vec<_>>(), vec![2, 3, 4]);
    assert_eq!(argmax_tokens(a.view(), &cs, 3).len(), 1);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib table_decode::tests`
Expected: FAIL to compile (module/functions missing).

- [ ] **Step 3: Implement** `argmax_tokens` and `decode_structure_tokens` per the outline; module header carries the `bobine` attribution for `argmax_tokens`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib table_decode::tests`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/table_decode.rs src/lib.rs src/orient.rs
git commit -m "feat(table): decode SLANet_plus structure tokens into a span-aware grid"
```

---

### Task 8: Cell boxes, line-to-cell assignment, recogniser end to end

**Files:**
- Modify: `src/table_decode.rs` (add functions)
- Modify: `src/table.rs:21-115` (`TableCell.score`, `TableStructure.unassigned_lines`, `has_spans`, `to_markdown`), `:317-432` (`recognize`, `recognize_at`, remove placeholder)
- Modify: `src/doc_pipeline.rs:126-210` (table condition and `process_table`)
- Test: `src/table_decode.rs`, `src/table.rs`, `tests/doc_models_test.rs`

**Interfaces:**
- Consumes: Task 6 (`table_input`, `split_table_outputs`, `charset`), Task 7 (`argmax_tokens`, `decode_structure_tokens`).
- Produces:
  - `TableCell { …, pub score: f32 }`; `bbox` is absolute page pixels (`table::BBox`).
  - `TableStructure { rows, num_rows, num_cols, pub unassigned_lines: Vec<TextResult> }`; `TableStructure::has_spans(&self) -> bool`.
  - `table_decode::quad_to_bbox(quad: &[f32], crop_w: u32, crop_h: u32, origin: (i32, i32)) -> Option<table::BBox>`
  - `table_decode::build_structure(grid: &DecodedGrid, tokens: &[StepToken], loc: ArrayView2<f32>, crop_w: u32, crop_h: u32, origin: (i32, i32)) -> TableStructure`
  - `table_decode::assign_lines_to_cells(structure: &mut TableStructure, lines: Vec<TextResult>)`
  - `TableStructureRecognizer::recognize_at(&mut self, crop: &Mat, origin_x: i32, origin_y: i32) -> Result<TableStructure, EngineError>`; `recognize(&mut self, img: &Mat)` = `recognize_at(img, 0, 0)`.
  - `doc_pipeline::lines_from_ocr(boxes: &[[Point2f; 4]], txts: &[String], scores: &[f32], offset: (i32, i32)) -> Vec<TextResult>` (`pub(crate)`)

Rules pinned here:
- `quad_to_bbox`: quad values are normalised to the 488 canvas, so crop pixels = `q · max(crop_w, crop_h)` (equivalent to `bobine` `rapid_table.rs:346-366`); x clamped to `[0, crop_w]`, y to `[0, crop_h]`; xyxy = min/max of the 4 points, converted with `as i32` (truncation); add `origin`. All 8 values `< 1e-6` → `None`.
- `build_structure`: cell box = `quad_to_bbox(loc[tokens[cell.token_index].step])`; step beyond `loc` rows or `None` → degenerate box at the origin (`x_min = x_max = origin.0`, `y_min = y_max = origin.1`), which never receives lines. `score` = `tokens[cell.token_index].prob`. Rows grouped by `row`, cells by `col`.
- `assign_lines_to_cells` (spec §6.4): line centre = `(frame.left + frame.width / 2, frame.top + frame.height / 2)`. Candidate cells whose box contains the centre (inclusive) → the smallest-area one. Otherwise the cell with the highest IoU between the line frame and the cell box, if that IoU > 0; ties → first in row-major order. Otherwise → `unassigned_lines`. Lines of one cell are ordered: same band when vertical centres differ by ≤ half the smaller line height, bands top to bottom, left to right inside a band; texts trimmed and joined with one space into `cell.text`.
- `to_markdown` (GFM): exactly `num_cols` cells per row (missing positions empty), header separator `|---|` × `num_cols`, `|` in text escaped as `\|`, newlines as spaces. `to_html` unchanged.

- [ ] **Step 1: Write the failing unit tests**

```rust
fn line(text: &str, l: f32, t: f32, w: f32, h: f32) -> TextResult {
    TextResult { text: text.into(), score: 0.9, box_points: [(l, t), (l + w, t), (l + w, t + h), (l, t + h)], frame: Frame::new(w, h, t, l) }
}
fn cell(r: usize, c: usize, b: [i32; 4], cs: usize, text: &str) -> TableCell {
    TableCell { bbox: BBox { x_min: b[0], y_min: b[1], x_max: b[2], y_max: b[3] }, row: r, col: c, row_span: 1, col_span: cs, text: text.into(), score: 1.0 }
}
fn table_2x2() -> TableStructure {
    TableStructure { num_rows: 2, num_cols: 2, unassigned_lines: vec![], rows: vec![
        TableRow { cells: vec![cell(0, 0, [0, 0, 100, 50], 1, ""), cell(0, 1, [100, 0, 200, 50], 1, "")] },
        TableRow { cells: vec![cell(1, 0, [0, 50, 100, 100], 1, ""), cell(1, 1, [100, 50, 200, 100], 1, "")] }] }
}
#[test]
fn quad_scales_by_long_side_and_offsets() {
    let b = quad_to_bbox(&[0.1, 0.1, 0.5, 0.1, 0.5, 0.2, 0.1, 0.2], 976, 488, (300, 700)).unwrap();
    assert_eq!((b.x_min, b.y_min, b.x_max, b.y_max), (300 + 97, 700 + 97, 300 + 488, 700 + 195));
    assert!(quad_to_bbox(&[0.0; 8], 976, 488, (0, 0)).is_none());
}
#[test]
fn assign_by_centre_then_iou_then_unassigned() {
    let mut s = table_2x2();
    assign_lines_to_cells(&mut s, vec![
        line("Цена", 120.0, 10.0, 60.0, 20.0),     // centre in (0,1)
        line("12", 90.0, 60.0, 30.0, 20.0),        // centre (105,70) in (1,1)
        line("край", 195.0, 95.0, 30.0, 20.0),     // centre (210,105) outside; IoU > 0 only with (1,1)
        line("извън", 400.0, 400.0, 30.0, 20.0),   // no overlap
    ]);
    assert_eq!(s.rows[0].cells[1].text, "Цена");
    assert_eq!(s.rows[1].cells[1].text, "12 край");
    assert_eq!(s.unassigned_lines.iter().map(|l| l.text.as_str()).collect::<Vec<_>>(), vec!["извън"]);
}
#[test]
fn lines_in_one_cell_join_in_reading_order() {
    let mut s = table_2x2();
    assign_lines_to_cells(&mut s, vec![line("б", 50.0, 5.0, 20.0, 10.0), line("в", 5.0, 30.0, 20.0, 10.0), line("а", 5.0, 6.0, 20.0, 10.0)]);
    assert_eq!(s.rows[0].cells[0].text, "а б в");
}
#[test]
fn overlapping_cells_pick_smallest_containing() {
    let mut s = table_2x2();
    s.rows[0].cells.insert(0, cell(0, 0, [0, 0, 200, 100], 2, ""));
    assign_lines_to_cells(&mut s, vec![line("а", 10.0, 10.0, 20.0, 10.0)]);
    assert_eq!((s.rows[0].cells[0].text.as_str(), s.rows[0].cells[1].text.as_str()), ("", "а"));
}
#[test]
fn assign_with_zero_cells_leaves_all_unassigned() {
    let mut s = TableStructure { rows: vec![], num_rows: 0, num_cols: 0, unassigned_lines: vec![] };
    assign_lines_to_cells(&mut s, vec![line("а", 0.0, 0.0, 10.0, 10.0), line("б", 20.0, 0.0, 10.0, 10.0)]);
    assert_eq!(s.unassigned_lines.len(), 2);
}
#[test]
fn markdown_pads_rows_and_escapes_pipes() {
    let s = TableStructure { num_rows: 2, num_cols: 3, unassigned_lines: vec![], rows: vec![
        TableRow { cells: vec![cell(0, 0, [0; 4], 1, "а"), cell(0, 1, [0; 4], 1, "б"), cell(0, 2, [0; 4], 1, "в")] },
        TableRow { cells: vec![cell(1, 0, [0; 4], 1, "x | y"), cell(1, 1, [0; 4], 1, "z")] }] };
    assert_eq!(s.to_markdown(), "| а | б | в |\n|---|---|---|\n| x \\| y | z |  |\n");
    assert!(!s.has_spans());
}
#[test]
fn build_structure_degenerate_box_for_missing_quad() {
    let grid = decode_structure_tokens(&["<tr>", "<td></td>", "</tr>"]);
    let toks = vec![StepToken { step: 1, token: "<tr>".into(), prob: 0.9 }, StepToken { step: 2, token: "<td></td>".into(), prob: 0.8 },
                    StepToken { step: 3, token: "</tr>".into(), prob: 0.9 }];
    let s = build_structure(&grid, &toks, ndarray::Array2::<f32>::zeros((5, 8)).view(), 100, 50, (300, 700));
    let c = &s.rows[0].cells[0];
    assert_eq!((c.bbox.x_min, c.bbox.y_min, c.bbox.x_max, c.bbox.y_max, c.score), (300, 700, 300, 700, 0.8));
}
```

`to_markdown` row format: `"| " + cells.join(" | ") + " |"`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib -- table_decode::tests table::tests`
Expected: FAIL to compile (`quad_to_bbox`, `unassigned_lines` missing).

- [ ] **Step 3: Implement**

Functions per the rules. `recognize_at`: `table_input` → `run_with_inputs({"x"})` → `split_table_outputs(outputs, charset.len())` → `argmax_tokens(…, max_tokens)` → `decode_structure_tokens` → `build_structure(…, crop size, (origin_x, origin_y))`. In `doc_pipeline.rs`: the table branch runs when `table_recognizer.is_some()` (drop the "both must be set" condition at `:128`); `process_table` OCRs the table crop once with `OcrRunOptions { orientation: Some(false), ..Default::default() }`, converts lines with `lines_from_ocr(…, (x_min, y_min))`, calls `recognize_at(&crop, x_min, y_min)` and `assign_lines_to_cells`; the percent-of-crop code (`:170-206`) is deleted. Update the `TableCell` literals in `table.rs` tests (`score: 1.0`) and the `unassigned_lines` field.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib`
Expected: PASS.

- [ ] **Step 5: Write the table model test** (`tests/doc_models_test.rs`)

```rust
#[test]
#[ignore = "requires models/DocOCR (scripts/download_doc_models.sh)"]
fn table_fixture_has_3x4_grid_with_one_merged_cell() {
    let truth: serde_json::Value = serde_json::from_slice(&std::fs::read(fixture("table_3x4_merged.json")).unwrap()).unwrap();
    let crop = rusto::image_impl::imread(fixture("table_3x4_merged.png")).unwrap();
    let mut rec = TableStructureRecognizer::new(TableStructureConfig::slanet_plus(doc_model("table_slanetplus.rten"))).unwrap();
    let s = rec.recognize(&crop).unwrap();
    assert_eq!((s.num_rows, s.num_cols), (3, 4));
    let spans: Vec<_> = s.rows.iter().flat_map(|r| &r.cells).filter(|c| c.col_span > 1 || c.row_span > 1).collect();
    assert_eq!(spans.len(), 1);
    assert_eq!((spans[0].row, spans[0].col, spans[0].col_span), (0, 0, truth["merged"]["col_span"].as_u64().unwrap() as usize));
    let (w, h) = (truth["width"].as_i64().unwrap() as i32, truth["height"].as_i64().unwrap() as i32);
    assert!(s.rows.iter().flat_map(|r| &r.cells).all(|c| c.bbox.x_max <= w && c.bbox.y_max <= h && c.bbox.x_max > c.bbox.x_min));
}
```

- [ ] **Step 6: Run the model test**

Run: `cargo test --test doc_models_test -- --ignored table_fixture_has_3x4_grid_with_one_merged_cell`
Expected: PASS. Also run it with `RUSTO_DOC_MODELS` pointing to a directory that holds only `table_slanetplus.onnx`: PASS (metadata dictionary path).

- [ ] **Step 7: Commit**

```bash
git add src/table.rs src/table_decode.rs src/doc_pipeline.rs tests/doc_models_test.rs CHANGELOG.md
git commit -m "feat(table): SLANet_plus cell boxes in page pixels and OCR line assignment to cells"
```
CHANGELOG: Added `**table**: SLANet_plus structure with rowspan/colspan, cell boxes, cell text, unassigned_lines (decoder translated from bobine 0.7.3 rapid_table.rs, MIT)`; Changed `**table**: TableCell.bbox is absolute page pixels (percent-of-crop convention removed); TableCell.score added`.

---

### Task 9: Markdown rendering of `DocResult`

**Files:**
- Modify: `src/doc_pipeline.rs:273-391` (`impl DocResult`: replace `to_markdown`; remove `is_list_line` if unused)
- Test: `src/doc_pipeline.rs` (unit)

**Interfaces:**
- Consumes: `DocRegion`, `RegionRole` (Tasks 4–5), `TableStructure::has_spans`, `to_markdown`, `to_html` (Task 8).
- Produces: `DocResult::to_markdown(&self) -> String`.

Rendering rules (spec §6.3 table; exact strings are plan decisions). Walk `regions` in `index` order; each produces one block; blocks are joined with `"\n\n"`; non-empty output ends with exactly one `"\n"`; regions with an OCR role whose text is `None`, empty or whitespace produce no block (Visual, Formula and Algorithm regions always produce one; a Table region produces none only when it has no cells, no unassigned lines and no text).
- DocTitle `# {text}`; SectionTitle `## {text}`; Paragraph and VerticalText `{text}`.
- Footnote: no block in place; collected as `[^{n}]: {text}` (n from 1 in reading order), emitted as one block after the body.
- Caption: `*{text}*`, inserted directly after the block of the nearest preceding region with label `Image`, `Chart` or `Table`; if there is none, in place.
- Table: `table` with cells → `to_html()` if `has_spans()`, else `to_markdown()` without its trailing newline; `table` with zero cells → the `unassigned_lines` texts joined with a space as a paragraph; `table` is `None` → `text` as a paragraph.
- Visual: `![{label}](#region-{index})`.
- Formula: ```` ```formula\nregion-{index} [{x_min},{y_min},{x_max},{y_max}]\n``` ````; Algorithm the same with `algorithm`.
- FurnitureText / FurnitureImage: no block in place; collected as list items `- {label}: {text}` or `- {label}: ![{label}](#region-{index})`; emitted last as `## Page furniture\n\n{items joined by "\n"}`.

- [ ] **Step 1: Write the failing unit tests** (helper `reg(index, LayoutType, text)` builds a `DocRegion` with bbox `[1,2,3,4]`)

```rust
#[test]
fn markdown_of_empty_result_is_empty() { assert_eq!(result(vec![]).to_markdown(), ""); }
#[test]
fn markdown_headings_and_paragraphs() {
    let r = result(vec![reg(0, LayoutType::DocTitle, "Заглавие"), reg(1, LayoutType::ParagraphTitle, "Раздел"), reg(2, LayoutType::Text, "Абзац.")]);
    assert_eq!(r.to_markdown(), "# Заглавие\n\n## Раздел\n\nАбзац.\n");
}
#[test]
fn caption_moves_after_nearest_preceding_visual() {
    let r = result(vec![reg(0, LayoutType::Image, ""), reg(1, LayoutType::Text, "т"), reg(2, LayoutType::FigureTitle, "Фиг. 1")]);
    assert_eq!(r.to_markdown(), "![image](#region-0)\n\n*Фиг. 1*\n\nт\n");
}
#[test]
fn caption_without_preceding_visual_stays_in_place() {
    let r = result(vec![reg(0, LayoutType::FigureTitle, "Фиг. 1"), reg(1, LayoutType::Image, "")]);
    assert_eq!(r.to_markdown(), "*Фиг. 1*\n\n![image](#region-1)\n");
}
#[test]
fn footnotes_and_page_furniture_go_last() {
    let r = result(vec![reg(0, LayoutType::Header, "Горен"), reg(1, LayoutType::Text, "Тяло"), reg(2, LayoutType::Footnote, "Бележка"), reg(3, LayoutType::Number, "3")]);
    assert_eq!(r.to_markdown(), "Тяло\n\n[^1]: Бележка\n\n## Page furniture\n\n- header: Горен\n- number: 3\n");
}
#[test]
fn formula_block_references_region() {
    assert_eq!(result(vec![reg(4, LayoutType::DisplayFormula, "")]).to_markdown(), "```formula\nregion-4 [1,2,3,4]\n```\n");
}
#[test]
fn table_with_span_renders_html_without_span_gfm() {
    let mut r = result(vec![reg(0, LayoutType::Table, "")]);
    r.regions[0].table = Some(span_table());          // helper: 2x2 grid, cell (0,0) col_span 2
    let md = r.to_markdown();
    assert!(md.starts_with("<table>") && md.contains("colspan=\"2\""), "{md}");
    r.regions[0].table = Some(plain_table());         // helper: 2x2 grid, no spans
    assert!(r.to_markdown().starts_with("| "));
}
#[test]
fn table_without_cells_renders_unassigned_text() {
    let mut r = result(vec![reg(0, LayoutType::Table, "")]);
    r.regions[0].table = Some(TableStructure { rows: vec![], num_rows: 0, num_cols: 0, unassigned_lines: vec![text_line("а"), text_line("б")] });
    assert_eq!(r.to_markdown(), "а б\n");
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib doc_pipeline::tests`
Expected: FAIL (old renderer iterates `blocks`).

- [ ] **Step 3: Implement** `to_markdown` per the rules (private helpers allowed).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib doc_pipeline::tests`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/doc_pipeline.rs
git commit -m "feat(doc): markdown rendering by layout role with captions, footnotes, tables and page furniture"
```

---

### Task 10: Pipeline integration, end-to-end test, library example

**Files:**
- Modify: `src/doc_pipeline.rs:23-28` (pipeline fields), `:41-151` (`new`, `run`), `:153-270` (`process_table`, `crop_image`)
- Create: `examples/document_layout.rs`
- Create: `tests/doc_pipeline_test.rs`
- Modify: `src/lib.rs:72` (export new types)

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `DocPipeline::run_with_image(&mut self, image: &Mat) -> Result<(DocResult, Mat), EngineError>` (returns the corrected page)
  - `DocPipeline::run<P: AsRef<Path>>(&mut self, image_path: P) -> Result<DocResult, EngineError>` (`imread` + `run_with_image`)
  - `fn mean_score(lines: &[TextResult]) -> f32` (private; 0.0 for empty)

Run order:
1. Orientation: if configured, `classify`, `decide_orientation(out, orient_min_confidence)`, rotate with `Orientation::rotate_image` when `applied_degrees != 0`; stage `Ran { items: 1 }`; else `orientation: None`, stage Skipped `"orientation model not configured"`.
2. Layout on the corrected page, then `drop_nested_text_regions`; stage `Ran { items: regions.len() }`; `index` = position after this step.
3. Per region by `role()`: OCR roles → crop, `run_on_mat_with_options(&crop, &OcrRunOptions { orientation: Some(false), ..Default::default() })`, `lines_from_ocr(…, (x_min, y_min))`, `text` = texts joined with a space. VerticalText → OCR the crop rotated 90° clockwise and 90° counter-clockwise, keep the result with the higher `mean_score` (tie → clockwise), map points back with `rotate_point_back(applied 90 or 270, rotated crop size)` and then add the region offset. Table → as Task 8; `text = None` when `table` is `Some`. Visual, Formula, Algorithm, FurnitureImage → no OCR.
4. Per-region errors do not abort the page: the region keeps `text: None` (or `table: None` with its OCR text), and the stage becomes `Failed { reason: "{n} of {m} regions failed: {first error}" }`. Stage Ocr `Ran { items: regions OCR'd }`; stage Table `Ran { items: tables }`, Skipped `"table model not configured"`, or Skipped `"no table regions"`.
5. `blocks` filled for compatibility (one `DocBlock` per region; `html` = `table.to_html()`).
`DocPipeline::new`: `validate_config`; under `use-opencv` return `EngineError::Engine("DocPipeline is not supported with the OpenCV backend")`; `TableDetector` is still constructed when configured and stays unused (spec §6.4).

Plan decision (spec §7): per-region OCR is the default and the only mode in this plan; the §8 gate may later switch strategy after the external evaluation.

- [ ] **Step 1: Write the failing tests** (`tests/doc_pipeline_test.rs`; both `#[ignore = "requires RUSTO_TEST_MODELS and models/DocOCR"]`)

```rust
fn pipeline() -> DocPipeline {
    let ocr_dir = PathBuf::from(std::env::var("RUSTO_TEST_MODELS").expect("RUSTO_TEST_MODELS"));
    let cfg = DocPipelineConfig::new(LayoutConfig::default(doc_model("layout_ppdoclayoutv3.rten")),
            InitializeConfig::ppv5(ocr_dir.join("det.onnx"), ocr_dir.join("rec.onnx"), ocr_dir.join("dict.txt")))
        .with_orientation(OrientConfig::rapid_orientation_v2(doc_model("orient_rapidai_v2.rten")))
        .with_table_recognizer(TableStructureConfig::slanet_plus(doc_model("table_slanetplus.rten")));
    DocPipeline::new(cfg).unwrap()
}
#[test] #[ignore = "requires RUSTO_TEST_MODELS and models/DocOCR"]
fn rotated_page_end_to_end() {
    let upright = image::open(fixture("page_text.png")).unwrap();
    let (r, corrected) = pipeline().run_with_image(&Mat::new(upright.rotate90())).unwrap();
    assert_eq!(r.orientation.unwrap().applied_degrees, 270);
    assert_eq!((r.width, r.height, corrected.cols(), corrected.rows()), (1240, 1754, 1240, 1754));
    let t = r.regions.iter().find_map(|g| g.table.as_ref()).expect("table");
    assert_eq!((t.num_rows, t.num_cols), (3, 4));
    let md = r.to_markdown();
    assert!(md.starts_with("# ") || md.starts_with("## "), "{md}");
    assert!(md.contains("<table>") && md.contains("340"), "{md}");
    assert!(r.stages.iter().all(|s| !matches!(s.state, StageState::Failed { .. })));
}
#[test] #[ignore = "requires RUSTO_TEST_MODELS and models/DocOCR"]
fn blank_page_yields_no_regions() {
    let blank = Mat::new(image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(1240, 1754, image::Rgb([255, 255, 255]))));
    let (r, _) = pipeline().run_with_image(&blank).unwrap();
    assert!(r.regions.is_empty());
    assert_eq!(r.to_markdown(), "");
    assert!(r.stages.iter().any(|s| s.stage == Stage::Layout && matches!(s.state, StageState::Ran { items: 0, .. })));
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `RUSTO_TEST_MODELS=<dir with det.onnx, rec.onnx, dict.txt> cargo test --test doc_pipeline_test -- --ignored`
Expected: FAIL to compile (`run_with_image` missing).

- [ ] **Step 3: Implement** the run order above.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib && RUSTO_TEST_MODELS=<dir with PP-OCRv5 det.onnx, rec.onnx, dict.txt> cargo test --test doc_pipeline_test -- --ignored`
Expected: PASS.

- [ ] **Step 5: Write `examples/document_layout.rs`**

Library use without the CLI: positional args `<image> <det> <rec> <dict> [doc-models-dir]`; builds `DocPipelineConfig::new(...).with_orientation(...).with_table_recognizer(...)`, calls `run_with_image`, prints `serde_json::to_string_pretty(&result)`, then `result.to_markdown()`, then one stderr line per `StageStatus`.

- [ ] **Step 6: Run the example and measure**

Run: `cargo build --release --example document_layout && /usr/bin/time -l target/release/examples/document_layout tests/fixtures/doc/page_text.png <det> <rec> <dict>`
Expected: JSON then Markdown on stdout; on an M-series Mac stderr shows orientation ≤ 10 ms, layout ≤ 800 ms, table ≤ 100 ms; `maximum resident set size` < 2 147 483 648. Record the numbers in the commit message body. (300 dpi timing is checked on a real page outside the repo, spec §8.)

- [ ] **Step 7: Commit**

```bash
git add src/doc_pipeline.rs src/lib.rs examples/document_layout.rs tests/doc_pipeline_test.rs CHANGELOG.md
git commit -m "feat(doc): DocPipeline owns page orientation and routes regions by layout role"
```
CHANGELOG: Added `**doc**: DocPipeline::run_with_image, corrected-page coordinates, per-region OCR, vertical text, stage status; examples/document_layout.rs`.

---

### Task 11: CLI flags and `document` output

**Files:**
- Modify: `src/main.rs:5-58` (flags, `OutputFormat`), `:60-119` (`main`)
- Test: `src/main.rs` (`#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes: `DocPipeline`, `DocPipelineConfig`, `OrientConfig::rapid_orientation_v2`, `TableStructureConfig::slanet_plus`, `image_impl::imwrite`.
- Produces: flags `--enable-layout`, `--layout-model <PATH>` (`requires = "enable_layout"`, default `models/DocOCR/layout_ppdoclayoutv3.rten`), `--enable-table` (`requires = "enable_layout"`), `--table-model <PATH>` (`requires = "enable_table"`, default `models/DocOCR/table_slanetplus.rten`), `--save-corrected <PATH>` (`requires = "enable_layout"`); `--orient-model` gets `requires = "enable_orient"` and default `models/DocOCR/orient_rapidai_v2.rten`; `OutputFormat::Document`.
  - `enum RunPlan { Ocr { orient: Option<PathBuf> }, Document { orient: Option<PathBuf>, layout: PathBuf, table: Option<PathBuf>, save_corrected: Option<PathBuf>, markdown: bool } }`
  - `fn plan_run(cli: &Cli) -> Result<RunPlan, String>`

Rules: `--format document` without `--enable-layout` → `Err("--format document requires --enable-layout")`; with `--enable-layout`, only `document` and `markdown` are accepted, else `Err("with --enable-layout use --format document or --format markdown")`. OCR mode with `--enable-orient`: `InitializeConfig.orient = Some(OrientConfig::rapid_orientation_v2(path))`, `global.use_orient = true`, run options `orientation: Some(true)` (boxes stay in original-scan coordinates, as `RustO` does today). The unwarp warning stays for `--enable-unwarp`/`--unwarp-model` only. Document mode: `DocPipelineConfig::new(LayoutConfig::default(layout), ocr)` + orientation/table if set; `run_with_image`; `--save-corrected` writes the returned `Mat` with `imwrite`; `document` prints `serde_json::to_string_pretty(&result)`, `markdown` prints `result.to_markdown()`. Plan decisions: the strict `requires` rules, the default model paths, and `--save-corrected` only in document mode.

- [ ] **Step 1: Write the failing tests**

```rust
fn parse(extra: &[&str]) -> Result<Cli, clap::Error> {
    let mut a = vec!["rusto", "--det-model", "d", "--rec-model", "r", "--dict", "k", "img.png"];
    a.extend_from_slice(extra);
    Cli::try_parse_from(a)
}
#[test]
fn existing_invocation_is_plain_ocr() { assert!(matches!(plan_run(&parse(&[]).unwrap()), Ok(RunPlan::Ocr { orient: None }))); }
#[test]
fn enable_orient_uses_default_model() {
    let Ok(RunPlan::Ocr { orient: Some(p) }) = plan_run(&parse(&["--enable-orient"]).unwrap()) else { panic!() };
    assert_eq!(p, PathBuf::from("models/DocOCR/orient_rapidai_v2.rten"));
}
#[test]
fn document_needs_layout() {
    assert_eq!(plan_run(&parse(&["--format", "document"]).unwrap()).err().unwrap(), "--format document requires --enable-layout");
}
#[test]
fn document_mode_collects_models() {
    let Ok(RunPlan::Document { table: Some(t), markdown: false, save_corrected: Some(s), .. }) =
        plan_run(&parse(&["--enable-layout", "--enable-table", "--format", "document", "--save-corrected", "c.png"]).unwrap()) else { panic!() };
    assert_eq!((t, s), (PathBuf::from("models/DocOCR/table_slanetplus.rten"), PathBuf::from("c.png")));
}
#[test]
fn dependent_flags_are_rejected_by_clap() {
    assert!(parse(&["--table-model", "t.rten"]).is_err());
    assert!(parse(&["--enable-table"]).is_err());
    assert!(parse(&["--save-corrected", "c.png"]).is_err());
    assert!(parse(&["--orient-model", "o.rten"]).is_err());
}
#[test]
fn layout_mode_rejects_tsv() { assert!(plan_run(&parse(&["--enable-layout", "--format", "tsv"]).unwrap()).is_err()); }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --bin rusto-rs`
Expected: FAIL to compile (`plan_run` missing).

- [ ] **Step 3: Implement** flags, `plan_run`, and `main` dispatch.

- [ ] **Step 4: Run tests, then the binary**

Run: `cargo test --bin rusto-rs && cargo run --release -- --det-model <det> --rec-model <rec> --dict <dict> --enable-orient --enable-layout --enable-table --format markdown tests/fixtures/doc/page_text.png`
Expected: tests PASS; Markdown with a heading and a table on stdout.

- [ ] **Step 5: Commit**

```bash
git add src/main.rs CHANGELOG.md
git commit -m "feat(cli): --enable-layout, --enable-table, --save-corrected, document output; functional --enable-orient"
```

---

### Task 12: Documentation, version 0.4.0, final verification

**Files:**
- Modify: `README.md` (section "Document layout, tables and orientation")
- Create: `docs/content/en/03.advance-guide/09.document-layout-and-tables.md` (front matter as in `05.classification-and-orientation.md`)
- Modify: `docs/content/en/02.how-to/06.use-cli.md` (new flags), `docs/content/en/03.advance-guide/05.classification-and-orientation.md` (RapidAI v2 page model, `rapid_orientation_v2`)
- Modify: `CHANGELOG.md` (`[Unreleased]` → `[0.4.0] - <date>`), version files via `scripts/bump_version.py`

**Interfaces:**
- Consumes: all public API above. Produces: none.

- [ ] **Step 1: Write the docs** — model download (`scripts/download_doc_models.sh --convert-rten`), library example (from `examples/document_layout.rs`), CLI example, label→Markdown table (spec §6.3), coordinate space and `rotate_box_back`, `applied_degrees` convention, OpenCV backend unsupported, licences and attributions (RapidAI, PaddlePaddle, `bobine` MIT notice). Plan decision: English pages only; `docs/content/id` is left to upstream translators.

- [ ] **Step 2: Bump the version**

Run: `python3 scripts/bump_version.py 0.4.0 --dry-run`, inspect the file list, then `python3 scripts/bump_version.py 0.4.0 --no-stage`.
Expected: `Cargo.toml` `version = "0.4.0"` and the library package files updated; `MODELS_VERSION` unchanged (`0.3.2`). Rename `## [Unreleased]` to `## [0.4.0] - <date>` if the script did not.

- [ ] **Step 3: Full verification**

Run, in order:
- `cargo test --workspace` → all pass (ignored tests reported as ignored).
- `cargo clippy --lib --bins --tests --examples 2>&1 | grep -E "src/(layout|table|table_decode|orient|doc_pipeline|engine|types|main)\.rs"` → no output.
- `cargo build --examples` → success.
- `RUSTO_TEST_MODELS=<dir> cargo test --test doc_models_test --test doc_pipeline_test -- --ignored` → all pass.
- `python3 scripts/bump_version.py --dry-run patch` → no consistency error (CI step in `.github/workflows/ci.yml`).
- `cargo check --no-default-features --features use-opencv` only if OpenCV is installed; otherwise state in the report that the OpenCV build is not verified locally (CI does not build it).

- [ ] **Step 4: Commit**

```bash
git add README.md docs/content/en CHANGELOG.md Cargo.toml Cargo.lock <files listed by bump_version.py>
git commit -m "docs(doc): document layout, tables and orientation; release 0.4.0"
```

---

## Plan decisions (summary)

1. `OrientPreprocess` enum on `OrientConfig` (default `Stretch` = unchanged CLS/legacy behaviour) plus `OrientConfig::rapid_orientation_v2`; the shared `OrientClassifier` also serves the CLS model, so the fix cannot replace the old pre-processing.
2. `applied_degrees` is the clockwise rotation applied to the scan; `PageOrientation` adds `detected_degrees`; `DocResult.orientation` is `None` when the stage is skipped.
3. Reading order from the layout output's 7th column; no XY-cut port and therefore no `ppocr-rs` code or attribution.
4. Layout resize uses `INTER_CUBIC` (yml `interp: 2`), RGB for all three models as in the spike.
5. `LayoutType` keeps legacy variants and reuses `Text/Table/Header/Footer/Reference` for the matching V3 labels; one `RegionRole` map drives OCR routing, nesting and Markdown.
6. `DocPipelineConfig::new(layout, ocr)` (two arguments); `StageStatus` = `{ stage, state: Ran{elapsed_ms, items} | Skipped{reason} | Failed{reason} }`; per-region failures do not abort the page.
7. Table dictionary via new `metadata_value` (not `get_character_list`, which would change the recogniser); embedded 48-token constant for `.rten`.
8. `max_tokens` 500 from the official yml; output tensors identified by last dimension; location head must have 8 values; SLANeXt model types rejected by `TableStructureRecognizer::new`.
9. Cells with an all-zero quad keep a degenerate box at the table origin; centre ties go to the smallest containing cell; IoU ties to the first cell in row-major order.
10. Fixtures committed as plain PNGs (no LFS in this repository); rotations generated in tests.
11. `.gitignore` and `MODELS_VERSION` unchanged; the download script does not auto-install `rten-convert`.
12. CLI: strict `requires` relations, default model paths under `models/DocOCR`, `--save-corrected` only with `--enable-layout`, layout mode accepts only `document`/`markdown`.
13. Version bump and docs in the last task; English docs only.
