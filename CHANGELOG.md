# Changelog

All notable changes to RustO! will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased upstream]

Entries by other contributors that landed in the history before this branch and are not part of the document layout, tables and orientation work. They were removed from 0.4.0 because they do not describe this branch; they are kept here until upstream assigns them to a release.

### Changed
- **flutter**: Poll asynchronously for OCR completion in example widget test; add inline code documentation (@byrizki)

### Fixed
- **bindings**: Synchronize UniFFI generated bindings checksums and enhance flutter failure diagnostics (@byrizki)
- Flutter example ci, pypi release action, dart & flutter pub score, flutter ci release (@byrizki)

## [0.4.0] - 2026-10-31

### Added
- **doc**: scripts/download_doc_models.sh downloads and verifies RapidAI orientation v2, PP-DocLayoutV3 and SLANet_plus models (Apache-2.0)
- **orient**: OrientConfig::rapid_orientation_v2 (resize-short 256, centre crop 224, ImageNet normalisation), PageOrientation, rotate_box_back
- **layout**: PP-DocLayoutV3 support (25 labels, [N,7] output, reading order)
- **table**: SLANet_plus table structure model type (`TableStructureConfig::slanet_plus`, 488 px keep-ratio input `x`, 500 tokens), embedded structure dictionary `SLANET_PLUS_DICT` with ONNX `character` metadata override, `EngineSession::metadata_value`; pre-processing and dictionary translated from bobine 0.7.3 `rapid_table.rs` (MIT)
- **table**: SLANet_plus structure with rowspan/colspan, cell boxes, cell text, unassigned_lines (`argmax_tokens` is translated from bobine's `rapid_table.rs`, MIT; the grid decoder is new code)
- **doc**: DocPipeline::run_with_image, corrected-page coordinates, per-region OCR, vertical text, stage status; examples/document_layout.rs
- **cli**: `--enable-layout`, `--layout-model`, `--enable-table`, `--table-model`, `--save-corrected`, `--format document`; `--enable-orient` now runs orientation (plain OCR keeps original-scan coordinates); `--orient-model` requires `--enable-orient` and defaults to `models/DocOCR/orient_rapidai_v2.rten`

### Changed
- **doc**: Behaviour change: `DocResult::to_markdown` is a new renderer by layout role: `# ` for `doc_title`, `## ` for `paragraph_title`, `[^n]` footnote definitions after the body, header/footer/number regions under `## Page furniture`, and GFM tables (HTML `<table>` when spans exist). The legacy `blocks` renderer is removed
- **doc**: Caption rule: a `figure_title` caption is moved behind an image, chart or table only when that block is the immediately preceding body block; otherwise it stays in model reading order
- **doc**: Breaking (struct literals): `LayoutRegion` gains `reading_order` and `OrientConfig` gains `preprocess`; code that builds them with struct literals must add the fields
- **layout**: Breaking: layout outputs of shape `[N,6]` (the earlier 10-class decoder with NMS, including PP-DocLayout_plus-L) are no longer accepted; only the PP-DocLayoutV3 `[N,7]` output is decoded. `LayoutDetector::new` of an older model fails with a clear `[N,7]` error
- **api**: `OrientConfig::rapid_orientation_v2` and `LayoutConfig::default` take `impl Into<PathBuf>`, so a `&str` works; `PathBuf` callers still compile, but a call that passes `"…".into()` now needs the type annotation removed
- **api**: `DocStage`, `StageState` and `RegionRole` are #[non_exhaustive]; a `match` on them outside the crate needs a `_ =>` arm
- **api**: `table::BBox` is now the same type as `layout::BBox` (re-exported), so `rotate_box_back` accepts table cell boxes; `layout::BBox` is `Copy + PartialEq + Eq + Serialize`
- **cli**: `--save-corrected` accepts only `.png`, `.jpg` and `.jpeg` (case-insensitive) and rejects other extensions before models load
- **doc**: `DocPipelineConfig::table_detector` is kept, but the stub cell detector is never run: `DocPipeline::new` does not load its model and only checks that a configured path exists; `examples/doc_pipeline_demo.rs` no longer needs an rtdetr model
- **doc**: DocPipelineConfig is #[non_exhaustive]; use DocPipelineConfig::new(layout, ocr) and with_* builders
- **layout**: LayoutConfig defaults 800 px, mean 0/std 1, conf_thresh 0.5 (was 0.4, unused); LayoutType is #[non_exhaustive]
- **table**: TableStructureConfig and TableModelType are #[non_exhaustive]; the default config is now SLANet_plus (`models/DocOCR/table_slanetplus.rten`) and the recogniser rejects the SLANeXt model types
- **table**: TableCell.bbox is absolute page pixels (percent-of-crop convention removed); TableCell.score added
- **cli**: Behaviour change: `--orient-model` without `--enable-orient` is now a clap error; `--enable-orient` now runs orientation (before, the flag had no effect); `--layout-model`, `--table-model` and `--save-corrected` also require their enabling flag
- **orient**: The model loader selects `.rten`/`.onnx` by extension and rejects other extensions, also for the legacy line classifier
- **docs**: Added "Document Layout, Tables & Orientation" guide; CLI and orientation guides updated; documentation is English only (`docs/content/id` not updated)
- **tests**: Ignored timing test `tests/doc_timing_test.rs` (orientation, layout, table structure only, full page)

### Fixed
- **table**: A quad with a NaN or infinite value no longer produces a cell box (`quad_to_bbox` returns `None`)
- **table**: Place colspan cells only where the whole span is free
- **orient**: Do not re-apply softmax to models that already output probabilities

### Known limitations
- Layout, orientation and table stages are unsupported with the `use-opencv` backend. `cargo check --features use-opencv` was not run: not checked, OpenCV not installed on the build machine; CI does not build it
- All boxes of `DocResult` are in corrected-page space; check `DocResult::stages` (an orientation `Failed` stage means the page was not rotated) and use `rotate_box_back` with `orientation.applied_degrees` for scan coordinates
- Markdown output does not escape OCR text
- Formulas, algorithms, images, charts and seals are located but not recognised
- Table structure is not attempted when OCR of the table crop fails

### Attribution
- SLANet_plus pre-processing, structure dictionary handling and `argmax_tokens` are translated from `bobine` 0.7.3 (`src/rapid_table.rs`, https://github.com/opticsWolf/bobine), used under its MIT option; the grid decoder is new code; Copyright (c) 2026 opticsWolf. Notice in the header of `src/table.rs`
- Models (downloaded by `scripts/download_doc_models.sh`, not bundled): RapidAI orientation v2, PaddlePaddle PP-DocLayoutV3 and SLANet_plus, all Apache-2.0

### Performance (Apple M5 Pro, release, 1240x1754 page, median of 5 after warm-up)
- Orientation 7 ms, layout 474 ms, table structure only 21 ms (945x184 crop), table stage including crop OCR about 300 ms, whole page about 1.9 s, peak RSS about 1.7 GB

## [0.3.2] - 2026-10-25

### Added
- **scripts**: Tag committer github user in changelog entries (@byrizki)
- **scripts**: Update bump_version.py to cover all bindings, packages, examples, and docs (@byrizki)
- **bindings**: Unify generate-bindings pipeline across all targets (@byrizki)
- **dotnet**: Migrate to auto-generated uniffi-bindgen-cs bindings (@byrizki)
- **dart**: Migrate to auto-generated uniffi-bindgen-dart bindings (@byrizki)
- **core**: Pin uniffi 0.31 and automate C header generation with cbindgen (@byrizki)
- **core**: Add contiguous copy fast path, spatial C FFI, bindings, and docs for decoded RGB input (@byrizki)
- Add validated decoded RGB input for Rust and C callers (@tacman)
- **ci**: Add model package verifications and binding contract tests (@byrizki)
- **verify**: Add go/flutter/dart/java package verification and verify actual image across all bindings (@byrizki)
- **flutter**: Verify real Flutter APK, FFI spatial pipeline, and live Maestro E2E test (@byrizki)
- Add packed packages content verifier, CI improvements, and Flutter example app with Maestro (@byrizki)
- **rn-android**: Enable end-to-end APK build and Maestro verification without mocks (@byrizki)
- **bindings**: Add Obj-C/SPM iOS, unified RN Maestro test suite, and multi-language bindings (Node, Python, Go, Java, Dart/Flutter) (@byrizki)

### Changed
- Read release body from changelog in build workflow (@byrizki)
- **flutter**: Setup Java 17 for Maestro CLI execution in flutter-example job (@byrizki)
- Fix wasm loading in docs ci, model check in wasm test, and sequential docs deployment (@byrizki)
- Separate CI workflows for docs and library (@byrizki)
- Add CI workflow for PRs and push using reusable build workflow (@byrizki)
- Update package badges and align model package versions to 0.3.2 (@byrizki)
- Update documentation, quick-start guides, and code snippets across bindings (@byrizki)
- Replace Maven publishing with JitPack build trigger for Android and Java (@byrizki)
- Update build and publish workflows for multi-platform language bindings (@byrizki)
- **docs**: Upgrade nuxt, docus, and mermaid to latest versions (@byrizki)
- **docs**: Consolidate Snyk dependency upgrades for nuxt, docus, and mermaid (@byrizki)
- Humanize models guide, correct multilingual language scope, and fix diagrams (@byrizki)
- Add model zip download releases and consumer decision guide (@byrizki)
- Improve iOS, React Native iOS, and Models & Tiers guides (@byrizki)
- Add website guides and installation for node, python, go, java, and flutter (@byrizki)
- Add ecosystem packages documentation and quickstarts for node, python, go, java, and flutter (@byrizki)
- Format RGB integration test (@tacman)
- Update locks (@byrizki)
- Enable default ffi features, fix Dart FFI deprecation, and clean linting in Flutter and React Native example (@byrizki)
- Snyk security integration, planning documents, and documentation updates (@byrizki)

### Fixed
- **ios**: Add RuntimeOptions typealias for OcrRunOptions and implement parameterless init (@byrizki)
- **ios**: Use zero-argument init() for UniFFI structs to prevent redeclaration collisions (@byrizki)
- **bindings**: Resolve iOS Swift config inits, RN android local AARs, and flutter test FFI library resolution (@byrizki)
- **ffi**: Exclude internal uniffi rustbuffer symbols from canonical C header (@byrizki)
- Replace geo-clipper with clipper2-rust (@byrizki)
- Upgrade better-sqlite3 from 12.6.2 to 12.11.1 (@snyk-bot)
- FFI memory safety issues and update OCR pipeline coordinate mapping and detection logic (@byrizki)
- Fix bugs and optimize core image processing & contour detection (@google-labs-jules[bot])

## [0.3.1] - 2026-10-10

### Changed
- **docs**: Load rten models from jsdelivr cdn instead of bundling
- Update readme
- Update content
- Rename to @rustors for npm packages
- Release models-rten-v0.3.2
- Rename packages
- Release models-rten-v0.3.1
- Remove unpublish-models workflow as npm requires interactive 2FA
- Add workflow to unpublish npm model packages
- Set default to true for all model publish workflow dispatch options
- Selectively run packaging and artifact uploads based on checked options
- Auto-read MODELS_VERSION and run all checkable release/publish items in parallel
- Use absolute paths for npm publish to avoid git repo misinterpretation
- Update live demo docs
- Update demo content
- Improve live demo

### Fixed
- **ci**: Bundle Swift sources and LICENSE in RustO.xcframework.zip for CocoaPods
- Generate_npm_models
- **npm**: Include .rten models as primary files in npm packages
- **ci**: Streamline pipeline to download ONNX -> quantize to INT8 -> convert all to RTen
- **ci**: Preserve onnx models with --keep-onnx for INT8 quantization in publish-models workflow
- **scripts**: Support --all flag and python module fallback in quantize_ppocrv6_int8.py
- Separate models publishing

## [0.3.0] - 2026-10-07

### Added
- **packages**: Add @rustors/web, @rustors/react, and @rustors/model-* packages
- **wasm**: Add rten-wasm crate for high-performance browser OCR inference
- **core**: Add configurable image calibration, descreening, and optimization pipeline

### Changed
- Standardize session usage from RtenSession/MnnSession to EngineSession
- Update npm model generator and PowerShell scripts for ONNX/RTen
- Complete MNN to RTen migration across bindings, examples, and podspecs
- Update workflows and model scripts for RTen engine migration
- Update gitignore for debugging dumps and add architecture benchmark plans
- Update documentation, guides, and interactive live demo with RTen WASM
- **bindings**: Align Android, iOS, .NET, and React Native bindings with RTen and calibration options
- **engine**: Replace MNN with pure-Rust RTen backend and remove legacy MNN crates
- Fix formats
- Update node to 24

## [0.2.5] - 2026-08-20

### Added
- Add docs sites, improve configuration api

### Fixed
- Fix RN lint

## [0.2.4] - 2026-08-20

### Added
- Add several text detection configuration

### Fixed
- Fix ts lint

## [0.2.3] - 2026-08-19

### Changed
- Refactor build ci
- Use legacy yarn
- Update download script
- Refactor to use yarn

### Fixed
- Fix ocr source validation
- Fix ios test build
- Fix ios build
- Fix subworkflow run
- Fix build example gate
- Fix build example gate, rn 0.75+
- Fix rn deps install
- Fix example build gate
- Fix npm install examples gate

## [0.2.2] - 2026-08-14

### Changed
- Add example build gate
- Optimize & rename public apis

### Fixed
- Fix rn kotlin bridge

## [0.2.1] - 2026-08-13

### Changed
- Bump version

### Fixed
- Fix jitpack release

## [0.2.0] - 2026-08-13

### Changed
- Update publishing

## [0.1.7] - 2026-08-12

### Changed
- Update changelog

### Fixed
- Kotlin action

## [0.1.6] - 2026-08-12

### Added
- Separate model files from repo, download on-the-fly from rusto-rs-models releases

### Fixed
- Fix swift lint
- Fix C# build


## [0.1.5] - 2026-08-12

### Added
- Add PP OCR v6

### Changed
- Update cargo.toml
- Use maven central

### Fixed
- Fix crates publish with retry
- Fix sign key id

## [0.1.4] - 2026-08-12

### Changed
- Update script
- Use prebuilt MNN intelligently

### Fixed
- Fix release tag on build
- Fix ios/osx build
- Revert mnn submodule to upstream; inject cstdint via cmake flags instead
- Update mnn submodule - add cstdint for AppleClang 21 (macOS 26)
- Fix mac os build
- Android/ios file content uri resolver
- RN binding, build and ts types

## [0.1.0] - 2024-11-20

### Added
- 🎉 Initial release of RustO! (formerly rapidocr)
- Pure Rust OCR implementation based on RapidOCR
- PaddleOCR PPOCRv4/v5 model support via ONNX Runtime
- High-level API with `RapidOCR` and `RapidOCRConfig`
- CLI application with JSON, Text, and TSV output formats
- C FFI bindings for cross-language integration
- iOS CocoaPods support via Pod spec
- GitHub Actions CI/CD workflows
- Comprehensive documentation and examples

### Changed
- **Branding**: Renamed from `rapidocr` to `RustO!` (crate name: `rusto-rs`)
- **Library name**: Changed from `rapidocr` to `rusto` for imports
- **CLI name**: Updated from `rapidocr` to `rusto`
- **Documentation**: Complete rebranding with RustO! identity
- **Documentation**: Added comprehensive architecture explanation
- **Documentation**: Added PaddleOCR and RapidOCR acknowledgments

### Fixed
- **Warnings**: Cleaned up all 16 compiler warnings
  - Added `#![allow(dead_code)]` to experimental code in `contours.rs`
  - Fixed unused variable warnings
  - Suppressed warnings for reference implementations
- **Library API**: Fixed `lib.rs` to use correct `RapidOcr::new_ppv5()` method
- **Library API**: Simplified `RapidOCRConfig` to only include essential model paths
- **Library API**: Added `run()` method to `RapidOcr` for convenient file-based OCR
- **CLI**: Removed unused `use_opencv` flag from CLI arguments
- **CLI**: Updated to use simplified `RapidOCRConfig` structure
- **Build**: Removed old `rapidocr_json` binary in `src/bin/`
- **Build**: Clean builds with zero warnings ✅

### Build Status
- ✅ `cargo check` - Success with 0 warnings
- ✅ `cargo build` - Success with 0 warnings
- ✅ `cargo build --release` - Success
- ✅ CLI binary compiles correctly (`rusto`)
- ✅ Library API is functional
- ✅ FFI bindings ready

### Infrastructure
- ✅ iOS Pod spec created (`RustO.podspec`)
- ✅ GitHub Actions CI workflow (`.github/workflows/ci.yml`)
- ✅ GitHub Actions release workflow (`.github/workflows/release.yml`)
- ✅ Automated testing on Linux, macOS, and Windows
- ✅ Automated documentation deployment
- ✅ Security audits configured

---

## Project Status

**Version**: 0.4.0  
**Status**: Production Ready  
**Accuracy**: 99.3% OpenCV parity  
**Last Updated**: October 31, 2026
