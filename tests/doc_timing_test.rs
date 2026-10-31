//! Timing of the document stages (ignored; run with `--release`).
//!
//! `RUSTO_DOC_MODELS` holds the document models (default `models/DocOCR`).
//! `RUSTO_TEST_MODELS` holds `det.onnx`, `rec.onnx`, `dict.txt` for the
//! full-pipeline numbers. Numbers go to stderr: warm-up run, then the median of
//! `RUSTO_TIMING_RUNS` (default 5) runs.

use std::path::PathBuf;
use std::time::Instant;

use rusto::image_impl::Mat;
use rusto::{
    decide_orientation, DocPipeline, DocPipelineConfig, InitializeConfig, LayoutConfig,
    LayoutDetector, LayoutType, OrientClassifier, OrientConfig, DocStage, StageState,
    TableStructureConfig, TableStructureRecognizer,
};

fn fixture(name: &str) -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/doc")
        .join(name)
}

fn doc_model(name: &str) -> PathBuf {
    PathBuf::from(std::env::var("RUSTO_DOC_MODELS").unwrap_or("models/DocOCR".into())).join(name)
}

fn runs() -> usize {
    std::env::var("RUSTO_TIMING_RUNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5)
        .max(5)
}

/// One warm-up call, then the median (ms) of `runs()` timed calls.
fn median_ms(mut f: impl FnMut()) -> f64 {
    f();
    let mut t: Vec<f64> = (0..runs())
        .map(|_| {
            let s = Instant::now();
            f();
            s.elapsed().as_secs_f64() * 1000.0
        })
        .collect();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    t[t.len() / 2]
}

#[test]
#[ignore = "timing; requires models/DocOCR, run with --release --nocapture"]
fn model_stage_timings() {
    let page = image::open(fixture("page_text.png")).unwrap();
    let mat = Mat::new(page.clone());

    let mut orient = OrientClassifier::new(OrientConfig::rapid_orientation_v2(doc_model(
        "orient_rapidai_v2.rten",
    )))
    .unwrap();
    let ms = median_ms(|| {
        let out = orient.classify(&mat).unwrap();
        let _ = decide_orientation(&out, 0.6);
    });
    eprintln!("TIMING orientation (classify): {ms:.1} ms");

    let mut layout = LayoutDetector::new(LayoutConfig::default(doc_model(
        "layout_ppdoclayoutv3.rten",
    )))
    .unwrap();
    let ms = median_ms(|| {
        layout.detect(&mat).unwrap();
    });
    eprintln!("TIMING layout (detect): {ms:.1} ms");

    // Structure only: `recognize_at` runs the model and decodes, without any OCR.
    let regions = layout.detect(&mat).unwrap().regions;
    let table = regions
        .iter()
        .find(|r| r.layout_type == LayoutType::Table)
        .expect("table region");
    let b = table.bbox;
    let crop = Mat::new(page.crop_imm(
        b.x_min as u32,
        b.y_min as u32,
        (b.x_max - b.x_min) as u32,
        (b.y_max - b.y_min) as u32,
    ));
    let mut rec = TableStructureRecognizer::new(TableStructureConfig::slanet_plus(doc_model(
        "table_slanetplus.rten",
    )))
    .unwrap();
    let ms = median_ms(|| {
        rec.recognize_at(&crop, b.x_min, b.y_min).unwrap();
    });
    eprintln!(
        "TIMING table structure only (recognize_at, {}x{} crop): {ms:.1} ms",
        b.x_max - b.x_min,
        b.y_max - b.y_min
    );
}

#[test]
#[ignore = "timing; requires RUSTO_TEST_MODELS and models/DocOCR, run with --release --nocapture"]
fn pipeline_stage_timings() {
    let Ok(ocr_dir) = std::env::var("RUSTO_TEST_MODELS") else {
        eprintln!("skipped: set RUSTO_TEST_MODELS");
        return;
    };
    let ocr = PathBuf::from(ocr_dir);
    let cfg = DocPipelineConfig::new(
        LayoutConfig::default(doc_model("layout_ppdoclayoutv3.rten")),
        InitializeConfig::ppv5(
            ocr.join("det.onnx"),
            ocr.join("rec.onnx"),
            ocr.join("dict.txt"),
        ),
    )
    .with_orientation(OrientConfig::rapid_orientation_v2(doc_model(
        "orient_rapidai_v2.rten",
    )))
    .with_table_recognizer(TableStructureConfig::slanet_plus(doc_model(
        "table_slanetplus.rten",
    )));
    let mut pipeline = DocPipeline::new(cfg).unwrap();
    let mat = Mat::new(image::open(fixture("page_text.png")).unwrap());

    pipeline.run_with_image(&mat).unwrap();
    let mut totals = Vec::new();
    let mut table_stage = Vec::new();
    for _ in 0..runs() {
        let s = Instant::now();
        let (r, _) = pipeline.run_with_image(&mat).unwrap();
        totals.push(s.elapsed().as_secs_f64() * 1000.0);
        for st in &r.stages {
            if let (DocStage::Table, StageState::Ran { elapsed_ms, .. }) = (&st.stage, &st.state) {
                table_stage.push(*elapsed_ms);
            }
        }
    }
    let med = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    eprintln!(
        "TIMING full page (run_with_image): {:.0} ms",
        med(&mut totals)
    );
    eprintln!(
        "TIMING table stage incl. crop OCR: {:.0} ms",
        med(&mut table_stage)
    );
}
