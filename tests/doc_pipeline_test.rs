use std::path::PathBuf;

use rusto::image_impl::Mat;
use rusto::{
    DocPipeline, DocPipelineConfig, InitializeConfig, LayoutConfig, OrientConfig, Stage, StageState,
    TableStructureConfig,
};

fn fixture(name: &str) -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/doc").join(name)
}

fn doc_model(name: &str) -> PathBuf {
    PathBuf::from(std::env::var("RUSTO_DOC_MODELS").unwrap_or("models/DocOCR".into())).join(name)
}

/// `RUSTO_TEST_MODELS` must hold `det.onnx`, `rec.onnx` and a Cyrillic `dict.txt`
/// (PP-OCRv5 detector, Cyrillic recogniser).
fn pipeline() -> Option<DocPipeline> {
    let Ok(ocr_dir) = std::env::var("RUSTO_TEST_MODELS") else {
        eprintln!("skipped: set RUSTO_TEST_MODELS to a directory with det.onnx, rec.onnx, dict.txt");
        return None;
    };
    let ocr_dir = PathBuf::from(ocr_dir);
    let cfg = DocPipelineConfig::new(
        LayoutConfig::default(doc_model("layout_ppdoclayoutv3.rten")),
        InitializeConfig::ppv5(ocr_dir.join("det.onnx"), ocr_dir.join("rec.onnx"), ocr_dir.join("dict.txt")),
    )
    .with_orientation(OrientConfig::rapid_orientation_v2(doc_model("orient_rapidai_v2.rten")))
    .with_table_recognizer(TableStructureConfig::slanet_plus(doc_model("table_slanetplus.rten")));
    Some(DocPipeline::new(cfg).unwrap())
}

#[test]
#[ignore = "requires RUSTO_TEST_MODELS and models/DocOCR"]
fn rotated_page_end_to_end() {
    let upright = image::open(fixture("page_text.png")).unwrap();
    let Some(mut pipeline) = pipeline() else { return };
    let (r, corrected) = pipeline.run_with_image(&Mat::new(upright.rotate90())).unwrap();
    assert_eq!(r.orientation.unwrap().applied_degrees, 270);
    assert_eq!((r.width, r.height, corrected.cols(), corrected.rows()), (1240, 1754, 1240, 1754));
    let t = r.regions.iter().find_map(|g| g.table.as_ref()).expect("table");
    assert_eq!((t.num_rows, t.num_cols), (3, 4));
    let md = r.to_markdown();
    assert!(md.starts_with("# ") || md.starts_with("## "), "{md}");
    assert!(md.contains("<table>") && md.contains("340"), "{md}");
    assert!(r.stages.iter().all(|s| !matches!(s.state, StageState::Failed { .. })));
}

#[test]
#[ignore = "requires RUSTO_TEST_MODELS and models/DocOCR"]
fn blank_page_yields_no_regions() {
    let blank = Mat::new(image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(1240, 1754, image::Rgb([255, 255, 255]))));
    let Some(mut pipeline) = pipeline() else { return };
    let (r, _) = pipeline.run_with_image(&blank).unwrap();
    assert!(r.regions.is_empty());
    assert_eq!(r.to_markdown(), "");
    assert!(r.stages.iter().any(|s| s.stage == Stage::Layout && matches!(s.state, StageState::Ran { items: 0, .. })));
}

#[test]
#[ignore = "requires RUSTO_TEST_MODELS and models/DocOCR"]
fn upright_page_regions_in_order_with_text_in_reading_order() {
    let Some(mut pipeline) = pipeline() else { return };
    let (r, _) = pipeline.run_with_image(&Mat::new(image::open(fixture("page_text.png")).unwrap())).unwrap();
    assert_eq!(r.orientation.unwrap().applied_degrees, 0);
    let labels: Vec<_> = r.regions.iter().map(|g| g.label.name()).collect();
    assert_eq!(labels, ["paragraph_title", "text", "text", "table"], "{:?}", r.stages);
    assert!(r.regions.iter().enumerate().all(|(i, g)| g.index == i));
    // Words of the first line come out in reading order, not in detector order.
    let first = r.regions[1].text.as_deref().unwrap();
    assert!(first.starts_with("Настоящият документ описва"), "{first}");
    assert!(r.regions[3].text.is_none() && r.regions[3].lines.is_empty());
    assert_eq!(r.blocks.len(), 4);
    assert!(r.blocks[3].html.as_deref().unwrap().starts_with("<table>"));
    assert!(r.stages.iter().all(|s| matches!(s.state, StageState::Ran { .. })), "{:?}", r.stages);
}
