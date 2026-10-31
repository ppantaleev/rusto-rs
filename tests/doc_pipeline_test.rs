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

const TOL: f32 = 6.0;

fn inside(f: &rusto::Frame, b: &rusto::layout::BBox) -> bool {
    f.left >= b.x_min as f32 - TOL
        && f.top >= b.y_min as f32 - TOL
        && f.left + f.width <= b.x_max as f32 + TOL
        && f.top + f.height <= b.y_max as f32 + TOL
}

fn assert_in_page_space(r: &rusto::DocResult) {
    for g in &r.regions {
        // (a) text lines lie inside their region
        for l in &g.lines {
            assert!(inside(&l.frame, &g.bbox), "region {} line {:?} outside {:?}", g.index, l.frame, g.bbox);
        }
        // (b) cells and unassigned lines lie inside the table region
        if let Some(t) = &g.table {
            for c in t.rows.iter().flat_map(|row| &row.cells) {
                let f = rusto::Frame::new(
                    (c.bbox.x_max - c.bbox.x_min) as f32,
                    (c.bbox.y_max - c.bbox.y_min) as f32,
                    c.bbox.y_min as f32,
                    c.bbox.x_min as f32,
                );
                assert!(inside(&f, &g.bbox), "cell {:?} outside {:?}", c.bbox, g.bbox);
            }
            assert!(t.unassigned_lines.iter().all(|l| inside(&l.frame, &g.bbox)));
        }
    }
}

#[test]
#[ignore = "requires RUSTO_TEST_MODELS and models/DocOCR"]
fn rotated_page_gives_the_upright_result_in_corrected_page_space() {
    let Some(mut pipeline) = pipeline() else { return };
    let upright = image::open(fixture("page_text.png")).unwrap();
    let (a, _) = pipeline.run_with_image(&Mat::new(upright.clone())).unwrap();
    let (b, _) = pipeline.run_with_image(&Mat::new(upright.rotate90())).unwrap();
    assert_eq!(a.orientation.unwrap().applied_degrees, 0);
    assert_eq!(b.orientation.unwrap().applied_degrees, 270);
    assert_in_page_space(&a);
    assert_in_page_space(&b);

    // (c) same regions, texts and boxes as for the upright page
    assert_eq!((a.width, a.height), (b.width, b.height));
    assert_eq!(a.regions.len(), b.regions.len());
    for (x, y) in a.regions.iter().zip(&b.regions) {
        assert_eq!((x.index, x.label), (y.index, y.label));
        assert_eq!(x.text, y.text, "region {}", x.index);
        let d = |p: i32, q: i32| (p - q).abs() <= 2;
        assert!(
            d(x.bbox.x_min, y.bbox.x_min) && d(x.bbox.y_min, y.bbox.y_min) && d(x.bbox.x_max, y.bbox.x_max) && d(x.bbox.y_max, y.bbox.y_max),
            "region {}: {:?} vs {:?}",
            x.index,
            x.bbox,
            y.bbox
        );
        assert_eq!(x.lines.len(), y.lines.len(), "region {}", x.index);
        match (&x.table, &y.table) {
            (Some(s), Some(t)) => {
                assert_eq!((s.num_rows, s.num_cols), (t.num_rows, t.num_cols));
                let cells = |t: &rusto::TableStructure| {
                    t.rows.iter().flat_map(|r| r.cells.iter().map(|c| (c.text.clone(), c.bbox))).collect::<Vec<_>>()
                };
                let (cs, ct) = (cells(s), cells(t));
                assert_eq!(cs.len(), ct.len());
                for ((ta, ba), (tb, bb)) in cs.iter().zip(&ct) {
                    assert_eq!(ta, tb);
                    let d = |p: i32, q: i32| (p - q).abs() <= 3;
                    assert!(d(ba.x_min, bb.x_min) && d(ba.y_min, bb.y_min) && d(ba.x_max, bb.x_max) && d(ba.y_max, bb.y_max), "{ba:?} vs {bb:?}");
                }
            }
            (None, None) => {}
            _ => panic!("table missing on one page, region {}", x.index),
        }
    }
}
