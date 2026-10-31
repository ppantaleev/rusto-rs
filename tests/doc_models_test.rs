use std::path::PathBuf;

use rusto::image_impl::Mat;
use rusto::{decide_orientation, LayoutConfig, LayoutDetector, LayoutType, OrientClassifier, OrientConfig, RegionRole};

fn fixture(name: &str) -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/doc").join(name)
}

fn doc_model(name: &str) -> PathBuf {
    PathBuf::from(std::env::var("RUSTO_DOC_MODELS").unwrap_or("models/DocOCR".into())).join(name)
}

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

fn iou(a: &rusto::layout::BBox, b: &serde_json::Value) -> f64 {
    let b: Vec<f64> = b.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let (ax0, ay0, ax1, ay1) = (a.x_min as f64, a.y_min as f64, a.x_max as f64, a.y_max as f64);
    let iw = (ax1.min(b[2]) - ax0.max(b[0])).max(0.0);
    let ih = (ay1.min(b[3]) - ay0.max(b[1])).max(0.0);
    let inter = iw * ih;
    inter / ((ax1 - ax0) * (ay1 - ay0) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
}

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
