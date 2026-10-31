use std::path::PathBuf;

use rusto::image_impl::Mat;
use rusto::{decide_orientation, OrientClassifier, OrientConfig};

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
