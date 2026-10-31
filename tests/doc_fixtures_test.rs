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
