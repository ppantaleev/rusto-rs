//! Document layout, orientation and table structure from library code, without the CLI.
//!
//! ```text
//! cargo run --release --example document_layout -- \
//!     <image> <det.onnx> <rec.onnx> <dict.txt> [doc-models-dir]
//! ```
//!
//! `doc-models-dir` defaults to `models/DocOCR` (see `scripts/download_doc_models.sh`).
//! Prints the result as JSON and as Markdown on stdout and one line per stage on stderr.
//! All coordinates are in the corrected page that `run_with_image` also returns.

use std::path::PathBuf;

use rusto::image_impl::imread;
use rusto::{
    DocPipeline, DocPipelineConfig, InitializeConfig, LayoutConfig, OrientConfig, StageState, TableStructureConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 4 || args.len() > 5 {
        eprintln!("usage: document_layout <image> <det> <rec> <dict> [doc-models-dir]");
        std::process::exit(2);
    }
    let doc_dir = PathBuf::from(args.get(4).map_or("models/DocOCR", String::as_str));

    let config = DocPipelineConfig::new(
        LayoutConfig::default(doc_dir.join("layout_ppdoclayoutv3.rten")),
        InitializeConfig::ppv5(&args[1], &args[2], &args[3]),
    )
    .with_orientation(OrientConfig::rapid_orientation_v2(doc_dir.join("orient_rapidai_v2.rten")))
    .with_table_recognizer(TableStructureConfig::slanet_plus(doc_dir.join("table_slanetplus.rten")));
    let mut pipeline = DocPipeline::new(config)?;

    let image = imread(&args[0])?;
    let (result, _corrected_page) = pipeline.run_with_image(&image)?;

    println!("{}", serde_json::to_string_pretty(&result)?);
    println!("{}", result.to_markdown());
    for status in &result.stages {
        match &status.state {
            StageState::Ran { elapsed_ms, items } => {
                eprintln!("{:?}: ran, {elapsed_ms:.1} ms, {items} items", status.stage)
            }
            StageState::Skipped { reason } => eprintln!("{:?}: skipped, {reason}", status.stage),
            StageState::Failed { reason } => eprintln!("{:?}: failed, {reason}", status.stage),
        }
    }
    Ok(())
}
