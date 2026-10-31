use clap::{Parser, ValueEnum};
use rusto::image_impl::{imread, imwrite};
use rusto::{
    DetectTextResult, DocPipeline, DocPipelineConfig, ImageSource, InitializeConfig, LayoutConfig, OcrRunOptions,
    OrientConfig, RustO, TableStructureConfig,
};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "rusto")]
#[command(about = "RustO! - Pure Rust OCR powered by PaddleOCR engine", long_about = None)]
struct Cli {
    /// Path to detection model (ONNX)
    #[arg(long)]
    det_model: PathBuf,

    /// Path to recognition model (ONNX)
    #[arg(long)]
    rec_model: PathBuf,

    /// Path to dictionary file
    #[arg(long)]
    dict: PathBuf,

    /// Input image path
    image: PathBuf,

    /// Output format
    #[arg(short, long, value_enum, default_value_t = OutputFormat::Json)]
    format: OutputFormat,

    /// Enable document orientation classification
    #[arg(long)]
    enable_orient: bool,

    /// Path to orientation classification model (.rten, falls back to .onnx)
    #[arg(long, requires = "enable_orient", default_value = "models/DocOCR/orient_rapidai_v2.rten")]
    orient_model: PathBuf,

    /// Enable text unwarping
    #[arg(long)]
    enable_unwarp: bool,

    /// Path to text unwarping model (uvdoc.rten)
    #[arg(long)]
    unwarp_model: Option<PathBuf>,

    /// Enable document layout analysis (regions, reading order, Markdown structure)
    #[arg(long)]
    enable_layout: bool,

    /// Path to layout detection model (PP-DocLayoutV3, .rten, falls back to .onnx)
    #[arg(long, requires = "enable_layout", default_value = "models/DocOCR/layout_ppdoclayoutv3.rten")]
    layout_model: PathBuf,

    /// Enable table structure recognition for table regions
    #[arg(long, requires = "enable_layout")]
    enable_table: bool,

    /// Path to table structure model (SLANet_plus, .rten, falls back to .onnx)
    #[arg(long, requires = "enable_table", default_value = "models/DocOCR/table_slanetplus.rten")]
    table_model: PathBuf,

    /// Write the orientation-corrected page to this path (PNG or JPEG, by extension)
    #[arg(long, requires = "enable_layout")]
    save_corrected: Option<PathBuf>,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum OutputFormat {
    /// JSON output with full details
    Json,
    /// Plain text, one line per detected text
    Text,
    /// TSV format: text\tscore\tx1,y1,x2,y2,x3,y3,x4,y4
    Tsv,
    /// Markdown format with metadata and ordering
    Markdown,
    /// Plain text ordered by position
    TextOrdered,
    /// Document JSON with regions, reading order and tables (needs --enable-layout)
    Document,
}

/// What the run does, derived from the flags.
#[derive(Debug)]
enum RunPlan {
    /// Plain OCR; `orient` is the page orientation model, if enabled.
    Ocr { orient: Option<PathBuf> },
    /// Layout pipeline; `markdown` selects Markdown instead of JSON output.
    Document {
        orient: Option<PathBuf>,
        layout: PathBuf,
        table: Option<PathBuf>,
        save_corrected: Option<PathBuf>,
        markdown: bool,
    },
}

fn plan_run(cli: &Cli) -> Result<RunPlan, String> {
    let orient = cli.enable_orient.then(|| cli.orient_model.clone());
    if !cli.enable_layout {
        if cli.format == OutputFormat::Document {
            return Err("--format document requires --enable-layout".to_string());
        }
        return Ok(RunPlan::Ocr { orient });
    }
    if !matches!(cli.format, OutputFormat::Document | OutputFormat::Markdown) {
        return Err("with --enable-layout use --format document or --format markdown".to_string());
    }
    if let Some(path) = &cli.save_corrected {
        let ext = path.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase);
        if !matches!(ext.as_deref(), Some("png" | "jpg" | "jpeg")) {
            return Err(format!(
                "--save-corrected: unsupported extension for {}; use .png, .jpg or .jpeg",
                path.display()
            ));
        }
    }
    Ok(RunPlan::Document {
        orient,
        layout: cli.layout_model.clone(),
        table: cli.enable_table.then(|| cli.table_model.clone()),
        save_corrected: cli.save_corrected.clone(),
        markdown: cli.format == OutputFormat::Markdown,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let plan = plan_run(&cli).map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

    if cli.enable_unwarp || cli.unwarp_model.is_some() {
        eprintln!("Warning: --enable-unwarp/--unwarp-model are not supported and are ignored.");
    }

    let mut config = InitializeConfig::ppv5(&cli.det_model, &cli.rec_model, &cli.dict);
    let orient_enabled = match plan {
        RunPlan::Document { orient, layout, table, save_corrected, markdown } => {
            let mut doc_config = DocPipelineConfig::new(LayoutConfig::default(layout), config);
            if let Some(path) = orient {
                doc_config = doc_config.with_orientation(OrientConfig::rapid_orientation_v2(path));
            }
            if let Some(path) = table {
                doc_config = doc_config.with_table_recognizer(TableStructureConfig::slanet_plus(path));
            }
            let mut pipeline = DocPipeline::new(doc_config)?;
            eprintln!("Processing image...");
            let image = imread(&cli.image)?;
            let (result, corrected) = pipeline.run_with_image(&image)?;
            if let Some(path) = save_corrected {
                imwrite(&path, &corrected)?;
            }
            eprintln!("✓ Document analysis completed");
            if markdown {
                println!("{}", result.to_markdown());
            } else {
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
            return Ok(());
        }
        RunPlan::Ocr { orient } => {
            if let Some(path) = &orient {
                config.orient = Some(OrientConfig::rapid_orientation_v2(path.clone()));
                config.global.use_orient = true;
            }
            orient.is_some()
        }
    };

    let mut ocr = RustO::initialize(config)?;

    // Run OCR
    eprintln!("Processing image...");
    let detected = ocr.detect_text(
        &ImageSource::Path(cli.image.clone()),
        &OcrRunOptions { orientation: orient_enabled.then_some(true), ..OcrRunOptions::default() },
    )?;
    let DetectTextResult::Structured(results) = detected else {
        unreachable!("default OCR output is structured lines");
    };
    eprintln!("✓ OCR completed");

    // Output projected text results.
    match cli.format {
        OutputFormat::Json => println!("{}", serde_json::to_string_pretty(&results)?),
        OutputFormat::Text | OutputFormat::TextOrdered => {
            for result in &results {
                println!("{}", result.text);
            }
        }
        OutputFormat::Tsv => {
            for result in &results {
                let box_points = result.box_points;
                println!(
                    "{}\t{:.3}\t{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1}",
                    result.text,
                    result.score,
                    box_points[0].0,
                    box_points[0].1,
                    box_points[1].0,
                    box_points[1].1,
                    box_points[2].0,
                    box_points[2].1,
                    box_points[3].0,
                    box_points[3].1,
                );
            }
        }
        OutputFormat::Document => unreachable!("rejected by plan_run without --enable-layout"),
        OutputFormat::Markdown => {
            println!("```");
            for result in &results {
                println!("{}", result.text);
            }
            println!("```");
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(extra: &[&str]) -> Result<Cli, clap::Error> {
        let mut a = vec!["rusto", "--det-model", "d", "--rec-model", "r", "--dict", "k", "img.png"];
        a.extend_from_slice(extra);
        Cli::try_parse_from(a)
    }
    #[test]
    fn existing_invocation_is_plain_ocr() {
        assert!(matches!(plan_run(&parse(&[]).unwrap()), Ok(RunPlan::Ocr { orient: None })));
    }
    #[test]
    fn enable_orient_uses_default_model() {
        let Ok(RunPlan::Ocr { orient: Some(p) }) = plan_run(&parse(&["--enable-orient"]).unwrap()) else { panic!() };
        assert_eq!(p, PathBuf::from("models/DocOCR/orient_rapidai_v2.rten"));
    }
    #[test]
    fn document_needs_layout() {
        assert_eq!(
            plan_run(&parse(&["--format", "document"]).unwrap()).err().unwrap(),
            "--format document requires --enable-layout"
        );
    }
    #[test]
    fn document_mode_collects_models() {
        let Ok(RunPlan::Document { table: Some(t), markdown: false, save_corrected: Some(s), .. }) = plan_run(
            &parse(&["--enable-layout", "--enable-table", "--format", "document", "--save-corrected", "c.png"]).unwrap(),
        ) else {
            panic!()
        };
        assert_eq!((t, s), (PathBuf::from("models/DocOCR/table_slanetplus.rten"), PathBuf::from("c.png")));
    }
    #[test]
    fn save_corrected_extension_is_checked_before_model_loading() {
        let bmp = parse(&["--enable-layout", "--format", "document", "--save-corrected", "c.bmp"]).unwrap();
        assert!(plan_run(&bmp).err().unwrap().contains("--save-corrected"));
        let none = parse(&["--enable-layout", "--format", "document", "--save-corrected", "c"]).unwrap();
        assert!(plan_run(&none).is_err());
        for ok in ["c.PNG", "c.jpg", "c.JPEG"] {
            let cli = parse(&["--enable-layout", "--format", "document", "--save-corrected", ok]).unwrap();
            assert!(matches!(plan_run(&cli), Ok(RunPlan::Document { save_corrected: Some(_), .. })), "{ok}");
        }
    }
    #[test]
    fn dependent_flags_are_rejected_by_clap() {
        assert!(parse(&["--table-model", "t.rten"]).is_err());
        assert!(parse(&["--enable-table"]).is_err());
        assert!(parse(&["--save-corrected", "c.png"]).is_err());
        assert!(parse(&["--orient-model", "o.rten"]).is_err());
    }
    #[test]
    fn layout_mode_rejects_tsv() {
        assert!(plan_run(&parse(&["--enable-layout", "--format", "tsv"]).unwrap()).is_err());
    }
    #[test]
    fn layout_markdown_and_orient_are_planned() {
        let Ok(RunPlan::Document { orient: Some(_), markdown: true, table: None, .. }) =
            plan_run(&parse(&["--enable-layout", "--enable-orient", "--format", "markdown"]).unwrap())
        else {
            panic!()
        };
    }
}
