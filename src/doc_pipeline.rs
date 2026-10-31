use std::path::Path;

#[cfg(feature = "use-opencv")]
use opencv::{core, core::Point2f, imgproc, prelude::*};

#[cfg(not(feature = "use-opencv"))]
use crate::image_impl::{self, Mat, Point2f, Rect};

use crate::config::InitializeConfig;
use crate::engine::EngineError;
use crate::layout::{BBox, LayoutDetector, LayoutRegion, LayoutType, RegionRole};
use crate::orient::PageOrientation;
use crate::rusto_ocr::RustO;
use crate::table::{TableDetector, TableDetectorConfig, TableStructure, TableStructureConfig, TableStructureRecognizer};
use crate::table_decode::assign_lines_to_cells;
use crate::types::{Frame, LayoutConfig, OrientConfig};
use crate::TextResult;
use serde::Serialize;
use std::time::Instant;

/// Configuration of the document pipeline.
///
/// `#[non_exhaustive]`: build it with [`DocPipelineConfig::new`] and the
/// `with_*` builders.
#[non_exhaustive]
pub struct DocPipelineConfig {
    pub layout: LayoutConfig,
    pub ocr: InitializeConfig,
    pub table_detector: Option<TableDetectorConfig>,
    pub table_recognizer: Option<TableStructureConfig>,
    /// Page orientation model; `None` skips the orientation stage.
    pub orient: Option<OrientConfig>,
    /// The orientation correction is applied only at or above this confidence
    /// (0.0..=1.0). Independent of `OrientConfig::conf_thresh`.
    pub orient_min_confidence: f32,
}

impl DocPipelineConfig {
    /// Layout and OCR are required; the other stages are off by default and
    /// `orient_min_confidence` is 0.6.
    pub fn new(layout: LayoutConfig, ocr: InitializeConfig) -> Self {
        Self {
            layout,
            ocr,
            table_detector: None,
            table_recognizer: None,
            orient: None,
            orient_min_confidence: 0.6,
        }
    }

    pub fn with_orientation(mut self, orient: OrientConfig) -> Self {
        self.orient = Some(orient);
        self
    }

    pub fn with_orient_min_confidence(mut self, min_confidence: f32) -> Self {
        self.orient_min_confidence = min_confidence;
        self
    }

    pub fn with_table_recognizer(mut self, recognizer: TableStructureConfig) -> Self {
        self.table_recognizer = Some(recognizer);
        self
    }

    pub fn with_table_detector(mut self, detector: TableDetectorConfig) -> Self {
        self.table_detector = Some(detector);
        self
    }
}

fn validate_config(cfg: &DocPipelineConfig) -> Result<(), EngineError> {
    if !(0.0..=1.0).contains(&cfg.orient_min_confidence) {
        // Also rejects NaN: no comparison with NaN is true.
        return Err(EngineError::Preprocess(
            "orient_min_confidence must be within 0..=1".to_string(),
        ));
    }
    Ok(())
}

/// A pipeline stage, as reported in [`StageStatus`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    Orientation,
    Layout,
    Ocr,
    Table,
}

/// Outcome of one stage.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum StageState {
    Ran { elapsed_ms: f64, items: usize },
    Skipped { reason: String },
    Failed { reason: String },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StageStatus {
    pub stage: Stage,
    #[serde(flatten)]
    pub state: StageState,
}

/// A region in reading order with its recognised content.
#[derive(Debug, Clone, Serialize)]
pub struct DocRegion {
    /// Position in reading order; anchor `#region-{index}`.
    pub index: usize,
    pub label: LayoutType,
    pub bbox: BBox,
    pub score: f32,
    pub text: Option<String>,
    pub lines: Vec<TextResult>,
    pub table: Option<TableStructure>,
}

pub struct DocPipeline {
    layout: LayoutDetector,
    ocr: RustO,
    // Table structure runs on the layout table region; the detector is not used yet.
    #[allow(dead_code)]
    table_detector: Option<TableDetector>,
    table_recognizer: Option<TableStructureRecognizer>,
}

#[derive(Debug, Clone)]
pub struct DocBlock {
    pub region: LayoutRegion,
    pub text: Option<String>,
    pub html: Option<String>, // For tables: HTML markup
}

/// Result of a document run. `width`/`height` are the corrected page size.
#[derive(Debug, Clone, Serialize)]
pub struct DocResult {
    pub width: u32,
    pub height: u32,
    pub orientation: Option<PageOrientation>,
    pub regions: Vec<DocRegion>,
    pub stages: Vec<StageStatus>,
    #[serde(skip)]
    pub blocks: Vec<DocBlock>,
}

impl DocPipeline {
    pub fn new(config: DocPipelineConfig) -> Result<Self, EngineError> {
        validate_config(&config)?;
        let layout = LayoutDetector::new(config.layout)?;
        let ocr = RustO::initialize(config.ocr)?;
        
        // Initialize table recognition components if configured
        let table_detector = if let Some(detector_config) = config.table_detector {
            Some(TableDetector::new(detector_config)?)
        } else {
            None
        };
        
        let table_recognizer = if let Some(recognizer_config) = config.table_recognizer {
            Some(TableStructureRecognizer::new(recognizer_config)?)
        } else {
            None
        };
        
        Ok(Self { 
            layout, 
            ocr,
            table_detector,
            table_recognizer,
        })
    }

    pub fn run<P: AsRef<Path>>(&mut self, image_path: P) -> Result<DocResult, EngineError> {
        #[cfg(feature = "use-opencv")]
        use crate::image_impl::imread;
        #[cfg(not(feature = "use-opencv"))]
        use crate::image_impl::imread;

        let img = imread(image_path)?;
        #[cfg(feature = "use-opencv")]
        let (page_w, page_h) = (img.cols() as u32, img.rows() as u32);
        #[cfg(not(feature = "use-opencv"))]
        let (page_w, page_h) = {
            let size = img.size()?;
            (size.width as u32, size.height as u32)
        };

        // 1. Layout Analysis
        let layout_start = Instant::now();
        let layout_output = self.layout.detect(&img)?;
        let mut regions = layout_output.regions;
        let layout_status = StageStatus {
            stage: Stage::Layout,
            state: StageState::Ran {
                elapsed_ms: layout_start.elapsed().as_secs_f64() * 1000.0,
                items: regions.len(),
            },
        };

        // Sort regions: top-to-bottom, left-to-right
        regions.sort_by(|a, b| {
            let y_diff = (a.bbox.y_min - b.bbox.y_min).abs();
            if y_diff < 20 {
                // Tolerance for same line
                a.bbox.x_min.cmp(&b.bbox.x_min)
            } else {
                a.bbox.y_min.cmp(&b.bbox.y_min)
            }
        });

        let mut blocks = Vec::new();
        let ocr_start = Instant::now();
        let mut table_start: Option<Instant> = None;
        let mut table_items = 0usize;

        // 2. Process each region
        for region in regions {
            match region.layout_type {
                // Interim routing by role; rewritten in the document pipeline task.
                lt if lt.role().needs_ocr() || lt.role() == RegionRole::Formula => {
                    // Crop and OCR
                    let crop = self.crop_image(&img, &region)?;
                    let ocr_res = self.ocr.run_on_mat_with_options(
                        &crop,
                        &crate::OcrRunOptions::default(),
                    )?;

                    // Combine text lines
                    let text = ocr_res.txts.join(" ");
                    blocks.push(DocBlock {
                        region,
                        text: Some(text),
                        html: None,
                    });
                }
                LayoutType::Figure => {
                    // Figures - just store region
                    blocks.push(DocBlock { 
                        region, 
                        text: None,
                        html: None,
                    });
                }
                LayoutType::Table => {
                    // Tables - run recognition if configured
                    if self.table_recognizer.is_some() {
                        table_start.get_or_insert_with(Instant::now);
                        table_items += 1;
                        match self.process_table(&img, &region) {
                            Ok(html) => {
                                blocks.push(DocBlock {
                                    region,
                                    text: None,
                                    html: Some(html),
                                });
                            }
                            Err(e) => {
                                eprintln!("Table recognition failed: {}", e);
                                blocks.push(DocBlock { region, text: None, html: None });
                            }
                        }
                    } else {
                        // No table recognition configured
                        blocks.push(DocBlock { region, text: None, html: None });
                    }
                }
                _ => {
                    blocks.push(DocBlock { region, text: None, html: None });
                }
            }
        }

        let ocr_ms = ocr_start.elapsed().as_secs_f64() * 1000.0;
        let ocr_items = blocks.iter().filter(|b| b.text.is_some()).count();
        let table_configured = self.table_recognizer.is_some();
        let table_status = StageStatus {
            stage: Stage::Table,
            state: if table_configured {
                StageState::Ran {
                    elapsed_ms: table_start.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1000.0),
                    items: table_items,
                }
            } else {
                StageState::Skipped { reason: "table model not configured".to_string() }
            },
        };
        let doc_regions = blocks
            .iter()
            .enumerate()
            .map(|(index, b)| DocRegion {
                index,
                label: b.region.layout_type,
                bbox: b.region.bbox,
                score: b.region.confidence,
                text: b.text.clone(),
                lines: Vec::new(),
                table: None,
            })
            .collect();
        let stages = vec![
            StageStatus {
                stage: Stage::Orientation,
                state: StageState::Skipped { reason: "orientation model not configured".to_string() },
            },
            layout_status,
            StageStatus {
                stage: Stage::Ocr,
                state: StageState::Ran { elapsed_ms: ocr_ms, items: ocr_items },
            },
            table_status,
        ];

        Ok(DocResult {
            width: page_w,
            height: page_h,
            orientation: None,
            regions: doc_regions,
            stages,
            blocks,
        })
    }

    /// Process a table region: OCR the crop once, recognise the structure in
    /// page pixels and assign the OCR lines to the cells.
    fn process_table(&mut self, img: &Mat, region: &LayoutRegion) -> Result<String, EngineError> {
        // `crop_image` clamps the crop to the page, so the crop origin is clamped alike.
        let (x_min, y_min) = (region.bbox.x_min.max(0), region.bbox.y_min.max(0));
        let table_crop = self.crop_image(img, region)?;

        let options = crate::OcrRunOptions { orientation: Some(false), ..Default::default() };
        let ocr = self.ocr.run_on_mat_with_options(&table_crop, &options)?;
        let lines = lines_from_ocr(&ocr.boxes, &ocr.txts, &ocr.scores, (x_min, y_min));

        let recognizer = self
            .table_recognizer
            .as_mut()
            .ok_or_else(|| EngineError::Engine("table recognizer is not configured".to_string()))?;
        let mut structure = recognizer.recognize_at(&table_crop, x_min, y_min)?;
        assign_lines_to_cells(&mut structure, lines);
        Ok(structure.to_html())
    }

    #[cfg(feature = "use-opencv")]
    fn crop_image(&self, img: &Mat, region: &LayoutRegion) -> Result<Mat, EngineError> {
        let rect = core::Rect::new(
            region.bbox.x_min,
            region.bbox.y_min,
            region.bbox.x_max - region.bbox.x_min,
            region.bbox.y_max - region.bbox.y_min,
        );

        // Ensure rect is within image bounds
        let img_w = img.cols();
        let img_h = img.rows();
        let x = rect.x.max(0).min(img_w - 1);
        let y = rect.y.max(0).min(img_h - 1);
        let w = rect.width.min(img_w - x);
        let h = rect.height.min(img_h - y);

        if w <= 0 || h <= 0 {
            return Err(EngineError::ImageError("Invalid crop region".to_string()));
        }

        let valid_rect = core::Rect::new(x, y, w, h);
        let cropped = Mat::roi(img, valid_rect)?;
        let mut dst = Mat::default();
        cropped.copy_to(&mut dst)?;
        Ok(dst)
    }

    #[cfg(not(feature = "use-opencv"))]
    fn crop_image(&self, img: &Mat, region: &LayoutRegion) -> Result<Mat, EngineError> {
        let rect = Rect {
            x: region.bbox.x_min,
            y: region.bbox.y_min,
            width: (region.bbox.x_max - region.bbox.x_min) as u32,
            height: (region.bbox.y_max - region.bbox.y_min) as u32,
        };

        // Ensure rect is within image bounds
        let size = img.size()?;
        let img_w = size.width;
        let img_h = size.height;

        let x = rect.x.max(0).min(img_w - 1);
        let y = rect.y.max(0).min(img_h - 1);
        let w = rect.width.min((img_w - x) as u32);
        let h = rect.height.min((img_h - y) as u32);

        if w == 0 || h == 0 {
            return Err(EngineError::ImageError("Invalid crop region".to_string()));
        }

        let valid_rect = Rect {
            x,
            y,
            width: w,
            height: h,
        };
        image_impl::crop(img, valid_rect).map_err(|e| EngineError::ImageError(e.to_string()))
    }
}

/// Convert OCR output to text lines. `offset` is added to every point, which
/// maps lines found in a crop back to page coordinates. The frame is the
/// axis-aligned box of the four points. Only as many lines as the shortest of
/// the three slices are returned.
pub(crate) fn lines_from_ocr(
    boxes: &[[Point2f; 4]],
    txts: &[String],
    scores: &[f32],
    offset: (i32, i32),
) -> Vec<TextResult> {
    let (dx, dy) = (offset.0 as f32, offset.1 as f32);
    boxes
        .iter()
        .zip(txts)
        .zip(scores)
        .map(|((quad, text), &score)| {
            let box_points = [0, 1, 2, 3].map(|i| (quad[i].x + dx, quad[i].y + dy));
            TextResult { text: text.clone(), score, frame: Frame::from_points(&box_points), box_points }
        })
        .collect()
}

impl DocResult {
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();
        let mut last_was_caption = false;

        for (idx, block) in self.blocks.iter().enumerate() {
            if let Some(text) = &block.text {
                if text.trim().is_empty() {
                    continue;
                }

                match block.region.layout_type {
                    LayoutType::Title => {
                        // Detect title level based on font size heuristics
                        // For now, use bbox height as proxy for font size
                        let height = block.region.bbox.y_max - block.region.bbox.y_min;
                        let level = if height > 100 {
                            1 // # Main title
                        } else if height > 60 {
                            2 // ## Section
                        } else {
                            3 // ### Subsection
                        };
                        let prefix = "#".repeat(level);
                        md.push_str(&format!("{} {}\n\n", prefix, text.trim()));
                    }
                    LayoutType::Header => {
                        // Headers are typically page headers - can be omitted or included
                        // For RapidDoc compatibility, include them but don't make them prominent
                        md.push_str(&format!("{}\n\n", text.trim()));
                    }
                    LayoutType::Footer => {
                        // Footers (page numbers, etc.) - typically skipped in content
                        // but included for completeness
                        md.push_str(&format!("---\n{}\n\n", text.trim()));
                    }
                    LayoutType::FigureCaption => {
                        // Figure captions should be in italics and on separate line
                        md.push_str(&format!("*{}*\n\n", text.trim()));
                        last_was_caption = true;
                    }
                    LayoutType::TableCaption => {
                        // Table captions should be in italics
                        md.push_str(&format!("*{}*\n\n", text.trim()));
                        last_was_caption = true;
                    }
                    LayoutType::Equation => {
                        // Equations - treat as regular text for now (LaTeX formatting ignored)
                        md.push_str(&format!("{}\n\n", text.trim()));
                    }
                    LayoutType::Reference => {
                        // References - keep as plain text but maybe add indicator
                        md.push_str(&format!("{}\n\n", text.trim()));
                    }
                    LayoutType::Text => {
                        // Check if this looks like a list
                        if Self::is_list_line(text) {
                            // Format as markdown list
                            md.push_str(&format!("{}\n", text.trim()));
                        } else {
                            // Regular paragraph
                            md.push_str(&format!("{}\n\n", text.trim()));
                            last_was_caption = false;
                        }
                    }
                    _ => {
                        md.push_str(&format!("{}\n\n", text.trim()));
                        last_was_caption = false;
                    }
                }
            } else {
                // Blocks without text (figures, tables)
                match block.region.layout_type {
                    LayoutType::Figure => {
                        // Check if next block is a figure caption
                        let has_caption = idx + 1 < self.blocks.len() 
                            && matches!(self.blocks[idx + 1].region.layout_type, LayoutType::FigureCaption);
                        
                        if !has_caption && !last_was_caption {
                            // No caption, show placeholder
                            md.push_str("![Figure]\n\n");
                        } else if !last_was_caption {
                            // Caption comes after, just show image
                            md.push_str("![Figure]\n");
                        }
                        // If caption was before, it's already added
                        last_was_caption = false;
                    }
                    LayoutType::Table => {
                        // Check if we have HTML table markup (from table recognition)
                        if let Some(html) = &block.html {
                            // Insert HTML table directly (RapidDoc style)
                            md.push_str(&format!("\n{}\n\n", html));
                        } else {
                            // No table recognition - show placeholder
                            md.push_str("| Table | (Structure recognition not yet implemented) |\n");
                            md.push_str("|-------|---------------------------------------------|\n\n");
                        }
                        last_was_caption = false;
                    }
                    _ => {}
                }
            }
        }

        md
    }

    /// Detect if a line looks like a list item
    fn is_list_line(text: &str) -> bool {
        let trimmed = text.trim();
        // Check for common list patterns
        trimmed.starts_with("- ")
            || trimmed.starts_with("* ")
            || trimmed.starts_with("• ")
            || (trimmed.len() > 2 
                && trimmed.chars().next().unwrap().is_ascii_digit()
                && (trimmed.chars().nth(1) == Some('.') || trimmed.chars().nth(1) == Some(')')))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> DocPipelineConfig {
        DocPipelineConfig::new(LayoutConfig::default("l.rten".into()), InitializeConfig::ppv5("d", "r", "k"))
    }

    #[test]
    fn config_new_has_spec_defaults() {
        let c = cfg();
        assert!(c.orient.is_none() && c.table_recognizer.is_none() && c.table_detector.is_none());
        assert_eq!(c.orient_min_confidence, 0.6);
        let c = c
            .with_orient_min_confidence(0.7)
            .with_orientation(OrientConfig::rapid_orientation_v2("o.rten".into()));
        assert_eq!(c.orient_min_confidence, 0.7);
        assert!(c.orient.is_some());
    }

    #[test]
    fn config_rejects_out_of_range_orient_confidence() {
        for v in [-0.1, 1.1, f32::NAN] {
            assert!(validate_config(&cfg().with_orient_min_confidence(v)).is_err());
        }
        assert!(validate_config(&cfg().with_orient_min_confidence(1.0)).is_ok());
    }

    #[test]
    fn doc_result_json_shape() {
        let r = DocResult {
            width: 10,
            height: 20,
            orientation: Some(PageOrientation { applied_degrees: 270, detected_degrees: 90, confidence: 0.92 }),
            regions: vec![DocRegion {
                index: 0,
                label: LayoutType::ParagraphTitle,
                bbox: BBox { x_min: 1, y_min: 2, x_max: 3, y_max: 4 },
                score: 0.9,
                text: Some("Раздел".into()),
                lines: vec![],
                table: None,
            }],
            stages: vec![StageStatus {
                stage: Stage::Table,
                state: StageState::Skipped { reason: "table model not configured".into() },
            }],
            blocks: vec![],
        };
        let v: serde_json::Value = serde_json::to_value(&r).unwrap();
        assert_eq!(v["orientation"]["applied_degrees"], 270);
        assert_eq!(v["regions"][0]["label"], "paragraph_title");
        assert_eq!(v["stages"][0]["stage"], "table");
        assert_eq!(v["stages"][0]["state"], "skipped");
        assert!(v.get("blocks").is_none());
    }

    #[test]
    fn stage_status_serialises_each_variant_flat() {
        let ran = serde_json::to_value(StageStatus {
            stage: Stage::Ocr,
            state: StageState::Ran { elapsed_ms: 1.5, items: 3 },
        })
        .unwrap();
        assert_eq!(ran, serde_json::json!({"stage": "ocr", "state": "ran", "elapsed_ms": 1.5, "items": 3}));
        let failed = serde_json::to_value(StageStatus {
            stage: Stage::Orientation,
            state: StageState::Failed { reason: "boom".into() },
        })
        .unwrap();
        assert_eq!(failed, serde_json::json!({"stage": "orientation", "state": "failed", "reason": "boom"}));
        let skipped = serde_json::to_value(StageStatus {
            stage: Stage::Layout,
            state: StageState::Skipped { reason: "x".into() },
        })
        .unwrap();
        assert_eq!(skipped, serde_json::json!({"stage": "layout", "state": "skipped", "reason": "x"}));
    }

    #[cfg(not(feature = "use-opencv"))]
    #[test]
    fn lines_from_ocr_adds_offset_and_builds_frame() {
        let p = |x, y| Point2f { x, y };
        let boxes = [[p(10.0, 5.0), p(40.0, 5.0), p(40.0, 25.0), p(10.0, 25.0)]];
        let lines = lines_from_ocr(&boxes, &["ab".to_string()], &[0.75], (100, 200));
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].text, "ab");
        assert_eq!(lines[0].score, 0.75);
        assert_eq!(lines[0].box_points[0], (110.0, 205.0));
        assert_eq!(lines[0].box_points[2], (140.0, 225.0));
        assert_eq!(lines[0].frame, Frame::new(30.0, 20.0, 205.0, 110.0));
        // Slices of unequal length give the shortest.
        assert!(lines_from_ocr(&boxes, &[], &[0.5], (0, 0)).is_empty());
    }

    /// Whole pipeline of `process_table` on the 3x4 fixture with a real OCR and
    /// the real table model. `RUSTO_TEST_MODELS` must hold `det.onnx`,
    /// `rec.onnx` and a Cyrillic `dict.txt` (PP-OCRv5 detector, Cyrillic recogniser).
    #[cfg(not(feature = "use-opencv"))]
    #[test]
    #[ignore = "requires models/DocOCR and RUSTO_TEST_MODELS with Cyrillic OCR models"]
    fn table_fixture_text_lands_in_cells() {
        use crate::table::TableStructureConfig;

        let Ok(dir) = std::env::var("RUSTO_TEST_MODELS") else {
            eprintln!("skipped: set RUSTO_TEST_MODELS to a directory with det.onnx, rec.onnx, dict.txt");
            return;
        };
        let dir = std::path::PathBuf::from(dir);
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let fixture = root.join("tests/fixtures/doc/table_3x4_merged");
        let truth: serde_json::Value =
            serde_json::from_slice(&std::fs::read(fixture.with_extension("json")).unwrap()).unwrap();
        let img = image_impl::imread(fixture.with_extension("png")).unwrap();

        let mut ocr = RustO::initialize(InitializeConfig::ppv5(dir.join("det.onnx"), dir.join("rec.onnx"), dir.join("dict.txt"))).unwrap();
        let options = crate::OcrRunOptions { orientation: Some(false), ..Default::default() };
        let out = ocr.run_on_mat_with_options(&img, &options).unwrap();
        let lines = lines_from_ocr(&out.boxes, &out.txts, &out.scores, (0, 0));
        let mut rec = TableStructureRecognizer::new(TableStructureConfig::slanet_plus(root.join("models/DocOCR/table_slanetplus.rten"))).unwrap();
        let mut s = rec.recognize_at(&img, 0, 0).unwrap();
        assign_lines_to_cells(&mut s, lines.clone());

        let texts: Vec<Vec<String>> = s.rows.iter().map(|r| r.cells.iter().map(|c| c.text.clone()).collect()).collect();
        eprintln!("grid {}x{}, unassigned {}: {texts:?}", s.num_rows, s.num_cols, s.unassigned_lines.len());
        eprintln!("markdown:\n{}", s.to_markdown());
        assert_eq!((s.num_rows, s.num_cols), (3, 4));
        let expected: Vec<Vec<String>> = truth["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r.as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect())
            .collect();
        // Texts are compared per row in cell order; the merged header cell is one entry.
        assert_eq!(texts, expected);
        assert!(s.unassigned_lines.is_empty());
    }
}
