use std::path::Path;

#[cfg(feature = "use-opencv")]
use opencv::{core, core::Point2f, imgproc, prelude::*};

#[cfg(not(feature = "use-opencv"))]
use crate::image_impl::{self, Mat, Point2f, Rect};

use crate::config::InitializeConfig;
use crate::engine::EngineError;
use crate::layout::{drop_nested_text_regions, BBox, LayoutDetector, LayoutRegion, LayoutType, RegionRole};
use crate::orient::{decide_orientation, rotate_point_back, OrientClassifier, Orientation, PageOrientation};
use crate::rusto_ocr::RustO;
use crate::table::{TableDetector, TableDetectorConfig, TableStructure, TableStructureConfig, TableStructureRecognizer};
use crate::table_decode::{assign_lines_to_cells, reading_order};
use crate::types::{Frame, LayoutConfig, OrientConfig};
use crate::TextResult;
use serde::Serialize;
use std::time::{Duration, Instant};

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

/// A region in reading order with its recognised content. All coordinates
/// (box, lines, table cells) are in the corrected page, see [`DocPipeline::run_with_image`].
#[derive(Debug, Clone, Serialize)]
pub struct DocRegion {
    /// Position in reading order; anchor `#region-{index}`.
    pub index: usize,
    pub label: LayoutType,
    pub bbox: BBox,
    pub score: f32,
    pub text: Option<String>,
    /// OCR lines of the region. Empty for a recognised table: its lines live in
    /// the cells and in `TableStructure::unassigned_lines`.
    pub lines: Vec<TextResult>,
    /// Set for a table region whose structure was recognised; `text` is then `None`.
    pub table: Option<TableStructure>,
}

pub struct DocPipeline {
    orient: Option<OrientClassifier>,
    orient_min_confidence: f32,
    layout: LayoutDetector,
    ocr: RustO,
    /// Constructed when configured but never run: the table structure model
    /// runs on the layout `Table` regions (spec section 6.4), so a separate
    /// table detector has no role yet. The field keeps the model loaded and
    /// validated so a later strategy can use it.
    _table_detector: Option<TableDetector>,
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

/// How the pipeline processes a layout region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Route {
    /// OCR the crop.
    Ocr,
    /// OCR the crop rotated both ways and keep the better reading.
    VerticalOcr,
    /// Table structure model plus OCR of the crop.
    Table,
    /// No recognition.
    Skip,
}

fn route(role: RegionRole, table_configured: bool) -> Route {
    match role {
        RegionRole::VerticalText => Route::VerticalOcr,
        // Without a table model the table text is still read as plain text.
        RegionRole::Table if table_configured => Route::Table,
        r if r.needs_ocr() || r == RegionRole::Table => Route::Ocr,
        _ => Route::Skip,
    }
}

/// Mean recognition score of `lines`; 0.0 when there are none.
fn mean_score(lines: &[TextResult]) -> f32 {
    if lines.is_empty() {
        return 0.0;
    }
    lines.iter().map(|l| l.score).sum::<f32>() / lines.len() as f32
}

/// Keep the clockwise reading unless the counter-clockwise one scores
/// strictly higher on [`mean_score`].
fn pick_vertical(clockwise: Vec<TextResult>, counter_clockwise: Vec<TextResult>) -> Vec<TextResult> {
    if mean_score(&counter_clockwise) > mean_score(&clockwise) {
        counter_clockwise
    } else {
        clockwise
    }
}

/// Map lines found in a crop rotated by `applied_degrees` clockwise
/// (`rotated_w x rotated_h` pixels) back into the unrotated crop, then add the
/// crop origin `offset`. `lines` must be in rotated-crop coordinates.
fn unrotate_lines(
    mut lines: Vec<TextResult>,
    applied_degrees: u16,
    rotated_w: f32,
    rotated_h: f32,
    offset: (i32, i32),
) -> Vec<TextResult> {
    let (dx, dy) = (offset.0 as f32, offset.1 as f32);
    for line in &mut lines {
        line.box_points = line.box_points.map(|(x, y)| {
            let (bx, by) = rotate_point_back(x, y, applied_degrees, rotated_w, rotated_h);
            (bx + dx, by + dy)
        });
        line.frame = Frame::from_points(&line.box_points);
    }
    lines
}

fn join_texts(lines: &[TextResult]) -> String {
    lines.iter().map(|l| l.text.as_str()).collect::<Vec<_>>().join(" ")
}

/// Read `crop` after rotating it 90 degrees clockwise and 90 degrees
/// counter-clockwise (vertical text), keep the reading with the higher
/// [`mean_score`] (tie: clockwise) and return its lines in the unrotated
/// crop's coordinates shifted by `offset`. `ocr` reads one rotated crop and
/// returns lines in that crop's coordinates.
fn ocr_both_rotations(
    crop: &Mat,
    offset: (i32, i32),
    mut ocr: impl FnMut(&Mat) -> Result<Vec<TextResult>, EngineError>,
) -> Result<Vec<TextResult>, EngineError> {
    // `Orientation::Rotate270.rotate_image` turns the image 90 degrees clockwise,
    // `Rotate90` 90 degrees counter-clockwise.
    let mut read = |orientation: Orientation, applied: u16| -> Result<Vec<TextResult>, EngineError> {
        let rotated = orientation.rotate_image(crop)?;
        let (rw, rh) = mat_size(&rotated)?;
        Ok(unrotate_lines(ocr(&rotated)?, applied, rw as f32, rh as f32, offset))
    };
    let clockwise = read(Orientation::Rotate270, 90)?;
    let counter_clockwise = read(Orientation::Rotate90, 270)?;
    Ok(pick_vertical(clockwise, counter_clockwise))
}

/// Clamp `bbox` to a `page_w x page_h` page. Returns `(x, y, width, height)`,
/// or `None` when nothing of the box lies on the page.
fn clamp_crop(bbox: &BBox, page_w: i32, page_h: i32) -> Option<(i32, i32, u32, u32)> {
    let x0 = bbox.x_min.max(0);
    let y0 = bbox.y_min.max(0);
    let x1 = bbox.x_max.min(page_w);
    let y1 = bbox.y_max.min(page_h);
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    Some((x0, y0, (x1 - x0) as u32, (y1 - y0) as u32))
}

/// Time, region count and first error of the regions that went through one
/// processing branch (plain OCR or tables). Durations of the branches add up
/// without overlap because each region is timed in exactly one branch.
#[derive(Debug, Default)]
struct BranchStats {
    elapsed: Duration,
    total: usize,
    ok: usize,
    first_error: Option<String>,
}

impl BranchStats {
    fn record(&mut self, elapsed: Duration, outcome: Result<(), &EngineError>) {
        self.elapsed += elapsed;
        self.total += 1;
        match outcome {
            Ok(()) => self.ok += 1,
            Err(e) => {
                self.first_error.get_or_insert_with(|| e.to_string());
            }
        }
    }

    /// `Failed` when any region failed, else `Skipped` with `empty_reason`
    /// when the branch saw no region and a reason is given, else `Ran`.
    fn state(&self, empty_reason: Option<&str>) -> StageState {
        if let Some(first) = &self.first_error {
            return StageState::Failed {
                reason: format!("{} of {} regions failed: {}", self.total - self.ok, self.total, first),
            };
        }
        match empty_reason {
            Some(reason) if self.total == 0 => StageState::Skipped { reason: reason.to_string() },
            _ => StageState::Ran { elapsed_ms: self.elapsed.as_secs_f64() * 1000.0, items: self.ok },
        }
    }
}

/// OCR lines of a table crop and the outcome of the structure model.
struct TableOutcome {
    lines: Vec<TextResult>,
    structure: Result<TableStructure, EngineError>,
}

#[cfg(feature = "use-opencv")]
fn mat_size(img: &Mat) -> Result<(i32, i32), EngineError> {
    Ok((img.cols(), img.rows()))
}

#[cfg(not(feature = "use-opencv"))]
fn mat_size(img: &Mat) -> Result<(i32, i32), EngineError> {
    let size = img.size().map_err(|e| EngineError::ImageError(e.to_string()))?;
    Ok((size.width, size.height))
}

impl DocPipeline {
    /// Fails with `EngineError::Engine` under the `use-opencv` feature: the
    /// document pipeline needs the pure Rust image backend.
    pub fn new(config: DocPipelineConfig) -> Result<Self, EngineError> {
        validate_config(&config)?;
        #[cfg(feature = "use-opencv")]
        return Err(EngineError::Engine("DocPipeline is not supported with the OpenCV backend".to_string()));

        #[cfg(not(feature = "use-opencv"))]
        {
            let orient = config.orient.map(OrientClassifier::new).transpose()?;
            let layout = LayoutDetector::new(config.layout)?;
            let ocr = RustO::initialize(config.ocr)?;
            let table_detector = config.table_detector.map(TableDetector::new).transpose()?;
            let table_recognizer = config.table_recognizer.map(TableStructureRecognizer::new).transpose()?;

            Ok(Self {
                orient,
                orient_min_confidence: config.orient_min_confidence,
                layout,
                ocr,
                _table_detector: table_detector,
                table_recognizer,
            })
        }
    }

    pub fn run<P: AsRef<Path>>(&mut self, image_path: P) -> Result<DocResult, EngineError> {
        let img = crate::image_impl::imread(image_path)?;
        Ok(self.run_with_image(&img)?.0)
    }

    /// Run the pipeline on a page image. Returns the result and the corrected
    /// page: the input rotated by `result.orientation.applied_degrees`
    /// clockwise. Every box in the result (regions, lines, table cells) is in
    /// the pixels of the corrected page, so callers must crop or draw on the
    /// returned `Mat`, not on the input.
    ///
    /// Stages run as orientation, layout, then per region OCR or table
    /// recognition. A failure of one region does not abort the page; it shows
    /// in the stage status. Orientation and layout failures: a failed
    /// orientation stage is reported and the input page is used as it is; a
    /// failed layout stage returns the error.
    pub fn run_with_image(&mut self, image: &Mat) -> Result<(DocResult, Mat), EngineError> {
        // 1. Orientation
        let orient_start = Instant::now();
        let mut orientation = None;
        let mut corrected: Option<Mat> = None;
        let orientation_state = match self.orient.as_mut() {
            None => StageState::Skipped { reason: "orientation model not configured".to_string() },
            Some(clf) => match clf.classify(image) {
                Err(e) => StageState::Failed { reason: e.to_string() },
                Ok(out) => {
                    let decided = decide_orientation(&out, self.orient_min_confidence);
                    let rotated = if decided.applied_degrees != 0 {
                        out.orientation.rotate_image(image).map(Some)
                    } else {
                        Ok(None)
                    };
                    match rotated {
                        Err(e) => StageState::Failed { reason: e.to_string() },
                        Ok(page) => {
                            corrected = page;
                            orientation = Some(decided);
                            StageState::Ran { elapsed_ms: orient_start.elapsed().as_secs_f64() * 1000.0, items: 1 }
                        }
                    }
                }
            },
        };
        let corrected = corrected.unwrap_or_else(|| image.clone());
        let (page_w, page_h) = mat_size(&corrected)?;

        // 2. Layout; the model order is the reading order.
        let layout_start = Instant::now();
        let detected = self.layout.detect(&corrected)?;
        let layout_regions = drop_nested_text_regions(detected.regions);
        let layout_state = StageState::Ran {
            elapsed_ms: layout_start.elapsed().as_secs_f64() * 1000.0,
            items: layout_regions.len(),
        };

        // 3. Per region
        let table_configured = self.table_recognizer.is_some();
        let mut ocr_stats = BranchStats::default();
        let mut table_stats = BranchStats::default();
        let mut regions = Vec::with_capacity(layout_regions.len());
        let mut blocks = Vec::with_capacity(layout_regions.len());
        for (index, layout_region) in layout_regions.into_iter().enumerate() {
            let mut region = DocRegion {
                index,
                label: layout_region.layout_type,
                bbox: layout_region.bbox,
                score: layout_region.confidence,
                text: None,
                lines: Vec::new(),
                table: None,
            };
            let started = Instant::now();
            let route = route(region.label.role(), table_configured);
            match route {
                Route::Skip => {}
                Route::Ocr | Route::VerticalOcr => {
                    let outcome = self.ocr_region(&corrected, &region.bbox, route == Route::VerticalOcr);
                    ocr_stats.record(started.elapsed(), outcome.as_ref().map(|_| ()));
                    if let Ok(lines) = outcome {
                        region.text = Some(join_texts(&lines));
                        region.lines = lines;
                    }
                }
                Route::Table => {
                    let outcome = self.process_table(&corrected, &region.bbox);
                    let result = match &outcome {
                        Ok(o) => o.structure.as_ref().map(|_| ()),
                        Err(e) => Err(e),
                    };
                    table_stats.record(started.elapsed(), result);
                    match outcome {
                        Ok(TableOutcome { structure: Ok(table), .. }) => region.table = Some(table),
                        // The structure failed but the text was read: keep it as plain text.
                        Ok(TableOutcome { lines, structure: Err(_) }) => {
                            region.text = Some(join_texts(&lines));
                            region.lines = lines;
                        }
                        Err(_) => {}
                    }
                }
            }
            blocks.push(DocBlock {
                region: LayoutRegion {
                    bbox: region.bbox,
                    layout_type: region.label,
                    confidence: region.score,
                    reading_order: layout_region.reading_order,
                },
                text: region.text.clone(),
                html: region.table.as_ref().map(TableStructure::to_html),
            });
            regions.push(region);
        }

        let table_state = if !table_configured {
            StageState::Skipped { reason: "table model not configured".to_string() }
        } else {
            table_stats.state(Some("no table regions"))
        };
        let stages = vec![
            StageStatus { stage: Stage::Orientation, state: orientation_state },
            StageStatus { stage: Stage::Layout, state: layout_state },
            StageStatus { stage: Stage::Ocr, state: ocr_stats.state(None) },
            StageStatus { stage: Stage::Table, state: table_state },
        ];

        let result = DocResult {
            width: page_w.max(0) as u32,
            height: page_h.max(0) as u32,
            orientation,
            regions,
            stages,
            blocks,
        };
        Ok((result, corrected))
    }

    /// Crop `bbox` from the page. Returns the crop and its origin in the page.
    fn crop_region(&self, page: &Mat, bbox: &BBox) -> Result<(Mat, i32, i32), EngineError> {
        let (page_w, page_h) = mat_size(page)?;
        let (x, y, w, h) = clamp_crop(bbox, page_w, page_h)
            .ok_or_else(|| EngineError::ImageError("Invalid crop region".to_string()))?;
        Ok((crop_mat(page, x, y, w, h)?, x, y))
    }

    /// OCR one crop with page orientation off (the pipeline already corrected
    /// the page) and return lines shifted by `offset`. The detector returns
    /// boxes sorted only by integer top-left y, so words of one text line can
    /// interleave; the lines are put in reading order (bands of equal height
    /// top to bottom, each band left to right).
    fn ocr_lines(&mut self, crop: &Mat, offset: (i32, i32)) -> Result<Vec<TextResult>, EngineError> {
        let options = crate::OcrRunOptions { orientation: Some(false), ..Default::default() };
        let ocr = self.ocr.run_on_mat_with_options(crop, &options)?;
        let mut lines = lines_from_ocr(&ocr.boxes, &ocr.txts, &ocr.scores, offset);
        reading_order(&mut lines);
        Ok(lines)
    }

    /// OCR a text region; `vertical` tries the crop rotated 90 degrees
    /// clockwise and counter-clockwise. Lines are in page coordinates.
    fn ocr_region(&mut self, page: &Mat, bbox: &BBox, vertical: bool) -> Result<Vec<TextResult>, EngineError> {
        let (crop, x, y) = self.crop_region(page, bbox)?;
        if !vertical {
            return self.ocr_lines(&crop, (x, y));
        }
        ocr_both_rotations(&crop, (x, y), |rotated| self.ocr_lines(rotated, (0, 0)))
    }

    /// Process a table region: OCR the crop once, recognise the structure in
    /// page pixels and assign the OCR lines to the cells. `Err` only when the
    /// crop or the OCR fails; a structure failure comes back inside the outcome
    /// together with the OCR lines.
    fn process_table(&mut self, page: &Mat, bbox: &BBox) -> Result<TableOutcome, EngineError> {
        let (crop, x, y) = self.crop_region(page, bbox)?;
        let lines = self.ocr_lines(&crop, (x, y))?;
        let recognizer = self
            .table_recognizer
            .as_mut()
            .ok_or_else(|| EngineError::Engine("table recognizer is not configured".to_string()))?;
        let structure = recognizer.recognize_at(&crop, x, y).map(|mut structure| {
            assign_lines_to_cells(&mut structure, lines.clone());
            structure
        });
        Ok(TableOutcome { lines, structure })
    }
}

#[cfg(feature = "use-opencv")]
fn crop_mat(img: &Mat, x: i32, y: i32, w: u32, h: u32) -> Result<Mat, EngineError> {
    let cropped = Mat::roi(img, core::Rect::new(x, y, w as i32, h as i32))?;
    let mut dst = Mat::default();
    cropped.copy_to(&mut dst)?;
    Ok(dst)
}

#[cfg(not(feature = "use-opencv"))]
fn crop_mat(img: &Mat, x: i32, y: i32, w: u32, h: u32) -> Result<Mat, EngineError> {
    image_impl::crop(img, Rect { x, y, width: w, height: h }).map_err(|e| EngineError::ImageError(e.to_string()))
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
    /// Markdown of the page, regions in `index` order. Footnotes follow the
    /// body as `[^n]` definitions, header/footer/number regions come last under
    /// `## Page furniture`, and a caption moves behind the nearest preceding
    /// image, chart or table. Empty `regions` give an empty string.
    pub fn to_markdown(&self) -> String {
        let mut regions: Vec<&DocRegion> = self.regions.iter().collect();
        regions.sort_by_key(|r| r.index);

        let mut body: Vec<String> = Vec::new();
        let mut footnotes: Vec<String> = Vec::new();
        let mut furniture: Vec<String> = Vec::new();
        // Insert position for captions: just behind the block of the nearest
        // preceding image, chart or table.
        let mut anchor: Option<usize> = None;

        for r in regions {
            let text = r.text.as_deref().map(str::trim).unwrap_or("");
            let [x0, y0, x1, y1] = [r.bbox.x_min, r.bbox.y_min, r.bbox.x_max, r.bbox.y_max];
            let block = match r.label.role() {
                RegionRole::DocTitle => (!text.is_empty()).then(|| format!("# {text}")),
                RegionRole::SectionTitle => (!text.is_empty()).then(|| format!("## {text}")),
                RegionRole::Paragraph | RegionRole::VerticalText => {
                    (!text.is_empty()).then(|| text.to_string())
                }
                RegionRole::Footnote => {
                    if !text.is_empty() {
                        footnotes.push(format!("[^{}]: {}", footnotes.len() + 1, text));
                    }
                    None
                }
                RegionRole::Caption => {
                    if text.is_empty() {
                        continue;
                    }
                    let caption = format!("*{text}*");
                    match anchor {
                        Some(pos) => {
                            body.insert(pos, caption);
                            anchor = Some(pos + 1);
                        }
                        None => body.push(caption),
                    }
                    continue;
                }
                RegionRole::Table => Self::table_block(r, text),
                RegionRole::Visual => Some(format!("![{}](#region-{})", r.label.name(), r.index)),
                RegionRole::Formula => {
                    Some(format!("```formula\nregion-{} [{x0},{y0},{x1},{y1}]\n```", r.index))
                }
                RegionRole::Algorithm => {
                    Some(format!("```algorithm\nregion-{} [{x0},{y0},{x1},{y1}]\n```", r.index))
                }
                RegionRole::FurnitureText => {
                    if !text.is_empty() {
                        furniture.push(format!("- {}: {}", r.label.name(), text));
                    }
                    None
                }
                RegionRole::FurnitureImage => {
                    furniture.push(format!(
                        "- {}: ![{}](#region-{})",
                        r.label.name(),
                        r.label.name(),
                        r.index
                    ));
                    None
                }
            };
            let is_anchor = matches!(r.label, LayoutType::Image | LayoutType::Chart | LayoutType::Table);
            if let Some(b) = block {
                body.push(b);
                if is_anchor {
                    anchor = Some(body.len());
                }
            } else if is_anchor {
                anchor = None;
            }
        }

        if !footnotes.is_empty() {
            body.push(footnotes.join("\n"));
        }
        if !furniture.is_empty() {
            body.push(format!("## Page furniture\n\n{}", furniture.join("\n")));
        }
        if body.is_empty() {
            return String::new();
        }
        let mut md = body.join("\n\n");
        md.push('\n');
        md
    }

    fn table_block(r: &DocRegion, text: &str) -> Option<String> {
        if let Some(t) = &r.table {
            if t.rows.iter().any(|row| !row.cells.is_empty()) {
                return Some(if t.has_spans() {
                    t.to_html()
                } else {
                    t.to_markdown().trim_end_matches('\n').to_string()
                });
            }
            let lines: Vec<&str> =
                t.unassigned_lines.iter().map(|l| l.text.trim()).filter(|s| !s.is_empty()).collect();
            if !lines.is_empty() {
                return Some(lines.join(" "));
            }
        }
        (!text.is_empty()).then(|| text.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> DocPipelineConfig {
        DocPipelineConfig::new(LayoutConfig::default("l.rten".into()), InitializeConfig::ppv5("d", "r", "k"))
    }

    fn reg(index: usize, label: LayoutType, text: &str) -> DocRegion {
        DocRegion {
            index,
            label,
            bbox: BBox { x_min: 1, y_min: 2, x_max: 3, y_max: 4 },
            score: 0.9,
            text: Some(text.to_string()),
            lines: vec![],
            table: None,
        }
    }

    fn result(regions: Vec<DocRegion>) -> DocResult {
        DocResult { width: 10, height: 20, orientation: None, regions, stages: vec![], blocks: vec![] }
    }

    fn text_line(t: &str) -> TextResult {
        let pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        TextResult { text: t.to_string(), score: 1.0, frame: Frame::from_points(&pts), box_points: pts }
    }

    fn grid(col_span: usize) -> TableStructure {
        use crate::table::{TableCell, TableRow};
        let cell = |row: usize, col: usize, span: usize| TableCell {
            bbox: crate::table::BBox { x_min: 0, y_min: 0, x_max: 1, y_max: 1 },
            row,
            col,
            row_span: 1,
            col_span: span,
            text: format!("{row}{col}"),
            score: 1.0,
        };
        let first = if col_span > 1 { vec![cell(0, 0, col_span)] } else { vec![cell(0, 0, 1), cell(0, 1, 1)] };
        TableStructure {
            rows: vec![TableRow { cells: first }, TableRow { cells: vec![cell(1, 0, 1), cell(1, 1, 1)] }],
            num_rows: 2,
            num_cols: 2,
            unassigned_lines: vec![],
        }
    }

    #[test]
    fn markdown_of_empty_result_is_empty() { assert_eq!(result(vec![]).to_markdown(), ""); }
    #[test]
    fn markdown_headings_and_paragraphs() {
        let r = result(vec![reg(0, LayoutType::DocTitle, "Заглавие"), reg(1, LayoutType::ParagraphTitle, "Раздел"), reg(2, LayoutType::Text, "Абзац.")]);
        assert_eq!(r.to_markdown(), "# Заглавие\n\n## Раздел\n\nАбзац.\n");
    }
    #[test]
    fn caption_moves_after_nearest_preceding_visual() {
        let r = result(vec![reg(0, LayoutType::Image, ""), reg(1, LayoutType::Text, "т"), reg(2, LayoutType::FigureTitle, "Фиг. 1")]);
        assert_eq!(r.to_markdown(), "![image](#region-0)\n\n*Фиг. 1*\n\nт\n");
    }
    #[test]
    fn caption_without_preceding_visual_stays_in_place() {
        let r = result(vec![reg(0, LayoutType::FigureTitle, "Фиг. 1"), reg(1, LayoutType::Image, "")]);
        assert_eq!(r.to_markdown(), "*Фиг. 1*\n\n![image](#region-1)\n");
    }
    #[test]
    fn footnotes_and_page_furniture_go_last() {
        let r = result(vec![reg(0, LayoutType::Header, "Горен"), reg(1, LayoutType::Text, "Тяло"), reg(2, LayoutType::Footnote, "Бележка"), reg(3, LayoutType::Number, "3")]);
        assert_eq!(r.to_markdown(), "Тяло\n\n[^1]: Бележка\n\n## Page furniture\n\n- header: Горен\n- number: 3\n");
    }
    #[test]
    fn formula_block_references_region() {
        assert_eq!(result(vec![reg(4, LayoutType::DisplayFormula, "")]).to_markdown(), "```formula\nregion-4 [1,2,3,4]\n```\n");
    }
    #[test]
    fn table_with_span_renders_html_without_span_gfm() {
        let mut r = result(vec![reg(0, LayoutType::Table, "")]);
        r.regions[0].table = Some(grid(2));
        let md = r.to_markdown();
        assert!(md.starts_with("<table>") && md.contains("colspan=\"2\""), "{md}");
        r.regions[0].table = Some(grid(1));
        assert!(r.to_markdown().starts_with("| "));
    }
    #[test]
    fn table_without_cells_renders_unassigned_text() {
        let mut r = result(vec![reg(0, LayoutType::Table, "")]);
        r.regions[0].table = Some(TableStructure { rows: vec![], num_rows: 0, num_cols: 0, unassigned_lines: vec![text_line("а"), text_line("б")] });
        assert_eq!(r.to_markdown(), "а б\n");
    }
    #[test]
    fn markdown_skips_blank_text_and_handles_missing_text() {
        let mut none = reg(0, LayoutType::Text, "");
        none.text = None;
        let r = result(vec![none, reg(1, LayoutType::Text, "  "), reg(2, LayoutType::Footnote, " "), reg(3, LayoutType::Header, ""), reg(4, LayoutType::Table, "")]);
        assert_eq!(r.to_markdown(), "");
    }
    #[test]
    fn footnotes_are_numbered_in_reading_order_and_furniture_images_listed() {
        let r = result(vec![reg(0, LayoutType::Footnote, "а"), reg(1, LayoutType::Algorithm, ""), reg(2, LayoutType::VisionFootnote, "б"), reg(3, LayoutType::HeaderImage, "")]);
        assert_eq!(
            r.to_markdown(),
            "```algorithm\nregion-1 [1,2,3,4]\n```\n\n[^1]: а\n[^2]: б\n\n## Page furniture\n\n- header_image: ![header_image](#region-3)\n"
        );
    }


    fn scored(text: &str, score: f32) -> TextResult {
        let mut l = text_line(text);
        l.score = score;
        l
    }

    #[test]
    fn mean_score_is_zero_for_no_lines_and_averages_otherwise() {
        assert_eq!(mean_score(&[]), 0.0);
        assert!((mean_score(&[scored("a", 0.5), scored("b", 1.0)]) - 0.75).abs() < 1e-6);
    }

    #[test]
    fn routing_follows_role_and_table_model() {
        use RegionRole as R;
        for r in [R::DocTitle, R::SectionTitle, R::Paragraph, R::Footnote, R::Caption, R::FurnitureText] {
            assert_eq!(route(r, false), Route::Ocr, "{r:?}");
            assert_eq!(route(r, true), Route::Ocr, "{r:?}");
        }
        assert_eq!(route(R::VerticalText, false), Route::VerticalOcr);
        assert_eq!(route(R::Table, true), Route::Table);
        assert_eq!(route(R::Table, false), Route::Ocr);
        for r in [R::Visual, R::Formula, R::Algorithm, R::FurnitureImage] {
            assert_eq!(route(r, true), Route::Skip, "{r:?}");
        }
    }

    #[test]
    fn vertical_reading_prefers_higher_mean_score_and_clockwise_on_tie() {
        let cw = vec![scored("cw", 0.6)];
        let ccw = vec![scored("ccw", 0.9)];
        assert_eq!(pick_vertical(cw.clone(), ccw.clone())[0].text, "ccw");
        assert_eq!(pick_vertical(ccw.clone(), cw.clone())[0].text, "ccw");
        assert_eq!(pick_vertical(vec![scored("cw", 0.8)], vec![scored("ccw", 0.8)])[0].text, "cw");
        assert!(pick_vertical(vec![], vec![]).is_empty());
        assert_eq!(pick_vertical(vec![], vec![scored("ccw", 0.1)])[0].text, "ccw");
    }

    #[test]
    fn unrotate_lines_maps_both_rotations_back_and_adds_offset() {
        // Crop 100 wide x 40 high, point (10, 5) in the crop.
        let line = |p: (f32, f32)| {
            let mut l = text_line("x");
            l.box_points = [p; 4];
            l.frame = Frame::from_points(&l.box_points);
            l
        };
        // Rotated 90 clockwise (40 x 100): (x, y) -> (h - y, x) = (35, 10).
        let back = unrotate_lines(vec![line((35.0, 10.0))], 90, 40.0, 100.0, (200, 300));
        assert_eq!(back[0].box_points[0], (210.0, 305.0));
        assert_eq!(back[0].frame, Frame::from_points(&back[0].box_points));
        // Rotated 90 counter-clockwise: (x, y) -> (y, w - x) = (5, 90).
        let back = unrotate_lines(vec![line((5.0, 90.0))], 270, 40.0, 100.0, (0, 0));
        assert_eq!(back[0].box_points[2], (10.0, 5.0));
    }

    #[cfg(not(feature = "use-opencv"))]
    #[test]
    fn both_rotations_are_read_and_the_better_one_is_mapped_back() {
        let crop = Mat::new(image::DynamicImage::new_rgb8(100, 40)); // 100 wide, 40 high
        let at = |text: &str, score: f32, p: (f32, f32)| {
            let mut l = scored(text, score);
            l.box_points = [p; 4];
            l.frame = Frame::from_points(&l.box_points);
            vec![l]
        };
        let mut sizes = Vec::new();
        let mut call = 0;
        // Clockwise reading scores 0.5, counter-clockwise 0.9. The same crop point
        // (10, 5) is (35, 10) in the clockwise and (5, 90) in the counter-clockwise rotation.
        let lines = ocr_both_rotations(&crop, (200, 300), |m| {
            sizes.push(mat_size(m).unwrap());
            call += 1;
            Ok(if call == 1 { at("cw", 0.5, (35.0, 10.0)) } else { at("ccw", 0.9, (5.0, 90.0)) })
        })
        .unwrap();
        assert_eq!(sizes, [(40, 100), (40, 100)]);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].text, "ccw");
        assert_eq!(lines[0].box_points[0], (210.0, 305.0));

        // Tie: clockwise wins, and an error of either reading is passed on.
        let mut call = 0;
        let lines = ocr_both_rotations(&crop, (0, 0), |_| {
            call += 1;
            Ok(at(if call == 1 { "cw" } else { "ccw" }, 0.7, (1.0, 1.0)))
        })
        .unwrap();
        assert_eq!(lines[0].text, "cw");
        assert!(ocr_both_rotations(&crop, (0, 0), |_| Err(EngineError::Engine("x".into()))).is_err());
    }

    /// A paragraph crop turned 90 degrees counter-clockwise is read upright by
    /// the clockwise pass; the boxes land in the turned crop, shifted by `offset`.
    #[cfg(not(feature = "use-opencv"))]
    #[test]
    #[ignore = "requires RUSTO_TEST_MODELS with Cyrillic OCR models"]
    fn vertical_crop_is_read_in_the_better_rotation() {
        let Ok(dir) = std::env::var("RUSTO_TEST_MODELS") else {
            eprintln!("skipped: set RUSTO_TEST_MODELS to a directory with det.onnx, rec.onnx, dict.txt");
            return;
        };
        let dir = std::path::PathBuf::from(dir);
        let page = image_impl::imread(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/doc/page_text.png")).unwrap();
        let upright = image_impl::crop(&page, Rect { x: 92, y: 262, width: 837, height: 150 }).unwrap();
        let sideways = Orientation::Rotate90.rotate_image(&upright).unwrap(); // 150 wide, 837 high
        assert_eq!(mat_size(&sideways).unwrap(), (150, 837));

        let mut ocr = RustO::initialize(InitializeConfig::ppv5(dir.join("det.onnx"), dir.join("rec.onnx"), dir.join("dict.txt"))).unwrap();
        let options = crate::OcrRunOptions { orientation: Some(false), ..Default::default() };
        let lines = ocr_both_rotations(&sideways, (1000, 2000), |m| {
            let out = ocr.run_on_mat_with_options(m, &options)?;
            let mut l = lines_from_ocr(&out.boxes, &out.txts, &out.scores, (0, 0));
            reading_order(&mut l);
            Ok(l)
        })
        .unwrap();
        let text = join_texts(&lines);
        eprintln!("{text}");
        assert!(text.starts_with("Настоящият документ описва"), "{text}");
        for l in &lines {
            let f = &l.frame;
            assert!(f.left >= 990.0 && f.left + f.width <= 1160.0, "{f:?}");
            assert!(f.top >= 1990.0 && f.top + f.height <= 2850.0, "{f:?}");
        }
        // Turning counter-clockwise puts the top of the upright crop at the left edge.
        assert!(lines[0].frame.left < 1000.0 + 150.0 / 2.0, "{:?}", lines[0].frame);
        assert!(lines.last().unwrap().frame.left > 1000.0 + 150.0 / 2.0, "{:?}", lines.last().unwrap().frame);
    }

    #[test]
    fn crop_is_clamped_to_the_page() {
        let b = |x0, y0, x1, y1| BBox { x_min: x0, y_min: y0, x_max: x1, y_max: y1 };
        assert_eq!(clamp_crop(&b(10, 20, 50, 60), 100, 100), Some((10, 20, 40, 40)));
        assert_eq!(clamp_crop(&b(-5, -5, 30, 30), 100, 100), Some((0, 0, 30, 30)));
        assert_eq!(clamp_crop(&b(90, 90, 150, 150), 100, 100), Some((90, 90, 10, 10)));
        assert_eq!(clamp_crop(&b(100, 10, 120, 50), 100, 100), None);
        assert_eq!(clamp_crop(&b(10, 10, 10, 50), 100, 100), None);
    }

    #[test]
    fn branch_stats_assemble_stage_states() {
        let ms = Duration::from_millis;
        let mut s = BranchStats::default();
        assert_eq!(s.state(None), StageState::Ran { elapsed_ms: 0.0, items: 0 });
        assert_eq!(s.state(Some("no table regions")), StageState::Skipped { reason: "no table regions".into() });
        s.record(ms(10), Ok(()));
        s.record(ms(5), Ok(()));
        assert_eq!(s.state(Some("no table regions")), StageState::Ran { elapsed_ms: 15.0, items: 2 });
        let boom = EngineError::Engine("boom".into());
        s.record(ms(1), Err(&boom));
        s.record(ms(1), Err(&EngineError::Engine("second".into())));
        match s.state(None) {
            StageState::Failed { reason } => {
                assert!(reason.starts_with("2 of 4 regions failed: "), "{reason}");
                assert!(reason.ends_with("boom"), "{reason}");
            }
            other => panic!("{other:?}"),
        }
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
