use ndarray::{Array2, Array3, ArrayD, ArrayView2};
use serde::Serialize;
use std::collections::HashMap;
use std::time::Instant;

#[cfg(feature = "use-opencv")]
use opencv::prelude::*;

#[cfg(not(feature = "use-opencv"))]
use crate::image_impl::Mat;

use crate::engine::{load_stage_session, EngineError, EngineSession, InferenceSession};
use crate::types::LayoutConfig;

/// PP-DocLayoutV3 class names in model order.
///
/// Source: PaddleX `PP-DocLayoutV3/inference.yml` `label_list` (Apache-2.0).
pub const PP_DOCLAYOUT_V3_LABELS: [&str; 25] = [
    "abstract",
    "algorithm",
    "aside_text",
    "chart",
    "content",
    "display_formula",
    "doc_title",
    "figure_title",
    "footer",
    "footer_image",
    "footnote",
    "formula_number",
    "header",
    "header_image",
    "image",
    "inline_formula",
    "number",
    "paragraph_title",
    "reference",
    "reference_content",
    "seal",
    "table",
    "text",
    "vertical_text",
    "vision_footnote",
];

/// Pixel box with exclusive-style `min`/`max` corners in page coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BBox {
    pub x_min: i32,
    pub y_min: i32,
    pub x_max: i32,
    pub y_max: i32,
}

impl BBox {
    /// Area in pixels; 0 for an empty or inverted box.
    pub fn area(&self) -> i64 {
        let w = (self.x_max as i64 - self.x_min as i64).max(0);
        let h = (self.y_max as i64 - self.y_min as i64).max(0);
        w * h
    }

    /// Area of the overlap with `other`; 0 when the boxes do not overlap.
    pub fn intersection_area(&self, other: &BBox) -> i64 {
        let w = (self.x_max.min(other.x_max) as i64 - self.x_min.max(other.x_min) as i64).max(0);
        let h = (self.y_max.min(other.y_max) as i64 - self.y_min.max(other.y_min) as i64).max(0);
        w * h
    }
}

/// Layout element types.
///
/// The first ten variants come from the legacy 10-class model. The V3 labels
/// reuse `Text`, `Table`, `Header`, `Footer` and `Reference`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum LayoutType {
    Text = 0,
    Title = 1,
    Figure = 2,
    FigureCaption = 3,
    Table = 4,
    TableCaption = 5,
    Header = 6,
    Footer = 7,
    Reference = 8,
    Equation = 9,
    Abstract,
    Algorithm,
    AsideText,
    Chart,
    Content,
    DisplayFormula,
    DocTitle,
    FigureTitle,
    FooterImage,
    Footnote,
    FormulaNumber,
    HeaderImage,
    Image,
    InlineFormula,
    Number,
    ParagraphTitle,
    ReferenceContent,
    Seal,
    VerticalText,
    VisionFootnote,
}

/// How downstream stages treat a region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum RegionRole {
    DocTitle,
    SectionTitle,
    Paragraph,
    Footnote,
    VerticalText,
    Caption,
    Table,
    Visual,
    Formula,
    Algorithm,
    FurnitureText,
    FurnitureImage,
}

impl RegionRole {
    /// True when the region content is text that goes through OCR.
    pub fn needs_ocr(&self) -> bool {
        matches!(
            self,
            RegionRole::DocTitle
                | RegionRole::SectionTitle
                | RegionRole::Paragraph
                | RegionRole::Footnote
                | RegionRole::VerticalText
                | RegionRole::Caption
                | RegionRole::FurnitureText
        )
    }
}

impl LayoutType {
    /// Legacy 10-class model mapping.
    pub fn from_class(class_id: usize) -> Option<Self> {
        match class_id {
            0 => Some(LayoutType::Text),
            1 => Some(LayoutType::Title),
            2 => Some(LayoutType::Figure),
            3 => Some(LayoutType::FigureCaption),
            4 => Some(LayoutType::Table),
            5 => Some(LayoutType::TableCaption),
            6 => Some(LayoutType::Header),
            7 => Some(LayoutType::Footer),
            8 => Some(LayoutType::Reference),
            9 => Some(LayoutType::Equation),
            _ => None,
        }
    }

    /// PP-DocLayoutV3 class index (position in [`PP_DOCLAYOUT_V3_LABELS`]).
    pub fn from_v3_class(class_id: usize) -> Option<Self> {
        use LayoutType::*;
        Some(match class_id {
            0 => Abstract,
            1 => Algorithm,
            2 => AsideText,
            3 => Chart,
            4 => Content,
            5 => DisplayFormula,
            6 => DocTitle,
            7 => FigureTitle,
            8 => Footer,
            9 => FooterImage,
            10 => Footnote,
            11 => FormulaNumber,
            12 => Header,
            13 => HeaderImage,
            14 => Image,
            15 => InlineFormula,
            16 => Number,
            17 => ParagraphTitle,
            18 => Reference,
            19 => ReferenceContent,
            20 => Seal,
            21 => Table,
            22 => Text,
            23 => VerticalText,
            24 => VisionFootnote,
            _ => return None,
        })
    }

    pub fn name(&self) -> &'static str {
        match self {
            LayoutType::Text => "text",
            LayoutType::Title => "title",
            LayoutType::Figure => "figure",
            LayoutType::FigureCaption => "figure_caption",
            LayoutType::Table => "table",
            LayoutType::TableCaption => "table_caption",
            LayoutType::Header => "header",
            LayoutType::Footer => "footer",
            LayoutType::Reference => "reference",
            LayoutType::Equation => "equation",
            LayoutType::Abstract => "abstract",
            LayoutType::Algorithm => "algorithm",
            LayoutType::AsideText => "aside_text",
            LayoutType::Chart => "chart",
            LayoutType::Content => "content",
            LayoutType::DisplayFormula => "display_formula",
            LayoutType::DocTitle => "doc_title",
            LayoutType::FigureTitle => "figure_title",
            LayoutType::FooterImage => "footer_image",
            LayoutType::Footnote => "footnote",
            LayoutType::FormulaNumber => "formula_number",
            LayoutType::HeaderImage => "header_image",
            LayoutType::Image => "image",
            LayoutType::InlineFormula => "inline_formula",
            LayoutType::Number => "number",
            LayoutType::ParagraphTitle => "paragraph_title",
            LayoutType::ReferenceContent => "reference_content",
            LayoutType::Seal => "seal",
            LayoutType::VerticalText => "vertical_text",
            LayoutType::VisionFootnote => "vision_footnote",
        }
    }

    /// The downstream role of this label.
    pub fn role(&self) -> RegionRole {
        match self {
            LayoutType::DocTitle => RegionRole::DocTitle,
            LayoutType::ParagraphTitle | LayoutType::Title => RegionRole::SectionTitle,
            LayoutType::Text
            | LayoutType::Content
            | LayoutType::Abstract
            | LayoutType::AsideText
            | LayoutType::Reference
            | LayoutType::ReferenceContent => RegionRole::Paragraph,
            LayoutType::Footnote | LayoutType::VisionFootnote => RegionRole::Footnote,
            LayoutType::VerticalText => RegionRole::VerticalText,
            LayoutType::FigureTitle | LayoutType::FigureCaption | LayoutType::TableCaption => RegionRole::Caption,
            LayoutType::Table => RegionRole::Table,
            LayoutType::Image | LayoutType::Chart | LayoutType::Seal | LayoutType::Figure => RegionRole::Visual,
            LayoutType::DisplayFormula
            | LayoutType::InlineFormula
            | LayoutType::FormulaNumber
            | LayoutType::Equation => RegionRole::Formula,
            LayoutType::Algorithm => RegionRole::Algorithm,
            LayoutType::Header | LayoutType::Footer | LayoutType::Number => RegionRole::FurnitureText,
            LayoutType::HeaderImage | LayoutType::FooterImage => RegionRole::FurnitureImage,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct LayoutRegion {
    pub bbox: BBox,
    pub layout_type: LayoutType,
    pub confidence: f32,
    /// Model reading-order value (column 7 of the PP-DocLayoutV3 output).
    /// Only the relative order is meaningful; the values are not contiguous.
    pub reading_order: f32,
}

pub struct LayoutOutput {
    pub regions: Vec<LayoutRegion>,
    pub elapse: f64,
}

/// Build the PP-DocLayoutV3 inputs: `image` `[1,3,T,T]`, `im_shape` `[[T,T]]`
/// and `scale_factor` `[[T/h, T/w]]`.
///
/// Pre-processing follows PaddleX `inference.yml`: resize to `T x T` without
/// keeping the aspect ratio (`interp: 2`, cubic), RGB, `p / 255` (mean 0, std 1).
#[cfg(not(feature = "use-opencv"))]
pub(crate) fn layout_inputs(image: &Mat, target_size: i32) -> Result<HashMap<String, ArrayD<f32>>, EngineError> {
    use crate::image_impl::{resize, Size, INTER_CUBIC};

    let size = image.size()?;
    let (h, w) = (size.height, size.width);
    if h <= 0 || w <= 0 || target_size <= 0 {
        return Err(EngineError::Preprocess(format!(
            "invalid layout input: image {w}x{h}, target size {target_size}"
        )));
    }
    let t = target_size as usize;

    let mut resized = Mat::default();
    resize(image, &mut resized, Size::new(target_size, target_size), INTER_CUBIC)?;

    let hw = t * t;
    let mut data = vec![0.0f32; 3 * hw];
    for y in 0..t {
        for x in 0..t {
            let pixel = resized.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                data[c * hw + y * t + x] = pixel[c] as f32 / 255.0;
            }
        }
    }
    let image_t = Array3::from_shape_vec((3, t, t), data)
        .map_err(|e| EngineError::Preprocess(e.to_string()))?
        .insert_axis(ndarray::Axis(0));

    let tf = target_size as f32;
    let im_shape = Array2::from_shape_vec((1, 2), vec![tf, tf]).map_err(EngineError::ShapeError)?;
    let scale_factor =
        Array2::from_shape_vec((1, 2), vec![tf / h as f32, tf / w as f32]).map_err(EngineError::ShapeError)?;

    let mut inputs = HashMap::new();
    inputs.insert("image".to_string(), image_t.into_dyn());
    inputs.insert("im_shape".to_string(), im_shape.into_dyn());
    inputs.insert("scale_factor".to_string(), scale_factor.into_dyn());
    Ok(inputs)
}

#[cfg(feature = "use-opencv")]
pub(crate) fn layout_inputs(_image: &Mat, _target_size: i32) -> Result<HashMap<String, ArrayD<f32>>, EngineError> {
    Err(EngineError::Engine(
        "layout detection is not supported with the OpenCV backend".to_string(),
    ))
}

/// Select the `[N,7]` detection tensor
/// (`class, score, x1, y1, x2, y2, reading_order`) from the model outputs.
pub(crate) fn find_layout_rows(outputs: &HashMap<String, ArrayD<f32>>) -> Result<ArrayView2<'_, f32>, EngineError> {
    for tensor in outputs.values() {
        if tensor.ndim() == 2 && tensor.shape()[1] == 7 {
            return tensor
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(EngineError::ShapeError);
        }
    }
    Err(EngineError::OutputError(
        "layout model output must be [N,7] (PP-DocLayoutV3): no such tensor among outputs".to_string(),
    ))
}

/// Decode `[class, score, x1, y1, x2, y2, reading_order]` rows.
///
/// Boxes are already in original-image pixels (the model applies
/// `scale_factor`), so there is no rescale and no NMS. Rows below
/// `conf_thresh`, with an unknown class, or with an empty box after clamping to
/// the page are dropped. The result is ordered by the model reading order,
/// ties by `(y_min, x_min)`.
pub(crate) fn decode_layout_rows(
    rows: ArrayView2<f32>,
    conf_thresh: f32,
    img_w: i32,
    img_h: i32,
) -> Result<Vec<LayoutRegion>, EngineError> {
    if rows.ncols() != 7 {
        return Err(EngineError::OutputError(format!(
            "layout model output must be [N,7] (PP-DocLayoutV3), got {} columns",
            rows.ncols()
        )));
    }
    let (fw, fh) = (img_w as f32, img_h as f32);
    let mut regions = Vec::new();
    for row in rows.outer_iter() {
        let score = row[1];
        if score.is_nan() || score < conf_thresh {
            continue;
        }
        let class = row[0];
        if !class.is_finite() || class < 0.0 {
            continue;
        }
        let Some(layout_type) = LayoutType::from_v3_class(class as usize) else {
            continue;
        };
        let x_min = row[2].clamp(0.0, fw) as i32;
        let y_min = row[3].clamp(0.0, fh) as i32;
        let x_max = row[4].clamp(0.0, fw) as i32;
        let y_max = row[5].clamp(0.0, fh) as i32;
        if x_max <= x_min || y_max <= y_min {
            continue;
        }
        regions.push(LayoutRegion {
            bbox: BBox { x_min, y_min, x_max, y_max },
            layout_type,
            confidence: score,
            reading_order: row[6],
        });
    }
    regions.sort_by(|a, b| {
        a.reading_order
            .total_cmp(&b.reading_order)
            .then(a.bbox.y_min.cmp(&b.bbox.y_min))
            .then(a.bbox.x_min.cmp(&b.bbox.x_min))
    });
    Ok(regions)
}

/// Drop OCR-bound regions that lie (at least 80 % of their area) inside a
/// `Table`, `Image`, `Chart` or `Seal` region; the container is OCR'd or
/// recognised as a whole.
pub fn drop_nested_text_regions(regions: Vec<LayoutRegion>) -> Vec<LayoutRegion> {
    let containers: Vec<BBox> = regions
        .iter()
        .filter(|r| {
            matches!(
                r.layout_type,
                LayoutType::Table | LayoutType::Image | LayoutType::Chart | LayoutType::Seal
            )
        })
        .map(|r| r.bbox)
        .collect();
    regions
        .into_iter()
        .filter(|r| {
            if !r.layout_type.role().needs_ocr() {
                return true;
            }
            let area = r.bbox.area();
            if area == 0 {
                return true;
            }
            // intersection / area >= 0.8 in integers
            !containers.iter().any(|c| r.bbox.intersection_area(c) * 10 >= area * 8)
        })
        .collect()
}

pub struct LayoutDetector {
    session: EngineSession,
    config: LayoutConfig,
}

impl LayoutDetector {
    pub fn new(config: LayoutConfig) -> Result<Self, EngineError> {
        let session = load_stage_session("layout", &config.model_path, &config.engine_cfg)?;
        Ok(Self { session, config })
    }

    /// Construct layout detector with a custom inference session
    pub fn with_custom_session(config: LayoutConfig, session: impl InferenceSession + 'static) -> Self {
        Self {
            session: EngineSession::from_custom(session),
            config,
        }
    }

    pub fn detect(&mut self, image: &Mat) -> Result<LayoutOutput, EngineError> {
        let start = Instant::now();

        #[cfg(feature = "use-opencv")]
        let (orig_h, orig_w) = (image.rows(), image.cols());

        #[cfg(not(feature = "use-opencv"))]
        let size = image.size()?;
        #[cfg(not(feature = "use-opencv"))]
        let (orig_h, orig_w) = (size.height, size.width);

        let inputs = layout_inputs(image, self.config.target_size)?;
        let outputs = self.session.run_with_inputs(inputs)?;
        let rows = find_layout_rows(&outputs)?;
        let regions = decode_layout_rows(rows, self.config.conf_thresh, orig_w, orig_h)?;

        Ok(LayoutOutput {
            regions,
            elapse: start.elapsed().as_secs_f64(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::LayoutConfig;

    fn solid(w: u32, h: u32, v: u8) -> Mat {
        Mat::new(image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(w, h, image::Rgb([v, v, v]))))
    }

    #[test]
    fn v3_label_map_is_complete() {
        for (i, label) in PP_DOCLAYOUT_V3_LABELS.iter().enumerate() {
            let t = LayoutType::from_v3_class(i).unwrap();
            assert_eq!(t.name(), *label);
            assert_eq!(serde_json::to_string(&t).unwrap(), format!("\"{label}\""));
        }
        assert!(LayoutType::from_v3_class(25).is_none());
        assert_eq!(LayoutType::from_v3_class(21).unwrap().role(), RegionRole::Table);
        assert_eq!(LayoutType::from_v3_class(16).unwrap().role(), RegionRole::FurnitureText);
    }

    #[test]
    fn role_map_follows_spec() {
        use LayoutType as L;
        use RegionRole as R;
        let expect = [
            (L::DocTitle, R::DocTitle), (L::ParagraphTitle, R::SectionTitle),
            (L::Text, R::Paragraph), (L::Content, R::Paragraph), (L::Abstract, R::Paragraph),
            (L::AsideText, R::Paragraph), (L::Reference, R::Paragraph), (L::ReferenceContent, R::Paragraph),
            (L::Footnote, R::Footnote), (L::VisionFootnote, R::Footnote), (L::VerticalText, R::VerticalText),
            (L::FigureTitle, R::Caption), (L::Table, R::Table),
            (L::Image, R::Visual), (L::Chart, R::Visual), (L::Seal, R::Visual),
            (L::DisplayFormula, R::Formula), (L::InlineFormula, R::Formula), (L::FormulaNumber, R::Formula),
            (L::Algorithm, R::Algorithm), (L::Header, R::FurnitureText), (L::Footer, R::FurnitureText),
            (L::Number, R::FurnitureText), (L::HeaderImage, R::FurnitureImage), (L::FooterImage, R::FurnitureImage),
            (L::Title, R::SectionTitle), (L::Figure, R::Visual), (L::FigureCaption, R::Caption),
            (L::TableCaption, R::Caption), (L::Equation, R::Formula),
        ];
        for (t, r) in expect {
            assert_eq!(t.role(), r, "{t:?}");
        }
        let ocr: Vec<_> = [R::DocTitle, R::SectionTitle, R::Paragraph, R::Footnote, R::VerticalText, R::Caption, R::FurnitureText]
            .iter().map(|r| r.needs_ocr()).collect();
        assert!(ocr.iter().all(|b| *b));
        for r in [R::Table, R::Visual, R::Formula, R::Algorithm, R::FurnitureImage] {
            assert!(!r.needs_ocr(), "{r:?}");
        }
    }

    fn row(c: f32, s: f32, b: [f32; 4], ro: f32) -> [f32; 7] { [c, s, b[0], b[1], b[2], b[3], ro] }
    fn rows(r: &[[f32; 7]]) -> ndarray::Array2<f32> { ndarray::Array2::from_shape_vec((r.len(), 7), r.concat()).unwrap() }

    #[test]
    fn decode_keeps_threshold_orders_by_column_seven_and_does_not_rescale() {
        let a = rows(&[row(22.0, 0.95, [232., 440., 2297., 692.], 186.0), row(17.0, 0.50, [233., 1197., 822., 1256.], 20.0),
                       row(22.0, 0.49, [0., 0., 10., 10.], 1.0), row(30.0, 0.99, [0., 0., 10., 10.], 2.0)]);
        let r = decode_layout_rows(a.view(), 0.5, 2480, 3508).unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!(r[0].layout_type, LayoutType::ParagraphTitle);
        assert_eq!((r[1].bbox.x_min, r[1].bbox.x_max), (232, 2297));
    }

    #[test]
    fn decode_breaks_order_ties_by_position() {
        let a = rows(&[row(22.0, 0.9, [50., 100., 60., 110.], 5.0), row(22.0, 0.9, [10., 100., 20., 110.], 5.0),
                       row(22.0, 0.9, [90., 10., 99., 20.], 5.0)]);
        let r = decode_layout_rows(a.view(), 0.5, 200, 200).unwrap();
        let xs: Vec<_> = r.iter().map(|r| r.bbox.x_min).collect();
        assert_eq!(xs, vec![90, 10, 50]);
    }

    #[test]
    fn decode_clamps_and_drops_degenerate() {
        let a = rows(&[row(22.0, 0.9, [-5., 10., 3000., 50.], 1.0), row(22.0, 0.9, [100., 100., 100., 200.], 2.0)]);
        let r = decode_layout_rows(a.view(), 0.5, 2480, 3508).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!((r[0].bbox.x_min, r[0].bbox.x_max), (0, 2480));
    }

    #[test]
    fn decode_rejects_six_column_output() {
        let mut out = HashMap::new();
        out.insert("o".to_string(), ndarray::Array2::<f32>::zeros((3, 6)).into_dyn());
        let msg = find_layout_rows(&out).err().unwrap().to_string();
        assert!(msg.contains("[N,7]"), "{msg}");
        let six = ndarray::Array2::<f32>::zeros((3, 6));
        assert!(decode_layout_rows(six.view(), 0.5, 100, 100).is_err());
    }

    #[test]
    fn find_layout_rows_picks_the_rank_two_seven_column_tensor() {
        let mut out = HashMap::new();
        out.insert("masks".to_string(), ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(&[3, 7, 7])));
        out.insert("rows".to_string(), ndarray::Array2::<f32>::zeros((4, 7)).into_dyn());
        assert_eq!(find_layout_rows(&out).unwrap().dim(), (4, 7));
    }

    #[test]
    fn layout_inputs_tiny_image() {
        let m = layout_inputs(&solid(10, 20, 255), 800).unwrap();
        assert_eq!(m["image"].shape(), &[1, 3, 800, 800]);
        assert_eq!(m["im_shape"].iter().copied().collect::<Vec<_>>(), vec![800.0, 800.0]);
        assert_eq!(m["scale_factor"].iter().copied().collect::<Vec<_>>(), vec![40.0, 80.0]);
        assert!((m["image"][[0, 1, 5, 5]] - 1.0).abs() < 1e-6);
    }

    fn reg(t: LayoutType, b: [i32; 4]) -> LayoutRegion {
        LayoutRegion { bbox: BBox { x_min: b[0], y_min: b[1], x_max: b[2], y_max: b[3] }, layout_type: t, confidence: 0.9, reading_order: 0.0 }
    }

    #[test]
    fn nested_text_inside_table_is_dropped_overlapping_text_kept() {
        let out = drop_nested_text_regions(vec![
            reg(LayoutType::Table, [0, 0, 100, 100]),
            reg(LayoutType::Text, [10, 10, 90, 50]),       // 100 % inside: dropped
            reg(LayoutType::Text, [80, 0, 180, 100]),      // 20 % inside: kept
            reg(LayoutType::Text, [90, 0, 190, 100]),      // overlaps the previous text: kept
            reg(LayoutType::Text, [0, 0, 100, 125]),       // exactly 80 % inside: dropped
        ]);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn nested_text_just_below_eighty_percent_is_kept() {
        // 79 % inside the image region
        let out = drop_nested_text_regions(vec![
            reg(LayoutType::Image, [0, 0, 100, 100]),
            reg(LayoutType::Text, [0, 0, 100, 126]),
            reg(LayoutType::Text, [0, 0, 100, 124]),       // 80.6 % inside: dropped
        ]);
        assert_eq!(out.len(), 2);
        assert_eq!(out[1].bbox.y_max, 126);
    }

    #[test]
    fn nesting_drop_only_applies_to_ocr_regions_and_known_containers() {
        let out = drop_nested_text_regions(vec![
            reg(LayoutType::Seal, [0, 0, 100, 100]),
            reg(LayoutType::Chart, [200, 0, 300, 100]),
            reg(LayoutType::Image, [400, 0, 500, 100]),
            reg(LayoutType::Image, [10, 10, 20, 20]),          // visual inside seal: kept
            reg(LayoutType::Text, [210, 10, 290, 90]),         // inside chart: dropped
            reg(LayoutType::Header, [410, 10, 490, 90]),       // furniture text inside image: dropped
            reg(LayoutType::Algorithm, [10, 10, 90, 90]),      // not OCR-bound: kept
            reg(LayoutType::Text, [600, 600, 700, 700]),       // inside nothing: kept
            reg(LayoutType::Figure, [600, 0, 700, 100]),       // legacy figure is not a container
            reg(LayoutType::Text, [610, 10, 690, 90]),         // kept
        ]);
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn layout_config_defaults_follow_inference_yml() {
        let c = LayoutConfig::default("m.rten");
        assert_eq!((c.target_size, c.conf_thresh, c.mean, c.std), (800, 0.5, [0.0; 3], [1.0; 3]));
    }

    #[test]
    fn bbox_geometry() {
        let a = BBox { x_min: 0, y_min: 0, x_max: 10, y_max: 10 };
        let b = BBox { x_min: 5, y_min: 5, x_max: 20, y_max: 8 };
        assert_eq!(a.area(), 100);
        assert_eq!(a.intersection_area(&b), 15);
        assert_eq!(a.intersection_area(&BBox { x_min: 20, y_min: 20, x_max: 30, y_max: 30 }), 0);
    }
}
