//! Table detection and SLANet_plus table structure recognition.
//!
//! The SLANet_plus pre-processing (keep-ratio resize to 488, ImageNet
//! normalisation, zero padding to the bottom and right), the structure
//! dictionary construction and the output-head matching are translated from
//! `src/rapid_table.rs` of `bobine` 0.7.3 (<https://github.com/opticsWolf/bobine>),
//! used under its MIT option.
//!
//! MIT License
//!
//! Copyright (c) 2026 opticsWolf
//!
//! Permission is hereby granted, free of charge, to any person obtaining a copy
//! of this software and associated documentation files (the "Software"), to deal
//! in the Software without restriction, including without limitation the rights
//! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//! copies of the Software, and to permit persons to whom the Software is
//! furnished to do so, subject to the following conditions:
//!
//! The above copyright notice and this permission notice shall be included in all
//! copies or substantial portions of the Software.
//!
//! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//! SOFTWARE.

use crate::engine::{load_stage_session, EngineError, EngineSession, InferenceSession};
use crate::types::EngineConfig;
use ndarray::{Array2, Array3, Array4, ArrayD, Axis};
use std::collections::HashMap;
use std::path::PathBuf;

#[cfg(feature = "use-opencv")]
use opencv::prelude::*;

#[cfg(not(feature = "use-opencv"))]
use crate::image_impl::Mat;

/// Bounding box for table cells
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct BBox {
    pub x_min: i32,
    pub y_min: i32,
    pub x_max: i32,
    pub y_max: i32,
}

/// Table cell information
#[derive(Debug, Clone, serde::Serialize)]
pub struct TableCell {
    pub bbox: BBox,
    pub row: usize,
    pub col: usize,
    pub row_span: usize,
    pub col_span: usize,
    pub text: String,
}

/// Table row
#[derive(Debug, Clone, serde::Serialize)]
pub struct TableRow {
    pub cells: Vec<TableCell>,
}

/// Complete table structure
#[derive(Debug, Clone, serde::Serialize)]
pub struct TableStructure {
    pub rows: Vec<TableRow>,
    pub num_rows: usize,
    pub num_cols: usize,
}

impl TableStructure {
    /// Generate HTML table from structure
    pub fn to_html(&self) -> String {
        let mut html = String::from("<table>\n");
        
        for row in &self.rows {
            html.push_str("  <tr>\n");
            for cell in &row.cells {
                let colspan = if cell.col_span > 1 {
                    format!(" colspan=\"{}\"", cell.col_span)
                } else {
                    String::new()
                };
                let rowspan = if cell.row_span > 1 {
                    format!(" rowspan=\"{}\"", cell.row_span)
                } else {
                    String::new()
                };
                
                // Escape HTML special characters
                let text = cell.text
                    .replace('&', "&amp;")
                    .replace('<', "&lt;")
                    .replace('>', "&gt;")
                    .replace('"', "&quot;");
                
                html.push_str(&format!("    <td{}{}>{}</td>\n", colspan, rowspan, text));
            }
            html.push_str("  </tr>\n");
        }
        
        html.push_str("</table>");
        html
    }

    /// Generate markdown table from structure (simpler format, no spans)
    pub fn to_markdown(&self) -> String {
        if self.rows.is_empty() {
            return String::new();
        }

        let mut md = String::new();

        // Header row
        if let Some(first_row) = self.rows.first() {
            md.push_str("| ");
            for cell in &first_row.cells {
                md.push_str(&format!("{} | ", cell.text.trim()));
            }
            md.push('\n');

            // Separator
            md.push('|');
            for _ in &first_row.cells {
                md.push_str("---|");
            }
            md.push('\n');
        }

        // Data rows
        for row in self.rows.iter().skip(1) {
            md.push_str("| ");
            for cell in &row.cells {
                md.push_str(&format!("{} | ", cell.text.trim()));
            }
            md.push('\n');
        }

        md
    }
}

/// Configuration for table detection
#[derive(Debug, Clone)]
pub struct TableDetectorConfig {
    pub model_path: std::path::PathBuf,
    pub conf_threshold: f32,
    pub iou_threshold: f32,
    pub model_type: TableModelType,
}

#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum TableModelType {
    Wireless, // Borderless tables (SLANeXt; not supported by the recogniser)
    Wired,    // Bordered tables (SLANeXt; not supported by the recogniser)
    /// SLANet_plus (PaddleX / RapidAI `slanet-plus`): 488 px keep-ratio input,
    /// 8-value cell location head, 50 structure classes.
    SlanetPlus,
}

impl Default for TableDetectorConfig {
    fn default() -> Self {
        Self {
            model_path: std::path::PathBuf::from("models/DocOCR/rtdetr-wireless.rten"),
            conf_threshold: 0.5,
            iou_threshold: 0.5,
            model_type: TableModelType::Wireless,
        }
    }
}

/// RT-DETR table detector
pub struct TableDetector {
    _session: EngineSession,
    _config: TableDetectorConfig,
}

impl TableDetector {
    pub fn new(config: TableDetectorConfig) -> Result<Self, EngineError> {
        let engine_config = EngineConfig::default();
        let session = EngineSession::from_path(&config.model_path, &engine_config)?;
        
        Ok(Self { _session: session, _config: config })
    }

    pub fn with_custom_session(config: TableDetectorConfig, session: impl InferenceSession + 'static) -> Self {
        Self {
            _session: EngineSession::from_custom(session),
            _config: config,
        }
    }

    /// Detect table cells in an image
    pub fn detect(&mut self, img: &Mat) -> Result<Vec<BBox>, EngineError> {
        // Preprocess image for RT-DETR (640x640)
        let (input_tensor, scale_x, scale_y, orig_w, orig_h) = self.preprocess(img)?;
        
        // Run inference
        let mut inputs = HashMap::new();
        inputs.insert("image".to_string(), input_tensor);
        
        let outputs = self._session.run_with_inputs(inputs)?;
        
        // Postprocess: decode bounding boxes
        self.postprocess(outputs, scale_x, scale_y, orig_w, orig_h)
    }

    fn preprocess(&self, image: &Mat) -> Result<(ArrayD<f32>, f32, f32, i32, i32), EngineError> {
        #[cfg(feature = "use-opencv")]
        let (orig_h, orig_w) = (image.rows(), image.cols());
        
        #[cfg(not(feature = "use-opencv"))]
        let size = image.size()?;
        #[cfg(not(feature = "use-opencv"))]
        let (orig_h, orig_w) = (size.height, size.width);
        
        // RT-DETR expects 640x640 input
        let target_size = 640;
        let scale = (target_size as f32) / (orig_h.max(orig_w) as f32);
        let new_h = (orig_h as f32 * scale) as i32;
        let new_w = (orig_w as f32 * scale) as i32;
        
        #[cfg(not(feature = "use-opencv"))]
        {
            use crate::image_impl::{resize, Size, INTER_LINEAR};
            
            let mut resized = Mat::default();
            resize(image, &mut resized, Size::new(new_w, new_h), INTER_LINEAR)?;
            
            // ImageNet normalization
            let mean = [0.485f32, 0.456, 0.406];
            let std = [0.229f32, 0.224, 0.225];
            
            // Pad to 640x640
            let mut data = vec![0.0f32; 3 * target_size * target_size];
            
            for y in 0..new_h as usize {
                for x in 0..new_w as usize {
                    let pixel = resized.get_pixel(x as u32, y as u32);
                    for c in 0..3 {
                        let val = pixel[c] as f32 / 255.0;
                        let normalized = (val - mean[c]) / std[c];
                        data[c * target_size * target_size + y * target_size + x] = normalized;
                    }
                }
            }
            
            let arr = Array3::from_shape_vec((3, target_size, target_size), data)
                .map_err(|e| EngineError::Preprocess(e.to_string()))?;
            
            Ok((arr.insert_axis(ndarray::Axis(0)).into_dyn(), scale, scale, orig_w, orig_h))
        }
        
        #[cfg(feature = "use-opencv")]
        {
            use opencv::{core, imgproc};
            
            let mut resized = Mat::default();
            imgproc::resize(
                image,
                &mut resized,
                core::Size::new(new_w, new_h),
                0.0,
                0.0,
                imgproc::INTER_LINEAR,
            )?;
            
            // ImageNet normalization
            let mean = [0.485f32, 0.456, 0.406];
            let std = [0.229f32, 0.224, 0.225];
            
            let hw = target_size * target_size;
            let mut data = vec![0.0f32; 3 * hw];
            
            for y in 0..new_h as usize {
                for x in 0..new_w as usize {
                    let pix = resized.at_2d::<core::Vec3b>(y as i32, x as i32)?;
                    let idx = y * target_size + x;
                    for c in 0..3 {
                        let val = pix[c] as f32 / 255.0;
                        let normalized = (val - mean[c]) / std[c];
                        data[c * hw + idx] = normalized;
                    }
                }
            }
            
            let arr = Array3::from_shape_vec((3, target_size, target_size), data)
                .map_err(|e| EngineError::Preprocess(e.to_string()))?;
            
            Ok((arr.insert_axis(ndarray::Axis(0)).into_dyn(), scale, scale, orig_w, orig_h))
        }
    }

    fn postprocess(&self, outputs: HashMap<String, ArrayD<f32>>, scale_x: f32, scale_y: f32, orig_w: i32, orig_h: i32) -> Result<Vec<BBox>, EngineError> {
        // RT-DETR typically outputs: boxes [N, 4] and scores [N, num_classes]
        // Find the output tensors
        let boxes = outputs.values().next()
            .ok_or_else(|| EngineError::OutputError("No output tensor found".to_string()))?;
        
        // For now, return empty as actual RT-DETR output parsing requires
        // understanding the specific model output format
        // This is a placeholder that can be extended when testing with actual model
        let _ = (scale_x, scale_y, orig_w, orig_h, boxes);
        Ok(Vec::new())
    }
}

/// Configuration for table structure recognition
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TableStructureConfig {
    pub model_path: PathBuf,
    pub model_type: TableModelType,
    /// Side of the square model input; the crop is resized so its longer side
    /// equals this value and padded to a square.
    pub input_size: u32,
    /// Maximum number of structure tokens taken from the output.
    pub max_tokens: usize,
}

impl TableStructureConfig {
    /// SLANet_plus configuration: 488 px input and at most 500 tokens
    /// (`max_text_length` of the official `inference.yml`).
    pub fn slanet_plus(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            model_type: TableModelType::SlanetPlus,
            input_size: 488,
            max_tokens: 500,
        }
    }
}

impl Default for TableStructureConfig {
    fn default() -> Self {
        Self::slanet_plus("models/DocOCR/table_slanetplus.rten")
    }
}

/// SLANet_plus structure dictionary, in the order stored in the `character`
/// metadata of the RapidAI `slanet-plus.onnx` (`<td>` already merged into
/// `<td></td>`). Class 0 is `sos`, classes 1..=48 are these tokens and class 49
/// is `eos`.
pub const SLANET_PLUS_DICT: [&str; 48] = [
    "<thead>",
    "</thead>",
    "<tbody>",
    "</tbody>",
    "<tr>",
    "</tr>",
    "<td",
    ">",
    "</td>",
    " colspan=\"2\"",
    " colspan=\"3\"",
    " colspan=\"4\"",
    " colspan=\"5\"",
    " colspan=\"6\"",
    " colspan=\"7\"",
    " colspan=\"8\"",
    " colspan=\"9\"",
    " colspan=\"10\"",
    " colspan=\"11\"",
    " colspan=\"12\"",
    " colspan=\"13\"",
    " colspan=\"14\"",
    " colspan=\"15\"",
    " colspan=\"16\"",
    " colspan=\"17\"",
    " colspan=\"18\"",
    " colspan=\"19\"",
    " colspan=\"20\"",
    " rowspan=\"2\"",
    " rowspan=\"3\"",
    " rowspan=\"4\"",
    " rowspan=\"5\"",
    " rowspan=\"6\"",
    " rowspan=\"7\"",
    " rowspan=\"8\"",
    " rowspan=\"9\"",
    " rowspan=\"10\"",
    " rowspan=\"11\"",
    " rowspan=\"12\"",
    " rowspan=\"13\"",
    " rowspan=\"14\"",
    " rowspan=\"15\"",
    " rowspan=\"16\"",
    " rowspan=\"17\"",
    " rowspan=\"18\"",
    " rowspan=\"19\"",
    " rowspan=\"20\"",
    "<td></td>",
];

/// Build the class-index to token table: `sos`, the dictionary, `eos`.
///
/// `metadata` is the raw `character` model metadata (one token per line); when
/// absent (`.rten` files drop custom metadata) the embedded
/// [`SLANET_PLUS_DICT`] is used. `<td>` is removed and `<td></td>` appended when
/// missing, so the official Paddle dictionary file gives the same result.
pub(crate) fn structure_charset(metadata: Option<&str>) -> Vec<String> {
    let mut tokens: Vec<String> = match metadata.filter(|m| !m.trim().is_empty()) {
        Some(m) => m.split('\n').map(str::to_string).collect(),
        None => SLANET_PLUS_DICT.iter().map(|t| t.to_string()).collect(),
    };
    tokens.retain(|t| t != "<td>");
    if !tokens.iter().any(|t| t == "<td></td>") {
        tokens.push("<td></td>".to_string());
    }
    let mut charset = Vec::with_capacity(tokens.len() + 2);
    charset.push("sos".to_string());
    charset.extend(tokens);
    charset.push("eos".to_string());
    charset
}

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

#[cfg(feature = "use-opencv")]
pub(crate) fn table_input(_crop: &Mat, _input_size: u32) -> Result<(ArrayD<f32>, f32), EngineError> {
    Err(EngineError::Engine(
        "table structure recognition is not supported with the OpenCV backend".to_string(),
    ))
}

/// Build the SLANet_plus input `x` of shape `[1, 3, S, S]` (`S = input_size`).
///
/// The crop is resized, keeping its aspect ratio, so the longer side is `S`
/// (`INTER_LINEAR`, each side rounded and clamped to `1..=S`), converted to RGB,
/// normalised with `(p / 255 - mean) / std` (ImageNet) and placed in the
/// top-left corner; the remaining bottom/right area is zero (padding is applied
/// after normalisation). Returns the tensor and `ratio = S / max(w, h)`, which
/// maps model-input coordinates back to the crop.
#[cfg(not(feature = "use-opencv"))]
pub(crate) fn table_input(crop: &Mat, input_size: u32) -> Result<(ArrayD<f32>, f32), EngineError> {
    use crate::image_impl::{resize, Size, INTER_LINEAR};

    if input_size == 0 {
        return Err(EngineError::Preprocess("table input_size must be greater than zero".to_string()));
    }
    let size = crop.size().map_err(|e| EngineError::Preprocess(e.to_string()))?;
    let (w, h) = (size.width, size.height);
    if w <= 0 || h <= 0 {
        return Err(EngineError::Preprocess("table crop is empty".to_string()));
    }
    let s = input_size as usize;
    let ratio = input_size as f32 / w.max(h) as f32;
    let new_w = ((w as f64 * ratio as f64).round() as i32).clamp(1, input_size as i32);
    let new_h = ((h as f64 * ratio as f64).round() as i32).clamp(1, input_size as i32);

    let mut resized = Mat::default();
    resize(crop, &mut resized, Size::new(new_w, new_h), INTER_LINEAR)?;

    let plane = s * s;
    let mut data = vec![0.0f32; 3 * plane];
    for y in 0..new_h as usize {
        for x in 0..new_w as usize {
            let pixel = resized.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                data[c * plane + y * s + x] = (pixel[c] as f32 / 255.0 - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
            }
        }
    }
    let arr = Array4::from_shape_vec((1, 3, s, s), data).map_err(|e| EngineError::Preprocess(e.to_string()))?;
    Ok((arr.into_dyn(), ratio))
}

/// Pick the structure and location heads of batch 0 out of the model outputs.
///
/// Output names differ between exports (`save_infer_model/scale_*` in the
/// RapidAI model, `fetch_name_*` in the Paddle one), so the heads are matched by
/// their last dimension: `num_classes` for the structure logits, 8 for the cell
/// location (four corner points). Returns `(structure [T, num_classes], loc [T, 8])`.
// Consumed by the structure decoder.
#[allow(dead_code)]
pub(crate) fn split_table_outputs(
    outputs: HashMap<String, ArrayD<f32>>,
    num_classes: usize,
) -> Result<(Array2<f32>, Array2<f32>), EngineError> {
    let mut structure = None;
    let mut loc = None;
    let mut seen: Vec<String> = Vec::new();
    for (name, arr) in &outputs {
        seen.push(format!("{name}{:?}", arr.shape()));
        if arr.ndim() != 3 || arr.shape()[0] == 0 {
            continue;
        }
        match arr.shape()[2] {
            n if n == num_classes => structure = Some(arr),
            8 => loc = Some(arr),
            _ => {}
        }
    }
    seen.sort();
    let take = |arr: &ArrayD<f32>| -> Result<Array2<f32>, EngineError> {
        arr.index_axis(Axis(0), 0)
            .to_owned()
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| EngineError::OutputError(e.to_string()))
    };
    let structure = structure.ok_or_else(|| {
        EngineError::OutputError(format!(
            "table model has no structure head with {num_classes} classes (outputs: {})",
            seen.join(", ")
        ))
    })?;
    let loc = loc.ok_or_else(|| {
        EngineError::OutputError(format!(
            "table model has no location head with 8 values (outputs: {})",
            seen.join(", ")
        ))
    })?;
    Ok((take(structure)?, take(loc)?))
}

/// SLANet_plus table structure recognizer
pub struct TableStructureRecognizer {
    session: EngineSession,
    config: TableStructureConfig,
    charset: Vec<String>,
}

impl TableStructureRecognizer {
    pub fn new(config: TableStructureConfig) -> Result<Self, EngineError> {
        Self::check_model_type(&config)?;
        let engine_config = EngineConfig::default();
        let session = load_stage_session("table", &config.model_path, &engine_config)?;
        Ok(Self::from_session(config, session))
    }

    pub fn with_custom_session(config: TableStructureConfig, session: impl InferenceSession + 'static) -> Self {
        Self::from_session(config, EngineSession::from_custom(session))
    }

    fn check_model_type(config: &TableStructureConfig) -> Result<(), EngineError> {
        match config.model_type {
            TableModelType::SlanetPlus => Ok(()),
            _ => Err(EngineError::Engine(
                "TableStructureRecognizer supports TableModelType::SlanetPlus only".to_string(),
            )),
        }
    }

    fn from_session(config: TableStructureConfig, session: EngineSession) -> Self {
        let charset = structure_charset(session.metadata_value("character").as_deref());
        Self { session, config, charset }
    }

    /// Class-index to token table (`sos`, dictionary, `eos`).
    // Consumed by the structure decoder.
    #[allow(dead_code)]
    pub(crate) fn charset(&self) -> &[String] {
        &self.charset
    }

    /// Recognize table structure from image
    /// Returns a grid structure that can be filled with OCR text
    pub fn recognize(&mut self, img: &Mat) -> Result<TableStructure, EngineError> {
        let (input_tensor, _ratio) = table_input(img, self.config.input_size)?;

        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), input_tensor);

        let outputs = self.session.run_with_inputs(inputs)?;

        // Postprocess: parse structure tokens
        self.postprocess(outputs, img)
    }

    fn postprocess(&self, _outputs: HashMap<String, ArrayD<f32>>, _img: &Mat) -> Result<TableStructure, EngineError> {
        // SLANet outputs structure tokens that need to be decoded
        // This requires understanding the specific model output format
        // For now, return a simple 1x1 structure as placeholder
        
        // Create a simple single-cell structure
        let cell = TableCell {
            bbox: BBox { x_min: 0, y_min: 0, x_max: 100, y_max: 100 },
            row: 0,
            col: 0,
            row_span: 1,
            col_span: 1,
            text: String::new(), // Will be filled by OCR
        };
        
        Ok(TableStructure {
            rows: vec![TableRow { cells: vec![cell] }],
            num_rows: 1,
            num_cols: 1,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[cfg(not(feature = "use-opencv"))]
    fn solid(w: u32, h: u32, v: u8) -> Mat {
        Mat::new(image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(w, h, image::Rgb([v, v, v]))))
    }

    #[test]
    fn slanet_plus_config_defaults() {
        let c = TableStructureConfig::default();
        assert_eq!(c.model_path, PathBuf::from("models/DocOCR/table_slanetplus.rten"));
        assert!(matches!(c.model_type, TableModelType::SlanetPlus));
        assert_eq!((c.input_size, c.max_tokens), (488, 500));
    }

    #[test]
    fn charset_from_constant_and_metadata() {
        let c = structure_charset(None);
        assert_eq!(c.len(), 50);
        assert_eq!((c[0].as_str(), c[49].as_str(), c[48].as_str()), ("sos", "eos", "<td></td>"));
        let m = structure_charset(Some("<tr>\n<td>\n</td>"));
        assert_eq!(m, vec!["sos", "<tr>", "</td>", "<td></td>", "eos"]);
    }

    #[test]
    fn dictionary_constant_layout() {
        assert_eq!(SLANET_PLUS_DICT.len(), 48);
        assert_eq!(SLANET_PLUS_DICT[0], "<thead>");
        assert_eq!(SLANET_PLUS_DICT[6], "<td");
        assert_eq!(SLANET_PLUS_DICT[9], " colspan=\"2\"");
        assert_eq!(SLANET_PLUS_DICT[27], " colspan=\"20\"");
        assert_eq!(SLANET_PLUS_DICT[28], " rowspan=\"2\"");
        assert_eq!(SLANET_PLUS_DICT[46], " rowspan=\"20\"");
        assert_eq!(SLANET_PLUS_DICT[47], "<td></td>");
        assert!(!SLANET_PLUS_DICT.contains(&"<td>"));
    }

    #[cfg(not(feature = "use-opencv"))]
    #[test]
    fn table_input_keeps_ratio_and_pads_bottom_right() {
        let (t, ratio) = table_input(&solid(976, 488, 255), 488).unwrap();
        assert_eq!(t.shape(), &[1, 3, 488, 488]);
        assert!((ratio - 0.5).abs() < 1e-6);
        assert!((t[[0, 0, 0, 0]] - (1.0 - 0.485) / 0.229).abs() < 1e-3);
        assert!((t[[0, 2, 243, 487]] - (1.0 - 0.406) / 0.225).abs() < 1e-3); // last image row/column
        assert_eq!(t[[0, 0, 244, 10]], 0.0); // first padded row
        assert_eq!(t[[0, 0, 300, 10]], 0.0); // below the 244-px resized image
    }

    #[cfg(not(feature = "use-opencv"))]
    #[test]
    fn table_input_tall_crop_pads_right() {
        let (t, ratio) = table_input(&solid(244, 976, 0), 488).unwrap();
        assert!((ratio - 0.5).abs() < 1e-6);
        assert!((t[[0, 1, 487, 121]] - (0.0 - 0.456) / 0.224).abs() < 1e-3);
        assert_eq!(t[[0, 1, 10, 122]], 0.0); // first padded column
    }

    #[cfg(not(feature = "use-opencv"))]
    #[test]
    fn table_input_tiny_crop() {
        let (t, ratio) = table_input(&solid(3, 2, 0), 488).unwrap();
        assert_eq!(t.shape(), &[1, 3, 488, 488]);
        assert!((ratio - 488.0 / 3.0).abs() < 1e-3);
    }

    #[cfg(not(feature = "use-opencv"))]
    #[test]
    fn table_input_one_pixel_wide_strip() {
        let (t, _) = table_input(&solid(1, 4000, 128), 488).unwrap();
        assert_eq!(t.shape(), &[1, 3, 488, 488]); // width clamps to at least 1 px
    }

    #[cfg(feature = "use-opencv")]
    #[test]
    fn table_input_not_supported_with_opencv() {
        let m = Mat::default();
        assert!(table_input(&m, 488).err().unwrap().to_string().contains("OpenCV"));
    }

    #[test]
    fn split_outputs_by_last_dimension() {
        let mut o = HashMap::new();
        o.insert("fetch_name_0".into(), ndarray::ArrayD::<f32>::zeros(vec![1, 42, 8]));
        o.insert("fetch_name_1".into(), ndarray::ArrayD::<f32>::zeros(vec![1, 42, 50]));
        let (s, l) = split_table_outputs(o, 50).unwrap();
        assert_eq!((s.dim(), l.dim()), ((42, 50), (42, 8)));
    }

    #[test]
    fn split_outputs_rejects_class_count_mismatch() {
        let mut o = HashMap::new();
        o.insert("a".into(), ndarray::ArrayD::<f32>::zeros(vec![1, 5, 8]));
        o.insert("b".into(), ndarray::ArrayD::<f32>::zeros(vec![1, 5, 30]));
        assert!(split_table_outputs(o, 50).err().unwrap().to_string().contains("50"));
    }

    #[test]
    fn split_outputs_rejects_four_value_location_head() {
        let mut o = HashMap::new();
        o.insert("a".into(), ndarray::ArrayD::<f32>::zeros(vec![1, 5, 4]));
        o.insert("b".into(), ndarray::ArrayD::<f32>::zeros(vec![1, 5, 50]));
        assert!(split_table_outputs(o, 50).err().unwrap().to_string().contains("8"));
    }

    #[test]
    fn recognizer_rejects_slanext_model_type() {
        let c = TableStructureConfig { model_type: TableModelType::Wired, ..TableStructureConfig::default() };
        assert!(TableStructureRecognizer::new(c).err().unwrap().to_string().contains("SlanetPlus"));
    }

    fn model_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/DocOCR").join(name)
    }

    #[test]
    #[ignore = "needs models/DocOCR/table_slanetplus.onnx and .rten (scripts/download_doc_models.sh)"]
    fn embedded_dictionary_matches_onnx_metadata() {
        let onnx = load_stage_session("table", &model_path("table_slanetplus.onnx"), &EngineConfig::default()).unwrap();
        let meta = onnx.metadata_value("character").expect("ONNX model carries `character` metadata");
        assert_eq!(meta.split('\n').count(), 48);
        assert_eq!(meta.split('\n').collect::<Vec<_>>(), SLANET_PLUS_DICT.to_vec());
        let from_meta = structure_charset(Some(&meta));
        assert_eq!(from_meta, structure_charset(None));
        assert_eq!(from_meta.len(), 50);
        assert_eq!((from_meta[0].as_str(), from_meta[49].as_str()), ("sos", "eos"));

        // `.rten` drops custom metadata, so the recogniser falls back to the constant.
        let rten = load_stage_session("table", &model_path("table_slanetplus.rten"), &EngineConfig::default()).unwrap();
        assert!(rten.metadata_value("character").is_none());
        let rec = TableStructureRecognizer::new(TableStructureConfig::slanet_plus(model_path("table_slanetplus.rten"))).unwrap();
        assert_eq!(rec.charset(), structure_charset(None).as_slice());
    }

    /// Runs the real model on the table crop used by the spike (page ТП-0239-p4,
    /// crop (346,718)-(2323,1732)). Override the page with `RUSTO_TABLE_PAGE`.
    #[cfg(not(feature = "use-opencv"))]
    #[test]
    #[ignore = "needs models/DocOCR/table_slanetplus.rten and the ТП-0239-p4 page image"]
    fn real_table_crop_gives_five_by_six_skeleton() {
        use crate::image_impl::{crop, imread, Rect};

        let page = std::env::var("RUSTO_TABLE_PAGE")
            .unwrap_or_else(|_| "/Users/petar/dev/private/peo/eval/extract/ocr/pages/ТП-0239-p4-corr.png".to_string());
        let img = imread(&page).unwrap();
        let region = crop(&img, Rect::new(346, 718, 2323 - 346, 1732 - 718)).unwrap();

        let cfg = TableStructureConfig::slanet_plus(model_path("table_slanetplus.rten"));
        let mut rec = TableStructureRecognizer::new(cfg.clone()).unwrap();
        let (x, ratio) = table_input(&region, cfg.input_size).unwrap();
        assert!((ratio - 488.0 / 1977.0).abs() < 1e-6);
        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), x);
        let outputs = rec.session.run_with_inputs(inputs).unwrap();
        for (name, arr) in &outputs {
            eprintln!("output {name}: {:?}", arr.shape());
        }
        let (structure, loc) = split_table_outputs(outputs, rec.charset().len()).unwrap();
        assert_eq!(structure.ncols(), 50);
        assert_eq!(loc.ncols(), 8);
        assert_eq!(structure.nrows(), loc.nrows());

        let mut tokens = Vec::new();
        for row in structure.rows() {
            let best = row.iter().enumerate().fold((0, f32::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0;
            if rec.charset()[best] == "eos" {
                break;
            }
            tokens.push(rec.charset()[best].as_str());
        }
        let count = |t: &str| tokens.iter().filter(|x| **x == t).count();
        eprintln!("T={} tokens before eos={}", structure.nrows(), tokens.len());
        assert_eq!((count("<tr>"), count("<td></td>")), (5, 30));
    }

    #[test]
    fn test_html_generation() {
        let structure = TableStructure {
            rows: vec![
                TableRow {
                    cells: vec![
                        TableCell {
                            bbox: BBox { x_min: 0, y_min: 0, x_max: 100, y_max: 30 },
                            row: 0,
                            col: 0,
                            row_span: 1,
                            col_span: 1,
                            text: "Header 1".to_string(),
                        },
                        TableCell {
                            bbox: BBox { x_min: 100, y_min: 0, x_max: 200, y_max: 30 },
                            row: 0,
                            col: 1,
                            row_span: 1,
                            col_span: 1,
                            text: "Header 2".to_string(),
                        },
                    ],
                },
                TableRow {
                    cells: vec![
                        TableCell {
                            bbox: BBox { x_min: 0, y_min: 30, x_max: 100, y_max: 60 },
                            row: 1,
                            col: 0,
                            row_span: 1,
                            col_span: 1,
                            text: "Data 1".to_string(),
                        },
                        TableCell {
                            bbox: BBox { x_min: 100, y_min: 30, x_max: 200, y_max: 60 },
                            row: 1,
                            col: 1,
                            row_span: 1,
                            col_span: 1,
                            text: "Data 2".to_string(),
                        },
                    ],
                },
            ],
            num_rows: 2,
            num_cols: 2,
        };

        let html = structure.to_html();
        assert!(html.contains("<table>"));
        assert!(html.contains("<td>Header 1</td>"));
        assert!(html.contains("<td>Data 2</td>"));
        assert!(html.contains("</table>"));
    }
}
