use std::time::Instant;
use ndarray::{Array3, Array4};

#[cfg(feature = "use-opencv")]
use opencv::core::Point2f;
#[cfg(feature = "use-opencv")]
use opencv::prelude::*;

#[cfg(not(feature = "use-opencv"))]
use crate::image_impl::{Mat, Point2f};

use crate::engine::{load_stage_session, EngineError, EngineSession, InferenceSession};
use crate::layout::BBox;
use crate::types::{OrientConfig, OrientPreprocess};

/// Orientation classification result: the model's class label.
///
/// Direction convention (verified with `orient_rapidai_v2` on all four
/// rotations of `tests/fixtures/doc/page_text.png`): label `N` means the
/// scanned content is rotated `N` degrees clockwise relative to upright
/// (`DynamicImage::rotate90` produces a label-90 scan). [`Orientation::rotate_image`]
/// undoes this by rotating the scan clockwise by
/// [`Orientation::correction_cw_degrees`] (label 90 -> 270 clockwise, label 270
/// -> 90 clockwise, label 180 -> 180).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Orientation {
    /// Label 0: the page is upright
    Normal = 0,
    /// Label 90
    Rotate90 = 1,
    /// Label 180
    Rotate180 = 2,
    /// Label 270
    Rotate270 = 3,
}

impl Orientation {
    pub fn from_class(class_id: usize, num_classes: usize) -> Self {
        if num_classes == 2 {
            // Binary classification (0 vs 180) common in PPOCR
            match class_id {
                0 => Orientation::Normal,
                1 => Orientation::Rotate180,
                _ => Orientation::Normal,
            }
        } else {
            // 4-class classification (0, 90, 180, 270)
            match class_id {
                0 => Orientation::Normal,
                1 => Orientation::Rotate90,
                2 => Orientation::Rotate180,
                3 => Orientation::Rotate270,
                _ => Orientation::Normal,
            }
        }
    }

    /// The model's class label in degrees: 0, 90, 180 or 270.
    pub fn label_degrees(&self) -> u16 {
        match self {
            Orientation::Normal => 0,
            Orientation::Rotate90 => 90,
            Orientation::Rotate180 => 180,
            Orientation::Rotate270 => 270,
        }
    }

    /// Clockwise rotation (0/90/180/270) that [`Orientation::rotate_image`]
    /// applies to the scan to obtain the corrected page.
    pub fn correction_cw_degrees(&self) -> u16 {
        match self {
            Orientation::Normal => 0,
            Orientation::Rotate90 => 270,
            Orientation::Rotate180 => 180,
            Orientation::Rotate270 => 90,
        }
    }

    pub fn degrees(&self) -> i32 {
        match self {
            Orientation::Normal => 0,
            Orientation::Rotate90 => -90,  // Counter-clockwise for image correction
            Orientation::Rotate180 => 180,
            Orientation::Rotate270 => -270,  // 90 clockwise for image correction
        }
    }

    /// Rotate bounding box points from corrected image space back to original image space
    pub fn rotate_box_points_back(&self, box_points: &mut [Point2f; 4], corr_w: f32, corr_h: f32) {
        match self {
            Orientation::Normal => {}
            Orientation::Rotate90 => {
                for p in box_points.iter_mut() {
                    let old_x = p.x;
                    let old_y = p.y;
                    p.x = corr_h - 1.0 - old_y;
                    p.y = old_x;
                }
            }
            Orientation::Rotate180 => {
                for p in box_points.iter_mut() {
                    p.x = corr_w - 1.0 - p.x;
                    p.y = corr_h - 1.0 - p.y;
                }
            }
            Orientation::Rotate270 => {
                for p in box_points.iter_mut() {
                    let old_x = p.x;
                    let old_y = p.y;
                    p.x = old_y;
                    p.y = corr_w - 1.0 - old_x;
                }
            }
        }
    }
    
    /// Apply rotation to correct the image orientation
    #[cfg(feature = "use-opencv")]
    pub fn rotate_image(&self, img: &Mat) -> Result<Mat, EngineError> {
        use opencv::core::Mat as CvMat;

        match self {
            Orientation::Normal => Ok(img.clone()),
            Orientation::Rotate90 => {
                let mut rotated = CvMat::default();
                opencv::core::rotate(img, &mut rotated, opencv::core::ROTATE_90_COUNTERCLOCKWISE)?;
                Ok(rotated)
            }
            Orientation::Rotate180 => {
                let mut rotated = CvMat::default();
                opencv::core::rotate(img, &mut rotated, opencv::core::ROTATE_180)?;
                Ok(rotated)
            }
            Orientation::Rotate270 => {
                let mut rotated = CvMat::default();
                opencv::core::rotate(img, &mut rotated, opencv::core::ROTATE_90_CLOCKWISE)?;
                Ok(rotated)
            }
        }
    }
    
    /// Apply rotation to correct the image orientation (Pure Rust)
    #[cfg(not(feature = "use-opencv"))]
    pub fn rotate_image(&self, img: &Mat) -> Result<Mat, EngineError> {
        use crate::image_impl::{rotate_90, rotate_180, rotate_270};
        
        match self {
            Orientation::Normal => Ok(img.clone()),
            Orientation::Rotate90 => rotate_270(img)
                .map_err(|e| EngineError::Preprocess(e.to_string())),
            Orientation::Rotate180 => rotate_180(img)
                .map_err(|e| EngineError::Preprocess(e.to_string())),
            Orientation::Rotate270 => rotate_90(img)
                .map_err(|e| EngineError::Preprocess(e.to_string())),
        }
    }
}

pub struct OrientOutput {
    pub orientation: Orientation,
    pub confidence: f32,
    pub elapse: f64,
}

pub struct OrientClassifier {
    session: EngineSession,
    pub config: OrientConfig,  // Public to allow access to confidence_threshold
}

impl OrientClassifier {
    pub fn new(config: OrientConfig) -> Result<Self, EngineError> {
        let session = load_stage_session("orientation classifier", &config.model_path, &config.engine_cfg)?;
        Ok(Self { session, config })
    }

    /// Construct orientation classifier with a custom inference session
    pub fn with_custom_session(config: OrientConfig, session: impl InferenceSession + 'static) -> Self {
        Self {
            session: EngineSession::from_custom(session),
            config,
        }
    }

    pub fn classify(&mut self, image: &Mat) -> Result<OrientOutput, EngineError> {
        let start = Instant::now();

        // Preprocess image
        let input_tensor = self.preprocess(image)?;

        // Run inference
        let output = self.session.run(input_tensor.into_dyn())?;

        // Postprocess
        let (orientation, confidence) = self.postprocess(output)?;

        let elapse = start.elapsed().as_secs_f64();

        Ok(OrientOutput {
            orientation,
            confidence,
            elapse,
        })
    }

    fn preprocess(&self, image: &Mat) -> Result<Array4<f32>, EngineError> {
        orient_input(image, &self.config)
    }

    fn postprocess(&self, output: ndarray::ArrayD<f32>) -> Result<(Orientation, f32), EngineError> {
        // Output shape can be [1, 2] (PPOCR) or [1, 4]
        let output_2d = output.into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| EngineError::OutputError(format!("Invalid output shape: {}", e)))?;

        let row: Vec<f32> = output_2d.row(0).to_vec();
        let num_classes = row.len();
        let probs = probabilities(&row);

        let (class_id, &confidence) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| EngineError::OutputError("No class found".to_string()))?;

        let orientation = Orientation::from_class(class_id, num_classes);

        Ok((orientation, confidence))
    }
}

/// Build the model input for `image` according to `cfg.preprocess`.
pub(crate) fn orient_input(image: &Mat, cfg: &OrientConfig) -> Result<Array4<f32>, EngineError> {
    match cfg.preprocess {
        OrientPreprocess::Stretch => stretch_input(image, cfg),
        OrientPreprocess::ShortSideCenterCrop { resize_short } => center_crop_input(image, cfg, resize_short),
    }
}

fn stretch_input(image: &Mat, cfg: &OrientConfig) -> Result<Array4<f32>, EngineError> {
    // Resize to model input size
    let [_, h, w] = cfg.orient_image_shape;
    #[cfg(feature = "use-opencv")]
    {
        use opencv::{core, imgproc};
        
        let mut resized = Mat::default();
        imgproc::resize(
            image,
            &mut resized,
            core::Size::new(w, h),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;

        let rows = resized.rows() as usize;
        let cols = resized.cols() as usize;
        let hw = rows * cols;
        
        let mut data = vec![0.0f32; 3 * hw];
        
        for i in 0..rows {
            for j in 0..cols {
                let pix = resized.at_2d::<core::Vec3b>(i as i32, j as i32)?;
                let idx = i * cols + j;
                data[idx] = pix[0] as f32 / 255.0;
                data[hw + idx] = pix[1] as f32 / 255.0;
                data[2 * hw + idx] = pix[2] as f32 / 255.0;
            }
        }

        let arr = Array3::from_shape_vec((3, h as usize, w as usize), data)
            .map_err(|e| EngineError::Preprocess(e.to_string()))?;
        
        Ok(arr.insert_axis(ndarray::Axis(0)))
    }

    #[cfg(not(feature = "use-opencv"))]
    {
        use crate::image_impl::{resize, Size, INTER_LINEAR};
        
        let mut resized = Mat::default();
        resize(image, &mut resized, Size::new(w, h), INTER_LINEAR)?;
        let (_orig_h, _orig_w) = (image.size()?.height, image.size()?.width);
        
        let hw = h as usize * w as usize;
        let mut data = vec![0.0f32; 3 * hw];
        
        for y in 0..h as usize {
            for x in 0..w as usize {
                let pixel = resized.get_pixel(x as u32, y as u32);
                let idx = y * w as usize + x;
                data[idx] = pixel[0] as f32 / 255.0;
                data[hw + idx] = pixel[1] as f32 / 255.0;
                data[2 * hw + idx] = pixel[2] as f32 / 255.0;
            }
        }

        let arr = Array3::from_shape_vec((3, h as usize, w as usize), data)
            .map_err(|e| EngineError::Preprocess(e.to_string()))?;
        
        Ok(arr.insert_axis(ndarray::Axis(0)))
    }
}

#[cfg(feature = "use-opencv")]
fn center_crop_input(_image: &Mat, _cfg: &OrientConfig, _resize_short: u32) -> Result<Array4<f32>, EngineError> {
    Err(EngineError::Engine(
        "orientation pre-processing ShortSideCenterCrop is not supported with the OpenCV backend".to_string(),
    ))
}

/// Resize so the short side equals `resize_short`, centre crop to the model
/// shape and apply `(p / 255 - mean) / std` on RGB. When the resized image is
/// smaller than the crop (only if `resize_short` < crop size) edge pixels are
/// repeated.
#[cfg(not(feature = "use-opencv"))]
fn center_crop_input(image: &Mat, cfg: &OrientConfig, resize_short: u32) -> Result<Array4<f32>, EngineError> {
    use crate::image_impl::{resize, Size, INTER_LINEAR};

    let [_, ch, cw] = cfg.orient_image_shape;
    let (ch, cw) = (ch.max(1) as usize, cw.max(1) as usize);
    let size = image.size().map_err(|e| EngineError::Preprocess(e.to_string()))?;
    let (w, h) = (size.width.max(1) as f64, size.height.max(1) as f64);
    let scale = resize_short as f64 / w.min(h);
    let nw = ((w * scale).round() as i32).max(1);
    let nh = ((h * scale).round() as i32).max(1);

    let mut resized = Mat::default();
    resize(image, &mut resized, Size::new(nw, nh), INTER_LINEAR)?;

    let (nw, nh) = (nw as usize, nh as usize);
    let off_x = nw.saturating_sub(cw) / 2;
    let off_y = nh.saturating_sub(ch) / 2;
    let hw = ch * cw;
    let mut data = vec![0.0f32; 3 * hw];
    for y in 0..ch {
        let sy = (off_y + y).min(nh - 1) as u32;
        for x in 0..cw {
            let sx = (off_x + x).min(nw - 1) as u32;
            let pixel = resized.get_pixel(sx, sy);
            let idx = y * cw + x;
            for c in 0..3 {
                data[c * hw + idx] = (pixel[c] as f32 / 255.0 - cfg.mean[c]) / cfg.std[c];
            }
        }
    }
    let arr = Array3::from_shape_vec((3, ch, cw), data)
        .map_err(|e| EngineError::Preprocess(e.to_string()))?;
    Ok(arr.insert_axis(ndarray::Axis(0)))
}

/// Turn one row of model output into class probabilities.
///
/// Some orientation models (for example PaddleClas PULC `rapid_orientation.onnx`)
/// already end in a `Softmax` node, so their output is a probability
/// distribution. Applying softmax a second time flattens it: a confident
/// `[0.9, 0.03, 0.03, 0.03]` becomes about `[0.45, 0.18, 0.18, 0.18]`, and the
/// confidence can never pass a threshold such as 0.9. Other models output raw
/// logits and need the softmax. A row counts as probabilities when every value
/// is in `[0, 1]` and the sum is within `1e-3` of 1.0; otherwise softmax is applied.
pub(crate) fn probabilities(logits: &[f32]) -> Vec<f32> {
    const EPS: f32 = 1e-6;
    let in_range = logits.iter().all(|&x| (-EPS..=1.0 + EPS).contains(&x));
    let sum: f32 = logits.iter().sum();
    if !logits.is_empty() && in_range && (sum - 1.0).abs() <= 1e-3 {
        return logits.to_vec();
    }

    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();
    logits
        .iter()
        .map(|&x| (x - max_logit).exp() / exp_sum)
        .collect()
}

/// Page orientation decision. `applied_degrees` is the clockwise rotation
/// applied to the scan to obtain the corrected page (0 when the classifier was
/// not confident enough); `detected_degrees` is the model's class label.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct PageOrientation {
    pub applied_degrees: u16,
    pub detected_degrees: u16,
    pub confidence: f32,
}

/// Apply the correction only when `out.confidence >= min_confidence`.
pub fn decide_orientation(out: &OrientOutput, min_confidence: f32) -> PageOrientation {
    let applied = if out.confidence >= min_confidence {
        out.orientation.correction_cw_degrees()
    } else {
        0
    };
    PageOrientation {
        applied_degrees: applied,
        detected_degrees: out.orientation.label_degrees(),
        confidence: out.confidence,
    }
}

/// Rotate a continuous pixel coordinate clockwise by `cw_degrees` on a
/// `w x h` image: 90 maps `(x, y)` to `(h - y, x)`, 180 to `(w - x, h - y)`,
/// 270 to `(y, w - x)`. Other angles return the input.
fn rotate_point(x: f32, y: f32, cw_degrees: u16, w: f32, h: f32) -> (f32, f32) {
    match cw_degrees {
        90 => (h - y, x),
        180 => (w - x, h - y),
        270 => (y, w - x),
        _ => (x, y),
    }
}

/// Counter-rotation of `degrees` (0/90/180/270), `None` for other angles.
fn inverse_degrees(degrees: u16) -> Option<u16> {
    match degrees {
        0 => Some(0),
        90 => Some(270),
        180 => Some(180),
        270 => Some(90),
        _ => None,
    }
}

/// Map a point from the corrected page (`corrected_w x corrected_h`) back to the
/// scan it was rotated from by `applied_degrees` clockwise.
pub fn rotate_point_back(x: f32, y: f32, applied_degrees: u16, corrected_w: f32, corrected_h: f32) -> (f32, f32) {
    match inverse_degrees(applied_degrees) {
        Some(inv) => rotate_point(x, y, inv, corrected_w, corrected_h),
        None => (x, y),
    }
}

/// Rotate an axis-aligned box clockwise by `cw_degrees` on a `src_w x src_h`
/// image. Degrees other than 0/90/180/270 return the input.
pub fn rotate_box(bbox: &BBox, cw_degrees: u16, src_w: u32, src_h: u32) -> BBox {
    let (w, h) = (src_w as i32, src_h as i32);
    match cw_degrees {
        90 => BBox { x_min: h - bbox.y_max, y_min: bbox.x_min, x_max: h - bbox.y_min, y_max: bbox.x_max },
        180 => BBox { x_min: w - bbox.x_max, y_min: h - bbox.y_max, x_max: w - bbox.x_min, y_max: h - bbox.y_min },
        270 => BBox { x_min: bbox.y_min, y_min: w - bbox.x_max, x_max: bbox.y_max, y_max: w - bbox.x_min },
        _ => *bbox,
    }
}

/// Map a box from the corrected page (`corrected_w x corrected_h`) back to the
/// scan: equal to `rotate_box(b, (360 - applied) % 360, ...)`.
pub fn rotate_box_back(bbox: &BBox, applied_degrees: u16, corrected_w: u32, corrected_h: u32) -> BBox {
    match inverse_degrees(applied_degrees) {
        Some(inv) => rotate_box(bbox, inv, corrected_w, corrected_h),
        None => *bbox,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn top(row: &[f32]) -> (usize, f32) {
        let probs = probabilities(row);
        let (i, &c) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        (i, c)
    }

    #[test]
    fn test_probabilities_pass_through() {
        let row = [0.9, 0.04, 0.03, 0.03];
        assert_eq!(probabilities(&row), row.to_vec());
        let (class, conf) = top(&row);
        assert_eq!(class, 0);
        assert!((conf - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_logits_are_softmaxed() {
        let (class, conf) = top(&[4.0, 0.0, 0.0, 0.0]);
        assert_eq!(class, 0);
        assert!((conf - 0.948).abs() < 1e-3);
    }

    #[test]
    fn test_two_class_probabilities_pass_through() {
        let (class, conf) = top(&[0.2, 0.8]);
        assert_eq!(class, 1);
        assert!((conf - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_rotate_box_points_back_90() {
        let mut pts = [
            Point2f::new(20.0, 10.0),
            Point2f::new(50.0, 10.0),
            Point2f::new(50.0, 30.0),
            Point2f::new(20.0, 30.0),
        ];
        // corrected width = 200, height = 100
        Orientation::Rotate90.rotate_box_points_back(&mut pts, 200.0, 100.0);
        // x_orig = corr_h - 1 - y = 100 - 1 - y = 99 - y
        // y_orig = x
        assert_eq!(pts[0].x, 89.0);
        assert_eq!(pts[0].y, 20.0);
        assert_eq!(pts[1].x, 89.0);
        assert_eq!(pts[1].y, 50.0);
    }

    #[test]
    fn test_rotate_box_points_back_180() {
        let mut pts = [
            Point2f::new(10.0, 20.0),
            Point2f::new(100.0, 20.0),
            Point2f::new(100.0, 50.0),
            Point2f::new(10.0, 50.0),
        ];
        Orientation::Rotate180.rotate_box_points_back(&mut pts, 200.0, 100.0);
        assert_eq!(pts[0].x, 189.0);
        assert_eq!(pts[0].y, 79.0);
    }

    #[test]
    fn test_rotate_box_points_back_270() {
        let mut pts = [
            Point2f::new(20.0, 10.0),
            Point2f::new(50.0, 10.0),
            Point2f::new(50.0, 30.0),
            Point2f::new(20.0, 30.0),
        ];
        // corr_w = 200, corr_h = 100
        // x_orig = y, y_orig = corr_w - 1 - x = 199 - x
        Orientation::Rotate270.rotate_box_points_back(&mut pts, 200.0, 100.0);
        assert_eq!(pts[0].x, 10.0);
        assert_eq!(pts[0].y, 179.0);
    }

    fn solid(w: u32, h: u32, v: u8) -> Mat { Mat::new(image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(w, h, image::Rgb([v, v, v])))) }
    fn out(o: Orientation, c: f32) -> OrientOutput { OrientOutput { orientation: o, confidence: c, elapse: 0.0 } }

    type Rect = (u32, u32, u32, u32);

    /// Black image with an RGB(255, 128, 0) rectangle `x0..x1` x `y0..y1`, then an optional black `hole`.
    fn rect_image(w: u32, h: u32, (x0, y0, x1, y1): Rect, hole: Option<Rect>) -> Mat {
        let mut img = image::RgbImage::from_pixel(w, h, image::Rgb([0, 0, 0]));
        for y in y0..y1 { for x in x0..x1 { img.put_pixel(x, y, image::Rgb([255, 128, 0])); } }
        if let Some((hx0, hy0, hx1, hy1)) = hole {
            for y in hy0..hy1 { for x in hx0..hx1 { img.put_pixel(x, y, image::Rgb([0, 0, 0])); } }
        }
        Mat::new(image::DynamicImage::ImageRgb8(img))
    }

    /// Every pixel of the crop must be inside the rectangle (so a plain stretch or a wrong
    /// crop offset would read black) and normalised per channel with ImageNet mean/std.
    fn assert_crop_is_rectangle(t: &ndarray::Array4<f32>, points: &[(usize, usize)]) {
        assert_eq!(t.shape(), &[1, 3, 224, 224]);
        let mean = [0.485f32, 0.456, 0.406];
        let std = [0.229f32, 0.224, 0.225];
        let rgb = [255.0f32, 128.0, 0.0];
        for c in 0..3 {
            let want = (rgb[c] / 255.0 - mean[c]) / std[c];
            for &(y, x) in points {
                assert!((t[[0, c, y, x]] - want).abs() < 1e-3, "channel {c} at ({y},{x}): {} vs {want}", t[[0, c, y, x]]);
            }
        }
    }

    #[test]
    fn orient_input_center_crop_uses_imagenet_normalisation() {
        // 512x256: short side already 256 (scale 1); crop is x 144..368, y 16..240.
        let cfg = OrientConfig::rapid_orientation_v2("unused.rten");
        let t = orient_input(&rect_image(512, 256, (144, 16, 368, 240), None), &cfg).unwrap();
        assert_crop_is_rectangle(&t, &[(0, 0), (0, 223), (223, 0), (223, 223), (112, 112)]);
    }
    #[test]
    fn orient_input_center_crop_resizes_short_side_to_256() {
        let cfg = OrientConfig::rapid_orientation_v2("unused.rten");
        // The rectangle is exactly the resized crop window in source pixels. A black hole covers the
        // crop that a missing resize would read, so skipping the resize reads black at the inset corners.
        // The checked points are inset by 4 px so the resize filter does not blend the rectangle edge.
        let pts = [(4, 4), (4, 219), (219, 4), (219, 219)];
        // Landscape 1024x512 -> 512x256: crop x 144..368, y 16..240 = source 288..736 x 32..480;
        // without resize the crop would be source 400..624 x 144..368.
        let t = orient_input(&rect_image(1024, 512, (288, 32, 736, 480), Some((400, 144, 624, 368))), &cfg).unwrap();
        assert_crop_is_rectangle(&t, &pts);
        // Portrait 512x1024 -> 256x512: crop x 16..240, y 144..368 = source 32..480 x 288..736;
        // without resize the crop would be source 144..368 x 400..624.
        let t = orient_input(&rect_image(512, 1024, (32, 288, 480, 736), Some((144, 400, 368, 624))), &cfg).unwrap();
        assert_crop_is_rectangle(&t, &pts);
    }
    #[test]
    fn orient_input_stretch_keeps_legacy_scaling() {
        // Legacy: p / 255, mean/std ignored. (p/255 - 0.5)/0.5 would give 1.0, 0.004, -1.0 here.
        let cfg = OrientConfig::default("unused.rten".into());
        let img = Mat::new(image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(50, 20, image::Rgb([255, 128, 0]))));
        let t = orient_input(&img, &cfg).unwrap();
        assert_eq!(t.shape(), &[1, 3, 224, 224]);
        assert!((t[[0, 0, 10, 10]] - 1.0).abs() < 1e-6);
        assert!((t[[0, 1, 10, 10]] - 128.0 / 255.0).abs() < 1e-6);
        assert!(t[[0, 2, 10, 10]].abs() < 1e-6);
    }
    #[test]
    fn orient_input_tiny_image() {
        let cfg = OrientConfig::rapid_orientation_v2("unused.rten");
        assert_eq!(orient_input(&solid(20, 10, 128), &cfg).unwrap().shape(), &[1, 3, 224, 224]);
    }
    #[test]
    fn decide_orientation_applies_at_exact_threshold() {
        let p = decide_orientation(&out(Orientation::Rotate180, 0.6), 0.6);
        assert_eq!((p.applied_degrees, p.detected_degrees), (180, 180));
    }
    #[test]
    fn decide_orientation_below_threshold_keeps_page() {
        let p = decide_orientation(&out(Orientation::Rotate90, 0.5999), 0.6);
        assert_eq!((p.applied_degrees, p.detected_degrees), (0, 90));
        assert!((p.confidence - 0.5999).abs() < 1e-6);
    }
    #[test]
    fn rotate_box_known_values_and_round_trip() {
        let b = BBox { x_min: 20, y_min: 10, x_max: 50, y_max: 30 };
        let r = rotate_box(&b, 90, 200, 100);
        assert_eq!((r.x_min, r.y_min, r.x_max, r.y_max), (70, 20, 90, 50));
        for d in [0u16, 90, 180, 270] {
            let (cw, ch) = if d % 180 == 0 { (200, 100) } else { (100, 200) };
            let back = rotate_box_back(&rotate_box(&b, d, 200, 100), d, cw, ch);
            assert_eq!((back.x_min, back.y_min, back.x_max, back.y_max), (20, 10, 50, 30), "degrees {d}");
        }
    }
    #[test]
    fn correction_is_inverse_of_label() {
        for o in [Orientation::Normal, Orientation::Rotate90, Orientation::Rotate180, Orientation::Rotate270] {
            assert_eq!((o.label_degrees() + o.correction_cw_degrees()) % 360, 0);
        }
    }
}
