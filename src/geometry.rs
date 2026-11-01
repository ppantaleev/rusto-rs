use std::collections::BTreeMap;

use crate::engine::EngineError;

#[cfg(feature = "use-opencv")]
use opencv::{core, imgproc, prelude::*};

#[cfg(feature = "use-opencv")]
use opencv::core::{Mat, Point2f};

#[cfg(feature = "use-opencv")]
type ImgResult<T> = opencv::Result<T>;

#[cfg(not(feature = "use-opencv"))]
use crate::image_impl::{
    self, Mat, Point2f, Result as ImgResult, Size, INTER_LINEAR, ROTATE_90_CLOCKWISE,
};

pub type OpRecord = BTreeMap<String, BTreeMap<String, f32>>;

pub fn map_boxes_to_original(
    dt_boxes: &mut [[Point2f; 4]],
    op_record: &OpRecord,
    ori_h: i32,
    ori_w: i32,
) {
    for (op, v) in op_record.iter().rev() {
        if op.contains("padding") {
            let top = *v.get("top").unwrap_or(&0.0);
            let left = *v.get("left").unwrap_or(&0.0);
            for box_pts in dt_boxes.iter_mut() {
                for p in box_pts.iter_mut() {
                    p.x -= left;
                    p.y -= top;
                }
            }
        } else if op.contains("preprocess") {
            let ratio_h = *v.get("ratio_h").unwrap_or(&1.0);
            let ratio_w = *v.get("ratio_w").unwrap_or(&1.0);
            for box_pts in dt_boxes.iter_mut() {
                for p in box_pts.iter_mut() {
                    p.x *= ratio_w;
                    p.y *= ratio_h;
                }
            }
        }
    }

    for box_pts in dt_boxes.iter_mut() {
        for p in box_pts.iter_mut() {
            if p.x < 0.0 {
                p.x = 0.0;
            }
            if p.y < 0.0 {
                p.y = 0.0;
            }
            if p.x > ori_w as f32 {
                p.x = ori_w as f32;
            }
            if p.y > ori_h as f32 {
                p.y = ori_h as f32;
            }
        }
    }
}

pub fn apply_vertical_padding(
    img: &Mat,
    mut op_record: OpRecord,
    width_height_ratio: f32,
    min_height: f32,
) -> Result<(Mat, OpRecord), EngineError> {
    let h = img.rows();
    let w = img.cols();

    let use_limit_ratio = if (width_height_ratio - (-1.0)).abs() < f32::EPSILON {
        false
    } else {
        (w as f32) / (h as f32) > width_height_ratio
    };

    if (h as f32) <= min_height || use_limit_ratio {
        let padding_h = get_padding_h(h, w, width_height_ratio, min_height);
        let padded = add_round_letterbox(img, (padding_h, padding_h, 0, 0))?;
        let mut m = BTreeMap::new();
        m.insert("top".to_string(), padding_h as f32);
        m.insert("left".to_string(), 0.0);
        op_record.insert("padding_1".to_string(), m);
        Ok((padded, op_record))
    } else {
        let mut m = BTreeMap::new();
        m.insert("top".to_string(), 0.0);
        m.insert("left".to_string(), 0.0);
        op_record.insert("padding_1".to_string(), m);
        Ok((img.clone(), op_record))
    }
}

fn get_padding_h(h: i32, w: i32, width_height_ratio: f32, min_height: f32) -> i32 {
    // Match Python: max(int(w / width_height_ratio), min_height) * 2
    let new_h = ((w as f32 / width_height_ratio) as i32).max(min_height as i32) * 2;
    (new_h - h).abs() / 2
}

/// Smallest sine of a vertex angle for a quad to count as a quad. Three vertices
/// closer to a line than this make the homography to the crop rectangle singular
/// or so ill-conditioned that the crop is noise. A real text box has angles near
/// 90 degrees; 0.01 is about 0.57 degrees.
const MIN_VERTEX_SINE: f32 = 0.01;

/// True when `get_rotate_crop_image` cannot produce a usable crop:
/// - a coordinate is not finite;
/// - the longer opposite edge in either direction is shorter than 1 px (the crop
///   size is truncated to whole pixels). Detection boxes are clamped to the page,
///   so a box lying outside it collapses onto the edge as a flat quad;
/// - two vertices coincide, or three vertices are (nearly) collinear, so the
///   homography onto the crop rectangle is singular. A quad can have a non-zero
///   bounding size and still be flat, for example `(12.9,271) (1019.1,250.8)
///   (1021.1,271) (14.8,271)`, where three vertices lie on one line;
/// - the vertices are not in a consistent winding order (self-intersecting or
///   bow-tie quad).
pub fn is_degenerate_quad(points: &[Point2f; 4]) -> bool {
    if points.iter().any(|p| !p.x.is_finite() || !p.y.is_finite()) {
        return true;
    }
    let w = (points[0].x - points[1].x)
        .hypot(points[0].y - points[1].y)
        .max((points[2].x - points[3].x).hypot(points[2].y - points[3].y));
    let h = (points[0].x - points[3].x)
        .hypot(points[0].y - points[3].y)
        .max((points[1].x - points[2].x).hypot(points[1].y - points[2].y));
    if w < 1.0 || h < 1.0 {
        return true;
    }
    // Sine and sign of the turn at each vertex, in f64 to keep tiny cross products exact.
    let mut sign = 0.0f64;
    for i in 0..4 {
        let prev = &points[(i + 3) % 4];
        let cur = &points[i];
        let next = &points[(i + 1) % 4];
        let (ax, ay) = ((cur.x - prev.x) as f64, (cur.y - prev.y) as f64);
        let (bx, by) = ((next.x - cur.x) as f64, (next.y - cur.y) as f64);
        let len = ax.hypot(ay) * bx.hypot(by);
        let cross = ax * by - ay * bx;
        if len <= 0.0 || (cross.abs() / len) < MIN_VERTEX_SINE as f64 {
            return true;
        }
        if sign == 0.0 {
            sign = cross.signum();
        } else if cross.signum() != sign {
            return true;
        }
    }
    false
}

/// A singular perspective matrix from the pure-Rust warp is a bad quad, not a
/// failure of the image: it gives an empty crop. Other errors still propagate.
#[cfg(not(feature = "use-opencv"))]
fn empty_if_singular(result: ImgResult<Mat>) -> ImgResult<Mat> {
    use image_impl::SingularMatrixError;
    match result {
        Err(e) if e.is::<SingularMatrixError>() => empty_crop(),
        other => other,
    }
}

#[cfg(feature = "use-opencv")]
fn empty_if_singular(result: ImgResult<Mat>) -> ImgResult<Mat> {
    result
}

/// An image with no pixels. The recogniser maps it to an empty line.
fn empty_crop() -> ImgResult<Mat> {
    #[cfg(feature = "use-opencv")]
    {
        Ok(Mat::default())
    }
    #[cfg(not(feature = "use-opencv"))]
    {
        Mat::from_rgb8(0, 0, Vec::new())
    }
}

/// Crop one detected text box. A degenerate or singular box gives an empty image instead of
/// an error, so one bad box cannot fail the recognition of the whole image and
/// the result list stays aligned with the detection boxes.
pub fn crop_text_box(img: &Mat, points: &[Point2f; 4]) -> ImgResult<Mat> {
    if is_degenerate_quad(points) {
        return empty_crop();
    }
    empty_if_singular(get_rotate_crop_image(img, points))
}

#[cfg(feature = "use-opencv")]
pub fn get_rotate_crop_image(img: &Mat, points: &[Point2f; 4]) -> ImgResult<Mat> {
    let w1 = (points[0].x - points[1].x).hypot(points[0].y - points[1].y);
    let w2 = (points[2].x - points[3].x).hypot(points[2].y - points[3].y);
    let img_crop_width = w1.max(w2) as i32;

    let h1 = (points[0].x - points[3].x).hypot(points[0].y - points[3].y);
    let h2 = (points[1].x - points[2].x).hypot(points[1].y - points[2].y);
    let img_crop_height = h1.max(h2) as i32;

    let pts_src = core::Mat::from_slice_2d(&[
        [points[0].x, points[0].y],
        [points[1].x, points[1].y],
        [points[2].x, points[2].y],
        [points[3].x, points[3].y],
    ])?;

    let pts_dst = core::Mat::from_slice_2d(&[
        [0.0f32, 0.0f32],
        [img_crop_width as f32, 0.0f32],
        [img_crop_width as f32, img_crop_height as f32],
        [0.0f32, img_crop_height as f32],
    ])?;

    let m = imgproc::get_perspective_transform(&pts_src, &pts_dst, 0)?;
    let mut dst = Mat::default();
    imgproc::warp_perspective(
        img,
        &mut dst,
        &m,
        core::Size::new(img_crop_width, img_crop_height),
        imgproc::INTER_CUBIC,
        core::BORDER_REPLICATE,
        core::Scalar::all(0.0),
    )?;

    let size = dst.size()?;
    let dst_h = size.height as f32;
    let dst_w = size.width as f32;
    if dst_h / dst_w >= 1.5 {
        let mut rotated = Mat::default();
        core::rotate(&dst, &mut rotated, core::ROTATE_90_CLOCKWISE)?;
        Ok(rotated)
    } else {
        Ok(dst)
    }
}

#[cfg(not(feature = "use-opencv"))]
pub fn get_rotate_crop_image(img: &Mat, points: &[Point2f; 4]) -> ImgResult<Mat> {
    let w1 = (points[0].x - points[1].x).hypot(points[0].y - points[1].y);
    let w2 = (points[2].x - points[3].x).hypot(points[2].y - points[3].y);
    let img_crop_width = w1.max(w2) as i32;

    let h1 = (points[0].x - points[3].x).hypot(points[0].y - points[3].y);
    let h2 = (points[1].x - points[2].x).hypot(points[1].y - points[2].y);
    let img_crop_height = h1.max(h2) as i32;

    let pts_src = [
        [points[0].x, points[0].y],
        [points[1].x, points[1].y],
        [points[2].x, points[2].y],
        [points[3].x, points[3].y],
    ];

    let pts_dst = [
        [0.0f32, 0.0f32],
        [img_crop_width as f32, 0.0f32],
        [img_crop_width as f32, img_crop_height as f32],
        [0.0f32, img_crop_height as f32],
    ];

    let m = image_impl::get_perspective_transform(&pts_src, &pts_dst)?;
    let mut dst = Mat::default();
    image_impl::warp_perspective(
        img,
        &mut dst,
        &m,
        Size::new(img_crop_width, img_crop_height),
        2, // INTER_CUBIC
        image_impl::BORDER_REPLICATE,
    )?;

    let size = dst.size()?;
    let dst_h = size.height as f32;
    let dst_w = size.width as f32;
    if dst_h / dst_w >= 1.5 {
        let mut rotated = Mat::default();
        image_impl::rotate(&dst, &mut rotated, ROTATE_90_CLOCKWISE)?;
        Ok(rotated)
    } else {
        Ok(dst)
    }
}

pub fn resize_image_within_bounds(
    img: &Mat,
    min_side_len: f32,
    max_side_len: f32,
) -> Result<(Mat, f32, f32), EngineError> {
    let size = img.size()?;
    let mut h = size.height;
    let mut w = size.width;

    let mut ratio_h = 1.0f32;
    let mut ratio_w = 1.0f32;

    let max_value = h.max(w) as f32;
    let mut img_out = img.clone();
    if max_value > max_side_len {
        let (resized, rh, rw) = reduce_max_side(&img_out, max_side_len)?;
        img_out = resized;
        ratio_h = rh;
        ratio_w = rw;
    }

    let size2 = img_out.size()?;
    h = size2.height;
    w = size2.width;
    let min_value = h.min(w) as f32;
    if min_value < min_side_len {
        let (resized, rh, rw) = increase_min_side(&img_out, min_side_len)?;
        img_out = resized;
        ratio_h = rh;
        ratio_w = rw;
    }

    Ok((img_out, ratio_h, ratio_w))
}

pub fn reduce_max_side(img: &Mat, max_side_len: f32) -> Result<(Mat, f32, f32), EngineError> {
    let size = img.size()?;
    let h = size.height as f32;
    let w = size.width as f32;

    let mut ratio = 1.0f32;
    if h.max(w) > max_side_len {
        ratio = if h > w {
            max_side_len / h
        } else {
            max_side_len / w
        };
    }

    // Match Python: int(h * ratio) truncates, not rounds
    let mut resize_h = (h * ratio) as i32;
    let mut resize_w = (w * ratio) as i32;

    resize_h = ((resize_h as f32 / 32.0).round() * 32.0) as i32;
    resize_w = ((resize_w as f32 / 32.0).round() * 32.0) as i32;

    if resize_w <= 0 || resize_h <= 0 {
        return Err(EngineError::Preprocess(
            "resize_w or resize_h is less than or equal to 0".to_string(),
        ));
    }

    #[cfg(feature = "use-opencv")]
    let dst = {
        let mut d = Mat::default();
        imgproc::resize(
            img,
            &mut d,
            core::Size::new(resize_w, resize_h),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;
        d
    };

    #[cfg(not(feature = "use-opencv"))]
    let dst = {
        let mut d = Mat::default();
        image_impl::resize(img, &mut d, Size::new(resize_w, resize_h), INTER_LINEAR)?;
        d
    };

    let ratio_h = h / resize_h as f32;
    let ratio_w = w / resize_w as f32;
    Ok((dst, ratio_h, ratio_w))
}

pub fn increase_min_side(img: &Mat, min_side_len: f32) -> Result<(Mat, f32, f32), EngineError> {
    let size = img.size()?;
    let h = size.height as f32;
    let w = size.width as f32;

    let mut ratio = 1.0f32;
    if h.min(w) < min_side_len {
        ratio = if h < w {
            min_side_len / h
        } else {
            min_side_len / w
        };
    }

    // Match Python: int(h * ratio) truncates
    let mut resize_h = (h * ratio) as i32;
    let mut resize_w = (w * ratio) as i32;

    resize_h = ((resize_h as f32 / 32.0).round() * 32.0) as i32;
    resize_w = ((resize_w as f32 / 32.0).round() * 32.0) as i32;

    if resize_w <= 0 || resize_h <= 0 {
        return Err(EngineError::Preprocess(
            "resize_w or resize_h is less than or equal to 0".to_string(),
        ));
    }

    #[cfg(feature = "use-opencv")]
    let dst = {
        let mut d = Mat::default();
        imgproc::resize(
            img,
            &mut d,
            core::Size::new(resize_w, resize_h),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;
        d
    };

    #[cfg(not(feature = "use-opencv"))]
    let dst = {
        let mut d = Mat::default();
        image_impl::resize(img, &mut d, Size::new(resize_w, resize_h), INTER_LINEAR)?;
        d
    };

    let ratio_h = h / resize_h as f32;
    let ratio_w = w / resize_w as f32;
    Ok((dst, ratio_h, ratio_w))
}

#[cfg(feature = "use-opencv")]
pub fn add_round_letterbox(img: &Mat, padding: (i32, i32, i32, i32)) -> Result<Mat, EngineError> {
    let mut dst = Mat::default();
    core::copy_make_border(
        img,
        &mut dst,
        padding.0,
        padding.1,
        padding.2,
        padding.3,
        core::BORDER_CONSTANT,
        core::Scalar::new(0.0, 0.0, 0.0, 0.0),
    )?;
    Ok(dst)
}

#[cfg(not(feature = "use-opencv"))]
pub fn add_round_letterbox(img: &Mat, padding: (i32, i32, i32, i32)) -> Result<Mat, EngineError> {
    use image::{Rgb, RgbImage};

    let rgb_img = img.to_rgb8();
    let (width, height) = rgb_img.dimensions();

    let new_width = width + padding.2 as u32 + padding.3 as u32;
    let new_height = height + padding.0 as u32 + padding.1 as u32;

    let mut new_img = RgbImage::from_pixel(new_width, new_height, Rgb([0, 0, 0]));

    // Copy original image to center
    for y in 0..height {
        for x in 0..width {
            let pixel = rgb_img.get_pixel(x, y);
            new_img.put_pixel(x + padding.3 as u32, y + padding.0 as u32, *pixel);
        }
    }

    Ok(Mat::new(image::DynamicImage::ImageRgb8(new_img)))
}

pub fn iou(box1: &[f32; 4], box2: &[f32; 4]) -> f32 {
    let x1 = box1[0];
    let y1 = box1[1];
    let x2 = box1[2];
    let y2 = box1[3];

    let x1_p = box2[0];
    let y1_p = box2[1];
    let x2_p = box2[2];
    let y2_p = box2[3];

    let x1_i = x1.max(x1_p);
    let y1_i = y1.max(y1_p);
    let x2_i = x2.min(x2_p);
    let y2_i = y2.min(y2_p);

    let inter_area = (x2_i - x1_i + 1.0).max(0.0) * (y2_i - y1_i + 1.0).max(0.0);

    let box1_area = (x2 - x1 + 1.0) * (y2 - y1 + 1.0);
    let box2_area = (x2_p - x1_p + 1.0) * (y2_p - y1_p + 1.0);

    inter_area / (box1_area + box2_area - inter_area)
}

pub fn nms(boxes: &[[f32; 6]], iou_same: f32, iou_diff: f32) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..boxes.len()).collect();
    indices.sort_by(|&a, &b| boxes[b][1].partial_cmp(&boxes[a][1]).unwrap());

    let mut selected_indices = Vec::new();

    while let Some(&current) = indices.first() {
        selected_indices.push(current);
        let current_box = &boxes[current];
        let current_class = current_box[0];
        let current_coords = [
            current_box[2],
            current_box[3],
            current_box[4],
            current_box[5],
        ];

        indices.remove(0);

        indices.retain(|&i| {
            let box_ = &boxes[i];
            let box_class = box_[0];
            let box_coords = [box_[2], box_[3], box_[4], box_[5]];
            let iou_value = iou(&current_coords, &box_coords);
            let threshold = if current_class == box_class {
                iou_same
            } else {
                iou_diff
            };

            iou_value < threshold
        });
    }

    selected_indices
}

#[cfg(all(test, not(feature = "use-opencv")))]
mod degenerate_box_tests {
    use super::*;

    fn quad(x0: f32, y0: f32, x1: f32, y1: f32) -> [Point2f; 4] {
        [
            Point2f::new(x0, y0),
            Point2f::new(x1, y0),
            Point2f::new(x1, y1),
            Point2f::new(x0, y1),
        ]
    }

    fn page() -> Mat {
        Mat::from_rgb8(100, 40, vec![200; 100 * 40 * 3]).unwrap()
    }

    #[test]
    fn flat_and_sub_pixel_and_non_finite_quads_are_degenerate() {
        assert!(is_degenerate_quad(&quad(10.0, 40.0, 54.0, 40.0)), "zero height");
        assert!(is_degenerate_quad(&quad(10.0, 5.0, 10.0, 30.0)), "zero width");
        assert!(is_degenerate_quad(&quad(10.0, 5.0, 10.9, 30.0)), "width < 1 px");
        assert!(is_degenerate_quad(&quad(10.0, 5.0, 40.0, 5.5)), "height < 1 px");
        assert!(is_degenerate_quad(&quad(f32::NAN, 5.0, 40.0, 20.0)), "non-finite");
        assert!(!is_degenerate_quad(&quad(10.0, 5.0, 54.0, 20.0)));
        assert!(!is_degenerate_quad(&quad(10.0, 5.0, 11.0, 6.0)), "exactly 1 px is kept");
    }

    #[test]
    fn crop_of_degenerate_box_is_an_empty_image_not_an_error() {
        let img = page();
        let crop = crop_text_box(&img, &quad(10.0, 40.0, 54.0, 40.0)).expect("no error");
        assert!(crop.rows() <= 0 || crop.cols() <= 0);
        let collapsed = [Point2f::new(7.0, 7.0); 4];
        let crop = crop_text_box(&img, &collapsed).expect("fully collapsed quad: no error");
        assert!(crop.rows() <= 0 || crop.cols() <= 0);
    }

    fn q(p: [(f32, f32); 4]) -> [Point2f; 4] {
        p.map(|(x, y)| Point2f::new(x, y))
    }

    /// Quads that the detector produced on real scanned pages (300 dpi): they pass the
    /// size test, but three or four vertices are collinear, so the homography is singular.
    const REAL_SINGULAR_QUADS: [[(f32, f32); 4]; 5] = [
        [(12.907542, 271.0), (1019.14856, 250.83234), (1021.054, 271.0), (14.813035, 271.0)],
        [(2920.0972, 233.0), (2961.058, 233.0), (2953.0774, 233.0), (2912.1162, 233.0)],
        [(2.1519995, 227.0), (2089.747, 202.23608), (2090.91, 227.0), (3.3146038, 227.0)],
        [(5.148195, 219.0), (2094.2307, 193.55786), (2095.3652, 219.0), (6.28249, 219.0)],
        [(0.22055307, 137.0), (322.9253, 137.0), (321.5627, 137.0), (0.0, 137.0)],
    ];

    #[test]
    fn real_page_singular_quads_give_an_empty_crop_not_an_error() {
        let img = Mat::from_rgb8(3000, 300, vec![200; 3000 * 300 * 3]).unwrap();
        for (i, raw) in REAL_SINGULAR_QUADS.iter().enumerate() {
            let b = q(*raw);
            assert!(is_degenerate_quad(&b), "real quad {i} must be rejected");
            let crop = crop_text_box(&img, &b).unwrap_or_else(|e| panic!("real quad {i}: {e}"));
            assert!(crop.rows() <= 0 || crop.cols() <= 0, "real quad {i}");
        }
    }

    #[test]
    fn collinear_coincident_and_self_intersecting_quads_are_degenerate() {
        let img = page();
        let cases: [(&str, [(f32, f32); 4]); 5] = [
            ("collinear diagonal", [(0.0, 0.0), (30.0, 30.0), (60.0, 60.0), (90.0, 90.0)]),
            ("three collinear", [(10.0, 5.0), (60.0, 5.0), (90.0, 5.0), (10.0, 25.0)]),
            ("coincident points", [(10.0, 5.0), (10.0, 5.0), (60.0, 25.0), (10.0, 25.0)]),
            ("coincident opposite", [(10.0, 5.0), (60.0, 5.0), (10.0, 5.0), (10.0, 25.0)]),
            ("bow-tie order", [(10.0, 5.0), (60.0, 25.0), (60.0, 5.0), (10.0, 25.0)]),
        ];
        for (name, raw) in cases {
            let b = q(raw);
            assert!(is_degenerate_quad(&b), "{name}");
            let crop = crop_text_box(&img, &b).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert!(crop.rows() <= 0 || crop.cols() <= 0, "{name}");
        }
    }

    #[test]
    fn legitimate_quads_are_not_degenerate() {
        // thin wide line, rotated line and a perspective-skewed line all stay valid
        assert!(!is_degenerate_quad(&q([(0.0, 0.0), (1000.0, 0.0), (1000.0, 2.0), (0.0, 2.0)])));
        assert!(!is_degenerate_quad(&q([(10.0, 10.0), (60.0, 30.0), (50.0, 50.0), (0.0, 30.0)])));
        assert!(!is_degenerate_quad(&q([(10.0, 5.0), (98.0, 8.0), (95.0, 30.0), (12.0, 27.0)])));
        // counter-clockwise order is still a valid (mirrored) quad
        assert!(!is_degenerate_quad(&q([(10.0, 5.0), (10.0, 25.0), (60.0, 25.0), (60.0, 5.0)])));
    }

    #[test]
    fn singular_warp_error_is_typed_for_direct_callers_and_becomes_an_empty_crop() {
        let singular = [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 0.0, 1.0]];
        let mut dst = Mat::default();
        let err = image_impl::warp_perspective(&page(), &mut dst, &singular, Size::new(8, 8), 2, 0)
            .expect_err("direct callers still get the error");
        assert!(err.downcast_ref::<image_impl::SingularMatrixError>().is_some());
        assert_eq!(err.to_string(), "Matrix is not invertible");

        let recovered = empty_if_singular(Err(err)).expect("singular warp becomes an empty crop");
        assert!(recovered.rows() <= 0 || recovered.cols() <= 0);
        let other: ImgResult<Mat> = Err("SVD failed".into());
        assert!(empty_if_singular(other).is_err(), "other errors still propagate");
        assert!(empty_if_singular(Ok(page())).unwrap().cols() == 100);
    }

    #[test]
    fn crop_of_valid_box_is_unchanged() {
        let img = page();
        let b = quad(10.0, 5.0, 54.0, 20.0);
        let crop = crop_text_box(&img, &b).unwrap();
        let direct = get_rotate_crop_image(&img, &b).unwrap();
        assert_eq!((crop.cols(), crop.rows()), (direct.cols(), direct.rows()));
        assert_eq!((crop.cols(), crop.rows()), (44, 15));
    }
}
