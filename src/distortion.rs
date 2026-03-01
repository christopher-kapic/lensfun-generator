use image::{DynamicImage, GrayImage, ImageBuffer, Rgb};
use indicatif::{ProgressBar, ProgressStyle};

use crate::similarity::distortion_similarity;

/// Apply the ptlens distortion model to an RGB image.
///
///   r_corrected = a*r^4 + b*r^3 + c*r^2 + (1-a-b-c)*r
///
/// where r is the normalized distance from the image center (0..1, with 1 = half-diagonal).
/// Uses bilinear interpolation for smooth remapping.
pub fn apply_distortion(
    rgb: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    a: f64,
    b: f64,
    c: f64,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let (width, height) = rgb.dimensions();
    let cx = width as f64 / 2.0;
    let cy = height as f64 / 2.0;
    let r_max = (cx * cx + cy * cy).sqrt();
    let d = 1.0 - a - b - c;

    let mut output = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let dx = x as f64 - cx;
            let dy = y as f64 - cy;
            let r = (dx * dx + dy * dy).sqrt() / r_max;

            if r < 1e-10 {
                output.put_pixel(x, y, *rgb.get_pixel(x, y));
                continue;
            }

            let r2 = r * r;
            let r3 = r2 * r;
            let r4 = r3 * r;
            let r_src = a * r4 + b * r3 + c * r2 + d * r;

            let scale = r_src / r;
            let src_x = cx + dx * scale;
            let src_y = cy + dy * scale;

            let sx = src_x.floor() as i64;
            let sy = src_y.floor() as i64;
            let fx = src_x - sx as f64;
            let fy = src_y - sy as f64;

            if sx < 0 || sy < 0 || sx + 1 >= width as i64 || sy + 1 >= height as i64 {
                output.put_pixel(x, y, Rgb([0, 0, 0]));
                continue;
            }

            let p00 = rgb.get_pixel(sx as u32, sy as u32);
            let p10 = rgb.get_pixel((sx + 1) as u32, sy as u32);
            let p01 = rgb.get_pixel(sx as u32, (sy + 1) as u32);
            let p11 = rgb.get_pixel((sx + 1) as u32, (sy + 1) as u32);

            let interpolate = |c: usize| -> u8 {
                let v = (1.0 - fx) * (1.0 - fy) * p00[c] as f64
                    + fx * (1.0 - fy) * p10[c] as f64
                    + (1.0 - fx) * fy * p01[c] as f64
                    + fx * fy * p11[c] as f64;
                v.clamp(0.0, 255.0) as u8
            };

            output.put_pixel(x, y, Rgb([interpolate(0), interpolate(1), interpolate(2)]));
        }
    }

    output
}

/// Find the largest centered rectangle in a distortion-corrected image that
/// contains no black (out-of-bounds) pixels. Returns (x, y, width, height).
///
/// We scan inward from each edge along the center axes to find where black
/// pixels end, giving us the valid crop region.
fn find_valid_crop(img: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> (u32, u32, u32, u32) {
    let (w, h) = img.dimensions();
    let cx = w / 2;
    let cy = h / 2;

    let is_black = |x: u32, y: u32| -> bool {
        let p = img.get_pixel(x, y);
        p[0] == 0 && p[1] == 0 && p[2] == 0
    };

    // Scan from left edge along center row
    let mut left = 0u32;
    for x in 0..cx {
        if is_black(x, cy) {
            left = x + 1;
        } else {
            break;
        }
    }

    // Scan from right edge
    let mut right = w;
    for x in (cx..w).rev() {
        if is_black(x, cy) {
            right = x;
        } else {
            break;
        }
    }

    // Scan from top edge along center column
    let mut top = 0u32;
    for y in 0..cy {
        if is_black(cx, y) {
            top = y + 1;
        } else {
            break;
        }
    }

    // Scan from bottom edge
    let mut bottom = h;
    for y in (cy..h).rev() {
        if is_black(cx, y) {
            bottom = y;
        } else {
            break;
        }
    }

    // Add a small margin to avoid edge artifacts
    let margin = 4u32;
    left = (left + margin).min(cx);
    top = (top + margin).min(cy);
    right = right.saturating_sub(margin).max(cx + 1);
    bottom = bottom.saturating_sub(margin).max(cy + 1);

    let crop_w = right - left;
    let crop_h = bottom - top;

    (left, top, crop_w, crop_h)
}

/// Crop a grayscale image to the given rectangle.
fn crop_gray(img: &GrayImage, x: u32, y: u32, w: u32, h: u32) -> GrayImage {
    DynamicImage::ImageLuma8(img.clone())
        .crop_imm(x, y, w, h)
        .into_luma8()
}

/// Automatically optimize distortion parameters by comparing corrected raw
/// against the camera's embedded JPEG preview using edge-weighted SSIM.
///
/// Uses coordinate descent with ternary search, alternating parameter order
/// across passes to avoid getting stuck in local optima.
///
/// Returns (a, b, c) distortion coefficients. The caller attaches the focal length.
pub fn optimize_distortion(raw_img: &DynamicImage, preview_img: &DynamicImage) -> (f64, f64, f64) {
    let raw_rgb = raw_img.to_rgb8();
    let target_gray = DynamicImage::ImageRgb8(preview_img.to_rgb8()).into_luma8();

    let max_passes = 10;
    let ternary_iters = 25;

    let pb = ProgressBar::new(max_passes as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{bar:40}] pass {pos}/{len}")
            .unwrap(),
    );
    pb.set_message("Optimizing distortion");

    let mut a = 0.0_f64;
    let mut b = 0.0_f64;
    let mut c = 0.0_f64;

    // Evaluate distortion parameters by:
    // 1. Applying the correction
    // 2. Cropping to the valid (non-black) region
    // 3. Cropping the target to the same region
    // 4. Comparing with edge-weighted SSIM
    let evaluate = |a: f64, b: f64, c: f64| -> f64 {
        let corrected = apply_distortion(&raw_rgb, a, b, c);
        let (cx, cy, cw, ch) = find_valid_crop(&corrected);
        if cw < 32 || ch < 32 {
            return 0.0; // Correction too extreme, almost no valid pixels
        }
        let corrected_gray = DynamicImage::ImageRgb8(corrected).into_luma8();
        let cropped_corrected = crop_gray(&corrected_gray, cx, cy, cw, ch);
        let cropped_target = crop_gray(&target_gray, cx, cy, cw, ch);
        distortion_similarity(&cropped_corrected, &cropped_target)
    };

    // Coarse grid search to find a good starting point
    let grid_points = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3];
    let mut best_score = f64::NEG_INFINITY;
    for &ga in &grid_points {
        let s = evaluate(ga, 0.0, 0.0);
        if s > best_score {
            best_score = s;
            a = ga;
        }
    }
    for &gb in &grid_points {
        let s = evaluate(a, gb, 0.0);
        if s > best_score {
            best_score = s;
            b = gb;
        }
    }
    for &gc in &grid_points {
        let s = evaluate(a, b, gc);
        if s > best_score {
            best_score = s;
            c = gc;
        }
    }

    for pass in 0..max_passes {
        let prev_a = a;
        let prev_b = b;
        let prev_c = c;

        // Alternate parameter order: forward on even passes, reverse on odd
        if pass % 2 == 0 {
            a = ternary_search(|val| evaluate(val, b, c), -0.3, 0.3, ternary_iters);
            b = ternary_search(|val| evaluate(a, val, c), -0.3, 0.3, ternary_iters);
            c = ternary_search(|val| evaluate(a, b, val), -0.5, 0.5, ternary_iters);
        } else {
            c = ternary_search(|val| evaluate(a, b, val), -0.5, 0.5, ternary_iters);
            b = ternary_search(|val| evaluate(a, val, c), -0.3, 0.3, ternary_iters);
            a = ternary_search(|val| evaluate(val, b, c), -0.3, 0.3, ternary_iters);
        }

        pb.inc(1);

        let delta = (a - prev_a).abs() + (b - prev_b).abs() + (c - prev_c).abs();
        if delta < 1e-6 {
            break;
        }
    }

    pb.finish_with_message("Distortion calibration complete");

    let final_score = evaluate(a, b, c);
    eprintln!("    Final similarity score: {:.6}", final_score);

    (a, b, c)
}

/// Ternary search for the maximum of a unimodal function on [lo, hi].
fn ternary_search<F>(f: F, mut lo: f64, mut hi: f64, iterations: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    for _ in 0..iterations {
        let m1 = lo + (hi - lo) / 3.0;
        let m2 = hi - (hi - lo) / 3.0;

        if f(m1) < f(m2) {
            lo = m1;
        } else {
            hi = m2;
        }
    }

    (lo + hi) / 2.0
}
