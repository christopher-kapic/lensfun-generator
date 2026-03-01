use image::{DynamicImage, ImageBuffer, Rgb};
use indicatif::{ProgressBar, ProgressStyle};

use crate::similarity::ssim;

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

/// Automatically optimize distortion parameters by comparing corrected raw
/// against the camera's embedded JPEG preview using SSIM.
///
/// Uses coordinate descent with ternary search: optimize a, b, c one at a time,
/// repeating passes until convergence.
/// Returns (a, b, c) distortion coefficients. The caller attaches the focal length.
pub fn optimize_distortion(raw_img: &DynamicImage, preview_img: &DynamicImage) -> (f64, f64, f64) {
    let raw_rgb = raw_img.to_rgb8();
    let target_gray = DynamicImage::ImageRgb8(preview_img.to_rgb8()).into_luma8();

    let max_passes = 5;
    let ternary_iters = 20;

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

    let evaluate = |a: f64, b: f64, c: f64| -> f64 {
        let corrected = apply_distortion(&raw_rgb, a, b, c);
        let corrected_gray = DynamicImage::ImageRgb8(corrected).into_luma8();
        ssim(&corrected_gray, &target_gray)
    };

    // First pass: coarse grid search to find a good starting region
    let grid_points = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3];
    let mut best_ssim = f64::NEG_INFINITY;
    for &ga in &grid_points {
        let s = evaluate(ga, 0.0, 0.0);
        if s > best_ssim {
            best_ssim = s;
            a = ga;
        }
    }
    for &gb in &grid_points {
        let s = evaluate(a, gb, 0.0);
        if s > best_ssim {
            best_ssim = s;
            b = gb;
        }
    }
    for &gc in &grid_points {
        let s = evaluate(a, b, gc);
        if s > best_ssim {
            best_ssim = s;
            c = gc;
        }
    }

    for _pass in 0..max_passes {
        let prev_a = a;
        let prev_b = b;
        let prev_c = c;

        // Optimize a
        a = ternary_search(|val| evaluate(val, b, c), -0.3, 0.3, ternary_iters);

        // Optimize b
        b = ternary_search(|val| evaluate(a, val, c), -0.3, 0.3, ternary_iters);

        // Optimize c
        c = ternary_search(|val| evaluate(a, b, val), -0.5, 0.5, ternary_iters);

        pb.inc(1);

        // Check convergence
        let delta = (a - prev_a).abs() + (b - prev_b).abs() + (c - prev_c).abs();
        if delta < 1e-6 {
            break;
        }
    }

    pb.finish_with_message("Distortion calibration complete");

    (a, b, c)
}

/// Ternary search for the maximum of a unimodal function on [lo, hi].
/// The function should return SSIM (higher = better).
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
