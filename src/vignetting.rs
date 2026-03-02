use anyhow::{bail, Context, Result};
use image::GenericImageView;
use nalgebra::{DMatrix, DVector};
use rawler::decoders::RawDecodeParams;
use std::path::Path;

use crate::image_util::is_raw_file;
use crate::models::VignettingParams;

/// Analyze a diffuser image and compute vignetting correction parameters.
///
/// Uses the raw sensor decode (not embedded preview) to get uncorrected vignetting data.
///
/// The vignetting model is: v(r) = 1 + k1*r^2 + k2*r^4 + k3*r^6
/// where r is normalized so r=1 at image corners.
///
/// Samples average brightness at concentric rings from center to corners,
/// then fits the polynomial using least squares.
pub fn analyze_vignetting(path: &Path, focal_length: f64, aperture: f64, distance: f64) -> Result<VignettingParams> {
    if !path.exists() {
        bail!("File not found: {}", path.display());
    }

    // Always use raw sensor data for vignetting analysis
    let img = if is_raw_file(path) {
        let params = RawDecodeParams::default();
        rawler::analyze::raw_to_srgb(path, &params)
            .with_context(|| format!("Failed to decode RAW file: {}", path.display()))?
    } else {
        image::open(path)
            .with_context(|| format!("Failed to open image: {}", path.display()))?
    };

    let (width, height) = img.dimensions();
    let rgb = img.to_rgb8();

    let cx = width as f64 / 2.0;
    let cy = height as f64 / 2.0;
    let r_max = (cx * cx + cy * cy).sqrt();

    let num_rings = 100;
    let mut ring_sum = vec![0.0_f64; num_rings];
    let mut ring_count = vec![0u64; num_rings];

    for y in 0..height {
        for x in 0..width {
            let dx = x as f64 - cx;
            let dy = y as f64 - cy;
            let r = (dx * dx + dy * dy).sqrt() / r_max;

            let ring_idx = (r * (num_rings as f64 - 1.0)).round() as usize;
            if ring_idx >= num_rings {
                continue;
            }

            let pixel = rgb.get_pixel(x, y);
            let brightness = (pixel[0] as f64 + pixel[1] as f64 + pixel[2] as f64) / 3.0;
            ring_sum[ring_idx] += brightness;
            ring_count[ring_idx] += 1;
        }
    }

    let mut ring_brightness: Vec<(f64, f64)> = Vec::new();
    for i in 0..num_rings {
        if ring_count[i] > 0 {
            let r = i as f64 / (num_rings as f64 - 1.0);
            let avg = ring_sum[i] / ring_count[i] as f64;
            ring_brightness.push((r, avg));
        }
    }

    if ring_brightness.is_empty() {
        bail!("No valid brightness data found in image");
    }

    let center_brightness = ring_brightness[0].1;
    if center_brightness < 1e-10 {
        bail!("Center brightness is zero — invalid diffuser image");
    }

    let normalized: Vec<(f64, f64)> = ring_brightness
        .iter()
        .map(|(r, b)| (*r, b / center_brightness))
        .collect();

    // Fit polynomial: v(r) = 1 + k1*r^2 + k2*r^4 + k3*r^6
    let n = normalized.len();
    let mut a_mat = DMatrix::<f64>::zeros(n, 3);
    let mut b_vec = DVector::<f64>::zeros(n);

    for (i, (r, v)) in normalized.iter().enumerate() {
        let r2 = r * r;
        let r4 = r2 * r2;
        let r6 = r4 * r2;
        a_mat[(i, 0)] = r2;
        a_mat[(i, 1)] = r4;
        a_mat[(i, 2)] = r6;
        b_vec[i] = v - 1.0;
    }

    let at_a = a_mat.transpose() * &a_mat;
    let at_b = a_mat.transpose() * &b_vec;

    let solution = at_a
        .lu()
        .solve(&at_b)
        .context("Failed to solve least squares system — matrix is singular")?;

    Ok(VignettingParams {
        k1: solution[0],
        k2: solution[1],
        k3: solution[2],
        focal_length,
        aperture,
        distance,
    })
}
