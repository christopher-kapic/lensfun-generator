use anyhow::Result;
use image::GrayImage;
use nalgebra::{DMatrix, DVector};
use std::path::Path;

use crate::image_util::load_raw_for_tca;
use crate::models::TcaParams;

/// Analyze a RAW image to compute TCA (Transverse Chromatic Aberration) parameters.
///
/// Measures the radial displacement of red and blue channels relative to green
/// at high-contrast edges. Returns vr and vb scaling factors for the lensfun
/// poly3 TCA model.
pub fn analyze_tca(path: &Path, focal_length: f64) -> Result<TcaParams> {
    let img = load_raw_for_tca(path)?;
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();

    // Split into R, G, B grayscale channels
    let mut red = GrayImage::new(width, height);
    let mut green = GrayImage::new(width, height);
    let mut blue = GrayImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let p = rgb.get_pixel(x, y);
            red.put_pixel(x, y, image::Luma([p[0]]));
            green.put_pixel(x, y, image::Luma([p[1]]));
            blue.put_pixel(x, y, image::Luma([p[2]]));
        }
    }

    let cx = width as f64 / 2.0;
    let cy = height as f64 / 2.0;
    let r_max = (cx * cx + cy * cy).sqrt();

    // Compute Sobel gradients on green channel
    let (grad_mag, grad_x, grad_y) = sobel_gradients(&green);

    // Find threshold for strong edges (top 5% by gradient magnitude)
    let threshold = gradient_threshold(&grad_mag, 0.05);

    eprintln!("    Edge threshold: {:.1}", threshold);

    // Collect radial shift measurements at edge pixels
    let margin = 10u32;
    let profile_half = 5i32; // 11-pixel profile
    let mut measurements: Vec<(f64, f64, f64)> = Vec::new(); // (r_norm, shift_red, shift_blue)

    for y in margin..height - margin {
        for x in margin..width - margin {
            let idx = (y * width + x) as usize;
            let mag = grad_mag[idx];
            if mag < threshold {
                continue;
            }

            let gx = grad_x[idx];
            let gy = grad_y[idx];

            // Gradient direction (perpendicular to edge)
            let len = (gx * gx + gy * gy).sqrt();
            if len < 1e-6 {
                continue;
            }
            let dx = gx / len;
            let dy = gy / len;

            // Sample profiles along gradient direction in each channel
            let profile_r = sample_profile(&red, x, y, dx, dy, profile_half, width, height);
            let profile_g = sample_profile(&green, x, y, dx, dy, profile_half, width, height);
            let profile_b = sample_profile(&blue, x, y, dx, dy, profile_half, width, height);

            if profile_r.is_none() || profile_g.is_none() || profile_b.is_none() {
                continue;
            }

            let profile_r = profile_r.unwrap();
            let profile_g = profile_g.unwrap();
            let profile_b = profile_b.unwrap();

            // Find sub-pixel edge crossing in each channel
            let edge_g = find_edge_crossing(&profile_g);
            let edge_r = find_edge_crossing(&profile_r);
            let edge_b = find_edge_crossing(&profile_b);

            if edge_g.is_none() || edge_r.is_none() || edge_b.is_none() {
                continue;
            }

            let pos_g = edge_g.unwrap();
            let pos_r = edge_r.unwrap();
            let pos_b = edge_b.unwrap();

            // Convert profile-space shifts to image-space radial shifts
            let shift_r = pos_r - pos_g;
            let shift_b = pos_b - pos_g;

            // Compute radial position of this edge pixel
            let px = x as f64 - cx;
            let py = y as f64 - cy;
            let r = (px * px + py * py).sqrt();
            let r_norm = r / r_max;

            // Only use edges that are reasonably radial (gradient direction
            // roughly aligned with radial direction from center)
            if r > 10.0 {
                let radial_dx = px / r;
                let radial_dy = py / r;
                let alignment = (dx * radial_dx + dy * radial_dy).abs();
                if alignment < 0.3 {
                    continue; // Edge is mostly tangential, skip
                }

                // Project the profile shift onto the radial direction
                let radial_shift_r = shift_r * alignment;
                let radial_shift_b = shift_b * alignment;

                measurements.push((r_norm, radial_shift_r, radial_shift_b));
            }
        }
    }

    eprintln!("    {} edge measurements collected", measurements.len());

    if measurements.len() < 50 {
        eprintln!("    WARNING: Too few edge measurements, using default TCA (no correction)");
        return Ok(TcaParams {
            vr: 1.0,
            vb: 1.0,
            focal_length,
        });
    }

    // Filter outliers using median ± 3*MAD
    let measurements = filter_tca_outliers(&measurements);
    eprintln!(
        "    {} measurements after outlier rejection",
        measurements.len()
    );

    if measurements.len() < 20 {
        eprintln!("    WARNING: Too few measurements after filtering, using default TCA");
        return Ok(TcaParams {
            vr: 1.0,
            vb: 1.0,
            focal_length,
        });
    }

    // Bin measurements by radial distance
    let num_bins = 50;
    let mut bin_r_shift: Vec<Vec<f64>> = vec![Vec::new(); num_bins];
    let mut bin_b_shift: Vec<Vec<f64>> = vec![Vec::new(); num_bins];

    for &(r_norm, shift_r, shift_b) in &measurements {
        let bin = ((r_norm * num_bins as f64).floor() as usize).min(num_bins - 1);
        bin_r_shift[bin].push(shift_r);
        bin_b_shift[bin].push(shift_b);
    }

    // Compute median shift per bin
    let mut binned_data: Vec<(f64, f64, f64)> = Vec::new(); // (r_norm, median_shift_r, median_shift_b)
    for i in 0..num_bins {
        if bin_r_shift[i].len() >= 3 && bin_b_shift[i].len() >= 3 {
            let r_center = (i as f64 + 0.5) / num_bins as f64;
            let med_r = median(&mut bin_r_shift[i]);
            let med_b = median(&mut bin_b_shift[i]);
            binned_data.push((r_center, med_r, med_b));
        }
    }

    eprintln!("    {} radial bins with data", binned_data.len());

    if binned_data.len() < 5 {
        eprintln!("    WARNING: Too few radial bins, using default TCA");
        return Ok(TcaParams {
            vr: 1.0,
            vb: 1.0,
            focal_length,
        });
    }

    // Fit linear model: shift(r) = (vr - 1) * r  →  vr = 1 + slope
    // The TCA model says r_channel = vr * r_green, so a pixel at radius r
    // in green appears at radius vr*r in red. The measured shift is
    // shift_r ≈ (vr - 1) * r (in pixels along the radial direction).
    // But shifts are in profile-sample units, so we need to convert to
    // fractional radial scaling.
    //
    // We fit: shift(r) / r_pixels = (v - 1)
    // where r_pixels = r_norm * r_max

    let (vr, vb) = fit_tca_model(&binned_data, r_max);

    eprintln!("    TCA result: vr={:.6}, vb={:.6}", vr, vb);

    Ok(TcaParams {
        vr,
        vb,
        focal_length,
    })
}

/// Compute Sobel gradients on a grayscale image.
/// Returns (magnitude, gx, gy) as flat vectors indexed by y*width+x.
fn sobel_gradients(img: &GrayImage) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (w, h) = img.dimensions();
    let n = (w * h) as usize;
    let mut mag = vec![0.0_f64; n];
    let mut gx = vec![0.0_f64; n];
    let mut gy = vec![0.0_f64; n];

    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let idx = (y * w + x) as usize;
            let p = |dx: i32, dy: i32| -> f64 {
                img.get_pixel((x as i32 + dx) as u32, (y as i32 + dy) as u32)[0] as f64
            };

            let sx = -p(-1, -1) + p(1, -1) - 2.0 * p(-1, 0) + 2.0 * p(1, 0) - p(-1, 1) + p(1, 1);
            let sy = -p(-1, -1) - 2.0 * p(0, -1) - p(1, -1) + p(-1, 1) + 2.0 * p(0, 1) + p(1, 1);

            gx[idx] = sx;
            gy[idx] = sy;
            mag[idx] = (sx * sx + sy * sy).sqrt();
        }
    }

    (mag, gx, gy)
}

/// Find the gradient magnitude threshold for the top `fraction` of pixels.
fn gradient_threshold(mag: &[f64], fraction: f64) -> f64 {
    let mut sorted: Vec<f64> = mag.iter().copied().filter(|&v| v > 0.0).collect();
    if sorted.is_empty() {
        return 0.0;
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((1.0 - fraction) * sorted.len() as f64) as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Sample an 11-pixel intensity profile along a direction from (cx, cy).
/// Returns None if any sample falls outside image bounds.
fn sample_profile(
    img: &GrayImage,
    cx: u32,
    cy: u32,
    dx: f64,
    dy: f64,
    half: i32,
    width: u32,
    height: u32,
) -> Option<Vec<f64>> {
    let mut profile = Vec::with_capacity((2 * half + 1) as usize);

    for i in -half..=half {
        let sx = cx as f64 + i as f64 * dx;
        let sy = cy as f64 + i as f64 * dy;

        let ix = sx.round() as i32;
        let iy = sy.round() as i32;

        if ix < 0 || iy < 0 || ix >= width as i32 || iy >= height as i32 {
            return None;
        }

        profile.push(img.get_pixel(ix as u32, iy as u32)[0] as f64);
    }

    Some(profile)
}

/// Find the sub-pixel edge crossing position in a profile.
/// The edge crossing is where the intensity crosses the midpoint between
/// the profile's min and max values.
/// Returns the position as an offset from the profile center (0 = center).
fn find_edge_crossing(profile: &[f64]) -> Option<f64> {
    let min_val = profile.iter().copied().fold(f64::MAX, f64::min);
    let max_val = profile.iter().copied().fold(f64::MIN, f64::max);

    let range = max_val - min_val;
    if range < 10.0 {
        // Not a strong enough edge
        return None;
    }

    let midpoint = (min_val + max_val) / 2.0;
    let half = (profile.len() / 2) as f64;

    // Find the pair of adjacent samples that straddle the midpoint
    for i in 0..profile.len() - 1 {
        let v0 = profile[i];
        let v1 = profile[i + 1];

        if (v0 <= midpoint && v1 >= midpoint) || (v0 >= midpoint && v1 <= midpoint) {
            // Linear interpolation to find sub-pixel crossing
            let denom = v1 - v0;
            if denom.abs() < 1e-10 {
                continue;
            }
            let t = (midpoint - v0) / denom;
            let pos = i as f64 + t;
            return Some(pos - half);
        }
    }

    None
}

/// Filter TCA measurements using median ± 3*MAD on the shift/r_norm ratio.
fn filter_tca_outliers(measurements: &[(f64, f64, f64)]) -> Vec<(f64, f64, f64)> {
    if measurements.len() < 10 {
        return measurements.to_vec();
    }

    // Compute shift ratios (shift / r_norm) for red and blue
    let mut ratios_r: Vec<f64> = measurements
        .iter()
        .filter(|(r, _, _)| *r > 0.05)
        .map(|(r, sr, _)| sr / r)
        .collect();
    let mut ratios_b: Vec<f64> = measurements
        .iter()
        .filter(|(r, _, _)| *r > 0.05)
        .map(|(r, _, sb)| sb / r)
        .collect();

    if ratios_r.is_empty() || ratios_b.is_empty() {
        return measurements.to_vec();
    }

    let med_r = median(&mut ratios_r);
    let med_b = median(&mut ratios_b);

    let mad_r = median_absolute_deviation(&ratios_r, med_r);
    let mad_b = median_absolute_deviation(&ratios_b, med_b);

    let thresh_r = (3.0 * mad_r).max(0.5); // Minimum threshold in ratio units
    let thresh_b = (3.0 * mad_b).max(0.5);

    measurements
        .iter()
        .filter(|(r, sr, sb)| {
            if *r <= 0.05 {
                return false;
            }
            let ratio_r = sr / r;
            let ratio_b = sb / r;
            (ratio_r - med_r).abs() <= thresh_r && (ratio_b - med_b).abs() <= thresh_b
        })
        .copied()
        .collect()
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if values.len() % 2 == 0 {
        (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
    } else {
        values[values.len() / 2]
    }
}

fn median_absolute_deviation(values: &[f64], med: f64) -> f64 {
    let mut deviations: Vec<f64> = values.iter().map(|v| (v - med).abs()).collect();
    median(&mut deviations)
}

/// Fit the TCA model from binned measurements.
/// For each bin with center r and median shifts (shift_r, shift_b):
///   shift(r) ≈ (v - 1) * r * r_max  (shift in pixels)
/// So: v = 1 + shift / (r * r_max)
///
/// We use weighted least squares: shift = (v-1) * r * r_max
/// fitting for (v-1) as a single parameter.
fn fit_tca_model(binned_data: &[(f64, f64, f64)], r_max: f64) -> (f64, f64) {
    let n = binned_data.len();
    let mut a_mat = DMatrix::<f64>::zeros(n, 1);
    let mut b_r = DVector::<f64>::zeros(n);
    let mut b_b = DVector::<f64>::zeros(n);
    let mut w_mat = DMatrix::<f64>::zeros(n, n);

    for (i, &(r_norm, shift_r, shift_b)) in binned_data.iter().enumerate() {
        // shift = (v - 1) * r_norm * r_max
        a_mat[(i, 0)] = r_norm * r_max;
        b_r[i] = shift_r;
        b_b[i] = shift_b;
        // Weight outer bins more (they have larger shifts and more signal)
        w_mat[(i, i)] = 1.0 + 4.0 * r_norm * r_norm;
    }

    let lambda = 0.01;

    let at_w = a_mat.transpose() * &w_mat;
    let at_w_a = &at_w * &a_mat + lambda * DMatrix::<f64>::identity(1, 1);
    let at_w_br = &at_w * &b_r;
    let at_w_bb = &at_w * &b_b;

    let vr_minus_1 = at_w_a
        .clone()
        .lu()
        .solve(&at_w_br)
        .map(|s| s[0])
        .unwrap_or(0.0);
    let vb_minus_1 = at_w_a
        .lu()
        .solve(&at_w_bb)
        .map(|s| s[0])
        .unwrap_or(0.0);

    let vr = 1.0 + vr_minus_1;
    let vb = 1.0 + vb_minus_1;

    // Clamp to reasonable range — TCA scaling is always very close to 1.0
    let vr = vr.clamp(0.995, 1.005);
    let vb = vb.clamp(0.995, 1.005);

    (vr, vb)
}
