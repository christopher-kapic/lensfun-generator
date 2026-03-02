use image::{DynamicImage, GrayImage, ImageBuffer, Rgb};
use nalgebra::{DMatrix, DVector};

/// Apply the ptlens distortion model to an RGB image.
///
///   r_corrected = a*r^4 + b*r^3 + c*r^2 + (1-a-b-c)*r
///
/// where r is the normalized distance from the image center (0..1, with 1 = half-diagonal).
/// Uses bilinear interpolation for smooth remapping.
#[allow(dead_code)]
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

/// Automatically determine distortion parameters by detecting feature points
/// in both the raw decode and camera preview, matching them, and solving
/// for the ptlens model coefficients via least-squares.
///
/// The ptlens model maps output (corrected) coordinates to source (distorted):
///   r_src = a*r^4 + b*r^3 + c*r^2 + (1-a-b-c)*r
///
/// This is linear in a, b, c, so given matched feature correspondences between
/// the distorted raw and the corrected preview, we can solve directly.
pub fn optimize_distortion(raw_img: &DynamicImage, preview_img: &DynamicImage) -> (f64, f64, f64) {
    let raw_gray = raw_img.to_luma8();
    let preview_gray = preview_img.to_luma8();

    let (w, h) = raw_gray.dimensions();
    let cx = w as f64 / 2.0;
    let cy = h as f64 / 2.0;
    let r_max = (cx * cx + cy * cy).sqrt();

    // Normalize both images for better matching across different tone curves
    let raw_norm = histogram_equalize(&raw_gray);
    let preview_norm = histogram_equalize(&preview_gray);

    eprintln!("    Detecting features...");
    let raw_corners = detect_harris_corners(&raw_norm, 500);
    let preview_corners = detect_harris_corners(&preview_norm, 500);
    eprintln!(
        "    Found {} raw corners, {} preview corners",
        raw_corners.len(),
        preview_corners.len()
    );

    eprintln!("    Matching features...");
    let matches = match_features(&raw_norm, &raw_corners, &preview_norm, &preview_corners);
    eprintln!("    {} matches found", matches.len());

    if matches.len() < 10 {
        eprintln!("    WARNING: Too few matches, falling back to zero correction");
        return (0.0, 0.0, 0.0);
    }

    // Build the linear system from matched points.
    //
    // The raw image is distorted, the preview is corrected.
    // ptlens model: r_src = a*r_out^4 + b*r_out^3 + c*r_out^2 + (1-a-b-c)*r_out
    //
    // raw point → distorted position (r_src)
    // preview point → corrected position (r_out)
    //
    // Rearranging:
    //   r_src - r_out = a*(r_out^4 - r_out) + b*(r_out^3 - r_out) + c*(r_out^2 - r_out)
    //
    // This is a linear system A*x = b where x = [a, b, c].

    // First, filter outliers using RANSAC-like approach
    let inliers = filter_outliers(&matches, cx, cy, r_max);
    eprintln!("    {} inliers after outlier rejection", inliers.len());

    if inliers.len() < 6 {
        eprintln!("    WARNING: Too few inliers, falling back to zero correction");
        return (0.0, 0.0, 0.0);
    }

    // Log the radial distribution of inliers so we can assess coverage
    let mut radial_bins = [0u32; 5]; // 0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0
    for &(rx, ry, _, _) in &inliers {
        let r = ((rx as f64 - cx).powi(2) + (ry as f64 - cy).powi(2)).sqrt() / r_max;
        let bin = (r * 5.0).min(4.0) as usize;
        radial_bins[bin] += 1;
    }
    eprintln!(
        "    Radial distribution: center={}, mid-inner={}, mid={}, mid-outer={}, edge={}",
        radial_bins[0], radial_bins[1], radial_bins[2], radial_bins[3], radial_bins[4]
    );

    // Step 1: Estimate the scale difference between raw and preview.
    //
    // The raw decode (raw_to_srgb) often has a slightly different field of view
    // than the camera's embedded JPEG preview (which may crop to the "active area").
    // Even a 2% FOV difference produces large, oscillating polynomial coefficients
    // if not accounted for. We estimate the scale from near-center matches where
    // distortion is minimal.
    let mut scale_ratios: Vec<f64> = Vec::new();
    for &(raw_x, raw_y, prev_x, prev_y) in &inliers {
        let r_raw = ((raw_x as f64 - cx).powi(2) + (raw_y as f64 - cy).powi(2)).sqrt() / r_max;
        let r_prev = ((prev_x as f64 - cx).powi(2) + (prev_y as f64 - cy).powi(2)).sqrt() / r_max;
        // Only use inner matches (r < 0.3) where distortion is negligible,
        // so the ratio reflects pure scale difference, not distortion.
        if r_prev > 0.05 && r_prev < 0.3 {
            scale_ratios.push(r_raw / r_prev);
        }
    }
    scale_ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let scale = if !scale_ratios.is_empty() {
        scale_ratios[scale_ratios.len() / 2] // median
    } else {
        1.0
    };
    eprintln!("    Estimated raw/preview scale factor: {:.6}", scale);

    // Step 2: Build the linear system with scale-corrected radii.
    //
    // After dividing r_raw by the scale factor, the remaining difference
    // between r_raw_corrected and r_preview is purely distortion.
    //
    // r_raw/scale - r_prev = a*(r_prev^4 - r_prev) + b*(r_prev^3 - r_prev) + c*(r_prev^2 - r_prev)
    let n = inliers.len();
    let mut a_mat = DMatrix::<f64>::zeros(n, 3);
    let mut b_vec = DVector::<f64>::zeros(n);
    let mut w_mat = DMatrix::<f64>::zeros(n, n);

    for (i, &(raw_x, raw_y, prev_x, prev_y)) in inliers.iter().enumerate() {
        let raw_dx = raw_x as f64 - cx;
        let raw_dy = raw_y as f64 - cy;
        let r_src = (raw_dx * raw_dx + raw_dy * raw_dy).sqrt() / r_max;
        let r_src_scaled = r_src / scale; // Remove scale difference

        let prev_dx = prev_x as f64 - cx;
        let prev_dy = prev_y as f64 - cy;
        let r_out = (prev_dx * prev_dx + prev_dy * prev_dy).sqrt() / r_max;

        if r_out < 1e-6 {
            continue;
        }

        let r_out2 = r_out * r_out;
        let r_out3 = r_out2 * r_out;
        let r_out4 = r_out3 * r_out;

        a_mat[(i, 0)] = r_out4 - r_out;
        a_mat[(i, 1)] = r_out3 - r_out;
        a_mat[(i, 2)] = r_out2 - r_out;
        b_vec[i] = r_src_scaled - r_out;

        // Weight by radial position — outer points matter more for distortion.
        let weight = 1.0 + 4.0 * r_out * r_out;
        w_mat[(i, i)] = weight;
    }

    // Two-stage fitting: first determine c (the dominant barrel/pincushion
    // term), then lock it and fit a + b for any residual higher-order
    // distortion. This prevents the higher-order terms from destabilizing
    // the primary correction.
    //
    // Each stage uses weighted least squares with mild regularization.
    let lambda = 0.1;

    // Stage 1: fit c only (column 2 of a_mat) — this captures the dominant
    // barrel/pincushion distortion and is locked for subsequent fitting.
    let col_c = a_mat.column(2).clone_owned();
    let col_c_mat = DMatrix::from_column_slice(n, 1, col_c.as_slice());
    let (c_raw, rmse_c) = solve_regularized(&col_c_mat, &b_vec, &w_mat, lambda);
    let c_val = c_raw[0];
    let c_final = -c_val;
    eprintln!("    Stage 1 (c only):  c={:.6}, RMSE={:.6}", c_final, rmse_c);

    // Stage 2: fit a + b with c locked. Subtract c's contribution from the
    // RHS so we solve for the residual that a and b must explain.
    let b_residual = &b_vec - c_val * &col_c;
    let cols_ab = a_mat.columns(0, 2).clone_owned(); // columns 0 (a) and 1 (b)
    let (ab_raw, rmse_ab) = solve_regularized(&cols_ab, &b_residual, &w_mat, lambda);
    let a_val = ab_raw[0];
    let b_val = ab_raw[1];
    eprintln!(
        "    Stage 2 (a+b|c):  a={:.6}, b={:.6}, c={:.6}, RMSE={:.6}",
        -a_val, -b_val, c_final, rmse_ab
    );

    // Choose the simplest model that fits well enough.
    // Only add a and b if they meaningfully improve on c alone (>50% RMSE reduction).
    let (a_final, b_final, stage);
    let improvement_ab = if rmse_c > 1e-12 { (rmse_c - rmse_ab) / rmse_c } else { 0.0 };

    if improvement_ab > 0.50 {
        a_final = -a_val;
        b_final = -b_val;
        stage = 2;
    } else {
        a_final = 0.0;
        b_final = 0.0;
        stage = 1;
    }

    let d = 1.0 - a_final - b_final - c_final;
    eprintln!(
        "    Selected stage {} (a+b improvement: {:.1}%)",
        stage,
        improvement_ab * 100.0
    );
    eprintln!("    d (linear term) = {:.6} (should be close to 1.0)", d);

    (a_final, b_final, c_final)
}

/// Solve a weighted least squares problem with Tikhonov regularization.
/// Returns (solution_vector, rmse).
fn solve_regularized(
    a: &DMatrix<f64>,
    b: &DVector<f64>,
    w: &DMatrix<f64>,
    lambda: f64,
) -> (DVector<f64>, f64) {
    let ncols = a.ncols();
    let nrows = a.nrows();
    let at_w = a.transpose() * w;
    let at_w_a = &at_w * a + lambda * DMatrix::<f64>::identity(ncols, ncols);
    let at_w_b = &at_w * b;

    match at_w_a.lu().solve(&at_w_b) {
        Some(solution) => {
            let residual = a * &solution - b;
            let rmse = (residual.dot(&residual) / nrows as f64).sqrt();
            (solution, rmse)
        }
        None => (DVector::zeros(ncols), f64::MAX),
    }
}

/// Simple histogram equalization to normalize brightness/contrast.
/// This helps match features between the raw decode (flat, low-contrast)
/// and the camera preview (punchy, contrasty S-curve).
fn histogram_equalize(img: &GrayImage) -> GrayImage {
    let (w, h) = img.dimensions();
    let total = (w * h) as f64;

    // Build histogram
    let mut hist = [0u32; 256];
    for p in img.pixels() {
        hist[p[0] as usize] += 1;
    }

    // Build CDF
    let mut cdf = [0.0_f64; 256];
    cdf[0] = hist[0] as f64 / total;
    for i in 1..256 {
        cdf[i] = cdf[i - 1] + hist[i] as f64 / total;
    }

    // Map pixels
    let mut output = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let v = img.get_pixel(x, y)[0] as usize;
            let mapped = (cdf[v] * 255.0).round().clamp(0.0, 255.0) as u8;
            output.put_pixel(x, y, image::Luma([mapped]));
        }
    }

    output
}

/// Detect Harris corners in a grayscale image.
/// Returns up to `max_corners` strongest corners as (x, y) positions.
fn detect_harris_corners(img: &GrayImage, max_corners: usize) -> Vec<(u32, u32)> {
    let (w, h) = img.dimensions();
    if w < 10 || h < 10 {
        return Vec::new();
    }

    // Compute image gradients (Sobel)
    let mut ix = vec![0.0_f64; (w * h) as usize];
    let mut iy = vec![0.0_f64; (w * h) as usize];

    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let idx = (y * w + x) as usize;
            let p = |dx: i32, dy: i32| -> f64 {
                img.get_pixel((x as i32 + dx) as u32, (y as i32 + dy) as u32)[0] as f64
            };
            ix[idx] = -p(-1, -1) + p(1, -1) - 2.0 * p(-1, 0) + 2.0 * p(1, 0) - p(-1, 1) + p(1, 1);
            iy[idx] = -p(-1, -1) - 2.0 * p(0, -1) - p(1, -1) + p(-1, 1) + 2.0 * p(0, 1) + p(1, 1);
        }
    }

    // Compute Harris response: R = det(M) - k * trace(M)^2
    // where M = sum over window of [[Ix^2, Ix*Iy], [Ix*Iy, Iy^2]]
    let k = 0.04;
    let win = 3i32; // window half-size
    let mut responses: Vec<(f64, u32, u32)> = Vec::new();

    // Skip edges of image to avoid border artifacts and ensure patches fit
    let margin = 16u32;
    for y in margin..h - margin {
        for x in margin..w - margin {
            let mut sxx = 0.0_f64;
            let mut syy = 0.0_f64;
            let mut sxy = 0.0_f64;

            for dy in -win..=win {
                for dx in -win..=win {
                    let idx = ((y as i32 + dy) as u32 * w + (x as i32 + dx) as u32) as usize;
                    sxx += ix[idx] * ix[idx];
                    syy += iy[idx] * iy[idx];
                    sxy += ix[idx] * iy[idx];
                }
            }

            let det = sxx * syy - sxy * sxy;
            let trace = sxx + syy;
            let r = det - k * trace * trace;

            if r > 0.0 {
                responses.push((r, x, y));
            }
        }
    }

    // Sort by response strength (strongest first)
    responses.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    // Non-maximum suppression: keep only the strongest corner within a radius
    let suppress_radius = 8u32;
    let mut corners = Vec::new();

    for &(_, x, y) in &responses {
        let too_close = corners.iter().any(|&(cx, cy): &(u32, u32)| {
            let dx = x as i32 - cx as i32;
            let dy = y as i32 - cy as i32;
            (dx * dx + dy * dy) < (suppress_radius * suppress_radius) as i32
        });

        if !too_close {
            corners.push((x, y));
            if corners.len() >= max_corners {
                break;
            }
        }
    }

    corners
}

/// Match features between two images using normalized cross-correlation (NCC)
/// on local patches. NCC is invariant to linear brightness/contrast changes.
///
/// Returns matched pairs: (raw_x, raw_y, preview_x, preview_y).
fn match_features(
    raw: &GrayImage,
    raw_corners: &[(u32, u32)],
    preview: &GrayImage,
    preview_corners: &[(u32, u32)],
) -> Vec<(u32, u32, u32, u32)> {
    let patch_half = 12i32; // 25x25 patches

    let mut matches = Vec::new();

    // For each raw corner, find the best matching preview corner
    for &(rx, ry) in raw_corners {
        let mut best_ncc = -1.0_f64;
        let mut second_ncc = -1.0_f64;
        let mut best_match = (0u32, 0u32);

        for &(px, py) in preview_corners {
            // Only consider matches within a reasonable search radius
            // (distortion shouldn't move points more than ~10% of image size)
            let dx = rx as i32 - px as i32;
            let dy = ry as i32 - py as i32;
            let dist2 = dx * dx + dy * dy;
            let max_dist = (raw.width().max(raw.height()) as i32) / 5;
            if dist2 > max_dist * max_dist {
                continue;
            }

            let ncc = compute_ncc(raw, rx, ry, preview, px, py, patch_half);

            if ncc > best_ncc {
                second_ncc = best_ncc;
                best_ncc = ncc;
                best_match = (px, py);
            } else if ncc > second_ncc {
                second_ncc = ncc;
            }
        }

        // Lowe's ratio test: best match must be significantly better than second best
        if best_ncc > 0.6 && (second_ncc < 0.0 || best_ncc > second_ncc * 1.2) {
            matches.push((rx, ry, best_match.0, best_match.1));
        }
    }

    matches
}

/// Compute normalized cross-correlation between two patches.
/// Returns a value in [-1, 1] where 1 = perfect match.
/// NCC is invariant to linear brightness and contrast differences.
fn compute_ncc(
    img_a: &GrayImage,
    ax: u32,
    ay: u32,
    img_b: &GrayImage,
    bx: u32,
    by: u32,
    half: i32,
) -> f64 {
    let mut sum_a = 0.0_f64;
    let mut sum_b = 0.0_f64;
    let mut n = 0.0_f64;

    // Compute means
    for dy in -half..=half {
        for dx in -half..=half {
            let va = img_a.get_pixel((ax as i32 + dx) as u32, (ay as i32 + dy) as u32)[0] as f64;
            let vb = img_b.get_pixel((bx as i32 + dx) as u32, (by as i32 + dy) as u32)[0] as f64;
            sum_a += va;
            sum_b += vb;
            n += 1.0;
        }
    }

    let mean_a = sum_a / n;
    let mean_b = sum_b / n;

    // Compute NCC
    let mut cov = 0.0_f64;
    let mut var_a = 0.0_f64;
    let mut var_b = 0.0_f64;

    for dy in -half..=half {
        for dx in -half..=half {
            let va = img_a.get_pixel((ax as i32 + dx) as u32, (ay as i32 + dy) as u32)[0] as f64
                - mean_a;
            let vb = img_b.get_pixel((bx as i32 + dx) as u32, (by as i32 + dy) as u32)[0] as f64
                - mean_b;
            cov += va * vb;
            var_a += va * va;
            var_b += vb * vb;
        }
    }

    let denom = (var_a * var_b).sqrt();
    if denom < 1e-10 {
        return 0.0;
    }

    cov / denom
}

/// Filter outlier matches using a simple median-based approach.
///
/// For each match, compute the radial displacement ratio. Outliers will have
/// ratios that deviate significantly from the median.
fn filter_outliers(
    matches: &[(u32, u32, u32, u32)],
    cx: f64,
    cy: f64,
    r_max: f64,
) -> Vec<(u32, u32, u32, u32)> {
    if matches.len() < 4 {
        return matches.to_vec();
    }

    // Compute radial displacement ratio for each match
    let mut ratios: Vec<(f64, usize)> = Vec::new();
    for (i, &(rx, ry, px, py)) in matches.iter().enumerate() {
        let r_raw = ((rx as f64 - cx).powi(2) + (ry as f64 - cy).powi(2)).sqrt() / r_max;
        let r_prev = ((px as f64 - cx).powi(2) + (py as f64 - cy).powi(2)).sqrt() / r_max;

        if r_prev > 0.05 {
            // Skip center points where ratio is unstable
            ratios.push((r_raw / r_prev, i));
        }
    }

    if ratios.is_empty() {
        return matches.to_vec();
    }

    // Find median ratio
    ratios.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let median = ratios[ratios.len() / 2].0;

    // Keep matches within 3 * MAD (median absolute deviation) of the median
    let mut deviations: Vec<f64> = ratios.iter().map(|(r, _)| (r - median).abs()).collect();
    deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mad = deviations[deviations.len() / 2];
    let threshold = (3.0 * mad).max(0.02); // At least 2% tolerance

    let inlier_indices: Vec<usize> = ratios
        .iter()
        .filter(|(r, _)| (r - median).abs() <= threshold)
        .map(|(_, i)| *i)
        .collect();

    inlier_indices.iter().map(|&i| matches[i]).collect()
}
