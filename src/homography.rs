use anyhow::{bail, Result};
use image::{DynamicImage, ImageBuffer, Rgb};
use nalgebra::Matrix3;
use rand::seq::SliceRandom;
use rand::thread_rng;

use crate::distortion::{detect_harris_corners, histogram_equalize, match_features};

/// Align a reference image to the raw image using homography estimation.
///
/// The reference image (e.g., from a phone) is taken from a different viewpoint,
/// so we estimate a homography to warp it into the raw image's coordinate system.
pub fn align_reference(raw: &DynamicImage, reference: &DynamicImage) -> Result<DynamicImage> {
    let raw_gray = raw.to_luma8();
    let ref_gray = reference.to_luma8();

    let raw_norm = histogram_equalize(&raw_gray);
    let ref_norm = histogram_equalize(&ref_gray);

    eprintln!("    Detecting features in raw image...");
    let raw_corners = detect_harris_corners(&raw_norm, 500);
    eprintln!("    Detecting features in reference image...");
    let ref_corners = detect_harris_corners(&ref_norm, 500);
    eprintln!(
        "    Found {} raw corners, {} reference corners",
        raw_corners.len(),
        ref_corners.len()
    );

    eprintln!("    Matching features (wider search radius)...");
    let matches = match_features(&raw_norm, &raw_corners, &ref_norm, &ref_corners, 0.4);
    eprintln!("    {} matches found", matches.len());

    if matches.len() < 8 {
        bail!(
            "Too few feature matches ({}) between raw and reference image. \
             Check that the reference image shows the same scene.",
            matches.len()
        );
    }

    let correspondences: Vec<(f64, f64, f64, f64)> = matches
        .iter()
        .map(|&(rx, ry, px, py)| (rx as f64, ry as f64, px as f64, py as f64))
        .collect();

    eprintln!("    Estimating homography with RANSAC...");
    let (h, inliers) = match estimate_homography_ransac(&correspondences, 1000, 3.0) {
        Some(result) => result,
        None => bail!(
            "RANSAC homography estimation failed. \
             Check that the reference image shows the same scene and has sufficient overlap."
        ),
    };

    eprintln!(
        "    Homography estimated with {} inliers out of {} matches",
        inliers.len(),
        matches.len()
    );

    eprintln!("    Warping reference image...");
    let warped = warp_image(
        &reference.to_rgb8(),
        &h,
        raw.width(),
        raw.height(),
    );

    Ok(DynamicImage::ImageRgb8(warped))
}

/// Estimate a homography from 4+ point correspondences using Direct Linear Transform (DLT).
///
/// Each correspondence is (src_x, src_y, dst_x, dst_y) mapping from source to destination.
/// The homography H maps dst -> src (so we can warp the reference into the raw frame).
///
/// Points are normalized for numerical stability before solving.
fn estimate_homography_dlt(correspondences: &[(f64, f64, f64, f64)]) -> Option<Matrix3<f64>> {
    let n = correspondences.len();
    if n < 4 {
        return None;
    }

    // Normalize source points (raw image coordinates)
    let (src_norm, t_src) = normalize_points(
        &correspondences.iter().map(|&(sx, sy, _, _)| (sx, sy)).collect::<Vec<_>>(),
    );
    // Normalize destination points (reference image coordinates)
    let (dst_norm, t_dst) = normalize_points(
        &correspondences.iter().map(|&(_, _, dx, dy)| (dx, dy)).collect::<Vec<_>>(),
    );

    // Build the 2N x 9 matrix A
    let rows = 2 * n;
    let mut a_data = vec![0.0_f64; rows * 9];

    for i in 0..n {
        let (sx, sy) = src_norm[i];
        let (dx, dy) = dst_norm[i];

        // Row 2i: [0, 0, 0, -dx, -dy, -1, sy*dx, sy*dy, sy]
        let row1 = 2 * i;
        a_data[row1 * 9 + 3] = -dx;
        a_data[row1 * 9 + 4] = -dy;
        a_data[row1 * 9 + 5] = -1.0;
        a_data[row1 * 9 + 6] = sy * dx;
        a_data[row1 * 9 + 7] = sy * dy;
        a_data[row1 * 9 + 8] = sy;

        // Row 2i+1: [dx, dy, 1, 0, 0, 0, -sx*dx, -sx*dy, -sx]
        let row2 = 2 * i + 1;
        a_data[row2 * 9 + 0] = dx;
        a_data[row2 * 9 + 1] = dy;
        a_data[row2 * 9 + 2] = 1.0;
        a_data[row2 * 9 + 6] = -sx * dx;
        a_data[row2 * 9 + 7] = -sx * dy;
        a_data[row2 * 9 + 8] = -sx;
    }

    // Create nalgebra matrix (column-major)
    let a_mat = nalgebra::DMatrix::from_row_slice(rows, 9, &a_data);

    // SVD: the homography is the last column of V (right singular vectors)
    let svd = a_mat.svd(false, true);
    let v_t = svd.v_t?;

    // Last row of V^T = last column of V
    let h_vec: Vec<f64> = (0..9).map(|j| v_t[(8, j)]).collect();

    let h_norm = Matrix3::new(
        h_vec[0], h_vec[1], h_vec[2],
        h_vec[3], h_vec[4], h_vec[5],
        h_vec[6], h_vec[7], h_vec[8],
    );

    // Denormalize: H = T_src^-1 * H_norm * T_dst
    let t_src_inv = t_src.try_inverse()?;
    let h = t_src_inv * h_norm * t_dst;

    // Normalize so h[(2,2)] = 1 (if non-zero)
    let scale = h[(2, 2)];
    if scale.abs() < 1e-10 {
        return None;
    }

    Some(h / scale)
}

/// Normalize a set of 2D points so that their centroid is at the origin
/// and their average distance from the origin is sqrt(2).
/// Returns (normalized_points, normalization_matrix).
fn normalize_points(points: &[(f64, f64)]) -> (Vec<(f64, f64)>, Matrix3<f64>) {
    let n = points.len() as f64;

    let mean_x: f64 = points.iter().map(|p| p.0).sum::<f64>() / n;
    let mean_y: f64 = points.iter().map(|p| p.1).sum::<f64>() / n;

    let avg_dist: f64 = points
        .iter()
        .map(|p| ((p.0 - mean_x).powi(2) + (p.1 - mean_y).powi(2)).sqrt())
        .sum::<f64>()
        / n;

    let scale = if avg_dist > 1e-10 {
        std::f64::consts::SQRT_2 / avg_dist
    } else {
        1.0
    };

    let normalized: Vec<(f64, f64)> = points
        .iter()
        .map(|p| ((p.0 - mean_x) * scale, (p.1 - mean_y) * scale))
        .collect();

    let t = Matrix3::new(
        scale, 0.0, -mean_x * scale,
        0.0, scale, -mean_y * scale,
        0.0, 0.0, 1.0,
    );

    (normalized, t)
}

/// Estimate homography using RANSAC for robustness to outliers.
///
/// Returns the best homography and the indices of inlier correspondences.
/// Requires at least 20 inliers for a valid result.
fn estimate_homography_ransac(
    correspondences: &[(f64, f64, f64, f64)],
    max_iter: usize,
    threshold: f64,
) -> Option<(Matrix3<f64>, Vec<usize>)> {
    let n = correspondences.len();
    if n < 4 {
        return None;
    }

    let mut rng = thread_rng();
    let indices: Vec<usize> = (0..n).collect();

    let mut best_inliers: Vec<usize> = Vec::new();
    let mut best_h: Option<Matrix3<f64>> = None;

    for _ in 0..max_iter {
        // Sample 4 random correspondences
        let sample: Vec<usize> = indices
            .choose_multiple(&mut rng, 4)
            .copied()
            .collect();

        let sample_corr: Vec<(f64, f64, f64, f64)> =
            sample.iter().map(|&i| correspondences[i]).collect();

        // Estimate homography from sample
        let h = match estimate_homography_dlt(&sample_corr) {
            Some(h) => h,
            None => continue,
        };

        // Count inliers
        let inliers: Vec<usize> = (0..n)
            .filter(|&i| {
                let (sx, sy, dx, dy) = correspondences[i];
                reprojection_error(&h, dx, dy, sx, sy) < threshold
            })
            .collect();

        if inliers.len() > best_inliers.len() {
            best_inliers = inliers;
            best_h = Some(h);
        }
    }

    // Require minimum 20 inliers
    if best_inliers.len() < 20 {
        eprintln!(
            "    RANSAC: only {} inliers found (minimum 20 required)",
            best_inliers.len()
        );
        return None;
    }

    // Re-estimate from all inliers for better accuracy
    let inlier_corr: Vec<(f64, f64, f64, f64)> = best_inliers
        .iter()
        .map(|&i| correspondences[i])
        .collect();

    let final_h = estimate_homography_dlt(&inlier_corr).or(best_h)?;

    // Recompute inliers with refined homography
    let final_inliers: Vec<usize> = (0..n)
        .filter(|&i| {
            let (sx, sy, dx, dy) = correspondences[i];
            reprojection_error(&final_h, dx, dy, sx, sy) < threshold
        })
        .collect();

    Some((final_h, final_inliers))
}

/// Compute the reprojection error for a single correspondence under homography H.
/// H maps (src_x, src_y) -> (dst_x, dst_y).
fn reprojection_error(h: &Matrix3<f64>, src_x: f64, src_y: f64, dst_x: f64, dst_y: f64) -> f64 {
    let w = h[(2, 0)] * src_x + h[(2, 1)] * src_y + h[(2, 2)];
    if w.abs() < 1e-10 {
        return f64::MAX;
    }

    let proj_x = (h[(0, 0)] * src_x + h[(0, 1)] * src_y + h[(0, 2)]) / w;
    let proj_y = (h[(1, 0)] * src_x + h[(1, 1)] * src_y + h[(1, 2)]) / w;

    ((proj_x - dst_x).powi(2) + (proj_y - dst_y).powi(2)).sqrt()
}

/// Warp an image using a homography via inverse mapping with bilinear interpolation.
///
/// H maps from the reference image to the raw image coordinate system.
/// For each pixel in the output, we find the corresponding position in the input.
fn warp_image(
    img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    h: &Matrix3<f64>,
    width: u32,
    height: u32,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let mut output = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(width, height);

    // We need H_inv to map from output (raw) coordinates back to input (reference) coordinates
    let h_inv = match h.try_inverse() {
        Some(inv) => inv,
        None => return output,
    };

    let (src_w, src_h) = img.dimensions();

    for y in 0..height {
        for x in 0..width {
            let xf = x as f64;
            let yf = y as f64;

            let w = h_inv[(2, 0)] * xf + h_inv[(2, 1)] * yf + h_inv[(2, 2)];
            if w.abs() < 1e-10 {
                continue;
            }

            let src_x = (h_inv[(0, 0)] * xf + h_inv[(0, 1)] * yf + h_inv[(0, 2)]) / w;
            let src_y = (h_inv[(1, 0)] * xf + h_inv[(1, 1)] * yf + h_inv[(1, 2)]) / w;

            let sx = src_x.floor() as i64;
            let sy = src_y.floor() as i64;
            let fx = src_x - sx as f64;
            let fy = src_y - sy as f64;

            if sx < 0 || sy < 0 || sx + 1 >= src_w as i64 || sy + 1 >= src_h as i64 {
                continue;
            }

            let p00 = img.get_pixel(sx as u32, sy as u32);
            let p10 = img.get_pixel((sx + 1) as u32, sy as u32);
            let p01 = img.get_pixel(sx as u32, (sy + 1) as u32);
            let p11 = img.get_pixel((sx + 1) as u32, (sy + 1) as u32);

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
