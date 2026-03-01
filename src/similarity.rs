use image::GrayImage;

/// Compute the Structural Similarity Index (SSIM) between two grayscale images.
/// Returns a value in [0, 1] where 1 means the images are identical.
///
/// Uses non-overlapping 8×8 block windows for speed. At 720px this runs in
/// a few milliseconds, which is sufficient since we call it hundreds of times
/// during the distortion optimization loop.
pub fn ssim(img_a: &GrayImage, img_b: &GrayImage) -> f64 {
    let (w, h) = img_a.dimensions();
    assert_eq!((w, h), img_b.dimensions(), "SSIM: images must be same size");

    // SSIM constants (from the original paper, using 8-bit dynamic range = 255)
    let c1 = (0.01_f64 * 255.0).powi(2);
    let c2 = (0.03_f64 * 255.0).powi(2);
    let window: u32 = 8;

    let mut ssim_sum = 0.0_f64;
    let mut count = 0u32;

    let mut y = 0u32;
    while y + window <= h {
        let mut x = 0u32;
        while x + window <= w {
            let (mean_a, mean_b, var_a, var_b, cov_ab) =
                window_stats(img_a, img_b, x, y, window);

            let numerator = (2.0 * mean_a * mean_b + c1) * (2.0 * cov_ab + c2);
            let denominator =
                (mean_a * mean_a + mean_b * mean_b + c1) * (var_a + var_b + c2);

            ssim_sum += numerator / denominator;
            count += 1;

            x += window;
        }
        y += window;
    }

    if count == 0 {
        return 0.0;
    }

    ssim_sum / count as f64
}

/// Compute mean, variance, and covariance for a window in both images.
fn window_stats(
    a: &GrayImage,
    b: &GrayImage,
    x0: u32,
    y0: u32,
    size: u32,
) -> (f64, f64, f64, f64, f64) {
    let n = (size * size) as f64;
    let mut sum_a = 0.0_f64;
    let mut sum_b = 0.0_f64;
    let mut sum_a2 = 0.0_f64;
    let mut sum_b2 = 0.0_f64;
    let mut sum_ab = 0.0_f64;

    for dy in 0..size {
        for dx in 0..size {
            let va = a.get_pixel(x0 + dx, y0 + dy)[0] as f64;
            let vb = b.get_pixel(x0 + dx, y0 + dy)[0] as f64;
            sum_a += va;
            sum_b += vb;
            sum_a2 += va * va;
            sum_b2 += vb * vb;
            sum_ab += va * vb;
        }
    }

    let mean_a = sum_a / n;
    let mean_b = sum_b / n;
    let var_a = sum_a2 / n - mean_a * mean_a;
    let var_b = sum_b2 / n - mean_b * mean_b;
    let cov_ab = sum_ab / n - mean_a * mean_b;

    (mean_a, mean_b, var_a, var_b, cov_ab)
}
