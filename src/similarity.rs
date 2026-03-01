use image::GrayImage;

/// Compute the Structural Similarity Index (SSIM) between two grayscale images.
/// Returns a value in [0, 1] where 1 means the images are identical.
///
/// Uses non-overlapping 8×8 block windows for speed.
pub fn ssim(img_a: &GrayImage, img_b: &GrayImage) -> f64 {
    let (w, h) = img_a.dimensions();
    assert_eq!((w, h), img_b.dimensions(), "SSIM: images must be same size");

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

/// Apply 3×3 Sobel edge detection to a grayscale image.
/// Returns an edge magnitude image (higher values = stronger edges).
pub fn sobel_edges(img: &GrayImage) -> GrayImage {
    let (w, h) = img.dimensions();
    let mut output = GrayImage::new(w, h);

    if w < 3 || h < 3 {
        return output;
    }

    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let p = |dx: i32, dy: i32| -> f64 {
                img.get_pixel((x as i32 + dx) as u32, (y as i32 + dy) as u32)[0] as f64
            };

            // Sobel kernels
            let gx = -p(-1, -1) + p(1, -1)
                - 2.0 * p(-1, 0) + 2.0 * p(1, 0)
                - p(-1, 1) + p(1, 1);

            let gy = -p(-1, -1) - 2.0 * p(0, -1) - p(1, -1)
                + p(-1, 1) + 2.0 * p(0, 1) + p(1, 1);

            let magnitude = (gx * gx + gy * gy).sqrt().clamp(0.0, 255.0) as u8;
            output.put_pixel(x, y, image::Luma([magnitude]));
        }
    }

    output
}

/// Combined similarity metric: blend of SSIM on luminance and SSIM on edge maps.
/// Edge SSIM is much more sensitive to geometric distortion (curved vs straight lines).
pub fn distortion_similarity(img_a: &GrayImage, img_b: &GrayImage) -> f64 {
    let luma_ssim = ssim(img_a, img_b);
    let edges_a = sobel_edges(img_a);
    let edges_b = sobel_edges(img_b);
    let edge_ssim = ssim(&edges_a, &edges_b);

    // Weight edges more heavily since that's what distortion affects most
    0.3 * luma_ssim + 0.7 * edge_ssim
}

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
