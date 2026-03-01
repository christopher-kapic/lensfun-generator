use anyhow::{Context, Result};
use image::imageops::FilterType;
use image::DynamicImage;
use rawler::decoders::RawDecodeParams;
use std::path::Path;

const RAW_EXTENSIONS: &[&str] = &[
    "nef", "arw", "cr2", "cr3", "dng", "raf", "rw2", "orf", "pef", "srw",
];

const MAX_DIM: u32 = 720;

pub fn is_raw_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| RAW_EXTENSIONS.contains(&ext.to_lowercase().as_str()))
        .unwrap_or(false)
}

/// Load raw sensor data (uncorrected, no in-camera corrections) and resize to 720px max.
pub fn load_raw_uncorrected(path: &Path) -> Result<DynamicImage> {
    let params = RawDecodeParams::default();
    let img = rawler::analyze::raw_to_srgb(path, &params)
        .with_context(|| format!("Failed to decode RAW file: {}", path.display()))?;
    Ok(resize(img, MAX_DIM))
}

/// Extract the camera's embedded JPEG preview (has in-camera lens corrections applied)
/// and resize to 720px max.
pub fn load_camera_preview(path: &Path) -> Result<DynamicImage> {
    let params = RawDecodeParams::default();
    let img = rawler::analyze::extract_preview_pixels(path, &params)
        .or_else(|_| rawler::analyze::extract_thumbnail_pixels(path, &params))
        .with_context(|| {
            format!(
                "Failed to extract embedded preview from: {}",
                path.display()
            )
        })?;
    Ok(resize(img, MAX_DIM))
}

/// Find all RAW files in a directory.
pub fn find_raw_files(dir: &Path) -> Result<Vec<std::path::PathBuf>> {
    let mut files = Vec::new();
    let entries = std::fs::read_dir(dir)
        .with_context(|| format!("Failed to read directory: {}", dir.display()))?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && is_raw_file(&path) {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

fn resize(img: DynamicImage, max_dim: u32) -> DynamicImage {
    if img.width() > max_dim || img.height() > max_dim {
        img.resize(max_dim, max_dim, FilterType::Lanczos3)
    } else {
        img
    }
}

/// Ensure two images have the same dimensions by center-cropping the larger one.
pub fn match_dimensions(a: &DynamicImage, b: &DynamicImage) -> (DynamicImage, DynamicImage) {
    let (aw, ah) = (a.width(), a.height());
    let (bw, bh) = (b.width(), b.height());

    if aw == bw && ah == bh {
        return (a.clone(), b.clone());
    }

    let target_w = aw.min(bw);
    let target_h = ah.min(bh);

    let crop_center = |img: &DynamicImage, tw: u32, th: u32| -> DynamicImage {
        let x = (img.width() - tw) / 2;
        let y = (img.height() - th) / 2;
        img.crop_imm(x, y, tw, th)
    };

    (
        crop_center(a, target_w, target_h),
        crop_center(b, target_w, target_h),
    )
}
