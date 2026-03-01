mod distortion;
mod exif;
mod image_util;
mod models;
mod similarity;
mod vignetting;
mod xml;

use anyhow::{bail, Context, Result};
use clap::Parser;
use dialoguer::Input;
use std::path::PathBuf;

use crate::exif::read_exif;
use crate::image_util::{find_raw_files, load_camera_preview, load_raw_uncorrected, match_dimensions};
use crate::models::{CalibrationProject, LensInfo, LensType};

#[derive(Parser)]
#[command(name = "lensfun-generate")]
#[command(about = "Generate lensfun lens calibration XML from RAW files")]
struct Cli {
    /// Path to directory containing distortion/ and vignette/ subdirectories
    path: PathBuf,

    /// Lens maker (auto-detected from EXIF if omitted)
    #[arg(long)]
    maker: Option<String>,

    /// Lens model (auto-detected from EXIF if omitted)
    #[arg(long)]
    model: Option<String>,

    /// Lens mount (e.g., "Sony E", "Canon RF")
    #[arg(long)]
    mount: Option<String>,

    /// Crop factor (default: 1.0)
    #[arg(long, default_value_t = 1.0)]
    crop_factor: f64,

    /// Output file path (default: print to stdout)
    #[arg(short, long)]
    output: Option<PathBuf>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let dir = &cli.path;

    if !dir.is_dir() {
        bail!("Not a directory: {}", dir.display());
    }

    // Discover files
    let distortion_dir = dir.join("distortion");
    let vignette_dir = dir.join("vignette");

    let distortion_files = if distortion_dir.is_dir() {
        find_raw_files(&distortion_dir)?
    } else {
        Vec::new()
    };

    let vignette_files = if vignette_dir.is_dir() {
        find_raw_files(&vignette_dir)?
    } else {
        Vec::new()
    };

    if distortion_files.is_empty() && vignette_files.is_empty() {
        bail!(
            "No RAW files found. Expected distortion/ and/or vignette/ subdirectories in {}",
            dir.display()
        );
    }

    // Read EXIF from first available file
    let first_file = distortion_files
        .first()
        .or(vignette_files.first())
        .unwrap();
    let exif_data = read_exif(first_file).unwrap_or_else(|e| {
        eprintln!("Warning: Could not read EXIF data: {e}");
        crate::exif::ExifData {
            focal_length: None,
            aperture: None,
            lens_model: None,
            camera_maker: None,
            camera_model: None,
        }
    });

    eprintln!("EXIF data from {}:", first_file.display());
    if let Some(ref m) = exif_data.camera_maker {
        eprintln!("  Camera maker: {m}");
    }
    if let Some(ref m) = exif_data.camera_model {
        eprintln!("  Camera model: {m}");
    }
    if let Some(ref m) = exif_data.lens_model {
        eprintln!("  Lens model: {m}");
    }
    if let Some(f) = exif_data.focal_length {
        eprintln!("  Focal length: {f}mm");
    }
    if let Some(a) = exif_data.aperture {
        eprintln!("  Aperture: f/{a}");
    }
    eprintln!();

    // Resolve lens info: CLI flags > EXIF > interactive prompt
    let maker = resolve_field(
        cli.maker.as_deref(),
        exif_data.camera_maker.as_deref(),
        "Lens maker",
    )?;
    let model = resolve_field(
        cli.model.as_deref(),
        exif_data.lens_model.as_deref(),
        "Lens model",
    )?;
    let mount = resolve_field(cli.mount.as_deref(), None, "Lens mount (e.g., Sony E)")?;

    let lens_info = LensInfo {
        maker,
        model,
        mount,
        crop_factor: cli.crop_factor,
        lens_type: LensType::Rectilinear,
    };

    // Distortion calibration
    let mut distortion_params = Vec::new();

    if !distortion_files.is_empty() {
        eprintln!(
            "Calibrating distortion ({} image{})...",
            distortion_files.len(),
            if distortion_files.len() == 1 { "" } else { "s" }
        );

        for dist_file in &distortion_files {
            eprintln!("\n  Processing: {}", dist_file.file_name().unwrap_or_default().to_string_lossy());

            let dist_exif = read_exif(dist_file).ok();
            let focal = dist_exif.as_ref().and_then(|e| e.focal_length).unwrap_or(0.0);

            eprintln!("    Focal length: {}mm", focal);
            eprintln!("    Decoding raw sensor data...");
            let raw_img = load_raw_uncorrected(dist_file)?;

            eprintln!("    Extracting camera preview...");
            let preview_img = load_camera_preview(dist_file)?;

            let (raw_img, preview_img) = match_dimensions(&raw_img, &preview_img);

            eprintln!(
                "    Raw: {}x{}, Preview: {}x{}",
                raw_img.width(),
                raw_img.height(),
                preview_img.width(),
                preview_img.height()
            );

            eprintln!("    Optimizing distortion parameters...");
            let (a, b, c) = distortion::optimize_distortion(&raw_img, &preview_img);
            eprintln!("    Result: a={:.6}, b={:.6}, c={:.6}", a, b, c);

            distortion_params.push(crate::models::DistortionParams {
                a,
                b,
                c,
                focal_length: focal,
            });
        }
    }

    // Vignetting calibration
    let mut vignetting_params = Vec::new();

    if !vignette_files.is_empty() {
        eprintln!("\nCalibrating vignetting ({} images)...", vignette_files.len());
        for file in &vignette_files {
            let vig_exif = read_exif(file).ok();
            let focal = vig_exif.as_ref().and_then(|e| e.focal_length).unwrap_or(0.0);
            let aperture = vig_exif.as_ref().and_then(|e| e.aperture).unwrap_or(0.0);

            eprintln!(
                "  Analyzing {} (f={}mm, F/{})...",
                file.file_name().unwrap_or_default().to_string_lossy(),
                focal,
                aperture
            );

            let params = vignetting::analyze_vignetting(file, focal, aperture)?;
            eprintln!(
                "    k1={:.6}, k2={:.6}, k3={:.6}",
                params.k1, params.k2, params.k3
            );
            vignetting_params.push(params);
        }
    }

    // Generate XML
    let project = CalibrationProject {
        lens_info,
        distortion_params,
        vignetting_params,
    };

    let xml_output = xml::generate_xml(&project)?;

    match cli.output {
        Some(ref path) => {
            std::fs::write(path, &xml_output)
                .with_context(|| format!("Failed to write output file: {}", path.display()))?;
            eprintln!("\nOutput written to: {}", path.display());
        }
        None => {
            eprintln!();
            println!("{xml_output}");
        }
    }

    Ok(())
}

/// Resolve a metadata field: CLI flag takes priority, then EXIF, then interactive prompt.
fn resolve_field(cli_value: Option<&str>, exif_value: Option<&str>, prompt: &str) -> Result<String> {
    if let Some(v) = cli_value {
        return Ok(v.to_string());
    }
    if let Some(v) = exif_value {
        return Ok(v.to_string());
    }
    let value: String = Input::new()
        .with_prompt(prompt)
        .interact_text()
        .context("Failed to read input")?;
    Ok(value)
}
