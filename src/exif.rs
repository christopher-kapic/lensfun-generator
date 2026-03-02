use anyhow::{Context, Result};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct ExifData {
    pub focal_length: Option<f64>,
    pub aperture: Option<f64>,
    pub focus_distance: Option<f64>,
    pub lens_model: Option<String>,
    pub camera_maker: Option<String>,
    pub camera_model: Option<String>,
}

pub fn read_exif(path: &Path) -> Result<ExifData> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open file: {}", path.display()))?;
    let mut reader = BufReader::new(file);

    let exif_reader = exif::Reader::new();
    let exif = exif_reader
        .read_from_container(&mut reader)
        .with_context(|| format!("Failed to read EXIF data from: {}", path.display()))?;

    let focal_length = exif
        .get_field(exif::Tag::FocalLength, exif::In::PRIMARY)
        .and_then(|f| match &f.value {
            exif::Value::Rational(v) if !v.is_empty() => Some(v[0].to_f64()),
            _ => None,
        });

    let aperture = exif
        .get_field(exif::Tag::FNumber, exif::In::PRIMARY)
        .and_then(|f| match &f.value {
            exif::Value::Rational(v) if !v.is_empty() => Some(v[0].to_f64()),
            _ => None,
        });

    let focus_distance = exif
        .get_field(exif::Tag::SubjectDistance, exif::In::PRIMARY)
        .and_then(|f| match &f.value {
            exif::Value::Rational(v) if !v.is_empty() => Some(v[0].to_f64()),
            _ => None,
        });

    let lens_model = exif
        .get_field(exif::Tag::LensModel, exif::In::PRIMARY)
        .map(|f| f.display_value().to_string().trim_matches('"').to_string());

    let camera_maker = exif
        .get_field(exif::Tag::Make, exif::In::PRIMARY)
        .map(|f| f.display_value().to_string().trim_matches('"').to_string());

    let camera_model = exif
        .get_field(exif::Tag::Model, exif::In::PRIMARY)
        .map(|f| f.display_value().to_string().trim_matches('"').to_string());

    Ok(ExifData {
        focal_length,
        aperture,
        focus_distance,
        lens_model,
        camera_maker,
        camera_model,
    })
}
