/// Lens type matching lensfun's lens type enumeration
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub enum LensType {
    Rectilinear,
    Fisheye,
    Panoramic,
    Equirectangular,
}

/// Core lens identification and metadata
#[derive(Clone, Debug)]
pub struct LensInfo {
    pub maker: String,
    pub model: String,
    pub mount: String,
    pub crop_factor: f64,
    pub lens_type: LensType,
}

/// Distortion correction parameters (ptlens model)
/// r_corrected = a*r^4 + b*r^3 + c*r^2 + (1-a-b-c)*r
#[derive(Clone, Debug)]
pub struct DistortionParams {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub focal_length: f64,
}

/// Vignetting correction parameters (pa model)
/// v(r) = 1 + k1*r^2 + k2*r^4 + k3*r^6
#[derive(Clone, Debug)]
pub struct VignettingParams {
    pub k1: f64,
    pub k2: f64,
    pub k3: f64,
    pub focal_length: f64,
    pub aperture: f64,
    pub distance: f64,
}

/// TCA (Transverse Chromatic Aberration) correction parameters (poly3 model)
/// Red and blue channel radial scaling factors relative to green.
#[derive(Clone, Debug)]
pub struct TcaParams {
    pub vr: f64,
    pub vb: f64,
    pub focal_length: f64,
}

/// Top-level calibration project containing all data needed for XML generation
#[derive(Clone, Debug)]
pub struct CalibrationProject {
    pub lens_info: LensInfo,
    pub distortion_params: Vec<DistortionParams>,
    pub vignetting_params: Vec<VignettingParams>,
    pub tca_params: Vec<TcaParams>,
}
