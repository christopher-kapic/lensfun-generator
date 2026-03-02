use anyhow::{Context, Result};
use quick_xml::events::{BytesDecl, BytesEnd, BytesStart, BytesText, Event};
use quick_xml::Writer;
use std::io::Cursor;

use crate::models::{CalibrationProject, LensType};

pub fn generate_xml(project: &CalibrationProject) -> Result<String> {
    let mut writer = Writer::new_with_indent(Cursor::new(Vec::new()), b' ', 4);

    writer
        .write_event(Event::Decl(BytesDecl::new("1.0", Some("UTF-8"), None)))
        .context("XML write error")?;

    let mut root = BytesStart::new("lensdatabase");
    root.push_attribute(("version", "2"));
    writer
        .write_event(Event::Start(root))
        .context("XML write error")?;

    writer
        .write_event(Event::Start(BytesStart::new("lens")))
        .context("XML write error")?;

    write_text_element(&mut writer, "maker", &project.lens_info.maker)?;
    write_text_element(&mut writer, "model", &project.lens_info.model)?;
    write_text_element(&mut writer, "mount", &project.lens_info.mount)?;
    write_text_element(
        &mut writer,
        "cropfactor",
        &project.lens_info.crop_factor.to_string(),
    )?;

    let lens_type_str = match project.lens_info.lens_type {
        LensType::Rectilinear => "rectilinear",
        LensType::Fisheye => "fisheye",
        LensType::Panoramic => "panoramic",
        LensType::Equirectangular => "equirectangular",
    };
    write_text_element(&mut writer, "type", lens_type_str)?;

    let has_calibration = !project.distortion_params.is_empty()
        || !project.vignetting_params.is_empty()
        || !project.tca_params.is_empty();

    if has_calibration {
        writer
            .write_event(Event::Start(BytesStart::new("calibration")))
            .context("XML write error")?;

        for dist in &project.distortion_params {
            let mut elem = BytesStart::new("distortion");
            elem.push_attribute(("model", "ptlens"));
            elem.push_attribute(("focal", &*format!("{}", dist.focal_length)));
            elem.push_attribute(("a", &*format!("{}", dist.a)));
            elem.push_attribute(("b", &*format!("{}", dist.b)));
            elem.push_attribute(("c", &*format!("{}", dist.c)));
            writer
                .write_event(Event::Empty(elem))
                .context("XML write error")?;
        }

        for vig in &project.vignetting_params {
            let mut elem = BytesStart::new("vignetting");
            elem.push_attribute(("model", "pa"));
            elem.push_attribute(("focal", &*format!("{}", vig.focal_length)));
            elem.push_attribute(("aperture", &*format!("{}", vig.aperture)));
            elem.push_attribute(("distance", &*format!("{}", vig.distance)));
            elem.push_attribute(("k1", &*format!("{}", vig.k1)));
            elem.push_attribute(("k2", &*format!("{}", vig.k2)));
            elem.push_attribute(("k3", &*format!("{}", vig.k3)));
            writer
                .write_event(Event::Empty(elem))
                .context("XML write error")?;
        }

        for tca in &project.tca_params {
            let mut elem = BytesStart::new("tca");
            elem.push_attribute(("model", "poly3"));
            elem.push_attribute(("focal", &*format!("{}", tca.focal_length)));
            elem.push_attribute(("vr", &*format!("{:.6}", tca.vr)));
            elem.push_attribute(("vb", &*format!("{:.6}", tca.vb)));
            writer
                .write_event(Event::Empty(elem))
                .context("XML write error")?;
        }

        writer
            .write_event(Event::End(BytesEnd::new("calibration")))
            .context("XML write error")?;
    }

    writer
        .write_event(Event::End(BytesEnd::new("lens")))
        .context("XML write error")?;

    writer
        .write_event(Event::End(BytesEnd::new("lensdatabase")))
        .context("XML write error")?;

    let result = writer.into_inner().into_inner();
    String::from_utf8(result).context("UTF-8 conversion error")
}

fn write_text_element(
    writer: &mut Writer<Cursor<Vec<u8>>>,
    tag: &str,
    text: &str,
) -> Result<()> {
    writer
        .write_event(Event::Start(BytesStart::new(tag)))
        .context("XML write error")?;
    writer
        .write_event(Event::Text(BytesText::new(text)))
        .context("XML write error")?;
    writer
        .write_event(Event::End(BytesEnd::new(tag)))
        .context("XML write error")?;
    Ok(())
}
