//! AES69 SOFA helpers.
//!
//! Reading is delegated to the slim `sofa-reader` crate.  Writing uses the
//! maintained HDF5 implementation so generated files are independently
//! consumable by standard SOFA/HDF5 readers.

#![cfg(feature = "io-sofa")]

pub use sofa_reader::{
    AttrValue, CoordinateSystem, HrtfData, Result as SofaResult, SimpleFreeFieldHrtf, SofaError,
    SofaFile, SofaReader, SofaWriter, SourcePosition,
};
use std::{path::Path, str::FromStr};

use hdf5::types::VarLenUnicode;

/// Write a standards-compliant `SimpleFreeFieldHRTF` SOFA file.
///
/// `sofa-reader` remains the read-side compatibility API.  Its compact HDF5
/// emitter is intentionally not used here: external HDF5 readers validate
/// object-header metadata that its private writer does not yet produce.
pub fn write_simple_free_field_hrtf<P: AsRef<Path>>(
    path: P,
    h: &SimpleFreeFieldHrtf<'_>,
) -> SofaResult<()> {
    validate_hrtf(h)?;
    let file = hdf5::File::create(path).map_err(hdf5_error)?;

    write_string_attribute(&file, "Conventions", "SOFA")?;
    write_string_attribute(&file, "Version", "2.1")?;
    write_string_attribute(&file, "SOFAConventions", "SimpleFreeFieldHRTF")?;
    write_string_attribute(&file, "SOFAConventionsVersion", "1.0")?;
    write_string_attribute(&file, "APIName", "sonium-xem")?;
    write_string_attribute(&file, "DataType", "TF")?;
    write_string_attribute(&file, "RoomType", "free field")?;
    write_string_attribute(&file, "Title", h.title.unwrap_or("HRTF"))?;

    let frequencies = file
        .new_dataset::<f64>()
        .shape(h.frequencies.len())
        .create("N")
        .map_err(hdf5_error)?;
    frequencies.write_raw(h.frequencies).map_err(hdf5_error)?;
    write_string_attribute(&frequencies, "Units", "hertz")?;

    write_dataset(
        &file,
        "Data.Real",
        &[h.measurements, h.receivers, h.frequencies.len()],
        h.real,
    )?;
    write_dataset(
        &file,
        "Data.Imag",
        &[h.measurements, h.receivers, h.frequencies.len()],
        h.imag,
    )?;

    let source = write_dataset(
        &file,
        "SourcePosition",
        &[h.measurements, 3],
        h.source_position,
    )?;
    write_coordinate_attributes(&source, h.source_coords)?;
    let receiver = write_dataset(
        &file,
        "ReceiverPosition",
        &[h.receivers, 3, 1],
        h.receiver_position,
    )?;
    write_coordinate_attributes(&receiver, h.receiver_coords)?;
    let listener = h.listener_position.unwrap_or([0.0, 0.0, 0.0]);
    let listener = write_dataset(&file, "ListenerPosition", &[1, 3], &listener)?;
    write_coordinate_attributes(&listener, CoordinateSystem::Cartesian)?;
    let emitter = write_dataset(&file, "EmitterPosition", &[3, 1], &[0.0, 0.0, 0.0])?;
    write_coordinate_attributes(&emitter, CoordinateSystem::Cartesian)?;
    Ok(())
}

fn write_dataset(
    file: &hdf5::File,
    name: &str,
    shape: &[usize],
    values: &[f64],
) -> SofaResult<hdf5::Dataset> {
    let dataset = file
        .new_dataset::<f64>()
        .shape(shape)
        .create(name)
        .map_err(hdf5_error)?;
    dataset.write_raw(values).map_err(hdf5_error)?;
    Ok(dataset)
}

fn write_string_attribute(location: &hdf5::Location, name: &str, value: &str) -> SofaResult<()> {
    let value = VarLenUnicode::from_str(value)
        .map_err(|error| SofaError::InvalidStructure(error.to_string()))?;
    location
        .new_attr::<VarLenUnicode>()
        .create(name)
        .map_err(hdf5_error)?
        .write_scalar(&value)
        .map_err(hdf5_error)
}

fn write_coordinate_attributes(
    dataset: &hdf5::Dataset,
    coordinates: CoordinateSystem,
) -> SofaResult<()> {
    let (kind, units) = match coordinates {
        CoordinateSystem::Cartesian => ("cartesian", "metre"),
        CoordinateSystem::Spherical => ("spherical", "degree, degree, metre"),
    };
    write_string_attribute(dataset, "Type", kind)?;
    write_string_attribute(dataset, "Units", units)
}

fn validate_hrtf(h: &SimpleFreeFieldHrtf<'_>) -> SofaResult<()> {
    let expected_data = h.measurements * h.receivers * h.frequencies.len();
    if h.real.len() != expected_data || h.imag.len() != expected_data {
        return Err(SofaError::InvalidStructure(
            "Data.Real and Data.Imag must have shape M×R×N".into(),
        ));
    }
    if h.source_position.len() != h.measurements * 3 {
        return Err(SofaError::InvalidStructure(
            "SourcePosition must have shape M×3".into(),
        ));
    }
    if h.receiver_position.len() != h.receivers * 3 {
        return Err(SofaError::InvalidStructure(
            "ReceiverPosition must have shape R×3".into(),
        ));
    }
    Ok(())
}

fn hdf5_error(error: hdf5::Error) -> SofaError {
    SofaError::InvalidStructure(format!("HDF5 SOFA write failed: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_minimal_hrtf() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("minimal.sofa");

        let real = vec![1.0, 2.0, 3.0, 4.0];
        let imag = vec![0.1, 0.2, 0.3, 0.4];
        let frequencies = vec![500.0, 1000.0];
        let source_position = vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
        let receiver_position = vec![0.0, 0.0, 0.0];

        let hrtf = SimpleFreeFieldHrtf {
            real: &real,
            imag: &imag,
            frequencies: &frequencies,
            source_position: &source_position,
            source_coords: CoordinateSystem::Cartesian,
            receiver_position: &receiver_position,
            receiver_coords: CoordinateSystem::Cartesian,
            listener_position: None,
            title: Some("minimal"),
            measurements: 2,
            receivers: 1,
        };

        write_simple_free_field_hrtf(&path, &hrtf).unwrap();

        let reader = hdf5::File::open(&path).unwrap();
        let convention = reader
            .attr("SOFAConventions")
            .unwrap()
            .read_scalar::<VarLenUnicode>()
            .unwrap();
        assert_eq!(convention.as_str(), "SimpleFreeFieldHRTF");
        assert_eq!(reader.dataset("Data.Real").unwrap().shape(), [2, 1, 2]);
        assert_eq!(reader.dataset("Data.Imag").unwrap().shape(), [2, 1, 2]);
        assert_eq!(
            reader
                .dataset("Data.Real")
                .unwrap()
                .read_raw::<f64>()
                .unwrap(),
            real
        );
        assert_eq!(
            reader
                .dataset("Data.Imag")
                .unwrap()
                .read_raw::<f64>()
                .unwrap(),
            imag
        );
        assert_eq!(
            reader.dataset("N").unwrap().read_raw::<f64>().unwrap(),
            frequencies
        );
        assert_eq!(
            reader
                .dataset("SourcePosition")
                .unwrap()
                .read_raw::<f64>()
                .unwrap(),
            source_position
        );
    }
}
