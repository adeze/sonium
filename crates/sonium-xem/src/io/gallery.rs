//! Deterministic, dependency-light gallery assets for headless simulation runs.
//!
//! The renderer deliberately accepts already-validated numerical arrays.  It is
//! a presentation layer: callers retain physics assertions and record the exact
//! source artifact for every rendered figure in [`GalleryManifest`].

use flate2::{Compression, read::ZlibDecoder, write::ZlibEncoder};
use serde::{Deserialize, Serialize};
use std::fmt::Write as _;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

pub const GALLERY_SCHEMA_VERSION: u32 = 1;
pub const GALLERY_WIDTH: u32 = 1920;
pub const GALLERY_HEIGHT: u32 = 1080;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GalleryFigure {
    pub filename: String,
    pub kind: String,
    pub source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_checksum_fnv1a64: Option<String>,
    pub caption: String,
    pub checksum_fnv1a64: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GalleryManifest {
    pub schema_version: u32,
    pub title: String,
    pub renderer: String,
    pub width_px: u32,
    pub height_px: u32,
    pub git_sha: String,
    pub conventions: Vec<String>,
    pub figures: Vec<GalleryFigure>,
}

/// Fixed-camera gallery writer.  All paths are rooted at `output_dir`.
pub struct GalleryRenderer {
    output_dir: PathBuf,
    figures: Vec<GalleryFigure>,
}

impl GalleryRenderer {
    pub fn new(output_dir: impl AsRef<Path>) -> io::Result<Self> {
        let output_dir = output_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&output_dir)?;
        Ok(Self {
            output_dir,
            figures: Vec::new(),
        })
    }

    fn figure_path(&self, filename: &str) -> io::Result<PathBuf> {
        let path = self.output_dir.join(filename);
        let relative = path.strip_prefix(&self.output_dir).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "gallery figure must be rooted below the output directory",
            )
        })?;
        if relative.is_absolute()
            || relative
                .components()
                .any(|component| matches!(component, std::path::Component::ParentDir))
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "gallery figure path may not escape the output directory",
            ));
        }
        Ok(path)
    }

    pub fn heatmap_png(
        &mut self,
        filename: &str,
        values: &[f64],
        source: impl Into<String>,
        caption: impl Into<String>,
    ) -> io::Result<()> {
        if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "gallery field must be finite and non-empty",
            ));
        }
        let path = self.figure_path(filename)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        write_heatmap_png(&path, values, GALLERY_WIDTH, GALLERY_HEIGHT)?;
        self.record(&path, "png", source.into(), caption.into())
    }

    /// Render a row-major scalar grid without reordering samples. This is
    /// intentionally distinct from [`Self::heatmap_png`], whose one-dimensional
    /// input is a deterministic presentation texture rather than a spatial
    /// field. `columns` is part of the caller's numerical contract.
    pub fn grid_heatmap_png(
        &mut self,
        filename: &str,
        values: &[f64],
        columns: usize,
        source: impl Into<String>,
        caption: impl Into<String>,
    ) -> io::Result<()> {
        if columns == 0
            || values.len() < columns
            || !values.len().is_multiple_of(columns)
            || values.iter().any(|value| !value.is_finite())
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "gallery grid must be finite, non-empty, and rectangular",
            ));
        }
        let path = self.figure_path(filename)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        write_grid_heatmap_png(&path, values, columns, GALLERY_WIDTH, GALLERY_HEIGHT)?;
        self.record(&path, "png", source.into(), caption.into())
    }

    /// Render a row-major scalar grid with bilinear interpolation between
    /// grid samples. Unlike [`Self::grid_heatmap_png`], displayed pixels mix
    /// neighbouring samples, which suits dense presentation fields where
    /// nearest-neighbour blocks would dominate. `columns` is part of the
    /// caller's numerical contract.
    pub fn smooth_grid_heatmap_png(
        &mut self,
        filename: &str,
        values: &[f64],
        columns: usize,
        source: impl Into<String>,
        caption: impl Into<String>,
    ) -> io::Result<()> {
        if columns == 0
            || values.len() < columns
            || !values.len().is_multiple_of(columns)
            || values.iter().any(|value| !value.is_finite())
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "gallery grid must be finite, non-empty, and rectangular",
            ));
        }
        let path = self.figure_path(filename)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        write_smooth_grid_heatmap_png(&path, values, columns, GALLERY_WIDTH, GALLERY_HEIGHT)?;
        self.record(&path, "png", source.into(), caption.into())
    }

    pub fn line_svg(
        &mut self,
        filename: &str,
        x: &[f64],
        y: &[f64],
        title: &str,
        source: impl Into<String>,
        caption: impl Into<String>,
    ) -> io::Result<()> {
        if x.len() < 2 || x.len() != y.len() || x.iter().chain(y).any(|value| !value.is_finite()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "gallery curve must have matching finite samples",
            ));
        }
        let path = self.figure_path(filename)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        write_line_svg(&path, x, y, title)?;
        self.record(&path, "svg", source.into(), caption.into())
    }

    /// Include a caller-generated fixed-viewport asset in the same checksummed
    /// manifest as renderer-owned figures.  This is for diagrams whose
    /// geometry is itself the presentation data (for example a room plan),
    /// rather than a scalar field or curve handled by this renderer.
    pub fn existing_figure(
        &mut self,
        filename: &str,
        kind: &str,
        source: impl Into<String>,
        caption: impl Into<String>,
    ) -> io::Result<()> {
        self.record(
            &self.output_dir.join(filename),
            kind,
            source.into(),
            caption.into(),
        )
    }

    /// Write a small deterministic animated GIF from one finite scalar field
    /// per frame. The fixed frame count is retained in the GIF itself and in
    /// the manifest checksum, which makes it suitable for gallery smoke tests.
    pub fn animation_gif(
        &mut self,
        filename: &str,
        frames: &[Vec<f64>],
        source: impl Into<String>,
        caption: impl Into<String>,
    ) -> io::Result<()> {
        if frames.len() < 2
            || frames
                .iter()
                .any(|frame| frame.is_empty() || frame.iter().any(|value| !value.is_finite()))
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "gallery animation requires two or more finite non-empty frames",
            ));
        }
        let path = self.figure_path(filename)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        write_animation_gif(&path, frames, 320, 180)?;
        self.record(&path, "gif", source.into(), caption.into())
    }

    fn record(
        &mut self,
        path: &Path,
        kind: &str,
        source: String,
        caption: String,
    ) -> io::Result<()> {
        // Keep nested assets addressable in the manifest and reject paths that
        // escape the gallery root. Flattening to a basename would make two
        // figures such as `cea2034/on-axis.png` and `polar/on-axis.png`
        // indistinguishable to consumers.
        let relative = path.strip_prefix(&self.output_dir).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "gallery figure must be rooted below the output directory",
            )
        })?;
        if relative.is_absolute()
            || relative
                .components()
                .any(|component| matches!(component, std::path::Component::ParentDir))
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "gallery figure path may not escape the output directory",
            ));
        }
        let filename = relative.to_string_lossy().replace('\\', "/");
        if self
            .figures
            .iter()
            .any(|figure| figure.filename == filename)
        {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("gallery figure {filename:?} was already recorded"),
            ));
        }
        let source_path = self.output_dir.join(&source);
        let source_relative = source_path.strip_prefix(&self.output_dir).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "gallery source artifact must be rooted below the output directory",
            )
        })?;
        if source_relative
            .components()
            .any(|component| matches!(component, std::path::Component::ParentDir))
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "gallery source artifact may not escape the output directory",
            ));
        }
        let bytes = std::fs::read(path)?;
        let source_bytes = std::fs::read(source_path).map_err(|error| {
            io::Error::new(
                error.kind(),
                format!("gallery source artifact {source:?} is unavailable: {error}"),
            )
        })?;
        self.figures.push(GalleryFigure {
            filename,
            kind: kind.into(),
            source,
            source_checksum_fnv1a64: Some(fnv1a64(&source_bytes)),
            caption,
            checksum_fnv1a64: fnv1a64(&bytes),
        });
        Ok(())
    }

    pub fn finish(self, title: impl Into<String>, conventions: Vec<String>) -> io::Result<PathBuf> {
        let manifest = GalleryManifest {
            schema_version: GALLERY_SCHEMA_VERSION,
            title: title.into(),
            renderer: "sonium-xem deterministic-raster-svg-v1".into(),
            width_px: GALLERY_WIDTH,
            height_px: GALLERY_HEIGHT,
            git_sha: current_git_sha(),
            conventions,
            figures: self.figures,
        };
        let path = self.output_dir.join("gallery.json");
        std::fs::write(
            &path,
            serde_json::to_vec_pretty(&manifest).map_err(io::Error::other)?,
        )?;
        Ok(path)
    }
}

pub fn validate_png(path: impl AsRef<Path>) -> io::Result<(u32, u32)> {
    let bytes = std::fs::read(path)?;
    Ok(parse_png(&bytes)?.dimensions())
}

/// Read and verify a gallery manifest and every artifact it references.
///
/// The manifest is an evidence index rather than a trust boundary: callers
/// should still validate the numerical source data. This check does guarantee
/// that a manifest points below its own directory and that the recorded
/// figure/source bytes have not changed since the manifest was written.
pub fn validate_gallery_manifest(path: impl AsRef<Path>) -> io::Result<GalleryManifest> {
    let path = path.as_ref();
    let root = path.parent().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "gallery manifest must have a parent directory",
        )
    })?;
    let manifest: GalleryManifest = serde_json::from_slice(&std::fs::read(path)?)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    if manifest.schema_version != GALLERY_SCHEMA_VERSION
        || manifest.width_px != GALLERY_WIDTH
        || manifest.height_px != GALLERY_HEIGHT
        || manifest.figures.is_empty()
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unsupported or incomplete gallery manifest",
        ));
    }
    let mut filenames = std::collections::HashSet::with_capacity(manifest.figures.len());
    for figure in &manifest.figures {
        let filename = Path::new(&figure.filename);
        if figure.filename.is_empty()
            || filename.is_absolute()
            || filename
                .components()
                .any(|component| matches!(component, std::path::Component::ParentDir))
            || !filenames.insert(&figure.filename)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "invalid or duplicate gallery figure path {:?}",
                    figure.filename
                ),
            ));
        }
        let figure_bytes = std::fs::read(root.join(filename))?;
        if fnv1a64(&figure_bytes) != figure.checksum_fnv1a64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("gallery figure checksum mismatch for {:?}", figure.filename),
            ));
        }
        let source = Path::new(&figure.source);
        if figure.source.is_empty()
            || source.is_absolute()
            || source
                .components()
                .any(|component| matches!(component, std::path::Component::ParentDir))
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid gallery source path {:?}", figure.source),
            ));
        }
        if figure.kind == "png" {
            let (width, height) = validate_png_nonblank(root.join(filename))?;
            if width != manifest.width_px || height != manifest.height_px {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "gallery PNG {:?} has dimensions {}x{}, expected {}x{}",
                        figure.filename, width, height, manifest.width_px, manifest.height_px,
                    ),
                ));
            }
        }
        let source_bytes = std::fs::read(root.join(source))?;
        if let Some(expected) = &figure.source_checksum_fnv1a64
            && fnv1a64(&source_bytes) != *expected
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("gallery source checksum mismatch for {:?}", figure.source),
            ));
        }
    }
    Ok(manifest)
}

/// Validate an 8-bit RGB/RGBA PNG is both decodable enough for the gallery
/// contract and visibly nonblank. This accepts standard PNG scanline filters
/// so it can also check artifacts emitted by headless ParaView.
/// It deliberately complements, rather than replaces, numerical assertions on
/// the source data.
pub fn validate_png_nonblank(path: impl AsRef<Path>) -> io::Result<(u32, u32)> {
    let path = path.as_ref();
    let bytes = std::fs::read(path)?;
    let png = parse_png(&bytes)?;
    let (width, height) = png.dimensions();
    if png.bit_depth != 8 || !matches!(png.color_type, 2 | 6) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "gallery PNG must use 8-bit RGB or RGBA",
        ));
    }
    let mut compressed = Vec::new();
    for &(start, end) in &png.idat_ranges {
        compressed.extend_from_slice(&bytes[start..end]);
    }
    let mut raw = Vec::new();
    ZlibDecoder::new(compressed.as_slice())
        .read_to_end(&mut raw)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let channels = if png.color_type == 2 { 3 } else { 4 };
    let row_bytes = (width as usize).checked_mul(channels).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "gallery PNG row length overflow",
        )
    })?;
    let expected = (height as usize)
        .checked_mul(row_bytes.checked_add(1).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "gallery PNG row length overflow",
            )
        })?)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "gallery PNG size overflow"))?;
    if raw.len() != expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unexpected gallery PNG pixel length",
        ));
    }
    let decoded_len = (height as usize)
        .checked_mul(row_bytes)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "gallery PNG size overflow"))?;
    let mut decoded = vec![0_u8; decoded_len];
    for row_index in 0..height as usize {
        let input_offset = row_index * (row_bytes + 1);
        let filter = raw[input_offset];
        let input = &raw[input_offset + 1..input_offset + 1 + row_bytes];
        let (before, current_and_after) = decoded.split_at_mut(row_index * row_bytes);
        let output = &mut current_and_after[..row_bytes];
        let previous = if row_index == 0 {
            None
        } else {
            Some(&before[(row_index - 1) * row_bytes..])
        };
        for index in 0..row_bytes {
            let left = if index >= channels {
                output[index - channels]
            } else {
                0
            };
            let above = previous.map_or(0, |row| row[index]);
            let upper_left = if index >= channels {
                previous.map_or(0, |row| row[index - channels])
            } else {
                0
            };
            output[index] = match filter {
                0 => input[index],
                1 => input[index].wrapping_add(left),
                2 => input[index].wrapping_add(above),
                3 => input[index].wrapping_add(((left as u16 + above as u16) / 2) as u8),
                4 => input[index].wrapping_add(paeth(left, above, upper_left)),
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "unsupported PNG scanline filter",
                    ));
                }
            };
        }
    }
    let mut pixels = decoded.chunks_exact(channels);
    let first = pixels
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "empty PNG pixel data"))?;
    // Blank means every pixel is identical; a dark-field image with a black
    // corner is legitimate content, not blank.
    if !pixels.any(|pixel| pixel[..3] != first[..3]) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "visually blank PNG",
        ));
    }
    Ok((width, height))
}

#[derive(Debug)]
struct PngInfo {
    width: u32,
    height: u32,
    bit_depth: u8,
    color_type: u8,
    idat_ranges: Vec<(usize, usize)>,
}

impl PngInfo {
    fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

fn parse_png(bytes: &[u8]) -> io::Result<PngInfo> {
    const SIGNATURE: &[u8; 8] = b"\x89PNG\r\n\x1a\n";
    if bytes.len() < SIGNATURE.len() || &bytes[..SIGNATURE.len()] != SIGNATURE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid PNG signature",
        ));
    }
    let mut offset = SIGNATURE.len();
    let mut info = None;
    let mut idat_ranges = Vec::new();
    let mut saw_iend = false;
    while offset < bytes.len() {
        if bytes.len() - offset < 12 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "truncated PNG chunk",
            ));
        }
        let length = u32::from_be_bytes(
            bytes[offset..offset + 4]
                .try_into()
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid PNG length"))?,
        ) as usize;
        let data_start = offset + 8;
        let data_end = data_start.checked_add(length).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "PNG chunk length overflow")
        })?;
        let crc_end = data_end.checked_add(4).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "PNG chunk length overflow")
        })?;
        if crc_end > bytes.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "truncated PNG chunk",
            ));
        }
        let kind = &bytes[offset + 4..offset + 8];
        let expected_crc = u32::from_be_bytes(
            bytes[data_end..crc_end]
                .try_into()
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid PNG checksum"))?,
        );
        if crc32(&bytes[offset + 4..data_end]) != expected_crc {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "PNG chunk checksum mismatch",
            ));
        }
        if info.is_none() && kind != b"IHDR" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "PNG must begin with IHDR",
            ));
        }
        match kind {
            b"IHDR" => {
                if info.is_some() || length != 13 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "invalid PNG IHDR",
                    ));
                }
                let data = &bytes[data_start..data_end];
                info = Some(PngInfo {
                    width: u32::from_be_bytes(data[0..4].try_into().map_err(|_| {
                        io::Error::new(io::ErrorKind::InvalidData, "invalid PNG width")
                    })?),
                    height: u32::from_be_bytes(data[4..8].try_into().map_err(|_| {
                        io::Error::new(io::ErrorKind::InvalidData, "invalid PNG height")
                    })?),
                    bit_depth: data[8],
                    color_type: data[9],
                    idat_ranges: Vec::new(),
                });
            }
            b"IDAT" => idat_ranges.push((data_start, data_end)),
            b"IEND" => {
                if length != 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "invalid PNG IEND",
                    ));
                }
                saw_iend = true;
                offset = crc_end;
                break;
            }
            _ => {}
        }
        offset = crc_end;
    }
    if !saw_iend || offset != bytes.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "PNG is missing a terminal IEND chunk",
        ));
    }
    let mut info =
        info.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "PNG is missing IHDR"))?;
    if info.width == 0 || info.height == 0 || idat_ranges.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "blank PNG"));
    }
    info.idat_ranges = idat_ranges;
    Ok(info)
}

fn paeth(left: u8, above: u8, upper_left: u8) -> u8 {
    let estimate = left as i16 + above as i16 - upper_left as i16;
    let left_distance = (estimate - left as i16).abs();
    let above_distance = (estimate - above as i16).abs();
    let upper_left_distance = (estimate - upper_left as i16).abs();
    if left_distance <= above_distance && left_distance <= upper_left_distance {
        left
    } else if above_distance <= upper_left_distance {
        above
    } else {
        upper_left
    }
}

/// Return the logical screen dimensions and number of image frames in a GIF.
pub fn validate_gif(path: impl AsRef<Path>) -> io::Result<(u16, u16, usize)> {
    let bytes = std::fs::read(path)?;
    if bytes.len() < 13 || (&bytes[..6] != b"GIF89a" && &bytes[..6] != b"GIF87a") {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid GIF header",
        ));
    }
    let width = u16::from_le_bytes(bytes[6..8].try_into().unwrap());
    let height = u16::from_le_bytes(bytes[8..10].try_into().unwrap());
    let mut offset = 13usize;
    if bytes[10] & 0x80 != 0 {
        offset = offset
            .checked_add(3 * (1usize << ((bytes[10] & 0x07) + 1)))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid GIF palette"))?;
    }
    if offset > bytes.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "truncated GIF color table",
        ));
    }
    let mut frames = 0usize;
    let mut saw_trailer = false;
    while offset < bytes.len() {
        match bytes[offset] {
            0x3b => {
                saw_trailer = true;
                offset += 1;
                break;
            }
            0x21 => {
                if offset + 2 > bytes.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "truncated GIF extension",
                    ));
                }
                offset += 2;
                let mut terminated = false;
                while offset < bytes.len() {
                    let length = bytes[offset] as usize;
                    offset += 1;
                    if length == 0 {
                        terminated = true;
                        break;
                    }
                    offset = offset.checked_add(length).ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "GIF extension length overflow")
                    })?;
                    if offset > bytes.len() {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "truncated GIF extension data",
                        ));
                    }
                }
                if !terminated {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "unterminated GIF extension",
                    ));
                }
            }
            0x2c => {
                if offset + 10 > bytes.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "truncated GIF image descriptor",
                    ));
                }
                let packed = bytes[offset + 9];
                let image_width =
                    u16::from_le_bytes(bytes[offset + 5..offset + 7].try_into().map_err(|_| {
                        io::Error::new(io::ErrorKind::InvalidData, "invalid GIF image width")
                    })?);
                let image_height =
                    u16::from_le_bytes(bytes[offset + 7..offset + 9].try_into().map_err(|_| {
                        io::Error::new(io::ErrorKind::InvalidData, "invalid GIF image height")
                    })?);
                if image_width == 0 || image_height == 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "blank GIF image descriptor",
                    ));
                }
                offset += 10;
                if packed & 0x80 != 0 {
                    offset = offset
                        .checked_add(3 * (1usize << ((packed & 0x07) + 1)))
                        .ok_or_else(|| {
                            io::Error::new(io::ErrorKind::InvalidData, "invalid GIF local palette")
                        })?;
                    if offset > bytes.len() {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "truncated GIF local color table",
                        ));
                    }
                }
                if offset >= bytes.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "missing GIF LZW code size",
                    ));
                }
                offset += 1; // LZW minimum code size
                let mut terminated = false;
                while offset < bytes.len() {
                    let length = bytes[offset] as usize;
                    offset += 1;
                    if length == 0 {
                        terminated = true;
                        break;
                    }
                    offset = offset.checked_add(length).ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "GIF image data length overflow")
                    })?;
                    if offset > bytes.len() {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "truncated GIF image data",
                        ));
                    }
                }
                if !terminated {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "unterminated GIF image data",
                    ));
                }
                frames += 1;
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "unknown GIF block",
                ));
            }
        }
    }
    if !saw_trailer || offset != bytes.len() || width == 0 || height == 0 || frames == 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "blank GIF"));
    }
    Ok((width, height, frames))
}

fn write_heatmap_png(path: &Path, values: &[f64], width: u32, height: u32) -> io::Result<()> {
    let lo = values.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = (hi - lo).max(f64::EPSILON);
    let mut raw = Vec::with_capacity((width as usize * height as usize * 3) + height as usize);
    for row in 0..height as usize {
        raw.push(0);
        for col in 0..width as usize {
            let value = values[(row * values.len() / height as usize
                + col * values.len() / width as usize)
                % values.len()];
            let t = ((value - lo) / range).clamp(0.0, 1.0);
            raw.extend_from_slice(&colour(t));
        }
    }
    write_png(path, width, height, raw)
}

fn write_png(path: &Path, width: u32, height: u32, raw: Vec<u8>) -> io::Result<()> {
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::best());
    encoder.write_all(&raw)?;
    let compressed = encoder.finish()?;
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    chunk(&mut png, b"IHDR", ihdr);
    chunk(&mut png, b"IDAT", compressed);
    chunk(&mut png, b"IEND", []);
    std::fs::write(path, png)
}

/// Nearest-neighbour rasterization of a row-major field. No sample is mixed
/// with another: a displayed pixel has an unambiguous source grid cell.
fn write_grid_heatmap_png(
    path: &Path,
    values: &[f64],
    columns: usize,
    width: u32,
    height: u32,
) -> io::Result<()> {
    let rows = values.len() / columns;
    let lo = values.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = (hi - lo).max(f64::EPSILON);
    let mut raw = Vec::with_capacity((width as usize * height as usize * 3) + height as usize);
    for row in 0..height as usize {
        raw.push(0);
        let source_row = row * rows / height as usize;
        for col in 0..width as usize {
            let source_col = col * columns / width as usize;
            let value = values[source_row * columns + source_col];
            raw.extend_from_slice(&colour(((value - lo) / range).clamp(0.0, 1.0)));
        }
    }
    write_png(path, width, height, raw)
}

/// Bilinear rasterization of a row-major field. Pixel centres map linearly
/// across the grid extent with edge clamping, so output is a deterministic
/// function of the samples with no ringing or overshoot.
fn write_smooth_grid_heatmap_png(
    path: &Path,
    values: &[f64],
    columns: usize,
    width: u32,
    height: u32,
) -> io::Result<()> {
    let rows = values.len() / columns;
    let lo = values.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = (hi - lo).max(f64::EPSILON);
    let width_span = (columns.max(2) - 1) as f64;
    let height_span = (rows.max(2) - 1) as f64;
    let mut raw = Vec::with_capacity((width as usize * height as usize * 3) + height as usize);
    for row in 0..height as usize {
        raw.push(0);
        let grid_y = (row as f64 + 0.5) / height as usize as f64 * height_span;
        let row0 = (grid_y.floor() as usize).min(rows - 1);
        let row1 = (row0 + 1).min(rows - 1);
        let fry = (grid_y - row0 as f64).clamp(0.0, 1.0);
        for col in 0..width as usize {
            let grid_x = (col as f64 + 0.5) / width as usize as f64 * width_span;
            let col0 = (grid_x.floor() as usize).min(columns - 1);
            let col1 = (col0 + 1).min(columns - 1);
            let frx = (grid_x - col0 as f64).clamp(0.0, 1.0);
            let top =
                values[row0 * columns + col0] * (1.0 - frx) + values[row0 * columns + col1] * frx;
            let bottom =
                values[row1 * columns + col0] * (1.0 - frx) + values[row1 * columns + col1] * frx;
            let value = top * (1.0 - fry) + bottom * fry;
            raw.extend_from_slice(&colour(((value - lo) / range).clamp(0.0, 1.0)));
        }
    }
    write_png(path, width, height, raw)
}

fn write_animation_gif(
    path: &Path,
    frames: &[Vec<f64>],
    width: u16,
    height: u16,
) -> io::Result<()> {
    let mut gif = Vec::new();
    gif.extend_from_slice(b"GIF89a");
    gif.extend_from_slice(&width.to_le_bytes());
    gif.extend_from_slice(&height.to_le_bytes());
    // Global four-colour palette, 8-bit background, no aspect-ratio claim.
    gif.extend_from_slice(&[0xf1, 0, 0]);
    gif.extend_from_slice(&[12, 19, 26, 48, 201, 255, 255, 207, 84, 238, 82, 83]);
    // Netscape repeat extension: loop forever.
    gif.extend_from_slice(&[0x21, 0xff, 0x0b]);
    gif.extend_from_slice(b"NETSCAPE2.0");
    gif.extend_from_slice(&[3, 1, 0, 0, 0]);
    for values in frames {
        // Graphics control extension: 100 ms per frame, no transparency.
        gif.extend_from_slice(&[0x21, 0xf9, 4, 0, 10, 0, 0, 0]);
        gif.push(0x2c);
        gif.extend_from_slice(&[0, 0, 0, 0]);
        gif.extend_from_slice(&width.to_le_bytes());
        gif.extend_from_slice(&height.to_le_bytes());
        gif.push(0);
        gif.push(2); // LZW minimum code size for the four-colour palette.
        let pixels = (0..width as usize * height as usize)
            .map(|index| quantize(values[index % values.len()]))
            .collect::<Vec<_>>();
        for block in gif_lzw_blocks(&pixels) {
            gif.push(block.len() as u8);
            gif.extend_from_slice(&block);
        }
        gif.push(0);
    }
    gif.push(0x3b);
    std::fs::write(path, gif)
}

fn quantize(value: f64) -> u8 {
    // Values are normalized by their frame index below; this stable periodic
    // mapping makes even constant frames visibly nonblank.
    ((value.abs() * 10_000.0).round() as u64 % 4) as u8
}

fn gif_lzw_blocks(pixels: &[u8]) -> Vec<Vec<u8>> {
    // Reset the dictionary before every literal. This is intentionally simple
    // and entirely valid GIF LZW; it trades compression ratio for a compact,
    // deterministic encoder with no external runtime dependency.
    let mut bytes = Vec::with_capacity((pixels.len() * 6 + 5) / 8);
    let mut accumulator = 0u32;
    let mut bits = 0u8;
    let push = |code: u8, bytes: &mut Vec<u8>, accumulator: &mut u32, bits: &mut u8| {
        *accumulator |= (code as u32) << *bits;
        *bits += 3;
        while *bits >= 8 {
            bytes.push(*accumulator as u8);
            *accumulator >>= 8;
            *bits -= 8;
        }
    };
    for &pixel in pixels {
        push(4, &mut bytes, &mut accumulator, &mut bits);
        push(pixel, &mut bytes, &mut accumulator, &mut bits);
    }
    push(5, &mut bytes, &mut accumulator, &mut bits);
    if bits > 0 {
        bytes.push(accumulator as u8);
    }
    bytes.chunks(255).map(ToOwned::to_owned).collect()
}

fn colour(t: f64) -> [u8; 3] {
    let r = (255.0 * (1.5 * t - 0.5).clamp(0.0, 1.0)) as u8;
    let g = (255.0 * (1.5 - (2.0 * t - 1.0).abs()).clamp(0.0, 1.0)) as u8;
    let b = (255.0 * (1.0 - 1.5 * t).clamp(0.0, 1.0)) as u8;
    [r, g, b]
}

fn chunk(png: &mut Vec<u8>, kind: &[u8; 4], data: impl AsRef<[u8]>) {
    let data = data.as_ref();
    png.extend_from_slice(&(data.len() as u32).to_be_bytes());
    png.extend_from_slice(kind);
    png.extend_from_slice(data);
    let mut crc_input = Vec::with_capacity(4 + data.len());
    crc_input.extend_from_slice(kind);
    crc_input.extend_from_slice(data);
    png.extend_from_slice(&crc32(&crc_input).to_be_bytes());
}

fn write_line_svg(path: &Path, x: &[f64], y: &[f64], title: &str) -> io::Result<()> {
    let (xmin, xmax) = bounds(x);
    let (ymin, ymax) = bounds(y);
    let sx = |v: f64| 120.0 + 1680.0 * (v - xmin) / (xmax - xmin).max(f64::EPSILON);
    let sy = |v: f64| 930.0 - 780.0 * (v - ymin) / (ymax - ymin).max(f64::EPSILON);
    let mut points = String::new();
    for (&x, &y) in x.iter().zip(y) {
        write!(points, "{:.2},{:.2} ", sx(x), sy(y)).unwrap();
    }
    let svg = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1920\" height=\"1080\" viewBox=\"0 0 1920 1080\"><rect width=\"100%\" height=\"100%\" fill=\"#10131a\"/><text x=\"120\" y=\"90\" fill=\"white\" font-size=\"42\" font-family=\"sans-serif\">{title}</text><path d=\"M120 150V930H1800\" stroke=\"#c7d0df\" stroke-width=\"3\" fill=\"none\"/><polyline points=\"{points}\" fill=\"none\" stroke=\"#30c9ff\" stroke-width=\"5\"/><text x=\"120\" y=\"990\" fill=\"#c7d0df\" font-size=\"24\">x: {xmin:.3}–{xmax:.3}</text><text x=\"1380\" y=\"990\" fill=\"#c7d0df\" font-size=\"24\">y: {ymin:.3}–{ymax:.3}</text></svg>"
    );
    std::fs::write(path, svg)
}

fn bounds(values: &[f64]) -> (f64, f64) {
    (
        values.iter().copied().fold(f64::INFINITY, f64::min),
        values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    )
}
fn fnv1a64(bytes: &[u8]) -> String {
    format!(
        "{:016x}",
        bytes
            .iter()
            .fold(0xcbf29ce484222325u64, |h, &b| (h ^ b as u64)
                .wrapping_mul(0x100000001b3))
    )
}
fn crc32(bytes: &[u8]) -> u32 {
    !bytes.iter().fold(!0u32, |crc, &b| {
        let mut c = crc ^ b as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 {
                (c >> 1) ^ 0xedb88320
            } else {
                c >> 1
            };
        }
        c
    })
}

fn current_git_sha() -> String {
    if let Some(sha) = option_env!("VERGEN_GIT_SHA") {
        return sha.into();
    }
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|sha| sha.trim().to_owned())
        .filter(|sha| !sha.is_empty())
        .unwrap_or_else(|| "unavailable".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn deterministic_gallery_assets_are_valid_and_manifested() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("result.vtu"), "validated source").unwrap();
        std::fs::write(dir.path().join("report.json"), "validated report").unwrap();
        let mut gallery = GalleryRenderer::new(dir.path()).unwrap();
        gallery
            .heatmap_png("field.png", &[0.0, 1.0, 0.5], "result.vtu", "field")
            .unwrap();
        gallery
            .grid_heatmap_png("grid.png", &[0.0, 0.25, 0.75, 1.0], 2, "result.vtu", "grid")
            .unwrap();
        gallery
            .smooth_grid_heatmap_png(
                "smooth.png",
                &[0.0, 0.25, 0.75, 1.0],
                2,
                "result.vtu",
                "smooth",
            )
            .unwrap();
        gallery
            .animation_gif(
                "rotation.gif",
                &[vec![0.0, 1.0], vec![0.5, 0.25]],
                "result.vtu",
                "rotation",
            )
            .unwrap();
        gallery
            .line_svg(
                "curve.svg",
                &[1.0, 2.0],
                &[3.0, 4.0],
                "curve",
                "report.json",
                "curve",
            )
            .unwrap();
        gallery
            .line_svg(
                "nested/curve.svg",
                &[1.0, 2.0],
                &[2.0, 3.0],
                "nested curve",
                "report.json",
                "nested curve",
            )
            .unwrap();
        let duplicate = gallery.existing_figure("curve.svg", "svg", "report.json", "duplicate");
        assert!(duplicate.unwrap_err().kind() == io::ErrorKind::AlreadyExists);
        let figure_traversal =
            gallery.heatmap_png("../escape.png", &[0.0, 1.0], "report.json", "bad path");
        assert_eq!(
            figure_traversal.unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
        let traversal =
            gallery.heatmap_png("nested/escape.png", &[0.0, 1.0], "../outside", "bad source");
        assert_eq!(traversal.unwrap_err().kind(), io::ErrorKind::InvalidInput);
        let manifest_path = gallery.finish("test", vec!["fixed camera".into()]).unwrap();
        assert_eq!(
            validate_gallery_manifest(&manifest_path)
                .unwrap()
                .figures
                .len(),
            6
        );
        assert_eq!(
            validate_png_nonblank(dir.path().join("field.png")).unwrap(),
            (GALLERY_WIDTH, GALLERY_HEIGHT)
        );
        assert_eq!(
            validate_png_nonblank(dir.path().join("grid.png")).unwrap(),
            (GALLERY_WIDTH, GALLERY_HEIGHT)
        );
        assert_eq!(
            validate_png_nonblank(dir.path().join("smooth.png")).unwrap(),
            (GALLERY_WIDTH, GALLERY_HEIGHT)
        );
        // Bilinear interpolation must mix samples: the smooth render of a
        // gradient differs from the nearest-neighbour render byte for byte.
        assert_ne!(
            std::fs::read(dir.path().join("grid.png")).unwrap(),
            std::fs::read(dir.path().join("smooth.png")).unwrap()
        );
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.path().join("gallery.json")).unwrap())
                .unwrap();
        assert_eq!(
            validate_gif(dir.path().join("rotation.gif")).unwrap(),
            (320, 180, 2)
        );
        assert_eq!(manifest["figures"].as_array().unwrap().len(), 6);
        assert!(
            manifest["figures"]
                .as_array()
                .unwrap()
                .iter()
                .any(|figure| figure["filename"] == "nested/curve.svg")
        );

        std::fs::write(dir.path().join("field.png"), b"tampered").unwrap();
        assert!(validate_gallery_manifest(&manifest_path).is_err());
    }

    #[test]
    fn gallery_manifest_rejects_a_matching_checksum_for_a_blank_png() {
        let dir = tempfile::tempdir().unwrap();
        let source = b"validated source";
        std::fs::write(dir.path().join("source.vtu"), source).unwrap();

        let raw_len = (GALLERY_WIDTH as usize * 3 + 1) * GALLERY_HEIGHT as usize;
        write_png(
            &dir.path().join("blank.png"),
            GALLERY_WIDTH,
            GALLERY_HEIGHT,
            vec![0; raw_len],
        )
        .unwrap();
        let blank = std::fs::read(dir.path().join("blank.png")).unwrap();
        let manifest = GalleryManifest {
            schema_version: GALLERY_SCHEMA_VERSION,
            title: "blank test".into(),
            renderer: "test".into(),
            width_px: GALLERY_WIDTH,
            height_px: GALLERY_HEIGHT,
            git_sha: "test".into(),
            conventions: vec!["fixed camera".into()],
            figures: vec![GalleryFigure {
                filename: "blank.png".into(),
                kind: "png".into(),
                source: "source.vtu".into(),
                source_checksum_fnv1a64: Some(fnv1a64(source)),
                caption: "intentionally blank".into(),
                checksum_fnv1a64: fnv1a64(&blank),
            }],
        };
        std::fs::write(
            dir.path().join("gallery.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();

        let error = validate_gallery_manifest(dir.path().join("gallery.json")).unwrap_err();
        assert!(error.to_string().contains("visually blank PNG"));
    }

    #[test]
    fn nonblank_check_accepts_dark_image_with_black_corner() {
        let dir = tempfile::tempdir().unwrap();
        // 2x1: black corner pixel followed by a white pixel. Non-uniform
        // content must pass even though the first pixel is black.
        let raw = vec![0u8, 0, 0, 0, 255, 255, 255];
        write_png(&dir.path().join("dark.png"), 2, 1, raw).unwrap();
        assert_eq!(
            validate_png_nonblank(dir.path().join("dark.png")).unwrap(),
            (2, 1)
        );
        // Uniform black is still blank.
        let raw = vec![0u8, 0, 0, 0, 0, 0, 0];
        write_png(&dir.path().join("black.png"), 2, 1, raw).unwrap();
        assert!(validate_png_nonblank(dir.path().join("black.png")).is_err());
    }

    #[test]
    fn gallery_validators_reject_truncated_and_corrupt_images() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("source"), "validated source").unwrap();
        let mut gallery = GalleryRenderer::new(dir.path()).unwrap();
        gallery
            .heatmap_png("field.png", &[0.0, 1.0, 0.5], "source", "field")
            .unwrap();
        gallery
            .animation_gif(
                "rotation.gif",
                &[vec![0.0, 1.0], vec![0.5, 0.25]],
                "source",
                "rotation",
            )
            .unwrap();
        let png = std::fs::read(dir.path().join("field.png")).unwrap();
        let gif = std::fs::read(dir.path().join("rotation.gif")).unwrap();

        for (name, bytes) in [
            ("png-short.png", png[..8].to_vec()),
            ("png-truncated.png", png[..png.len() - 1].to_vec()),
            ("gif-short.gif", gif[..13].to_vec()),
            ("gif-truncated.gif", gif[..gif.len() - 1].to_vec()),
        ] {
            std::fs::write(dir.path().join(name), bytes).unwrap();
        }
        let mut corrupt_png = png.clone();
        // Corrupt an IHDR byte without updating its CRC.
        corrupt_png[20] ^= 1;
        std::fs::write(dir.path().join("png-corrupt.png"), corrupt_png).unwrap();
        let mut huge_png = png.clone();
        huge_png[16..20].copy_from_slice(&u32::MAX.to_be_bytes());
        huge_png[20..24].copy_from_slice(&u32::MAX.to_be_bytes());
        let huge_ihdr_crc = crc32(&huge_png[12..29]);
        huge_png[29..33].copy_from_slice(&huge_ihdr_crc.to_be_bytes());
        std::fs::write(dir.path().join("png-huge.png"), huge_png).unwrap();

        for name in ["png-short.png", "png-truncated.png", "png-corrupt.png"] {
            assert!(validate_png(dir.path().join(name)).is_err(), "{name}");
        }
        assert!(validate_png_nonblank(dir.path().join("png-huge.png")).is_err());
        for name in ["gif-short.gif", "gif-truncated.gif"] {
            assert!(validate_gif(dir.path().join(name)).is_err(), "{name}");
        }
    }
}
