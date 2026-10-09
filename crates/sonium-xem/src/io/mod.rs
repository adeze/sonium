//! Unified output I/O for sonium simulators.
//!
//! Every field-producing binary writes mesh + solution data through a single
//! VTU/PVD pipeline that ParaView can browse without per-binary loaders.
//!
//! - [`vtu`]: VTK Unstructured Grid (.vtu) writer parameterized over a
//!   [`vtu::MeshView`] trait so both `sonium_xem::types::RoomMesh` and
//!   `sonium_fem::mesh::Mesh` feed the same writer.
//! - [`pvd`]: ParaView collection (.pvd) writer for time / frequency series.
//! - [`output_dir`]: resolves a single root directory and per-binary subdirs
//!   so ParaView can browse `<root>/<crate>/<binary>/`.

pub mod chunked;
pub mod external;
pub mod gallery;
pub mod output_dir;
pub mod pvd;
#[cfg(feature = "io-sofa")]
pub mod sofa;
pub mod stream;
pub mod vtu;

pub use chunked::{
    CHUNKED_BINARY_VERSION, ChunkedBinaryError, ChunkedBinaryReader, ChunkedBinaryWriter,
    DEFAULT_MAX_CHUNK_SIZE,
};
pub use external::{
    ExternalMultiphysicsDocument, ExternalReaderError, ExternalRoomDocument,
    read_external_multiphysics_document, read_external_multiphysics_value,
    read_external_room_document, read_external_room_document_checked,
    read_external_room_document_migrated, read_external_room_value,
    read_external_room_value_checked, read_external_room_value_migrated, validate_schema_document,
    validate_schema_envelope,
};
pub use gallery::{
    GalleryFigure, GalleryManifest, GalleryRenderer, validate_gallery_manifest, validate_gif,
    validate_png, validate_png_nonblank,
};
pub use output_dir::{
    binary_output_dir, sonium_output_root, try_binary_output_dir, try_sonium_output_root,
};
pub use pvd::{PvdCollection, validate_pvd_file, write_pvd};
pub use stream::{DEFAULT_MAX_JSON_RECORD_BYTES, JsonLinesReader, JsonLinesWriter};
pub use vtu::{CellKind, MeshView, VtuError, VtuWriter, validate_vtu_file};
