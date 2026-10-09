//! Versioned, bounded-memory chunked binary transport for large XEM payloads.
//!
//! JSON remains the human-readable control format.  Large field and result
//! buffers can use this framing format instead: every chunk is length-prefixed,
//! checksummed, and read into memory one at a time.

use std::io::{self, Read, Write};

/// Current on-disk format version.
pub const CHUNKED_BINARY_VERSION: u32 = 1;
/// Maximum chunk size accepted by [`ChunkedBinaryReader::new`].
pub const DEFAULT_MAX_CHUNK_SIZE: u64 = 256 * 1024 * 1024;

const MAGIC: &[u8; 16] = b"SONIUM-XEM-CBIN\0";
const FRAME_HEADER_BYTES: usize = 16;

/// Errors specific to the chunked container format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChunkedBinaryError {
    /// The stream does not begin with the XEM chunked-binary magic.
    InvalidMagic,
    /// The stream uses a format version this reader does not understand.
    UnsupportedVersion(u32),
    /// A frame claims more bytes than the configured memory bound.
    ChunkTooLarge { length: u64, maximum: u64 },
    /// A configured chunk bound cannot accept any payload.
    InvalidMaximumChunkSize,
    /// A frame's checksum does not match its payload.
    ChecksumMismatch { expected: u64, actual: u64 },
}

impl std::fmt::Display for ChunkedBinaryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidMagic => write!(f, "invalid XEM chunked-binary magic"),
            Self::UnsupportedVersion(version) => {
                write!(f, "unsupported XEM chunked-binary version {version}")
            }
            Self::ChunkTooLarge { length, maximum } => {
                write!(
                    f,
                    "chunk length {length} exceeds configured maximum {maximum}"
                )
            }
            Self::InvalidMaximumChunkSize => {
                write!(f, "configured XEM chunk maximum must be greater than zero")
            }
            Self::ChecksumMismatch { expected, actual } => write!(
                f,
                "chunk checksum mismatch: expected {expected:#018x}, got {actual:#018x}"
            ),
        }
    }
}

impl std::error::Error for ChunkedBinaryError {}

/// Streaming writer for a versioned sequence of binary chunks.
pub struct ChunkedBinaryWriter<W> {
    writer: W,
    maximum_chunk_size: u64,
    chunks_written: usize,
    bytes_written: u64,
}

impl<W: Write> ChunkedBinaryWriter<W> {
    /// Write the container header around an existing writer.
    pub fn new(writer: W) -> io::Result<Self> {
        Self::with_max_chunk_size(writer, DEFAULT_MAX_CHUNK_SIZE)
    }

    /// Write the container header with an explicit per-chunk memory bound.
    ///
    /// The bound is enforced before writing a frame header or payload, so a
    /// rejected chunk leaves the stream at a clean frame boundary.
    pub fn with_max_chunk_size(mut writer: W, maximum_chunk_size: u64) -> io::Result<Self> {
        if maximum_chunk_size == 0 {
            return Err(format_error(ChunkedBinaryError::InvalidMaximumChunkSize));
        }
        writer.write_all(MAGIC)?;
        writer.write_all(&CHUNKED_BINARY_VERSION.to_le_bytes())?;
        Ok(Self {
            writer,
            maximum_chunk_size,
            chunks_written: 0,
            bytes_written: 0,
        })
    }

    /// Append one chunk without retaining previous chunks in memory.
    pub fn write_chunk(&mut self, payload: &[u8]) -> io::Result<()> {
        let length = u64::try_from(payload.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "chunk length does not fit in u64",
            )
        })?;
        if length > self.maximum_chunk_size {
            return Err(format_error(ChunkedBinaryError::ChunkTooLarge {
                length,
                maximum: self.maximum_chunk_size,
            }));
        }
        let checksum = checksum(payload);
        self.writer.write_all(&length.to_le_bytes())?;
        self.writer.write_all(&checksum.to_le_bytes())?;
        self.writer.write_all(payload)?;
        self.chunks_written += 1;
        self.bytes_written = self
            .bytes_written
            .checked_add(length)
            .ok_or_else(|| io::Error::other("chunk byte count overflow"))?;
        Ok(())
    }

    /// Number of chunks successfully appended.
    pub const fn chunks_written(&self) -> usize {
        self.chunks_written
    }

    /// Number of payload bytes successfully appended.
    pub const fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    /// Return the configured per-chunk memory bound.
    pub const fn maximum_chunk_size(&self) -> u64 {
        self.maximum_chunk_size
    }

    /// Flush and return the wrapped writer.
    pub fn into_inner(mut self) -> io::Result<W> {
        self.writer.flush()?;
        Ok(self.writer)
    }
}

/// Streaming reader for a versioned sequence of binary chunks.
pub struct ChunkedBinaryReader<R> {
    reader: R,
    maximum_chunk_size: u64,
    chunks_read: usize,
    bytes_read: u64,
}

impl<R: Read> ChunkedBinaryReader<R> {
    /// Read and validate a container header using the default memory bound.
    pub fn new(reader: R) -> io::Result<Self> {
        Self::with_max_chunk_size(reader, DEFAULT_MAX_CHUNK_SIZE)
    }

    /// Read and validate a container header with an explicit chunk bound.
    pub fn with_max_chunk_size(mut reader: R, maximum_chunk_size: u64) -> io::Result<Self> {
        if maximum_chunk_size == 0 {
            return Err(format_error(ChunkedBinaryError::InvalidMaximumChunkSize));
        }
        let mut magic = [0u8; MAGIC.len()];
        reader.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(format_error(ChunkedBinaryError::InvalidMagic));
        }

        let mut version = [0u8; 4];
        reader.read_exact(&mut version)?;
        let version = u32::from_le_bytes(version);
        if version != CHUNKED_BINARY_VERSION {
            return Err(format_error(ChunkedBinaryError::UnsupportedVersion(
                version,
            )));
        }

        Ok(Self {
            reader,
            maximum_chunk_size,
            chunks_read: 0,
            bytes_read: 0,
        })
    }

    /// Read the next chunk, or `Ok(None)` at a clean end-of-stream.
    pub fn next_chunk(&mut self) -> io::Result<Option<Vec<u8>>> {
        let mut payload = Vec::new();
        match self.next_chunk_into(&mut payload)? {
            Some(_) => Ok(Some(payload)),
            None => Ok(None),
        }
    }

    /// Read the next chunk into caller-owned reusable storage.
    ///
    /// The returned slice borrows `payload`; subsequent calls overwrite the
    /// same buffer. This avoids a per-chunk allocation in large mesh/field
    /// streams while retaining the reader's per-chunk memory bound.
    pub fn next_chunk_into<'a>(
        &mut self,
        payload: &'a mut Vec<u8>,
    ) -> io::Result<Option<&'a [u8]>> {
        payload.clear();
        let mut frame_header = [0u8; FRAME_HEADER_BYTES];
        let read = read_frame_header(&mut self.reader, &mut frame_header)?;
        if read == 0 {
            return Ok(None);
        }
        if read != FRAME_HEADER_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "truncated XEM chunk frame header",
            ));
        }

        let mut length_bytes = [0u8; 8];
        length_bytes.copy_from_slice(&frame_header[..8]);
        let length = u64::from_le_bytes(length_bytes);
        if length > self.maximum_chunk_size {
            // Keep the stream positioned at the next frame so callers can
            // report the bounded-memory error and continue scanning records.
            let _ = discard_bytes(&mut self.reader, length);
            return Err(format_error(ChunkedBinaryError::ChunkTooLarge {
                length,
                maximum: self.maximum_chunk_size,
            }));
        }

        let mut checksum_bytes = [0u8; 8];
        checksum_bytes.copy_from_slice(&frame_header[8..]);
        let expected = u64::from_le_bytes(checksum_bytes);
        let length_usize = usize::try_from(length).map_err(|_| {
            // The configured bound may be larger than the host address space;
            // discard the frame payload before returning the conversion error.
            let _ = discard_bytes(&mut self.reader, length);
            io::Error::new(
                io::ErrorKind::InvalidData,
                "chunk length does not fit in the host address space",
            )
        })?;
        payload.resize(length_usize, 0);
        if let Err(error) = self.reader.read_exact(payload) {
            // Never expose a partially read frame through the reusable caller
            // buffer after an I/O failure. Keep the allocation for the next
            // frame without retaining untrusted bytes as a usable slice.
            payload.clear();
            return Err(error);
        }
        let actual = checksum(payload);
        if actual != expected {
            // A checksum failure consumes the complete frame but its contents
            // are not evidence. Clear the reusable view before returning the
            // error so callers cannot accidentally process it after handling
            // the failed read.
            payload.clear();
            return Err(format_error(ChunkedBinaryError::ChecksumMismatch {
                expected,
                actual,
            }));
        }

        self.chunks_read += 1;
        self.bytes_read = self
            .bytes_read
            .checked_add(length)
            .ok_or_else(|| io::Error::other("chunk byte count overflow"))?;
        Ok(Some(payload.as_slice()))
    }

    /// Number of chunks successfully read.
    pub const fn chunks_read(&self) -> usize {
        self.chunks_read
    }

    /// Number of payload bytes successfully read.
    pub const fn bytes_read(&self) -> u64 {
        self.bytes_read
    }

    /// Return the configured per-chunk memory bound.
    pub const fn maximum_chunk_size(&self) -> u64 {
        self.maximum_chunk_size
    }

    /// Return the wrapped reader.
    pub fn into_inner(self) -> R {
        self.reader
    }
}

fn read_frame_header<R: Read>(
    reader: &mut R,
    header: &mut [u8; FRAME_HEADER_BYTES],
) -> io::Result<usize> {
    let mut read = 0;
    while read < header.len() {
        let count = reader.read(&mut header[read..])?;
        if count == 0 {
            break;
        }
        read += count;
    }
    Ok(read)
}

fn discard_bytes<R: Read>(reader: &mut R, mut remaining: u64) -> io::Result<()> {
    let mut scratch = [0u8; 16 * 1024];
    while remaining > 0 {
        let request = remaining.min(scratch.len() as u64) as usize;
        reader.read_exact(&mut scratch[..request])?;
        remaining -= request as u64;
    }
    Ok(())
}

fn format_error(error: ChunkedBinaryError) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, error)
}

// FNV-1a is deliberately small and deterministic. It is an integrity check,
// not a cryptographic authentication mechanism.
fn checksum(payload: &[u8]) -> u64 {
    payload.iter().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_round_trip_incrementally() {
        let mut bytes = Vec::new();
        {
            let mut writer = ChunkedBinaryWriter::new(&mut bytes).expect("write header");
            writer.write_chunk(b"mesh chunk").expect("write mesh");
            writer
                .write_chunk(&[0, 1, 2, 3, 4])
                .expect("write field chunk");
            assert_eq!(writer.chunks_written(), 2);
            assert_eq!(writer.bytes_written(), 15);
            writer.into_inner().expect("flush writer");
        }

        let mut reader = ChunkedBinaryReader::new(bytes.as_slice()).expect("read header");
        assert_eq!(
            reader.next_chunk().expect("read mesh"),
            Some(b"mesh chunk".to_vec())
        );
        assert_eq!(
            reader.next_chunk().expect("read field"),
            Some(vec![0, 1, 2, 3, 4])
        );
        assert_eq!(reader.next_chunk().expect("read EOF"), None);
        assert_eq!(reader.chunks_read(), 2);
        assert_eq!(reader.bytes_read(), 15);
    }

    #[test]
    fn reader_rejects_oversized_chunks_before_allocation() {
        let mut bytes = Vec::new();
        let mut writer = ChunkedBinaryWriter::new(&mut bytes).expect("write header");
        writer.write_chunk(b"0123456789").expect("write chunk");
        writer.into_inner().expect("flush writer");

        let error = ChunkedBinaryReader::with_max_chunk_size(bytes.as_slice(), 4)
            .expect("read header")
            .next_chunk()
            .expect_err("chunk must exceed the configured bound");
        assert!(error.to_string().contains("exceeds configured maximum"));
    }

    #[test]
    fn oversized_reader_chunk_is_drained_before_the_next_frame() {
        let mut bytes = Vec::new();
        {
            let mut writer = ChunkedBinaryWriter::new(&mut bytes).expect("write header");
            writer.write_chunk(b"oversized").expect("write first chunk");
            writer.write_chunk(b"ok").expect("write second chunk");
            writer.into_inner().expect("flush writer");
        }
        let mut reader =
            ChunkedBinaryReader::with_max_chunk_size(bytes.as_slice(), 3).expect("read header");
        let error = reader
            .next_chunk()
            .expect_err("first chunk must exceed bound");
        assert!(error.to_string().contains("exceeds configured maximum"));
        assert_eq!(
            reader.next_chunk().expect("read second chunk"),
            Some(b"ok".to_vec())
        );
    }

    #[test]
    fn writer_rejects_oversized_chunks_before_writing_a_frame() {
        let mut bytes = Vec::new();
        let (error, chunks_written, maximum_chunk_size) = {
            let mut writer = ChunkedBinaryWriter::with_max_chunk_size(&mut bytes, 4)
                .expect("write bounded header");
            let error = writer
                .write_chunk(b"0123456789")
                .expect_err("writer must enforce its chunk bound");
            let chunks_written = writer.chunks_written();
            let maximum_chunk_size = writer.maximum_chunk_size();
            writer.into_inner().expect("flush bounded writer");
            (error, chunks_written, maximum_chunk_size)
        };
        assert!(error.to_string().contains("exceeds configured maximum"));
        assert_eq!(bytes.len(), 20);
        assert_eq!(chunks_written, 0);
        assert_eq!(maximum_chunk_size, 4);
    }

    #[test]
    fn zero_chunk_bounds_are_rejected_before_header_io() {
        let mut writer_bytes = Vec::new();
        let writer_error = match ChunkedBinaryWriter::with_max_chunk_size(&mut writer_bytes, 0) {
            Ok(_) => panic!("zero writer bound must be invalid"),
            Err(error) => error,
        };
        assert!(
            writer_error
                .to_string()
                .contains("must be greater than zero")
        );
        assert!(
            writer_bytes.is_empty(),
            "invalid writer must not emit a header"
        );

        let reader_error = match ChunkedBinaryReader::with_max_chunk_size([].as_slice(), 0) {
            Ok(_) => panic!("zero reader bound must be invalid"),
            Err(error) => error,
        };
        assert!(
            reader_error
                .to_string()
                .contains("must be greater than zero")
        );
    }

    #[test]
    fn reader_rejects_corrupted_payload() {
        let mut bytes = Vec::new();
        let mut writer = ChunkedBinaryWriter::new(&mut bytes).expect("write header");
        writer.write_chunk(b"payload").expect("write chunk");
        writer.into_inner().expect("flush writer");
        *bytes.last_mut().expect("payload byte") ^= 0x01;

        let error = ChunkedBinaryReader::new(bytes.as_slice())
            .expect("read header")
            .next_chunk()
            .expect_err("checksum must fail");
        assert!(error.to_string().contains("checksum mismatch"));
    }

    #[test]
    fn failed_frame_reads_clear_reusable_payload_storage() {
        let mut corrupted = Vec::new();
        let mut writer = ChunkedBinaryWriter::new(&mut corrupted).expect("write header");
        writer.write_chunk(b"payload").expect("write chunk");
        writer.into_inner().expect("flush writer");
        *corrupted.last_mut().expect("payload byte") ^= 0x01;

        let mut reader = ChunkedBinaryReader::new(corrupted.as_slice()).expect("read header");
        let mut reusable = vec![0xaa; 64];
        assert!(reader.next_chunk_into(&mut reusable).is_err());
        assert!(reusable.is_empty());
        assert_eq!(reader.chunks_read(), 0);
        assert_eq!(reader.bytes_read(), 0);

        let mut truncated = Vec::new();
        let mut writer = ChunkedBinaryWriter::new(&mut truncated).expect("write header");
        writer.write_chunk(b"payload").expect("write chunk");
        writer.into_inner().expect("flush writer");
        truncated.pop().expect("remove payload byte");

        let mut reader = ChunkedBinaryReader::new(truncated.as_slice()).expect("read header");
        reusable.extend_from_slice(b"stale");
        assert!(reader.next_chunk_into(&mut reusable).is_err());
        assert!(reusable.is_empty());
        assert_eq!(reader.chunks_read(), 0);
        assert_eq!(reader.bytes_read(), 0);
    }

    #[test]
    fn reusable_chunk_buffer_avoids_per_chunk_allocation() {
        let mut bytes = Vec::new();
        let mut writer = ChunkedBinaryWriter::new(&mut bytes).expect("write header");
        writer.write_chunk(b"first payload").expect("write first");
        writer.write_chunk(b"second payload").expect("write second");
        writer.into_inner().expect("flush writer");

        let mut reader = ChunkedBinaryReader::new(bytes.as_slice()).expect("read header");
        let mut reusable = Vec::with_capacity(64);
        let first_ptr = {
            let first = reader
                .next_chunk_into(&mut reusable)
                .expect("read first")
                .expect("first chunk");
            assert_eq!(first, b"first payload");
            first.as_ptr()
        };
        let capacity_after_first = reusable.capacity();
        let second_ptr = {
            let second = reader
                .next_chunk_into(&mut reusable)
                .expect("read second")
                .expect("second chunk");
            assert_eq!(second, b"second payload");
            second.as_ptr()
        };
        assert_eq!(reusable.capacity(), capacity_after_first);
        assert_eq!(second_ptr, first_ptr);
        assert_eq!(
            reader.next_chunk_into(&mut reusable).expect("read EOF"),
            None
        );
    }
}
