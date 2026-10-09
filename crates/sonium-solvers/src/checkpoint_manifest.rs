//! Portable sharding for solver checkpoints crossing device or rank boundaries.
//!
//! The solver checkpoint formats themselves are intentionally agnostic about
//! where a solve runs.  This envelope records that provenance and stores a
//! complete, ordered set of byte shards.  It is useful for distributed restart
//! (one shard per rank) and for device hand-off (the payload remains portable
//! host bytes while the residence tag records the originating backend).

use crate::{
    DeflatedCheckpoint, GcroDrCheckpoint, GmresBasisCheckpoint, GmresCheckpoint,
    MixedPrecisionBlockCheckpoint, MixedPrecisionCheckpoint,
};
use sonium_compute::{Backend, BackendError, DeviceBufferLike};
use sonium_solvers_core::traits::{
    AccuracyStatus, ComplexField, SolverError, SolverFailure, SolverStatus,
};
use std::fmt;

/// Version of the distributed checkpoint envelope.
pub const DISTRIBUTED_CHECKPOINT_FORMAT_VERSION: u16 = 1;
/// Magic prefix for [`DistributedCheckpoint`] packets.
pub const DISTRIBUTED_CHECKPOINT_MAGIC: &[u8; 8] = b"SDCKPT01";
/// Maximum complete payload accepted by the checked parser.
pub const MAX_DISTRIBUTED_CHECKPOINT_BYTES: usize = 256 * 1024 * 1024;
const MAX_DISTRIBUTED_CHECKPOINT_SHARDS: u32 = 1 << 20;

/// Backend that owned the checkpoint when it was captured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CheckpointResidence {
    /// CPU/host execution.
    Host = 0,
    /// CUDA device execution.  The packet still contains portable host bytes.
    Cuda = 1,
    /// Metal device execution.  The packet still contains portable host bytes.
    Metal = 2,
}

impl CheckpointResidence {
    fn from_tag(tag: u8) -> Result<Self, CheckpointManifestError> {
        match tag {
            0 => Ok(Self::Host),
            1 => Ok(Self::Cuda),
            2 => Ok(Self::Metal),
            _ => Err(CheckpointManifestError::Invalid("unknown residence tag")),
        }
    }

    const fn tag(self) -> u8 {
        self as u8
    }
}

/// A single rank/device shard of a checkpoint payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointShard {
    rank: u32,
    offset: u64,
    payload: Vec<u8>,
    checksum: u64,
}

impl CheckpointShard {
    /// Create a shard and compute its content checksum.
    pub fn new(rank: u32, offset: u64, payload: Vec<u8>) -> Result<Self, CheckpointManifestError> {
        let end = offset
            .checked_add(payload.len() as u64)
            .ok_or(CheckpointManifestError::Overflow)?;
        if end as usize > MAX_DISTRIBUTED_CHECKPOINT_BYTES {
            return Err(CheckpointManifestError::OversizedPayload);
        }
        let checksum = checksum64(&payload);
        Ok(Self {
            rank,
            offset,
            payload,
            checksum,
        })
    }

    /// Rank or device shard ordinal.
    pub const fn rank(&self) -> u32 {
        self.rank
    }

    /// Global byte offset of this shard.
    pub const fn offset(&self) -> u64 {
        self.offset
    }

    /// Portable payload bytes for this shard.
    pub fn payload(&self) -> &[u8] {
        &self.payload
    }

    /// FNV-1a checksum of [`Self::payload`].
    pub const fn checksum(&self) -> u64 {
        self.checksum
    }
}

/// A checked, versioned collection of rank/device checkpoint shards.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DistributedCheckpoint {
    residence: CheckpointResidence,
    world_size: u32,
    total_len: u64,
    payload_checksum: u64,
    shards: Vec<CheckpointShard>,
}

/// Backend-resident representation of a distributed checkpoint payload.
///
/// Device backends expose numeric buffer element types rather than byte
/// buffers, so payload bytes are packed into little-endian `u32` words. The
/// original byte length and checksum are retained for exact restart validation.
pub struct BackendCheckpointBuffer<B: Backend> {
    buffer: B::Buffer<u32>,
    byte_len: usize,
    checksum: u64,
}

/// Typed restore outcome for a checkpoint continuation callback.
///
/// The callback only runs after the complete device payload has been copied
/// back, length-checked, and checksum-checked. It receives both the decoded
/// portable bundle and the backend allocation so a solver family can rebuild
/// executable device state without treating unvalidated bytes as a
/// continuation packet.
#[derive(Debug)]
pub enum BackendCheckpointRestoreError<E> {
    /// Device transport or portable checkpoint validation failed.
    Transport(CheckpointManifestError),
    /// A solver-family decoder rejected an otherwise valid bundle.
    Decode(E),
}

/// A portable solver checkpoint that can be reconstructed after native device
/// transport.
///
/// Implementations are intentionally limited to checkpoint types with a
/// versioned decoder. This makes family admission explicit: a new solver
/// cannot claim device continuation merely because its bytes fit inside a
/// [`SolverCheckpointBundle`].
pub trait DeviceCheckpointState: Sized {
    /// Reconstruct this family's executable continuation state from the
    /// validated active-Krylov section.
    fn from_device_checkpoint_bundle(bundle: &SolverCheckpointBundle) -> Result<Self, SolverError>;
}

macro_rules! impl_device_checkpoint_state {
    ($state:ident) => {
        impl<T: ComplexField> DeviceCheckpointState for $state<T> {
            fn from_device_checkpoint_bundle(
                bundle: &SolverCheckpointBundle,
            ) -> Result<Self, SolverError> {
                Self::from_checkpoint_bytes(bundle.krylov_state())
            }
        }
    };
}

impl_device_checkpoint_state!(GmresCheckpoint);
impl_device_checkpoint_state!(GmresBasisCheckpoint);
impl_device_checkpoint_state!(DeflatedCheckpoint);
impl_device_checkpoint_state!(GcroDrCheckpoint);
impl_device_checkpoint_state!(MixedPrecisionCheckpoint);
impl_device_checkpoint_state!(MixedPrecisionBlockCheckpoint);

impl<E: fmt::Display> fmt::Display for BackendCheckpointRestoreError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transport(error) => error.fmt(formatter),
            Self::Decode(error) => {
                write!(formatter, "checkpoint continuation decode failed: {error}")
            }
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for BackendCheckpointRestoreError<E> {}

impl<B: Backend> BackendCheckpointBuffer<B> {
    /// Number of meaningful payload bytes represented by this buffer.
    pub const fn byte_len(&self) -> usize {
        self.byte_len
    }

    /// Backend-owned packed words, for backend-native synchronization.
    pub fn buffer(&self) -> &B::Buffer<u32> {
        &self.buffer
    }
}

impl DistributedCheckpoint {
    /// Split a validated compositional solver bundle into deterministic shards.
    ///
    /// The bundle is encoded with its native portable checkpoint format before
    /// sharding, so every rank receives bytes that can be checked again at the
    /// typed restart boundary.
    pub fn split_bundle(
        bundle: &SolverCheckpointBundle,
        world_size: u32,
        residence: CheckpointResidence,
    ) -> Result<Self, CheckpointManifestError> {
        let payload = bundle.to_checkpoint_bytes()?;
        Self::split_payload(&payload, world_size, residence)
    }

    /// Split a complete portable checkpoint payload into deterministic shards.
    pub fn split_payload(
        payload: &[u8],
        world_size: u32,
        residence: CheckpointResidence,
    ) -> Result<Self, CheckpointManifestError> {
        if payload.len() > MAX_DISTRIBUTED_CHECKPOINT_BYTES {
            return Err(CheckpointManifestError::OversizedPayload);
        }
        if world_size == 0 || world_size > MAX_DISTRIBUTED_CHECKPOINT_SHARDS {
            return Err(CheckpointManifestError::Invalid("invalid world size"));
        }

        let mut shards = Vec::with_capacity(world_size as usize);
        let total = payload.len() as u128;
        let world = world_size as u128;
        for rank in 0..world_size {
            let start = (total * rank as u128 / world) as usize;
            let end = (total * (rank as u128 + 1) / world) as usize;
            shards.push(CheckpointShard::new(
                rank,
                start as u64,
                payload[start..end].to_vec(),
            )?);
        }
        Self::from_shards(residence, world_size, shards)
    }

    /// Construct a manifest from caller-provided shards.
    ///
    /// Shards may arrive in any order, but must cover one contiguous payload
    /// exactly once, with one shard for each rank in `0..world_size`.
    pub fn from_shards(
        residence: CheckpointResidence,
        world_size: u32,
        mut shards: Vec<CheckpointShard>,
    ) -> Result<Self, CheckpointManifestError> {
        if world_size == 0 || world_size > MAX_DISTRIBUTED_CHECKPOINT_SHARDS {
            return Err(CheckpointManifestError::Invalid("invalid world size"));
        }
        if shards.len() != world_size as usize {
            return Err(CheckpointManifestError::Invalid(
                "shard count does not equal world size",
            ));
        }
        shards.sort_by_key(CheckpointShard::rank);
        for (expected_rank, shard) in shards.iter().enumerate() {
            if shard.rank != expected_rank as u32 {
                return Err(CheckpointManifestError::Invalid(
                    "rank set is not contiguous",
                ));
            }
            if checksum64(&shard.payload) != shard.checksum {
                return Err(CheckpointManifestError::ChecksumMismatch);
            }
        }

        let mut next_offset = 0_u64;
        let mut payload = Vec::new();
        for shard in &shards {
            if shard.offset != next_offset {
                return Err(CheckpointManifestError::Invalid(
                    "shards do not form a contiguous payload",
                ));
            }
            next_offset = next_offset
                .checked_add(shard.payload.len() as u64)
                .ok_or(CheckpointManifestError::Overflow)?;
            if next_offset as usize > MAX_DISTRIBUTED_CHECKPOINT_BYTES {
                return Err(CheckpointManifestError::OversizedPayload);
            }
            payload.extend_from_slice(&shard.payload);
        }

        Ok(Self {
            residence,
            world_size,
            total_len: next_offset,
            payload_checksum: checksum64(&payload),
            shards,
        })
    }

    /// Backend provenance recorded by the envelope.
    pub const fn residence(&self) -> CheckpointResidence {
        self.residence
    }

    /// Number of ranks/device shards represented.
    pub const fn world_size(&self) -> u32 {
        self.world_size
    }

    /// Complete payload length in bytes.
    pub const fn total_len(&self) -> u64 {
        self.total_len
    }

    /// Checksum of the reassembled payload.
    pub const fn payload_checksum(&self) -> u64 {
        self.payload_checksum
    }

    /// Shards in deterministic rank order.
    pub fn shards(&self) -> &[CheckpointShard] {
        &self.shards
    }

    /// Reassemble and validate the complete portable checkpoint payload.
    pub fn assemble_payload(&self) -> Result<Vec<u8>, CheckpointManifestError> {
        let rebuilt = Self::from_shards(self.residence, self.world_size, self.shards.clone())?;
        if rebuilt.total_len != self.total_len || rebuilt.payload_checksum != self.payload_checksum
        {
            return Err(CheckpointManifestError::ChecksumMismatch);
        }
        let mut payload = Vec::with_capacity(self.total_len as usize);
        for shard in &rebuilt.shards {
            payload.extend_from_slice(&shard.payload);
        }
        Ok(payload)
    }

    /// Reassemble and decode a typed compositional solver bundle.
    ///
    /// This validates the shard envelope and the bundle wire format before a
    /// caller can resume a solver, avoiding an opaque-byte restart boundary.
    pub fn assemble_bundle(&self) -> Result<SolverCheckpointBundle, CheckpointManifestError> {
        let payload = self.assemble_payload()?;
        SolverCheckpointBundle::from_checkpoint_bytes(&payload)
    }

    /// Re-shard this validated payload for a different restart topology.
    ///
    /// Repartitioning is deterministic and checksum-preserving: the complete
    /// payload is validated before new rank boundaries are computed.  The
    /// residence tag is explicit because a restart may move from host to a
    /// device (or between device backends) at the same time.  This method does
    /// not perform inter-rank transport; callers still own that hand-off.
    pub fn repartition(
        &self,
        world_size: u32,
        residence: CheckpointResidence,
    ) -> Result<Self, CheckpointManifestError> {
        let payload = self.assemble_payload()?;
        Self::split_payload(&payload, world_size, residence)
    }

    /// Repartition a typed solver bundle across a changed restart topology.
    ///
    /// The existing shards must first decode as a valid bundle; the payload is
    /// then re-sharded deterministically with a caller-selected residence tag.
    pub fn repartition_bundle(
        &self,
        world_size: u32,
        residence: CheckpointResidence,
    ) -> Result<Self, CheckpointManifestError> {
        self.assemble_bundle()?;
        self.repartition(world_size, residence)
    }

    /// Copy the validated payload into a backend-owned byte buffer.
    ///
    /// The manifest remains the source of truth for length and checksum; the
    /// backend boundary is checked so device allocation or transfer failures
    /// are returned instead of panicking. Native solver packets retain their
    /// own portable wire formats inside this byte-oriented envelope.
    pub fn copy_payload_to_backend<B: Backend>(
        &self,
        backend: &B,
    ) -> Result<BackendCheckpointBuffer<B>, CheckpointManifestError> {
        let payload = self.assemble_payload()?;
        let word_len = payload.len().div_ceil(std::mem::size_of::<u32>());
        let mut words = Vec::new();
        words
            .try_reserve_exact(word_len)
            .map_err(|error| CheckpointManifestError::Backend(error.to_string()))?;
        words.resize(word_len, 0_u32);
        for (index, chunk) in payload.chunks(std::mem::size_of::<u32>()).enumerate() {
            let mut bytes = [0_u8; std::mem::size_of::<u32>()];
            bytes[..chunk.len()].copy_from_slice(chunk);
            words[index] = u32::from_le_bytes(bytes);
        }
        let mut device = backend
            .raw_alloc::<u32>(word_len)
            .map_err(|error| CheckpointManifestError::Allocation(error.to_string()))?;
        backend
            .try_copy_h2d(&words, &mut device)
            .map_err(checkpoint_transfer_error)?;
        backend.synchronize().map_err(checkpoint_transfer_error)?;
        Ok(BackendCheckpointBuffer {
            buffer: device,
            byte_len: payload.len(),
            checksum: self.payload_checksum,
        })
    }

    /// Copy and validate a backend-owned payload during restart.
    ///
    /// Length, transfer, and FNV checksum checks all happen before the bytes
    /// are returned to the typed checkpoint decoder.
    pub fn restore_payload_from_backend<B: Backend>(
        &self,
        backend: &B,
        device: &BackendCheckpointBuffer<B>,
    ) -> Result<Vec<u8>, CheckpointManifestError> {
        let expected_len =
            usize::try_from(self.total_len).map_err(|_| CheckpointManifestError::Overflow)?;
        let expected_words = expected_len.div_ceil(std::mem::size_of::<u32>());
        if device.byte_len != expected_len || device.buffer.len() != expected_words {
            return Err(CheckpointManifestError::Invalid(
                "backend payload length does not match manifest",
            ));
        }
        let mut words = Vec::new();
        words
            .try_reserve_exact(expected_words)
            .map_err(|error| CheckpointManifestError::Backend(error.to_string()))?;
        words.resize(expected_words, 0_u32);
        backend
            .try_copy_d2h(&device.buffer, &mut words)
            .map_err(checkpoint_transfer_error)?;
        backend.synchronize().map_err(checkpoint_transfer_error)?;
        let mut payload = Vec::new();
        payload
            .try_reserve_exact(expected_len)
            .map_err(|error| CheckpointManifestError::Backend(error.to_string()))?;
        for word in words {
            payload.extend_from_slice(&word.to_le_bytes());
        }
        payload.truncate(expected_len);
        if device.checksum != self.payload_checksum || checksum64(&payload) != self.payload_checksum
        {
            return Err(CheckpointManifestError::ChecksumMismatch);
        }
        Ok(payload)
    }

    /// Serialize the manifest and all payload shards in a strict little-endian format.
    /// Restore and decode a typed solver bundle from a backend buffer.
    ///
    /// Backend length/transfer/checksum checks run before the bundle decoder,
    /// so device hand-off failures and malformed typed state fail closed at
    /// the restart boundary.
    pub fn restore_bundle_from_backend<B: Backend>(
        &self,
        backend: &B,
        device: &BackendCheckpointBuffer<B>,
    ) -> Result<SolverCheckpointBundle, CheckpointManifestError> {
        let payload = self.restore_payload_from_backend(backend, device)?;
        SolverCheckpointBundle::from_checkpoint_bytes(&payload)
    }

    /// Restore a typed executable continuation state from a device packet.
    ///
    /// This is transactional: no decoder is called unless the D2H transfer
    /// and checksums succeed, and failed transport drops temporary data before
    /// any partial state is published. The caller retains the device buffer,
    /// allowing the decoder to rebuild family-specific native state beside the
    /// verified portable Krylov/preconditioner/factor bundle.
    pub fn restore_bundle_from_backend_with<B: Backend, T, E>(
        &self,
        backend: &B,
        device: &BackendCheckpointBuffer<B>,
        restore: impl FnOnce(SolverCheckpointBundle, &B::Buffer<u32>) -> Result<T, E>,
    ) -> Result<T, BackendCheckpointRestoreError<E>> {
        let bundle = self
            .restore_bundle_from_backend(backend, device)
            .map_err(BackendCheckpointRestoreError::Transport)?;
        restore(bundle, device.buffer()).map_err(BackendCheckpointRestoreError::Decode)
    }

    /// Restore one of the explicitly admitted solver-family checkpoints after
    /// native device transport.
    pub fn restore_device_state<B: Backend, S: DeviceCheckpointState>(
        &self,
        backend: &B,
        device: &BackendCheckpointBuffer<B>,
    ) -> Result<S, BackendCheckpointRestoreError<SolverError>> {
        self.restore_bundle_from_backend_with(backend, device, |bundle, _| {
            S::from_device_checkpoint_bundle(&bundle)
        })
    }

    pub fn to_checkpoint_bytes(&self) -> Result<Vec<u8>, CheckpointManifestError> {
        let payload = self.assemble_payload()?;
        let mut output = Vec::new();
        output.extend_from_slice(DISTRIBUTED_CHECKPOINT_MAGIC);
        output.extend_from_slice(&DISTRIBUTED_CHECKPOINT_FORMAT_VERSION.to_le_bytes());
        output.push(self.residence.tag());
        output.push(0); // reserved flags
        output.extend_from_slice(&self.world_size.to_le_bytes());
        output.extend_from_slice(&(self.shards.len() as u32).to_le_bytes());
        output.extend_from_slice(&self.total_len.to_le_bytes());
        output.extend_from_slice(&self.payload_checksum.to_le_bytes());
        for shard in &self.shards {
            output.extend_from_slice(&shard.rank.to_le_bytes());
            output.extend_from_slice(&shard.offset.to_le_bytes());
            output.extend_from_slice(&(shard.payload.len() as u64).to_le_bytes());
            output.extend_from_slice(&shard.checksum.to_le_bytes());
            output.extend_from_slice(&shard.payload);
        }
        if output.len() > MAX_DISTRIBUTED_CHECKPOINT_BYTES {
            return Err(CheckpointManifestError::OversizedPayload);
        }
        // Keep the value alive so a future format extension cannot accidentally
        // serialize a manifest whose payload differs from its advertised bytes.
        debug_assert_eq!(checksum64(&payload), self.payload_checksum);
        Ok(output)
    }

    /// Decode and fully validate a serialized distributed checkpoint.
    pub fn from_checkpoint_bytes(bytes: &[u8]) -> Result<Self, CheckpointManifestError> {
        let mut reader = Reader::new(bytes);
        if reader.take(8)? != DISTRIBUTED_CHECKPOINT_MAGIC {
            return Err(CheckpointManifestError::Invalid("invalid checkpoint magic"));
        }
        if reader.u16()? != DISTRIBUTED_CHECKPOINT_FORMAT_VERSION {
            return Err(CheckpointManifestError::Invalid(
                "unsupported checkpoint version",
            ));
        }
        let residence = CheckpointResidence::from_tag(reader.u8()?)?;
        if reader.u8()? != 0 {
            return Err(CheckpointManifestError::Invalid(
                "reserved flags are non-zero",
            ));
        }
        let world_size = reader.u32()?;
        let shard_count = reader.u32()?;
        let total_len = reader.u64()?;
        let payload_checksum = reader.u64()?;
        if world_size == 0 || world_size > MAX_DISTRIBUTED_CHECKPOINT_SHARDS {
            return Err(CheckpointManifestError::Invalid("invalid world size"));
        }
        if shard_count != world_size {
            return Err(CheckpointManifestError::Invalid(
                "shard count does not equal world size",
            ));
        }
        if total_len as usize > MAX_DISTRIBUTED_CHECKPOINT_BYTES {
            return Err(CheckpointManifestError::OversizedPayload);
        }

        let mut shards = Vec::with_capacity(shard_count as usize);
        for _ in 0..shard_count {
            let rank = reader.u32()?;
            let offset = reader.u64()?;
            let length = reader.u64()?;
            let checksum = reader.u64()?;
            let length_usize =
                usize::try_from(length).map_err(|_| CheckpointManifestError::Overflow)?;
            if length_usize > MAX_DISTRIBUTED_CHECKPOINT_BYTES {
                return Err(CheckpointManifestError::OversizedPayload);
            }
            let payload = reader.take(length_usize)?.to_vec();
            let shard = CheckpointShard::new(rank, offset, payload)?;
            if shard.checksum != checksum {
                return Err(CheckpointManifestError::ChecksumMismatch);
            }
            shards.push(shard);
        }
        if !reader.is_empty() {
            return Err(CheckpointManifestError::TrailingBytes);
        }
        let manifest = Self::from_shards(residence, world_size, shards)?;
        if manifest.total_len != total_len || manifest.payload_checksum != payload_checksum {
            return Err(CheckpointManifestError::ChecksumMismatch);
        }
        Ok(manifest)
    }
}

/// A compositional solver restart packet.
///
/// `krylov_state` is the active iterate/basis packet, while the optional
/// sections carry the preconditioner and direct-factor state needed to resume
/// without silently rebuilding a different numerical operator.  Each section
/// retains its native checkpoint format; this envelope only provides ordering,
/// presence flags, and independent checksums.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolverCheckpointBundle {
    krylov_state: Vec<u8>,
    preconditioner_state: Option<Vec<u8>>,
    factor_state: Option<Vec<u8>>,
}

/// Version of the compositional solver checkpoint envelope.
pub const SOLVER_CHECKPOINT_BUNDLE_FORMAT_VERSION: u16 = 1;
/// Magic prefix for [`SolverCheckpointBundle`] packets.
pub const SOLVER_CHECKPOINT_BUNDLE_MAGIC: &[u8; 8] = b"SSBUNDL1";

impl SolverCheckpointBundle {
    /// Construct a bundle from native solver/preconditioner/factor packets.
    pub fn new(
        krylov_state: Vec<u8>,
        preconditioner_state: Option<Vec<u8>>,
        factor_state: Option<Vec<u8>>,
    ) -> Result<Self, CheckpointManifestError> {
        if krylov_state.is_empty() {
            return Err(CheckpointManifestError::Invalid("krylov state is empty"));
        }
        let total = section_total_len(
            &krylov_state,
            preconditioner_state.as_deref(),
            factor_state.as_deref(),
        )?;
        if total > MAX_DISTRIBUTED_CHECKPOINT_BYTES {
            return Err(CheckpointManifestError::OversizedPayload);
        }
        Ok(Self {
            krylov_state,
            preconditioner_state,
            factor_state,
        })
    }

    /// Native active-Krylov checkpoint bytes.
    pub fn krylov_state(&self) -> &[u8] {
        &self.krylov_state
    }

    /// Optional native preconditioner checkpoint bytes.
    pub fn preconditioner_state(&self) -> Option<&[u8]> {
        self.preconditioner_state.as_deref()
    }

    /// Optional native direct-factor checkpoint bytes.
    pub fn factor_state(&self) -> Option<&[u8]> {
        self.factor_state.as_deref()
    }

    /// Serialize the bundle with strict section lengths and checksums.
    pub fn to_checkpoint_bytes(&self) -> Result<Vec<u8>, CheckpointManifestError> {
        let total = section_total_len(
            &self.krylov_state,
            self.preconditioner_state.as_deref(),
            self.factor_state.as_deref(),
        )?;
        if total > MAX_DISTRIBUTED_CHECKPOINT_BYTES {
            return Err(CheckpointManifestError::OversizedPayload);
        }
        let mut output = Vec::with_capacity(total + 20);
        output.extend_from_slice(SOLVER_CHECKPOINT_BUNDLE_MAGIC);
        output.extend_from_slice(&SOLVER_CHECKPOINT_BUNDLE_FORMAT_VERSION.to_le_bytes());
        let mut flags = 0_u8;
        if self.preconditioner_state.is_some() {
            flags |= 1;
        }
        if self.factor_state.is_some() {
            flags |= 1 << 1;
        }
        output.push(flags);
        output.push(0); // reserved flags
        append_section(&mut output, &self.krylov_state)?;
        if let Some(section) = &self.preconditioner_state {
            append_section(&mut output, section)?;
        }
        if let Some(section) = &self.factor_state {
            append_section(&mut output, section)?;
        }
        Ok(output)
    }

    /// Decode and validate a compositional solver checkpoint.
    pub fn from_checkpoint_bytes(bytes: &[u8]) -> Result<Self, CheckpointManifestError> {
        let mut reader = Reader::new(bytes);
        if reader.take(8)? != SOLVER_CHECKPOINT_BUNDLE_MAGIC {
            return Err(CheckpointManifestError::Invalid("invalid bundle magic"));
        }
        if reader.u16()? != SOLVER_CHECKPOINT_BUNDLE_FORMAT_VERSION {
            return Err(CheckpointManifestError::Invalid(
                "unsupported bundle version",
            ));
        }
        let flags = reader.u8()?;
        if flags & !0b11 != 0 {
            return Err(CheckpointManifestError::Invalid("unknown bundle flags"));
        }
        if reader.u8()? != 0 {
            return Err(CheckpointManifestError::Invalid(
                "reserved bundle flags are non-zero",
            ));
        }
        let krylov_state = read_section(&mut reader)?;
        if krylov_state.is_empty() {
            return Err(CheckpointManifestError::Invalid("krylov state is empty"));
        }
        let preconditioner_state = if flags & 1 != 0 {
            Some(read_section(&mut reader)?)
        } else {
            None
        };
        let factor_state = if flags & (1 << 1) != 0 {
            Some(read_section(&mut reader)?)
        } else {
            None
        };
        if !reader.is_empty() {
            return Err(CheckpointManifestError::TrailingBytes);
        }
        Self::new(krylov_state, preconditioner_state, factor_state)
    }
}

fn section_total_len(
    krylov_state: &[u8],
    preconditioner_state: Option<&[u8]>,
    factor_state: Option<&[u8]>,
) -> Result<usize, CheckpointManifestError> {
    let mut total = 0_usize;
    for section in [Some(krylov_state), preconditioner_state, factor_state]
        .into_iter()
        .flatten()
    {
        if section.len() > MAX_DISTRIBUTED_CHECKPOINT_BYTES {
            return Err(CheckpointManifestError::OversizedPayload);
        }
        total = total
            .checked_add(16)
            .and_then(|value| value.checked_add(section.len()))
            .ok_or(CheckpointManifestError::Overflow)?;
    }
    Ok(total)
}

fn append_section(output: &mut Vec<u8>, section: &[u8]) -> Result<(), CheckpointManifestError> {
    output.extend_from_slice(
        &(u64::try_from(section.len()).map_err(|_| CheckpointManifestError::Overflow)?)
            .to_le_bytes(),
    );
    output.extend_from_slice(&checksum64(section).to_le_bytes());
    output.extend_from_slice(section);
    Ok(())
}

fn read_section(reader: &mut Reader<'_>) -> Result<Vec<u8>, CheckpointManifestError> {
    let length = usize::try_from(reader.u64()?).map_err(|_| CheckpointManifestError::Overflow)?;
    if length > MAX_DISTRIBUTED_CHECKPOINT_BYTES {
        return Err(CheckpointManifestError::OversizedPayload);
    }
    let expected_checksum = reader.u64()?;
    let section = reader.take(length)?.to_vec();
    if checksum64(&section) != expected_checksum {
        return Err(CheckpointManifestError::ChecksumMismatch);
    }
    Ok(section)
}

/// Errors raised while constructing or decoding a checkpoint manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckpointManifestError {
    Invalid(&'static str),
    Truncated,
    TrailingBytes,
    Overflow,
    OversizedPayload,
    ChecksumMismatch,
    /// Native device allocation failed before a checkpoint state was made
    /// visible to a caller.
    Allocation(String),
    /// A device became unusable while transporting checkpoint bytes.
    DeviceLost {
        backend: &'static str,
        operation: &'static str,
        reason: String,
    },
    /// A checked H2D/D2H transfer or synchronization failed.
    Transfer(String),
    /// Backwards-compatible catch-all for transports that cannot expose a
    /// more precise typed error.
    Backend(String),
}

impl fmt::Display for CheckpointManifestError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(reason) => write!(formatter, "invalid checkpoint manifest: {reason}"),
            Self::Truncated => formatter.write_str("truncated checkpoint manifest"),
            Self::TrailingBytes => formatter.write_str("checkpoint manifest has trailing bytes"),
            Self::Overflow => formatter.write_str("checkpoint manifest size overflow"),
            Self::OversizedPayload => {
                formatter.write_str("checkpoint manifest payload is oversized")
            }
            Self::ChecksumMismatch => formatter.write_str("checkpoint manifest checksum mismatch"),
            Self::Allocation(reason) => {
                write!(formatter, "checkpoint backend allocation failed: {reason}")
            }
            Self::DeviceLost {
                backend,
                operation,
                reason,
            } => write!(
                formatter,
                "checkpoint device lost during {operation} on {backend}: {reason}"
            ),
            Self::Transfer(reason) => {
                write!(formatter, "checkpoint backend transfer failed: {reason}")
            }
            Self::Backend(reason) => {
                write!(formatter, "checkpoint backend transfer failed: {reason}")
            }
        }
    }
}

impl std::error::Error for CheckpointManifestError {}

/// Map malformed or failed checkpoint transport into the shared solver
/// failure vocabulary.  A manifest error never establishes a solve
/// iterate, so its accuracy is intentionally left unchecked.
impl SolverFailure for CheckpointManifestError {
    fn status(&self) -> SolverStatus {
        SolverStatus::NumericalError
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        AccuracyStatus::NotChecked
    }
}

fn checkpoint_transfer_error(error: BackendError) -> CheckpointManifestError {
    match error {
        BackendError::DeviceLost {
            backend,
            operation,
            reason,
        } => CheckpointManifestError::DeviceLost {
            backend,
            operation,
            reason,
        },
        BackendError::Allocation { reason } => CheckpointManifestError::Allocation(reason),
        other => CheckpointManifestError::Transfer(other.to_string()),
    }
}

fn checksum64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

struct Reader<'a> {
    bytes: &'a [u8],
    cursor: usize,
}

impl<'a> Reader<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, cursor: 0 }
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], CheckpointManifestError> {
        let end = self
            .cursor
            .checked_add(length)
            .ok_or(CheckpointManifestError::Overflow)?;
        if end > self.bytes.len() {
            return Err(CheckpointManifestError::Truncated);
        }
        let result = &self.bytes[self.cursor..end];
        self.cursor = end;
        Ok(result)
    }

    fn u8(&mut self) -> Result<u8, CheckpointManifestError> {
        Ok(*self
            .take(1)?
            .first()
            .ok_or(CheckpointManifestError::Truncated)?)
    }

    fn u16(&mut self) -> Result<u16, CheckpointManifestError> {
        Ok(u16::from_le_bytes(
            self.take(2)?
                .try_into()
                .map_err(|_| CheckpointManifestError::Truncated)?,
        ))
    }

    fn u32(&mut self) -> Result<u32, CheckpointManifestError> {
        Ok(u32::from_le_bytes(
            self.take(4)?
                .try_into()
                .map_err(|_| CheckpointManifestError::Truncated)?,
        ))
    }

    fn u64(&mut self) -> Result<u64, CheckpointManifestError> {
        Ok(u64::from_le_bytes(
            self.take(8)?
                .try_into()
                .map_err(|_| CheckpointManifestError::Truncated)?,
        ))
    }

    const fn is_empty(&self) -> bool {
        self.cursor == self.bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sonium_compute::CpuBackend;

    fn payload() -> Vec<u8> {
        (0_u8..=31).collect()
    }

    #[test]
    fn split_and_assemble_is_deterministic() {
        let manifest =
            DistributedCheckpoint::split_payload(&payload(), 5, CheckpointResidence::Cuda).unwrap();
        assert_eq!(manifest.world_size(), 5);
        assert_eq!(manifest.residence(), CheckpointResidence::Cuda);
        assert_eq!(manifest.assemble_payload().unwrap(), payload());
        let bytes = manifest.to_checkpoint_bytes().unwrap();
        assert_eq!(bytes, manifest.to_checkpoint_bytes().unwrap());
        let restored = DistributedCheckpoint::from_checkpoint_bytes(&bytes).unwrap();
        assert_eq!(restored, manifest);
    }

    #[test]
    fn caller_shard_order_is_canonicalized() {
        let base =
            DistributedCheckpoint::split_payload(&payload(), 3, CheckpointResidence::Host).unwrap();
        let mut shards = base.shards().to_vec();
        shards.reverse();
        let reordered =
            DistributedCheckpoint::from_shards(CheckpointResidence::Host, 3, shards).unwrap();
        assert_eq!(reordered, base);
    }

    #[test]
    fn empty_payload_keeps_all_rank_boundaries() {
        let manifest =
            DistributedCheckpoint::split_payload(&[], 4, CheckpointResidence::Metal).unwrap();
        assert_eq!(manifest.shards().len(), 4);
        assert!(
            manifest
                .shards()
                .iter()
                .all(|shard| shard.payload().is_empty())
        );
        assert!(manifest.assemble_payload().unwrap().is_empty());
    }

    #[test]
    fn rejects_world_and_rank_shape_errors() {
        assert!(matches!(
            DistributedCheckpoint::split_payload(&payload(), 0, CheckpointResidence::Host),
            Err(CheckpointManifestError::Invalid("invalid world size"))
        ));
        let shard = CheckpointShard::new(1, 0, payload()).unwrap();
        assert!(matches!(
            DistributedCheckpoint::from_shards(CheckpointResidence::Host, 1, vec![shard]),
            Err(CheckpointManifestError::Invalid(
                "rank set is not contiguous"
            ))
        ));
    }

    #[test]
    fn rejects_gaps_duplicate_ranks_and_bad_checksums() {
        let base =
            DistributedCheckpoint::split_payload(&payload(), 2, CheckpointResidence::Host).unwrap();
        let mut gap = base.shards().to_vec();
        gap[1].offset += 1;
        assert!(matches!(
            DistributedCheckpoint::from_shards(CheckpointResidence::Host, 2, gap),
            Err(CheckpointManifestError::Invalid(
                "shards do not form a contiguous payload"
            ))
        ));
        let mut duplicate = base.shards().to_vec();
        duplicate[1].rank = 0;
        assert!(matches!(
            DistributedCheckpoint::from_shards(CheckpointResidence::Host, 2, duplicate),
            Err(CheckpointManifestError::Invalid(
                "rank set is not contiguous"
            ))
        ));
        let mut corrupted = base.shards().to_vec();
        corrupted[0].payload[0] ^= 1;
        assert_eq!(
            DistributedCheckpoint::from_shards(CheckpointResidence::Host, 2, corrupted),
            Err(CheckpointManifestError::ChecksumMismatch)
        );
    }

    #[test]
    fn rejects_wire_corruption_truncation_and_trailing_bytes() {
        let manifest =
            DistributedCheckpoint::split_payload(&payload(), 3, CheckpointResidence::Metal)
                .unwrap();
        let bytes = manifest.to_checkpoint_bytes().unwrap();
        let mut bad_magic = bytes.clone();
        bad_magic[0] ^= 1;
        assert!(matches!(
            DistributedCheckpoint::from_checkpoint_bytes(&bad_magic),
            Err(CheckpointManifestError::Invalid("invalid checkpoint magic"))
        ));
        let mut bad_flags = bytes.clone();
        bad_flags[11] = 1;
        assert!(matches!(
            DistributedCheckpoint::from_checkpoint_bytes(&bad_flags),
            Err(CheckpointManifestError::Invalid(
                "reserved flags are non-zero"
            ))
        ));
        assert_eq!(
            DistributedCheckpoint::from_checkpoint_bytes(&bytes[..bytes.len() - 1]),
            Err(CheckpointManifestError::Truncated)
        );
        let mut trailing = bytes;
        trailing.push(0);
        assert_eq!(
            DistributedCheckpoint::from_checkpoint_bytes(&trailing),
            Err(CheckpointManifestError::TrailingBytes)
        );
    }

    #[test]
    fn rejects_unknown_residence_and_checksum_header_mismatch() {
        let manifest =
            DistributedCheckpoint::split_payload(&payload(), 2, CheckpointResidence::Host).unwrap();
        let mut bytes = manifest.to_checkpoint_bytes().unwrap();
        bytes[10] = 99;
        assert!(matches!(
            DistributedCheckpoint::from_checkpoint_bytes(&bytes),
            Err(CheckpointManifestError::Invalid("unknown residence tag"))
        ));
        let mut bytes = manifest.to_checkpoint_bytes().unwrap();
        let checksum_offset = 8 + 2 + 1 + 1 + 4 + 4 + 8;
        bytes[checksum_offset] ^= 1;
        assert_eq!(
            DistributedCheckpoint::from_checkpoint_bytes(&bytes),
            Err(CheckpointManifestError::ChecksumMismatch)
        );
    }

    #[test]
    fn bundle_round_trips_all_native_sections() {
        let bundle = SolverCheckpointBundle::new(
            b"krylov-state".to_vec(),
            Some(b"preconditioner-state".to_vec()),
            Some(b"factor-state".to_vec()),
        )
        .unwrap();
        let bytes = bundle.to_checkpoint_bytes().unwrap();
        let restored = SolverCheckpointBundle::from_checkpoint_bytes(&bytes).unwrap();
        assert_eq!(restored, bundle);
        let sharded =
            DistributedCheckpoint::split_payload(&bytes, 3, CheckpointResidence::Metal).unwrap();
        let rebuilt =
            SolverCheckpointBundle::from_checkpoint_bytes(&sharded.assemble_payload().unwrap())
                .unwrap();
        assert_eq!(rebuilt, bundle);
    }

    #[test]
    fn bundle_supports_krylov_only_state() {
        let bundle = SolverCheckpointBundle::new(b"basis".to_vec(), None, None).unwrap();
        let restored =
            SolverCheckpointBundle::from_checkpoint_bytes(&bundle.to_checkpoint_bytes().unwrap())
                .unwrap();
        assert_eq!(restored.krylov_state(), b"basis");
        assert_eq!(restored.preconditioner_state(), None);
        assert_eq!(restored.factor_state(), None);
    }

    #[test]
    fn bundle_rejects_empty_state_unknown_flags_and_section_corruption() {
        assert!(matches!(
            SolverCheckpointBundle::new(Vec::new(), None, None),
            Err(CheckpointManifestError::Invalid("krylov state is empty"))
        ));
        let bundle =
            SolverCheckpointBundle::new(b"krylov".to_vec(), Some(b"preconditioner".to_vec()), None)
                .unwrap();
        let bytes = bundle.to_checkpoint_bytes().unwrap();
        let mut flags = bytes.clone();
        flags[10] = 0x80;
        assert!(matches!(
            SolverCheckpointBundle::from_checkpoint_bytes(&flags),
            Err(CheckpointManifestError::Invalid("unknown bundle flags"))
        ));
        let mut corrupted = bytes;
        corrupted[28] ^= 1;
        assert_eq!(
            SolverCheckpointBundle::from_checkpoint_bytes(&corrupted),
            Err(CheckpointManifestError::ChecksumMismatch)
        );
    }

    #[test]
    fn bundle_rejects_truncated_trailing_and_reserved_wire_state() {
        let bundle = SolverCheckpointBundle::new(
            b"krylov".to_vec(),
            Some(b"preconditioner".to_vec()),
            Some(b"factor".to_vec()),
        )
        .unwrap();
        let bytes = bundle.to_checkpoint_bytes().unwrap();
        assert_eq!(
            SolverCheckpointBundle::from_checkpoint_bytes(&bytes[..bytes.len() - 1]),
            Err(CheckpointManifestError::Truncated)
        );
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert_eq!(
            SolverCheckpointBundle::from_checkpoint_bytes(&trailing),
            Err(CheckpointManifestError::TrailingBytes)
        );
        let mut reserved = bytes;
        reserved[11] = 1;
        assert!(matches!(
            SolverCheckpointBundle::from_checkpoint_bytes(&reserved),
            Err(CheckpointManifestError::Invalid(
                "reserved bundle flags are non-zero"
            ))
        ));
    }

    #[test]
    fn backend_bundle_restore_is_checked_and_exposes_native_buffer_to_decoder() {
        let bundle = SolverCheckpointBundle::new(
            b"krylov".to_vec(),
            Some(b"preconditioner".to_vec()),
            Some(b"factors".to_vec()),
        )
        .unwrap();
        let manifest =
            DistributedCheckpoint::split_bundle(&bundle, 2, CheckpointResidence::Cuda).unwrap();
        let backend = CpuBackend::new();
        backend.reset_transfer_stats();
        let device = manifest.copy_payload_to_backend(&backend).unwrap();
        let restored = manifest
            .restore_bundle_from_backend_with(&backend, &device, |decoded, native| {
                assert_eq!(native.len(), device.byte_len().div_ceil(4));
                Ok::<_, &'static str>(decoded)
            })
            .unwrap();
        assert_eq!(restored, bundle);
        let transfers = backend.transfer_stats();
        assert!(transfers.h2d_bytes > 0);
        assert!(transfers.d2h_bytes > 0);
    }

    #[test]
    fn backend_failure_mapping_preserves_device_loss_identity() {
        let loss = checkpoint_transfer_error(BackendError::DeviceLost {
            backend: "cuda",
            operation: "copy_d2h",
            reason: "context destroyed".into(),
        });
        assert!(matches!(
            loss,
            CheckpointManifestError::DeviceLost {
                backend: "cuda",
                operation: "copy_d2h",
                ..
            }
        ));
        let transfer = checkpoint_transfer_error(BackendError::Transfer {
            backend: "cuda",
            operation: "copy_h2d",
            reason: "DMA failed".into(),
        });
        assert!(matches!(transfer, CheckpointManifestError::Transfer(_)));
        let allocation = checkpoint_transfer_error(BackendError::Allocation {
            reason: "out of memory".into(),
        });
        assert!(matches!(allocation, CheckpointManifestError::Allocation(_)));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_dma_restore_survives_backend_recreation() {
        use crate::{CsrMatrix, GmresCheckpoint, GmresConfig, SolverResultContract, try_gmres};
        use sonium_compute::CudaBackend;

        let operator = CsrMatrix::from_triplets(
            3,
            3,
            vec![
                (0, 0, 4.0),
                (0, 1, 1.0),
                (1, 0, 1.0),
                (1, 1, 3.0),
                (1, 2, 1.0),
                (2, 1, 1.0),
                (2, 2, 2.0),
            ],
        );
        let config = GmresConfig {
            max_iterations: 30,
            restart: 3,
            tolerance: 1.0e-12,
            ..GmresConfig::default()
        };
        let first = try_gmres(&operator, &[5.0, 0.0, 3.0], &config).unwrap();
        let checkpoint = GmresCheckpoint::from_solution(&first).unwrap();
        let bundle = SolverCheckpointBundle::new(
            checkpoint.to_checkpoint_bytes().unwrap(),
            Some(b"native-preconditioner".to_vec()),
            Some(b"native-factors".to_vec()),
        )
        .unwrap();
        let manifest =
            DistributedCheckpoint::split_bundle(&bundle, 3, CheckpointResidence::Cuda).unwrap();

        let first_device = match CudaBackend::new_default() {
            Ok(backend) => backend,
            Err(error) if std::env::var("SONIUM_REQUIRE_CUDA").is_ok() => {
                panic!("CUDA required by SONIUM_REQUIRE_CUDA but unavailable: {error}")
            }
            Err(_) => return,
        };
        first_device.reset_transfer_stats();
        let first_packet = manifest.copy_payload_to_backend(&first_device).unwrap();
        assert_eq!(
            manifest
                .restore_bundle_from_backend(&first_device, &first_packet)
                .unwrap(),
            bundle
        );
        let first_transfers = first_device.transfer_stats();
        assert!(first_transfers.h2d_bytes > 0 && first_transfers.d2h_bytes > 0);

        // Recreate the CUDA context and reconstruct the continuation packet
        // on the new device from the validated portable bundle.
        drop(first_packet);
        drop(first_device);
        let recreated_device = CudaBackend::new_default()
            .expect("CUDA backend recreation failed after successful initialization");
        let recreated_packet = manifest.copy_payload_to_backend(&recreated_device).unwrap();
        assert_eq!(
            recreated_packet.buffer().len(),
            recreated_packet.byte_len().div_ceil(4)
        );
        let restored: GmresCheckpoint<f64> = manifest
            .restore_device_state(&recreated_device, &recreated_packet)
            .unwrap();
        let rhs = [1.0, 7.0, 6.0];
        let fresh = try_gmres(&operator, &rhs, &config).unwrap();
        let resumed = restored.resume(&operator, &rhs, &config).unwrap();
        fresh.validate_contract().unwrap();
        resumed.validate_contract().unwrap();
        assert!(resumed.residual <= 1.0e-10);
        for (resumed, fresh) in resumed.x.iter().zip(&fresh.x) {
            assert!((*resumed - *fresh).abs() <= 1.0e-10);
        }
    }
}
