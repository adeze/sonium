//! Bounded-memory newline-delimited JSON streaming helpers.

use serde::Serialize;
use serde::de::DeserializeOwned;
use std::io::{self, BufRead, BufReader, Read, Write};

/// Default maximum UTF-8 JSON record size accepted by the streaming reader.
pub const DEFAULT_MAX_JSON_RECORD_BYTES: usize = 16 * 1024 * 1024;

/// Streaming writer for independent JSON records.
///
/// Each record is serialized and flushed as one newline-delimited JSON line;
/// the writer never accumulates the complete frequency sweep in memory.
pub struct JsonLinesWriter<W> {
    writer: W,
    max_record_bytes: usize,
    records_written: usize,
}

impl<W: Write> JsonLinesWriter<W> {
    /// Create an empty record stream around an existing writer.
    pub const fn new(writer: W) -> Self {
        Self::with_max_record_bytes(writer, DEFAULT_MAX_JSON_RECORD_BYTES)
    }

    /// Create a writer with an explicit per-record bound.
    ///
    /// Records are serialized before writing, so an oversized record is
    /// rejected without leaving a partial JSON line in the stream.
    pub const fn with_max_record_bytes(writer: W, max_record_bytes: usize) -> Self {
        Self {
            writer,
            max_record_bytes,
            records_written: 0,
        }
    }

    /// Serialize one record and append a newline.
    pub fn write_record<T: Serialize>(&mut self, record: &T) -> io::Result<()> {
        if self.max_record_bytes == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "JSON record maximum must be greater than zero",
            ));
        }
        let encoded = serde_json::to_vec(record).map_err(io::Error::other)?;
        if encoded.len() > self.max_record_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "JSON record is {} bytes, exceeding configured {}-byte limit",
                    encoded.len(),
                    self.max_record_bytes
                ),
            ));
        }
        self.writer.write_all(&encoded)?;
        self.writer.write_all(b"\n")?;
        self.records_written += 1;
        Ok(())
    }

    /// Number of records successfully written so far.
    pub const fn records_written(&self) -> usize {
        self.records_written
    }

    /// Return the configured per-record bound.
    pub const fn max_record_bytes(&self) -> usize {
        self.max_record_bytes
    }

    /// Return the wrapped writer.
    pub fn into_inner(self) -> W {
        self.writer
    }
}

/// Streaming reader for newline-delimited JSON records.
pub struct JsonLinesReader<R> {
    reader: BufReader<R>,
    max_record_bytes: usize,
}

impl<R: Read> JsonLinesReader<R> {
    /// Create a record reader around an existing readable stream.
    pub fn new(reader: R) -> Self {
        Self::with_max_record_bytes(reader, DEFAULT_MAX_JSON_RECORD_BYTES)
    }

    /// Create a reader with an explicit per-record memory bound.
    pub fn with_max_record_bytes(reader: R, max_record_bytes: usize) -> Self {
        Self {
            reader: BufReader::new(reader),
            max_record_bytes,
        }
    }

    /// Read and deserialize the next record, or `Ok(None)` at EOF.
    pub fn next_record<T: DeserializeOwned>(&mut self) -> io::Result<Option<T>> {
        let mut record = Vec::new();
        let Some(_) = self.next_record_into(&mut record)? else {
            return Ok(None);
        };
        serde_json::from_slice(&record)
            .map(Some)
            .map_err(io::Error::other)
    }

    /// Read the next JSON record into caller-owned reusable storage.
    ///
    /// The returned slice borrows `record`; subsequent calls overwrite the
    /// same buffer. This keeps newline-delimited result streams bounded
    /// without allocating a fresh byte vector for each record.
    pub fn next_record_into<'a>(
        &mut self,
        record: &'a mut Vec<u8>,
    ) -> io::Result<Option<&'a [u8]>> {
        if self.max_record_bytes == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "JSON record maximum must be greater than zero",
            ));
        }
        record.clear();
        loop {
            let (take, consume, finished, eof, oversized) = {
                let buffer = self.reader.fill_buf()?;
                if buffer.is_empty() {
                    (0, 0, false, true, false)
                } else if let Some(newline) = buffer.iter().position(|byte| *byte == b'\n') {
                    let oversized = record.len().saturating_add(newline) > self.max_record_bytes;
                    (newline, newline + 1, true, false, oversized)
                } else {
                    let oversized =
                        record.len().saturating_add(buffer.len()) > self.max_record_bytes;
                    let consume = if oversized {
                        self.max_record_bytes
                            .saturating_add(1)
                            .saturating_sub(record.len())
                            .min(buffer.len())
                    } else {
                        buffer.len()
                    };
                    (consume, consume, false, false, oversized)
                }
            };

            if take > 0 {
                let buffer = self.reader.fill_buf()?;
                record.extend_from_slice(&buffer[..take]);
            }
            if consume > 0 {
                self.reader.consume(consume);
            }
            if oversized {
                // Consume the rest of the rejected line so callers can
                // continue with the next record after handling the error.
                if !finished {
                    self.discard_record_tail()?;
                }
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "JSON record exceeds configured {}-byte limit",
                        self.max_record_bytes
                    ),
                ));
            }
            if finished {
                return Ok(Some(record.as_slice()));
            }
            if eof {
                return if record.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(record.as_slice()))
                };
            }
        }
    }

    fn discard_record_tail(&mut self) -> io::Result<()> {
        loop {
            let buffer = self.reader.fill_buf()?;
            if buffer.is_empty() {
                return Ok(());
            }
            if let Some(newline) = buffer.iter().position(|byte| *byte == b'\n') {
                self.reader.consume(newline + 1);
                return Ok(());
            }
            let length = buffer.len();
            self.reader.consume(length);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    struct Record {
        frequency: u32,
        value: f64,
    }

    #[test]
    fn json_lines_round_trip_without_collecting_records() {
        let mut bytes = Vec::new();
        {
            let mut writer = JsonLinesWriter::new(&mut bytes);
            writer
                .write_record(&Record {
                    frequency: 100,
                    value: 1.0,
                })
                .expect("write first record");
            writer
                .write_record(&Record {
                    frequency: 200,
                    value: 2.0,
                })
                .expect("write second record");
            assert_eq!(writer.records_written(), 2);
        }

        let mut reader = JsonLinesReader::new(bytes.as_slice());
        assert_eq!(
            reader.next_record::<Record>().expect("read first"),
            Some(Record {
                frequency: 100,
                value: 1.0,
            })
        );
        assert_eq!(
            reader.next_record::<Record>().expect("read second"),
            Some(Record {
                frequency: 200,
                value: 2.0,
            })
        );
        assert_eq!(reader.next_record::<Record>().expect("read EOF"), None);
    }

    #[test]
    fn json_lines_reader_rejects_oversized_records_before_deserialization() {
        let bytes = br#"{"frequency":100,"value":1.0}"#;
        let mut reader = JsonLinesReader::with_max_record_bytes(bytes.as_slice(), 8);
        let error = reader
            .next_record::<Record>()
            .expect_err("record must exceed configured limit");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("exceeds configured"));
    }

    #[test]
    fn json_lines_writer_rejects_oversized_records_without_partial_lines() {
        let mut bytes = Vec::new();
        {
            let mut writer = JsonLinesWriter::with_max_record_bytes(&mut bytes, 8);
            let error = writer
                .write_record(&Record {
                    frequency: 100,
                    value: 1.0,
                })
                .expect_err("writer must enforce its record bound");
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
            assert_eq!(writer.records_written(), 0);
            assert_eq!(writer.max_record_bytes(), 8);
        }
        assert!(bytes.is_empty());
    }

    #[test]
    fn zero_record_bounds_fail_closed() {
        let mut bytes = Vec::new();
        let mut writer = JsonLinesWriter::with_max_record_bytes(&mut bytes, 0);
        let error = writer
            .write_record(&Record {
                frequency: 100,
                value: 1.0,
            })
            .expect_err("zero writer bound must be invalid");
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(bytes.is_empty());

        let mut reader = JsonLinesReader::with_max_record_bytes(br"{}\n".as_slice(), 0);
        let error = reader
            .next_record::<Record>()
            .expect_err("zero reader bound must be invalid");
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn oversized_reader_record_is_drained_before_the_next_record() {
        let bytes = br#"{"frequency":100,"value":1.0,"extra":"too-long"}
{"frequency":200,"value":2.0}
"#;
        let mut reader = JsonLinesReader::with_max_record_bytes(bytes.as_slice(), 40);
        assert_eq!(
            reader
                .next_record::<Record>()
                .expect_err("first record must exceed bound")
                .kind(),
            io::ErrorKind::InvalidData
        );
        assert_eq!(
            reader.next_record::<Record>().expect("second record read"),
            Some(Record {
                frequency: 200,
                value: 2.0,
            })
        );
    }

    #[test]
    fn reusable_json_record_buffer_avoids_per_record_allocation() {
        let bytes = br#"{"frequency":100,"value":1.0}
{"frequency":200,"value":2.0}
"#;
        let mut reader = JsonLinesReader::new(bytes.as_slice());
        let mut reusable = Vec::with_capacity(64);
        let first_ptr = {
            let first = reader
                .next_record_into(&mut reusable)
                .expect("read first")
                .expect("first record");
            assert_eq!(first, br#"{"frequency":100,"value":1.0}"#);
            first.as_ptr()
        };
        let capacity = reusable.capacity();
        let second_ptr = {
            let second = reader
                .next_record_into(&mut reusable)
                .expect("read second")
                .expect("second record");
            assert_eq!(second, br#"{"frequency":200,"value":2.0}"#);
            second.as_ptr()
        };
        assert_eq!(reusable.capacity(), capacity);
        assert_eq!(second_ptr, first_ptr);
        assert_eq!(
            reader.next_record_into(&mut reusable).expect("read EOF"),
            None
        );
    }
}
