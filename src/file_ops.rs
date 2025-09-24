//! File operations module
//!
//! This module provides comprehensive file and directory operations,
//! including compression, format conversion, and advanced I/O utilities.

use crate::error::{CliError, ProcessingError, CliResult};
use std::fs;
use std::path::{Path, PathBuf};
use std::io::{self, Read, Write, BufReader, BufWriter};
use std::collections::HashMap;
use walkdir::WalkDir;
use flate2::{Compression, GzBuilder};
use bzip2::Compression as BzCompression;
use lz4::EncoderBuilder;
use zstd::Encoder;
use tempfile::NamedTempFile;
use fs_extra::{self, dir};
use path_absolutize::*;
use same_file::Handle;
use notify::{Watcher, RecursiveMode, watcher};
use std::sync::mpsc::channel;
use std::time::{Duration, SystemTime};
use tracing::{info, warn, error, debug};
use serde::{Deserialize, Serialize};

/// File operation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileStats {
    pub total_files: usize,
    pub total_size: u64,
    pub processed_files: usize,
    pub errors: usize,
    pub operation_time: Duration,
}

/// File operations manager
pub struct FileOps {
    buffer_size: usize,
    follow_symlinks: bool,
    preserve_permissions: bool,
}

impl FileOps {
    /// Create a new file operations instance
    pub fn new() -> Self {
        Self {
            buffer_size: 8192,
            follow_symlinks: false,
            preserve_permissions: true,
        }
    }

    /// Configure buffer size
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Configure symlink following
    pub fn with_follow_symlinks(mut self, follow: bool) -> Self {
        self.follow_symlinks = follow;
        self
    }

    /// Copy file with progress tracking
    pub fn copy_file(&self, from: &Path, to: &Path) -> CliResult<u64> {
        info!("Copying file from {:?} to {:?}", from, to);

        // Ensure destination directory exists
        if let Some(parent) = to.parent() {
            fs::create_dir_all(parent)?;
        }

        let mut reader = BufReader::with_capacity(self.buffer_size, fs::File::open(from)?);
        let mut writer = BufWriter::with_capacity(self.buffer_size, fs::File::create(to)?);

        let mut buffer = vec![0; self.buffer_size];
        let mut total_bytes = 0u64;

        loop {
            let bytes_read = reader.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }
            writer.write_all(&buffer[..bytes_read])?;
            total_bytes += bytes_read as u64;
        }

        writer.flush()?;
        debug!("Copied {} bytes", total_bytes);

        Ok(total_bytes)
    }

    /// Copy directory recursively
    pub fn copy_directory(&self, from: &Path, to: &Path) -> CliResult<FileStats> {
        info!("Copying directory from {:?} to {:?}", from, to);

        let start_time = std::time::Instant::now();

        let mut stats = FileStats {
            total_files: 0,
            total_size: 0,
            processed_files: 0,
            errors: 0,
            operation_time: Duration::default(),
        };

        // Use fs_extra for efficient directory copying
        let options = dir::CopyOptions {
            overwrite: true,
            skip_exist: false,
            buffer_size: self.buffer_size,
            copy_inside: false,
            content_only: false,
            depth: 0,
        };

        match fs_extra::dir::copy(from, to, &options) {
            Ok(bytes_copied) => {
                stats.total_size = bytes_copied;
                stats.processed_files = 1; // Directory itself
                // Count files in the copied directory
                if let Ok(count) = self.count_files(to) {
                    stats.total_files = count;
                    stats.processed_files = count;
                }
            }
            Err(e) => {
                error!("Failed to copy directory: {}", e);
                stats.errors = 1;
                return Err(CliError::Io(e));
            }
        }

        stats.operation_time = start_time.elapsed();
        info!("Directory copy completed: {} files, {} bytes in {:.2}s",
              stats.processed_files, stats.total_size, stats.operation_time.as_secs_f64());

        Ok(stats)
    }

    /// Move file or directory
    pub fn move_item(&self, from: &Path, to: &Path) -> CliResult<()> {
        info!("Moving item from {:?} to {:?}", from, to);

        if let Some(parent) = to.parent() {
            fs::create_dir_all(parent)?;
        }

        fs::rename(from, to)?;
        debug!("Item moved successfully");

        Ok(())
    }

    /// Remove file or directory
    pub fn remove_item(&self, path: &Path) -> CliResult<()> {
        info!("Removing item: {:?}", path);

        if path.is_dir() {
            fs::remove_dir_all(path)?;
        } else {
            fs::remove_file(path)?;
        }

        debug!("Item removed successfully");
        Ok(())
    }

    /// Get file metadata
    pub fn get_metadata(&self, path: &Path) -> CliResult<FileMetadata> {
        let metadata = fs::metadata(path)?;

        Ok(FileMetadata {
            path: path.to_path_buf(),
            size: metadata.len(),
            is_dir: metadata.is_dir(),
            is_file: metadata.is_file(),
            modified: metadata.modified()?,
            created: metadata.created().ok(),
            permissions: format!("{:?}", metadata.permissions()),
        })
    }

    /// List directory contents
    pub fn list_directory(&self, path: &Path, recursive: bool, max_depth: Option<usize>) -> CliResult<Vec<FileMetadata>> {
        let mut results = Vec::new();

        let walker = if recursive {
            WalkDir::new(path).max_depth(max_depth.unwrap_or(10))
        } else {
            WalkDir::new(path).max_depth(1)
        };

        for entry in walker {
            let entry = entry?;
            let metadata = self.get_metadata(entry.path())?;
            results.push(metadata);
        }

        Ok(results)
    }

    /// Find files by pattern
    pub fn find_files(&self, root: &Path, pattern: &str, recursive: bool) -> CliResult<Vec<PathBuf>> {
        let mut results = Vec::new();
        let glob_pattern = glob::Pattern::new(pattern)?;

        let walker = if recursive {
            WalkDir::new(root)
        } else {
            WalkDir::new(root).max_depth(1)
        };

        for entry in walker {
            let entry = entry?;
            let path = entry.path();

            if glob_pattern.matches_path(path) {
                results.push(path.to_path_buf());
            }
        }

        Ok(results)
    }

    /// Calculate directory size
    pub fn calculate_size(&self, path: &Path) -> CliResult<u64> {
        let mut total_size = 0u64;

        for entry in WalkDir::new(path) {
            let entry = entry?;
            if entry.file_type().is_file() {
                total_size += entry.metadata()?.len();
            }
        }

        Ok(total_size)
    }

    /// Count files in directory
    fn count_files(&self, path: &Path) -> CliResult<usize> {
        let mut count = 0;

        for entry in WalkDir::new(path) {
            let entry = entry?;
            if entry.file_type().is_file() {
                count += 1;
            }
        }

        Ok(count)
    }
}

/// File metadata structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileMetadata {
    pub path: PathBuf,
    pub size: u64,
    pub is_dir: bool,
    pub is_file: bool,
    pub modified: SystemTime,
    pub created: Option<SystemTime>,
    pub permissions: String,
}

/// Compression utilities
pub struct CompressionOps;

impl CompressionOps {
    /// Compress file using gzip
    pub fn compress_gzip(input: &Path, output: &Path, level: Compression) -> CliResult<u64> {
        info!("Compressing {:?} to {:?} with gzip", input, output);

        let mut input_file = fs::File::open(input)?;
        let output_file = fs::File::create(output)?;
        let mut encoder = GzBuilder::new()
            .filename(input.file_name().unwrap().to_str().unwrap())
            .write(output_file, level);

        let bytes_copied = io::copy(&mut input_file, &mut encoder)?;
        encoder.finish()?;

        debug!("Compressed {} bytes", bytes_copied);
        Ok(bytes_copied)
    }

    /// Decompress gzip file
    pub fn decompress_gzip(input: &Path, output: &Path) -> CliResult<u64> {
        info!("Decompressing gzip file {:?} to {:?}", input, output);

        use flate2::read::GzDecoder;
        let input_file = fs::File::open(input)?;
        let mut decoder = GzDecoder::new(input_file);
        let mut output_file = fs::File::create(output)?;

        let bytes_copied = io::copy(&mut decoder, &mut output_file)?;

        debug!("Decompressed {} bytes", bytes_copied);
        Ok(bytes_copied)
    }

    /// Compress file using bzip2
    pub fn compress_bzip2(input: &Path, output: &Path, level: BzCompression) -> CliResult<u64> {
        info!("Compressing {:?} to {:?} with bzip2", input, output);

        let input_file = fs::File::open(input)?;
        let output_file = fs::File::create(output)?;
        let mut encoder = bzip2::write::BzEncoder::new(output_file, level);

        let bytes_copied = io::copy(&mut BufReader::new(input_file), &mut encoder)?;
        encoder.finish()?;

        debug!("Compressed {} bytes", bytes_copied);
        Ok(bytes_copied)
    }

    /// Compress file using LZ4
    pub fn compress_lz4(input: &Path, output: &Path) -> CliResult<u64> {
        info!("Compressing {:?} to {:?} with LZ4", input, output);

        let input_file = fs::File::open(input)?;
        let output_file = fs::File::create(output)?;
        let mut encoder = EncoderBuilder::new()
            .level(4)
            .build(output_file)?;

        let bytes_copied = io::copy(&mut BufReader::new(input_file), &mut encoder)?;
        encoder.finish()?;

        debug!("Compressed {} bytes", bytes_copied);
        Ok(bytes_copied)
    }

    /// Compress file using Zstandard
    pub fn compress_zstd(input: &Path, output: &Path, level: i32) -> CliResult<u64> {
        info!("Compressing {:?} to {:?} with Zstandard", input, output);

        let input_file = fs::File::open(input)?;
        let output_file = fs::File::create(output)?;
        let mut encoder = Encoder::new(output_file, level)?;

        let bytes_copied = io::copy(&mut BufReader::new(input_file), &mut encoder)?;
        encoder.finish()?;

        debug!("Compressed {} bytes", bytes_copied);
        Ok(bytes_copied)
    }

    /// Auto-detect and decompress file
    pub fn decompress_auto(input: &Path, output: &Path) -> CliResult<u64> {
        let extension = input.extension()
            .and_then(|s| s.to_str())
            .unwrap_or("");

        match extension {
            "gz" | "gzip" => Self::decompress_gzip(input, output),
            "bz2" | "bzip2" => {
                // bzip2 decompression would be implemented here
                Err(CliError::Processing(ProcessingError::InvalidFormat(
                    "bzip2 decompression not yet implemented".to_string()
                )))
            }
            "lz4" => {
                // lz4 decompression would be implemented here
                Err(CliError::Processing(ProcessingError::InvalidFormat(
                    "lz4 decompression not yet implemented".to_string()
                )))
            }
            "zst" => {
                // zstd decompression would be implemented here
                Err(CliError::Processing(ProcessingError::InvalidFormat(
                    "zstd decompression not yet implemented".to_string()
                )))
            }
            _ => Err(CliError::Processing(ProcessingError::InvalidFormat(
                format!("Unsupported compression format: {}", extension)
            ))),
        }
    }
}

/// File watcher for monitoring changes
pub struct FileWatcher {
    watcher: notify::RecommendedWatcher,
    rx: std::sync::mpsc::Receiver<notify::DebouncedEvent>,
}

impl FileWatcher {
    /// Create a new file watcher
    pub fn new(delay: Duration) -> CliResult<Self> {
        let (tx, rx) = channel();
        let mut watcher = watcher(tx, delay)?;
        watcher.configure(notify::Config::default())?;

        Ok(Self { watcher, rx })
    }

    /// Watch a path for changes
    pub fn watch(&mut self, path: &Path, recursive: bool) -> CliResult<()> {
        let mode = if recursive { RecursiveMode::Recursive } else { RecursiveMode::NonRecursive };
        self.watcher.watch(path, mode)?;
        info!("Watching path: {:?}", path);
        Ok(())
    }

    /// Stop watching a path
    pub fn unwatch(&mut self, path: &Path) -> CliResult<()> {
        self.watcher.unwatch(path)?;
        info!("Stopped watching path: {:?}", path);
        Ok(())
    }

    /// Get the next event (blocking)
    pub fn next_event(&self) -> CliResult<notify::DebouncedEvent> {
        match self.rx.recv() {
            Ok(event) => Ok(event),
            Err(_) => Err(CliError::Processing(ProcessingError::ProcessingFailed(
                "File watcher channel closed".to_string()
            ))),
        }
    }

    /// Try to get the next event (non-blocking)
    pub fn try_next_event(&self) -> Option<notify::DebouncedEvent> {
        match self.rx.try_recv() {
            Ok(event) => Some(event),
            Err(std::sync::mpsc::TryRecvError::Empty) => None,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => None,
        }
    }
}

/// Temporary file utilities
pub struct TempFileOps;

impl TempFileOps {
    /// Create a temporary file with content
    pub fn create_temp_file(content: &[u8], extension: Option<&str>) -> CliResult<NamedTempFile> {
        let mut builder = NamedTempFile::new()?;
        if let Some(ext) = extension {
            builder = builder.suffix(ext);
        }

        let mut temp_file = builder.build()?;
        temp_file.write_all(content)?;
        temp_file.flush()?;

        Ok(temp_file)
    }

    /// Create a temporary directory
    pub fn create_temp_dir() -> CliResult<tempfile::TempDir> {
        let temp_dir = tempfile::tempdir()?;
        debug!("Created temporary directory: {:?}", temp_dir.path());
        Ok(temp_dir)
    }
}

/// File system utilities
pub struct FsUtils;

impl FsUtils {
    /// Get absolute path
    pub fn absolute_path(path: &Path) -> CliResult<PathBuf> {
        Ok(path.absolutize()?.to_path_buf())
    }

    /// Check if two paths refer to the same file
    pub fn is_same_file(path1: &Path, path2: &Path) -> CliResult<bool> {
        let handle1 = Handle::from_path(path1)?;
        let handle2 = Handle::from_path(path2)?;
        Ok(handle1 == handle2)
    }

    /// Get file extension
    pub fn get_extension(path: &Path) -> Option<String> {
        path.extension()?.to_str().map(|s| s.to_lowercase())
    }

    /// Change file extension
    pub fn change_extension(path: &Path, new_ext: &str) -> PathBuf {
        let mut new_path = path.to_path_buf();
        new_path.set_extension(new_ext);
        new_path
    }

    /// Ensure directory exists
    pub fn ensure_dir(path: &Path) -> CliResult<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        Ok(())
    }

    /// Get file stem (name without extension)
    pub fn get_stem(path: &Path) -> Option<String> {
        path.file_stem()?.to_str().map(|s| s.to_string())
    }
}

impl Default for FileOps {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_file_operations() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test.txt");
        let content = b"Hello, World!";

        // Create test file
        fs::write(&file_path, content).unwrap();

        let file_ops = FileOps::new();

        // Test metadata
        let metadata = file_ops.get_metadata(&file_path).unwrap();
        assert_eq!(metadata.size, content.len() as u64);
        assert!(metadata.is_file);

        // Test copy
        let copy_path = temp_dir.path().join("test_copy.txt");
        let bytes_copied = file_ops.copy_file(&file_path, &copy_path).unwrap();
        assert_eq!(bytes_copied, content.len() as u64);

        // Verify copy
        let copy_content = fs::read(&copy_path).unwrap();
        assert_eq!(copy_content, content);
    }

    #[test]
    fn test_compression() {
        let temp_dir = tempdir().unwrap();
        let input_path = temp_dir.path().join("input.txt");
        let output_path = temp_dir.path().join("output.txt.gz");

        let content = b"This is a test file for compression.";
        fs::write(&input_path, content).unwrap();

        // Test gzip compression
        let bytes_compressed = CompressionOps::compress_gzip(
            &input_path,
            &output_path,
            Compression::default()
        ).unwrap();

        assert_eq!(bytes_compressed, content.len() as u64);

        // Test gzip decompression
        let decompressed_path = temp_dir.path().join("decompressed.txt");
        let bytes_decompressed = CompressionOps::decompress_gzip(
            &output_path,
            &decompressed_path
        ).unwrap();

        let decompressed_content = fs::read(&decompressed_path).unwrap();
        assert_eq!(decompressed_content, content);
        assert_eq!(bytes_decompressed, content.len() as u64);
    }

    #[test]
    fn test_fs_utils() {
        let path = Path::new("/home/user/test.txt");

        assert_eq!(FsUtils::get_extension(path), Some("txt".to_string()));
        assert_eq!(FsUtils::get_stem(path), Some("test".to_string()));

        let new_path = FsUtils::change_extension(path, "md");
        assert_eq!(new_path, Path::new("/home/user/test.md"));
    }
}