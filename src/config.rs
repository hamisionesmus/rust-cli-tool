//! Configuration management for the CLI tool
//!
//! This module handles loading configuration from various sources:
//! - Command line arguments
//! - Configuration files (TOML, JSON, YAML)
//! - Environment variables
//! - Default values

use clap::{Arg, ArgMatches, Command};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Configuration error types
#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("TOML parsing error: {0}")]
    Toml(#[from] toml::de::Error),

    #[error("JSON parsing error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("YAML parsing error: {0}")]
    Yaml(#[from] serde_yaml::Error),

    #[error("Environment variable error: {0}")]
    EnvVar(#[from] env::VarError),

    #[error("Configuration validation error: {0}")]
    Validation(String),

    #[error("Unknown configuration format: {0}")]
    UnknownFormat(String),
}

/// Supported configuration file formats
#[derive(Debug, Clone, Copy)]
pub enum ConfigFormat {
    Toml,
    Json,
    Yaml,
}

impl ConfigFormat {
    /// Detect format from file extension
    pub fn from_path(path: &Path) -> Result<Self, ConfigError> {
        match path.extension().and_then(|s| s.to_str()) {
            Some("toml") => Ok(ConfigFormat::Toml),
            Some("json") => Ok(ConfigFormat::Json),
            Some("yaml") | Some("yml") => Ok(ConfigFormat::Yaml),
            Some(ext) => Err(ConfigError::UnknownFormat(ext.to_string())),
            None => Err(ConfigError::UnknownFormat("no extension".to_string())),
        }
    }
}

/// Main configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Input file path
    pub input_file: Option<String>,

    /// Output file path
    pub output_file: Option<String>,

    /// Number of worker threads
    pub workers: usize,

    /// Batch size for processing
    pub batch_size: usize,

    /// Processing timeout in seconds
    pub timeout: u64,

    /// Verbosity level (0-3)
    pub verbose: u8,

    /// Enable progress reporting
    pub progress: bool,

    /// Configuration file path
    pub config_file: Option<PathBuf>,

    /// Custom key-value pairs
    pub custom: HashMap<String, String>,

    /// Processing options
    pub processing: ProcessingConfig,

    /// Output options
    pub output: OutputConfig,

    /// Performance tuning
    pub performance: PerformanceConfig,
}

/// Processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    /// Enable parallel processing
    pub parallel: bool,

    /// Buffer size for I/O operations
    pub buffer_size: usize,

    /// Retry failed operations
    pub retry_failed: bool,

    /// Maximum retry attempts
    pub max_retries: u32,

    /// Enable compression
    pub compression: bool,

    /// Validation options
    pub validation: ValidationConfig,
}

/// Validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Enable input validation
    pub enabled: bool,

    /// Strict validation mode
    pub strict: bool,

    /// Custom validation rules
    pub rules: Vec<String>,
}

/// Output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Output format
    pub format: OutputFormat,

    /// Pretty print output
    pub pretty: bool,

    /// Include metadata
    pub metadata: bool,

    /// Compression level (0-9)
    pub compression_level: u8,
}

/// Output format options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputFormat {
    Json,
    Csv,
    Xml,
    Yaml,
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Memory limit in MB
    pub memory_limit: usize,

    /// CPU affinity
    pub cpu_affinity: Option<Vec<usize>>,

    /// I/O priority
    pub io_priority: String,

    /// Enable memory pooling
    pub memory_pool: bool,

    /// Cache size in MB
    pub cache_size: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            input_file: None,
            output_file: None,
            workers: num_cpus::get(),
            batch_size: 1000,
            timeout: 300,
            verbose: 1,
            progress: true,
            config_file: None,
            custom: HashMap::new(),
            processing: ProcessingConfig::default(),
            output: OutputConfig::default(),
            performance: PerformanceConfig::default(),
        }
    }
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            parallel: true,
            buffer_size: 8192,
            retry_failed: true,
            max_retries: 3,
            compression: false,
            validation: ValidationConfig::default(),
        }
    }
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            strict: false,
            rules: Vec::new(),
        }
    }
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            format: OutputFormat::Json,
            pretty: true,
            metadata: true,
            compression_level: 6,
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            memory_limit: 1024,
            cpu_affinity: None,
            io_priority: "normal".to_string(),
            memory_pool: true,
            cache_size: 256,
        }
    }
}

impl Config {
    /// Load configuration from command line arguments
    pub fn from_args() -> Result<Self, ConfigError> {
        let matches = Self::build_cli().get_matches();
        Self::from_matches(&matches)
    }

    /// Load configuration from matches
    pub fn from_matches(matches: &ArgMatches) -> Result<Self, ConfigError> {
        let mut config = Self::default();

        // Load from config file first
        if let Some(config_path) = matches.get_one::<String>("config") {
            let config_file = PathBuf::from(config_path);
            if config_file.exists() {
                config = Self::load_from_file(&config_file)?;
                config.config_file = Some(config_file);
            }
        }

        // Override with command line arguments
        if let Some(input) = matches.get_one::<String>("input") {
            config.input_file = Some(input.clone());
        }

        if let Some(output) = matches.get_one::<String>("output") {
            config.output_file = Some(output.clone());
        }

        if let Some(workers) = matches.get_one::<String>("workers") {
            config.workers = workers.parse().map_err(|_| {
                ConfigError::Validation("Invalid workers value".to_string())
            })?;
        }

        if let Some(batch_size) = matches.get_one::<String>("batch-size") {
            config.batch_size = batch_size.parse().map_err(|_| {
                ConfigError::Validation("Invalid batch-size value".to_string())
            })?;
        }

        config.verbose = matches.get_count("verbose") as u8;
        config.progress = !matches.get_flag("no-progress");

        // Load environment variables
        Self::load_from_env(&mut config)?;

        // Validate configuration
        config.validate()?;

        Ok(config)
    }

    /// Load configuration from file
    pub fn load_from_file(path: &Path) -> Result<Self, ConfigError> {
        let content = fs::read_to_string(path)?;
        let format = ConfigFormat::from_path(path)?;

        let config: Config = match format {
            ConfigFormat::Toml => toml::from_str(&content)?,
            ConfigFormat::Json => serde_json::from_str(&content)?,
            ConfigFormat::Yaml => serde_yaml::from_str(&content)?,
        };

        Ok(config)
    }

    /// Save configuration to file
    pub fn save_to_file(&self, path: &Path) -> Result<(), ConfigError> {
        let format = ConfigFormat::from_path(path)?;
        let content = match format {
            ConfigFormat::Toml => toml::to_string_pretty(self)?,
            ConfigFormat::Json => serde_json::to_string_pretty(self)?,
            ConfigFormat::Yaml => serde_yaml::to_string(self)?,
        };

        fs::write(path, content)?;
        Ok(())
    }

    /// Load configuration from environment variables
    fn load_from_env(config: &mut Config) -> Result<(), ConfigError> {
        if let Ok(workers) = env::var("CLI_WORKERS") {
            config.workers = workers.parse().unwrap_or(config.workers);
        }

        if let Ok(batch_size) = env::var("CLI_BATCH_SIZE") {
            config.batch_size = batch_size.parse().unwrap_or(config.batch_size);
        }

        if let Ok(verbose) = env::var("CLI_VERBOSE") {
            config.verbose = verbose.parse().unwrap_or(config.verbose);
        }

        Ok(())
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.workers == 0 {
            return Err(ConfigError::Validation("Workers must be greater than 0".to_string()));
        }

        if self.batch_size == 0 {
            return Err(ConfigError::Validation("Batch size must be greater than 0".to_string()));
        }

        if self.performance.memory_limit == 0 {
            return Err(ConfigError::Validation("Memory limit must be greater than 0".to_string()));
        }

        Ok(())
    }

    /// Build CLI interface
    fn build_cli() -> Command {
        Command::new("cli-tool")
            .version(env!("CARGO_PKG_VERSION"))
            .author(env!("CARGO_PKG_AUTHORS"))
            .about("High-performance CLI processing tool")
            .arg(
                Arg::new("config")
                    .short('c')
                    .long("config")
                    .value_name("FILE")
                    .help("Configuration file path")
            )
            .arg(
                Arg::new("input")
                    .short('i')
                    .long("input")
                    .value_name("FILE")
                    .help("Input file to process")
            )
            .arg(
                Arg::new("output")
                    .short('o')
                    .long("output")
                    .value_name("FILE")
                    .help("Output file")
            )
            .arg(
                Arg::new("workers")
                    .short('w')
                    .long("workers")
                    .value_name("NUM")
                    .help("Number of worker threads")
            )
            .arg(
                Arg::new("batch-size")
                    .short('b')
                    .long("batch-size")
                    .value_name("SIZE")
                    .help("Batch size for processing")
            )
            .arg(
                Arg::new("verbose")
                    .short('v')
                    .long("verbose")
                    .action(clap::ArgAction::Count)
                    .help("Increase verbosity level")
            )
            .arg(
                Arg::new("no-progress")
                    .long("no-progress")
                    .help("Disable progress reporting")
                    .action(clap::ArgAction::SetTrue)
            )
    }

    /// Get configuration summary
    pub fn summary(&self) -> String {
        format!(
            "Configuration:\n\
             - Workers: {}\n\
             - Batch Size: {}\n\
             - Timeout: {}s\n\
             - Verbose: {}\n\
             - Progress: {}\n\
             - Memory Limit: {}MB\n\
             - Cache Size: {}MB",
            self.workers,
            self.batch_size,
            self.timeout,
            self.verbose,
            self.progress,
            self.performance.memory_limit,
            self.performance.cache_size
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert!(config.workers > 0);
        assert_eq!(config.batch_size, 1000);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation() {
        let mut config = Config::default();
        config.workers = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_file_formats() {
        let config = Config::default();

        // Test TOML
        let toml_file = NamedTempFile::new().unwrap();
        config.save_to_file(toml_file.path()).unwrap();
        let loaded = Config::load_from_file(toml_file.path()).unwrap();
        assert_eq!(loaded.workers, config.workers);

        // Test JSON
        let json_file = NamedTempFile::with_suffix(".json").unwrap();
        config.save_to_file(json_file.path()).unwrap();
        let loaded = Config::load_from_file(json_file.path()).unwrap();
        assert_eq!(loaded.workers, config.workers);
    }
}