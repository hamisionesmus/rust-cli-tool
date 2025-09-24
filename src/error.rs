//! Error handling utilities
//!
//! This module provides comprehensive error handling with custom error types,
//! error chaining, and user-friendly error messages.

use std::fmt;
use std::io;
use std::num::ParseIntError;
use thiserror::Error;

/// Main error type for the CLI tool
#[derive(Error, Debug)]
pub enum CliError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),

    #[error("Processing error: {0}")]
    Processing(#[from] ProcessingError),

    #[error("Validation error: {0}")]
    Validation(#[from] ValidationError),

    #[error("Serialization error: {0}")]
    Serialization(#[from] SerializationError),

    #[error("Network error: {0}")]
    Network(#[from] NetworkError),

    #[error("Authentication error: {0}")]
    Auth(#[from] AuthError),

    #[error("Database error: {0}")]
    Database(#[from] DatabaseError),

    #[error("Timeout error: {0}")]
    Timeout(String),

    #[error("Interrupted by user")]
    Interrupted,

    #[error("Unknown error: {0}")]
    Unknown(String),
}

/// Configuration-related errors
#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("Missing required configuration: {field}")]
    MissingField { field: String },

    #[error("Invalid configuration value for {field}: {value}")]
    InvalidValue { field: String, value: String },

    #[error("Configuration file not found: {path}")]
    FileNotFound { path: String },

    #[error("Configuration file format error: {message}")]
    FormatError { message: String },

    #[error("Environment variable error: {var}")]
    EnvVarError { var: String },
}

/// Processing-related errors
#[derive(Error, Debug)]
pub enum ProcessingError {
    #[error("Input file error: {message}")]
    InputError { message: String },

    #[error("Output file error: {message}")]
    OutputError { message: String },

    #[error("Data processing failed: {message}")]
    ProcessingFailed { message: String },

    #[error("Memory limit exceeded: {limit}MB used")]
    MemoryLimitExceeded { limit: usize },

    #[error("Processing timeout after {seconds}s")]
    Timeout { seconds: u64 },

    #[error("Invalid data format: {format}")]
    InvalidFormat { format: String },

    #[error("Data validation failed: {reason}")]
    ValidationFailed { reason: String },
}

/// Validation-related errors
#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("Required field missing: {field}")]
    RequiredFieldMissing { field: String },

    #[error("Invalid field value: {field} = {value}")]
    InvalidFieldValue { field: String, value: String },

    #[error("Field out of range: {field} = {value}, expected {min}..{max}")]
    OutOfRange {
        field: String,
        value: String,
        min: String,
        max: String,
    },

    #[error("Invalid format: {field} = {value}")]
    InvalidFormat { field: String, value: String },

    #[error("Duplicate value: {field} = {value}")]
    DuplicateValue { field: String, value: String },
}

/// Serialization-related errors
#[derive(Error, Debug)]
pub enum SerializationError {
    #[error("JSON serialization error: {message}")]
    JsonError { message: String },

    #[error("YAML serialization error: {message}")]
    YamlError { message: String },

    #[error("XML serialization error: {message}")]
    XmlError { message: String },

    #[error("Binary serialization error: {message}")]
    BinaryError { message: String },

    #[error("Unsupported format: {format}")]
    UnsupportedFormat { format: String },
}

/// Network-related errors
#[derive(Error, Debug)]
pub enum NetworkError {
    #[error("Connection failed: {host}:{port}")]
    ConnectionFailed { host: String, port: u16 },

    #[error("DNS resolution failed: {host}")]
    DnsResolutionFailed { host: String },

    #[error("SSL/TLS error: {message}")]
    SslError { message: String },

    #[error("HTTP error: {status} - {message}")]
    HttpError { status: u16, message: String },

    #[error("Timeout: {operation} timed out after {seconds}s")]
    Timeout { operation: String, seconds: u64 },

    #[error("Rate limited: {retry_after}s")]
    RateLimited { retry_after: u64 },
}

/// Authentication-related errors
#[derive(Error, Debug)]
pub enum AuthError {
    #[error("Authentication failed: {reason}")]
    AuthenticationFailed { reason: String },

    #[error("Authorization failed: insufficient permissions")]
    AuthorizationFailed,

    #[error("Token expired")]
    TokenExpired,

    #[error("Invalid token: {reason}")]
    InvalidToken { reason: String },

    #[error("Account locked: {reason}")]
    AccountLocked { reason: String },

    #[error("Two-factor authentication required")]
    TwoFactorRequired,
}

/// Database-related errors
#[derive(Error, Debug)]
pub enum DatabaseError {
    #[error("Connection failed: {message}")]
    ConnectionFailed { message: String },

    #[error("Query failed: {query}")]
    QueryFailed { query: String },

    #[error("Transaction failed: {message}")]
    TransactionFailed { message: String },

    #[error("Constraint violation: {constraint}")]
    ConstraintViolation { constraint: String },

    #[error("Record not found: {table} id={id}")]
    RecordNotFound { table: String, id: String },

    #[error("Duplicate key: {key}")]
    DuplicateKey { key: String },
}

/// Result type alias for convenience
pub type CliResult<T> = Result<T, CliError>;

/// Error handling utilities
pub struct ErrorHandler;

impl ErrorHandler {
    /// Convert any error to CliError
    pub fn to_cli_error<E: std::error::Error + Send + Sync + 'static>(error: E) -> CliError {
        CliError::Unknown(error.to_string())
    }

    /// Handle errors with appropriate exit codes
    pub fn handle_error(error: &CliError) -> i32 {
        match error {
            CliError::Config(_) => {
                eprintln!("Configuration error: {}", error);
                1
            }
            CliError::Validation(_) => {
                eprintln!("Validation error: {}", error);
                2
            }
            CliError::Processing(_) => {
                eprintln!("Processing error: {}", error);
                3
            }
            CliError::Io(_) => {
                eprintln!("I/O error: {}", error);
                4
            }
            CliError::Network(_) => {
                eprintln!("Network error: {}", error);
                5
            }
            CliError::Auth(_) => {
                eprintln!("Authentication error: {}", error);
                6
            }
            CliError::Database(_) => {
                eprintln!("Database error: {}", error);
                7
            }
            CliError::Timeout(_) => {
                eprintln!("Timeout error: {}", error);
                8
            }
            CliError::Interrupted => {
                eprintln!("Operation interrupted by user");
                130
            }
            CliError::Serialization(_) => {
                eprintln!("Serialization error: {}", error);
                9
            }
            CliError::Unknown(_) => {
                eprintln!("Unknown error: {}", error);
                99
            }
        }
    }

    /// Log error with context
    pub fn log_error(error: &CliError, context: Option<&str>) {
        let level = match error {
            CliError::Config(_) | CliError::Validation(_) => log::Level::Warn,
            CliError::Processing(_) | CliError::Io(_) => log::Level::Error,
            _ => log::Level::Info,
        };

        let message = if let Some(ctx) = context {
            format!("{}: {}", ctx, error)
        } else {
            error.to_string()
        };

        log::log!(level, "{}", message);

        // Log stack trace for unknown errors in debug mode
        #[cfg(debug_assertions)]
        if let CliError::Unknown(_) = error {
            log::error!("Stack trace: {:?}", std::backtrace::Backtrace::capture());
        }
    }

    /// Create user-friendly error message
    pub fn user_friendly_message(error: &CliError) -> String {
        match error {
            CliError::Config(ConfigError::MissingField { field }) => {
                format!("Missing required configuration: {}. Please check your configuration file or command line arguments.", field)
            }
            CliError::Config(ConfigError::FileNotFound { path }) => {
                format!("Configuration file not found: {}. Please ensure the file exists or specify a different path.", path)
            }
            CliError::Processing(ProcessingError::InputError { message }) => {
                format!("Failed to process input: {}. Please check your input file format and try again.", message)
            }
            CliError::Processing(ProcessingError::MemoryLimitExceeded { limit }) => {
                format!("Memory limit exceeded ({}MB). Try reducing batch size or increasing memory limit.", limit)
            }
            CliError::Network(NetworkError::ConnectionFailed { host, port }) => {
                format!("Failed to connect to {}:{}. Please check your network connection and server status.", host, port)
            }
            CliError::Auth(AuthError::AuthenticationFailed { reason }) => {
                format!("Authentication failed: {}. Please check your credentials.", reason)
            }
            CliError::Timeout(msg) => {
                format!("Operation timed out: {}. This might be due to large data size or slow processing.", msg)
            }
            _ => format!("An error occurred: {}. Please check the logs for more details.", error),
        }
    }

    /// Suggest recovery actions for errors
    pub fn suggest_recovery(error: &CliError) -> Vec<String> {
        match error {
            CliError::Config(ConfigError::MissingField { field }) => vec![
                format!("Set the '{}' configuration option", field),
                "Check the configuration file syntax".to_string(),
                "Use --help to see available options".to_string(),
            ],
            CliError::Processing(ProcessingError::MemoryLimitExceeded { .. }) => vec![
                "Reduce the batch size using --batch-size".to_string(),
                "Increase memory limit in configuration".to_string(),
                "Process data in smaller chunks".to_string(),
            ],
            CliError::Network(NetworkError::ConnectionFailed { .. }) => vec![
                "Check network connectivity".to_string(),
                "Verify server is running and accessible".to_string(),
                "Check firewall settings".to_string(),
            ],
            CliError::Io(_) => vec![
                "Check file permissions".to_string(),
                "Ensure file paths are correct".to_string(),
                "Verify disk space availability".to_string(),
            ],
            _ => vec![
                "Check the logs for more detailed error information".to_string(),
                "Try running with --verbose for additional details".to_string(),
                "Contact support if the problem persists".to_string(),
            ],
        }
    }
}

/// Custom result type for operations that might fail
pub type Result<T> = std::result::Result<T, CliError>;

/// Convenience macro for error handling
#[macro_export]
macro_rules! bail {
    ($err:expr) => {
        return Err($err.into());
    };
    ($fmt:literal $(, $arg:expr)* $(,)?) => {
        return Err(CliError::Unknown(format!($fmt, $($arg),*)));
    };
}

/// Convenience macro for ensuring conditions
#[macro_export]
macro_rules! ensure {
    ($cond:expr, $err:expr) => {
        if !($cond) {
            bail!($err);
        }
    };
    ($cond:expr, $fmt:literal $(, $arg:expr)* $(,)?) => {
        if !($cond) {
            bail!($fmt, $($arg),* );
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let error = CliError::Config(ConfigError::MissingField {
            field: "input_file".to_string(),
        });
        assert!(error.to_string().contains("input_file"));
    }

    #[test]
    fn test_error_handler_exit_codes() {
        let config_error = CliError::Config(ConfigError::MissingField {
            field: "test".to_string(),
        });
        assert_eq!(ErrorHandler::handle_error(&config_error), 1);

        let validation_error = CliError::Validation(ValidationError::RequiredFieldMissing {
            field: "test".to_string(),
        });
        assert_eq!(ErrorHandler::handle_error(&validation_error), 2);
    }

    #[test]
    fn test_user_friendly_messages() {
        let error = CliError::Config(ConfigError::MissingField {
            field: "input_file".to_string(),
        });
        let message = ErrorHandler::user_friendly_message(&error);
        assert!(message.contains("configuration"));
        assert!(message.contains("input_file"));
    }

    #[test]
    fn test_recovery_suggestions() {
        let error = CliError::Processing(ProcessingError::MemoryLimitExceeded { limit: 1024 });
        let suggestions = ErrorHandler::suggest_recovery(&error);
        assert!(!suggestions.is_empty());
        assert!(suggestions[0].contains("batch-size"));
    }
}