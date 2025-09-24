use clap::{Arg, Command};
use std::process;
use tracing::{error, info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod cli;
mod config;
mod error;
mod processing;
mod networking;
mod file_ops;
mod metrics;
mod plugins;
mod security;
mod cache;
mod database;
mod cicd;

use crate::cli::Cli;
use crate::config::Config;
use crate::error::CliError;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "cli_tool=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("🚀 Starting CLI Tool v{}", env!("CARGO_PKG_VERSION"));

    // Parse command line arguments
    let cli = Cli::parse();

    // Load configuration
    let config = match Config::load(&cli.config_path) {
        Ok(config) => config,
        Err(e) => {
            error!("Failed to load configuration: {}", e);
            process::exit(1);
        }
    };

    // Execute the command
    if let Err(e) = cli.execute(config).await {
        error!("Command execution failed: {}", e);
        process::exit(1);
    }

    info!("✅ CLI Tool execution completed successfully");
    Ok(())
}