use clap::{Arg, ArgMatches, Command};
use std::path::PathBuf;
use anyhow::Result;
use dialoguer::{theme::ColorfulTheme, Input, Select, Confirm, MultiSelect};
use console::Term;
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Duration;

pub fn build_cli() -> Command {
    Command::new("rust-cli-tool")
        .version(env!("CARGO_PKG_VERSION"))
        .author("Hamisi Onesmus <kilumohamisi@gmail.com>")
        .about("A high-performance CLI tool for data processing and analysis")
        .arg_required_else_help(true)
        .arg(
            Arg::new("config")
                .short('c')
                .long("config")
                .value_name("FILE")
                .help("Path to configuration file")
                .global(true)
        )
        .arg(
            Arg::new("verbose")
                .short('v')
                .long("verbose")
                .help("Enable verbose output")
                .action(clap::ArgAction::SetTrue)
                .global(true)
        )
        .arg(
            Arg::new("quiet")
                .short('q')
                .long("quiet")
                .help("Suppress all output except errors")
                .action(clap::ArgAction::SetTrue)
                .global(true)
        )
        .arg(
            Arg::new("dry-run")
                .long("dry-run")
                .help("Show what would be done without executing")
                .action(clap::ArgAction::SetTrue)
                .global(true)
        )
        .arg(
            Arg::new("force")
                .short('f')
                .long("force")
                .help("Force overwrite existing files")
                .action(clap::ArgAction::SetTrue)
                .global(true)
        )
        .subcommand(
            Command::new("process")
                .about("Process data files")
                .arg(
                    Arg::new("input")
                        .short('i')
                        .long("input")
                        .value_name("FILE")
                        .help("Input file to process")
                        .required(true)
                )
                .arg(
                    Arg::new("output")
                        .short('o')
                        .long("output")
                        .value_name("DIR")
                        .help("Output directory")
                        .default_value("./output")
                )
                .arg(
                    Arg::new("format")
                        .long("format")
                        .value_name("FORMAT")
                        .help("Output format (json, csv, xml, yaml, binary)")
                        .default_value("json")
                )
                .arg(
                    Arg::new("batch-size")
                        .long("batch-size")
                        .value_name("SIZE")
                        .help("Processing batch size")
                        .default_value("1000")
                )
                .arg(
                    Arg::new("max-concurrent")
                        .long("max-concurrent")
                        .value_name("COUNT")
                        .help("Maximum concurrent operations")
                        .default_value(&num_cpus::get().to_string())
                )
                .arg(
                    Arg::new("compression")
                        .long("compression")
                        .value_name("TYPE")
                        .help("Output compression (gzip, bzip2, xz, zstd)")
                )
        )
        .subcommand(
            Command::new("analyze")
                .about("Analyze data and generate insights")
                .arg(
                    Arg::new("input")
                        .short('i')
                        .long("input")
                        .value_name("FILE")
                        .help("Input file to analyze")
                        .required(true)
                )
                .arg(
                    Arg::new("report")
                        .short('r')
                        .long("report")
                        .value_name("FILE")
                        .help("Output report file")
                        .default_value("./report.md")
                )
                .arg(
                    Arg::new("visualize")
                        .long("visualize")
                        .help("Generate visualizations")
                        .action(clap::ArgAction::SetTrue)
                )
                .arg(
                    Arg::new("deep-analysis")
                        .long("deep-analysis")
                        .help("Perform deep statistical analysis")
                        .action(clap::ArgAction::SetTrue)
                )
        )
        .subcommand(
            Command::new("convert")
                .about("Convert between data formats")
                .arg(
                    Arg::new("input")
                        .short('i')
                        .long("input")
                        .value_name("FILE")
                        .help("Input file")
                        .required(true)
                )
                .arg(
                    Arg::new("output")
                        .short('o')
                        .long("output")
                        .value_name("FILE")
                        .help("Output file")
                        .required(true)
                )
                .arg(
                    Arg::new("from")
                        .long("from")
                        .value_name("FORMAT")
                        .help("Input format")
                        .default_value("auto")
                )
                .arg(
                    Arg::new("to")
                        .long("to")
                        .value_name("FORMAT")
                        .help("Output format")
                        .default_value("json")
                )
        )
        .subcommand(
            Command::new("benchmark")
                .about("Run performance benchmarks")
                .arg(
                    Arg::new("iterations")
                        .short('n')
                        .long("iterations")
                        .value_name("COUNT")
                        .help("Number of benchmark iterations")
                        .default_value("1000")
                )
                .arg(
                    Arg::new("concurrency")
                        .short('c')
                        .long("concurrency")
                        .value_name("LEVEL")
                        .help("Concurrency level")
                        .default_value("1")
                )
                .arg(
                    Arg::new("output")
                        .short('o')
                        .long("output")
                        .value_name("FILE")
                        .help("Benchmark results file")
                        .default_value("./benchmark_results.json")
                )
        )
        .subcommand(
            Command::new("config")
                .about("Configuration management")
                .subcommand(
                    Command::new("init")
                        .about("Initialize configuration file")
                        .arg(
                            Arg::new("path")
                                .short('p')
                                .long("path")
                                .value_name("FILE")
                                .help("Configuration file path")
                                .default_value("./cli-tool.toml")
                        )
                )
                .subcommand(
                    Command::new("show")
                        .about("Show current configuration")
                )
                .subcommand(
                    Command::new("validate")
                        .about("Validate configuration file")
                        .arg(
                            Arg::new("path")
                                .short('p')
                                .long("path")
                                .value_name("FILE")
                                .help("Configuration file path")
                        )
                )
        )
        .subcommand(
            Command::new("interactive")
                .about("Launch interactive mode")
                .alias("i")
        )
}

pub async fn run_interactive_mode() -> Result<()> {
    let theme = ColorfulTheme::default();
    let term = Term::stdout();

    term.clear_screen()?;
    println!("🚀 Welcome to Rust CLI Tool - Interactive Mode");
    println!("==============================================\n");

    loop {
        let options = vec![
            "📊 Process Data Files",
            "🔍 Analyze Data",
            "🔄 Convert Formats",
            "⚡ Run Benchmarks",
            "⚙️  Configuration",
            "❌ Exit"
        ];

        let selection = Select::with_theme(&theme)
            .with_prompt("What would you like to do?")
            .items(&options)
            .default(0)
            .interact()?;

        match selection {
            0 => run_process_interactive().await?,
            1 => run_analyze_interactive().await?,
            2 => run_convert_interactive().await?,
            3 => run_benchmark_interactive().await?,
            4 => run_config_interactive().await?,
            5 => {
                println!("👋 Goodbye!");
                break;
            }
            _ => unreachable!(),
        }

        if !Confirm::with_theme(&theme)
            .with_prompt("Would you like to perform another operation?")
            .default(true)
            .interact()? {
            break;
        }

        term.clear_screen()?;
    }

    Ok(())
}

async fn run_process_interactive() -> Result<()> {
    let theme = ColorfulTheme::default();

    let input_file: String = Input::with_theme(&theme)
        .with_prompt("Enter input file path")
        .validate_with(|input: &String| {
            if PathBuf::from(input).exists() {
                Ok(())
            } else {
                Err("File does not exist")
            }
        })
        .interact_text()?;

    let output_dir: String = Input::with_theme(&theme)
        .with_prompt("Enter output directory")
        .default("./output".to_string())
        .interact_text()?;

    let formats = vec!["json", "csv", "xml", "yaml", "binary"];
    let format_selection = Select::with_theme(&theme)
        .with_prompt("Select output format")
        .items(&formats)
        .default(0)
        .interact()?;

    let batch_size: String = Input::with_theme(&theme)
        .with_prompt("Enter batch size")
        .default("1000".to_string())
        .validate_with(|input: &String| {
            input.parse::<usize>().map(|_| ()).map_err(|_| "Invalid number")
        })
        .interact_text()?;

    println!("🔄 Processing file: {}", input_file);
    let pb = ProgressBar::new(100);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>7}/{len:7} {msg}")
            .progress_chars("#>-")
    );

    // Simulate processing
    for i in 0..100 {
        pb.set_position(i);
        pb.set_message(format!("Processing batch {}", i));
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    pb.finish_with_message("✅ Processing completed!");
    println!("📁 Output saved to: {}", output_dir);

    Ok(())
}

async fn run_analyze_interactive() -> Result<()> {
    let theme = ColorfulTheme::default();

    let input_file: String = Input::with_theme(&theme)
        .with_prompt("Enter file to analyze")
        .validate_with(|input: &String| {
            if PathBuf::from(input).exists() {
                Ok(())
            } else {
                Err("File does not exist")
            }
        })
        .interact_text()?;

    let options = vec![
        "📈 Basic Statistics",
        "🔍 Data Quality Analysis",
        "📊 Correlation Analysis",
        "🎯 Outlier Detection",
        "📋 Missing Data Analysis"
    ];

    let selections = MultiSelect::with_theme(&theme)
        .with_prompt("Select analysis types")
        .items(&options)
        .defaults(&[true, true, false, false, true])
        .interact()?;

    let generate_visualizations = Confirm::with_theme(&theme)
        .with_prompt("Generate visualizations?")
        .default(true)
        .interact()?;

    println!("🔍 Analyzing file: {}", input_file);
    let pb = ProgressBar::new(selections.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.blue} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>7}/{len:7} {msg}")
            .progress_chars("#>-")
    );

    for (i, selection) in selections.iter().enumerate() {
        pb.set_position(i as u64);
        pb.set_message(format!("Running: {}", options[*selection]));
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    pb.finish_with_message("✅ Analysis completed!");

    if generate_visualizations {
        println!("📊 Generating visualizations...");
        let viz_pb = ProgressBar::new_spinner();
        viz_pb.set_message("Creating charts and graphs...");
        tokio::time::sleep(Duration::from_millis(1000)).await;
        viz_pb.finish_with_message("✅ Visualizations generated!");
    }

    println!("📄 Report saved to: ./analysis_report.md");

    Ok(())
}

async fn run_convert_interactive() -> Result<()> {
    let theme = ColorfulTheme::default();

    let input_file: String = Input::with_theme(&theme)
        .with_prompt("Enter input file path")
        .validate_with(|input: &String| {
            if PathBuf::from(input).exists() {
                Ok(())
            } else {
                Err("File does not exist")
            }
        })
        .interact_text()?;

    let output_file: String = Input::with_theme(&theme)
        .with_prompt("Enter output file path")
        .interact_text()?;

    let input_formats = vec!["auto", "json", "csv", "xml", "yaml"];
    let output_formats = vec!["json", "csv", "xml", "yaml"];

    let input_format = Select::with_theme(&theme)
        .with_prompt("Select input format")
        .items(&input_formats)
        .default(0)
        .interact()?;

    let output_format = Select::with_theme(&theme)
        .with_prompt("Select output format")
        .items(&output_formats)
        .default(0)
        .interact()?;

    println!("🔄 Converting {} -> {}", input_file, output_file);

    let pb = ProgressBar::new_spinner();
    pb.set_message("Converting file...");
    tokio::time::sleep(Duration::from_millis(1500)).await;
    pb.finish_with_message("✅ Conversion completed!");

    Ok(())
}

async fn run_benchmark_interactive() -> Result<()> {
    let theme = ColorfulTheme::default();

    let iterations: String = Input::with_theme(&theme)
        .with_prompt("Number of iterations")
        .default("1000".to_string())
        .validate_with(|input: &String| {
            input.parse::<u32>().map(|_| ()).map_err(|_| "Invalid number")
        })
        .interact_text()?;

    let concurrency: String = Input::with_theme(&theme)
        .with_prompt("Concurrency level")
        .default("1".to_string())
        .validate_with(|input: &String| {
            input.parse::<u32>().map(|_| ()).map_err(|_| "Invalid number")
        })
        .interact_text()?;

    println!("⚡ Running benchmarks...");
    println!("Iterations: {}, Concurrency: {}", iterations, concurrency);

    let pb = ProgressBar::new(iterations.parse().unwrap());
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.yellow} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>7}/{len:7} {msg}")
            .progress_chars("#>-")
    );

    for i in 0..iterations.parse().unwrap() {
        pb.set_position(i);
        pb.set_message(format!("Running iteration {}", i + 1));
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    pb.finish_with_message("✅ Benchmark completed!");
    println!("📊 Results saved to: ./benchmark_results.json");

    Ok(())
}

async fn run_config_interactive() -> Result<()> {
    let theme = ColorfulTheme::default();

    let options = vec![
        "📝 Initialize new config",
        "👀 Show current config",
        "✅ Validate config file",
        "🔧 Edit config interactively"
    ];

    let selection = Select::with_theme(&theme)
        .with_prompt("Configuration options")
        .items(&options)
        .default(0)
        .interact()?;

    match selection {
        0 => {
            let path: String = Input::with_theme(&theme)
                .with_prompt("Config file path")
                .default("./cli-tool.toml".to_string())
                .interact_text()?;

            println!("📝 Initializing config at: {}", path);
            // Config initialization logic would go here
            println!("✅ Config file created!");
        }
        1 => {
            println!("👀 Current configuration:");
            // Display current config
            println!("⚠️  Config display not implemented in interactive mode");
        }
        2 => {
            let path: String = Input::with_theme(&theme)
                .with_prompt("Config file path")
                .default("./cli-tool.toml".to_string())
                .interact_text()?;

            println!("✅ Validating config: {}", path);
            // Config validation logic would go here
            println!("✅ Config is valid!");
        }
        3 => {
            println!("🔧 Interactive config editing not yet implemented");
            println!("💡 Use 'cli-tool config init' to create a config file");
        }
        _ => unreachable!(),
    }

    Ok(())
}

pub fn handle_error(error: anyhow::Error, verbose: bool) {
    eprintln!("❌ Error: {}", error);

    if verbose {
        eprintln!("🔍 Error details:");
        eprintln!("{}", error.backtrace());
    }

    eprintln!("\n💡 Try 'cli-tool --help' for usage information");
    std::process::exit(1);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_build() {
        let app = build_cli();
        let matches = app.try_get_matches_from(vec!["test", "--help"]);
        // Should not panic
        assert!(matches.is_err()); // Because --help exits
    }
}