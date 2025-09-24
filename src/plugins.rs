//! Plugin system module
//!
//! This module provides a comprehensive plugin architecture that allows
//! extending the CLI tool functionality through dynamically loaded plugins.

use crate::error::{CliError, CliResult};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use libloading::{Library, Symbol};
use tracing::{info, warn, error, debug};

/// Plugin manager for loading and managing plugins
pub struct PluginManager {
    plugins: Arc<RwLock<HashMap<String, Box<dyn Plugin>>>>,
    plugin_paths: Vec<PathBuf>,
    loaded_libraries: Vec<Library>,
}

impl PluginManager {
    /// Create a new plugin manager
    pub fn new() -> Self {
        Self {
            plugins: Arc::new(RwLock::new(HashMap::new())),
            plugin_paths: Vec::new(),
            loaded_libraries: Vec::new(),
        }
    }

    /// Add a plugin search path
    pub fn add_plugin_path<P: AsRef<Path>>(&mut self, path: P) {
        self.plugin_paths.push(path.as_ref().to_path_buf());
    }

    /// Load all plugins from configured paths
    pub async fn load_plugins(&mut self) -> CliResult<()> {
        info!("Loading plugins from {} paths", self.plugin_paths.len());

        for path in &self.plugin_paths {
            if path.exists() {
                self.load_plugins_from_directory(path).await?;
            } else {
                warn!("Plugin path does not exist: {:?}", path);
            }
        }

        info!("Loaded {} plugins total", self.plugins.read().await.len());
        Ok(())
    }

    /// Load plugins from a specific directory
    async fn load_plugins_from_directory(&mut self, dir_path: &Path) -> CliResult<()> {
        debug!("Scanning directory for plugins: {:?}", dir_path);

        let entries = std::fs::read_dir(dir_path)?;
        for entry in entries {
            let entry = entry?;
            let path = entry.path();

            if self.is_plugin_file(&path) {
                match self.load_plugin_from_file(&path).await {
                    Ok(_) => info!("Loaded plugin: {:?}", path.file_name().unwrap()),
                    Err(e) => error!("Failed to load plugin {:?}: {}", path, e),
                }
            }
        }

        Ok(())
    }

    /// Check if a file is a plugin (based on extension and naming)
    fn is_plugin_file(&self, path: &Path) -> bool {
        if let Some(ext) = path.extension() {
            if ext == "so" || ext == "dll" || ext == "dylib" {
                return true;
            }
        }

        // Also check for .plugin files (our custom format)
        if let Some(ext) = path.extension() {
            if ext == "plugin" {
                return true;
            }
        }

        false
    }

    /// Load a plugin from a file
    async fn load_plugin_from_file(&mut self, path: &Path) -> CliResult<()> {
        debug!("Loading plugin from file: {:?}", path);

        // For now, we'll simulate plugin loading
        // In a real implementation, this would use libloading to load dynamic libraries

        let plugin_name = path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        // Create a mock plugin for demonstration
        let plugin = MockPlugin::new(&plugin_name);
        self.plugins.write().await.insert(plugin_name, Box::new(plugin));

        Ok(())
    }

    /// Get a plugin by name
    pub async fn get_plugin(&self, name: &str) -> Option<Box<dyn Plugin>> {
        self.plugins.read().await.get(name).cloned()
    }

    /// List all loaded plugins
    pub async fn list_plugins(&self) -> Vec<String> {
        self.plugins.read().await.keys().cloned().collect()
    }

    /// Execute a plugin command
    pub async fn execute_plugin(&self, name: &str, command: &str, args: Vec<String>) -> CliResult<String> {
        if let Some(plugin) = self.get_plugin(name).await {
            plugin.execute(command, args).await
        } else {
            Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Plugin '{}' not found", name)
            )))
        }
    }

    /// Unload all plugins
    pub async fn unload_plugins(&mut self) {
        info!("Unloading {} plugins", self.plugins.read().await.len());
        self.plugins.write().await.clear();
        self.loaded_libraries.clear();
    }
}

/// Plugin trait that all plugins must implement
#[async_trait::async_trait]
pub trait Plugin: Send + Sync {
    /// Get plugin metadata
    fn metadata(&self) -> PluginMetadata;

    /// Execute a plugin command
    async fn execute(&self, command: &str, args: Vec<String>) -> CliResult<String>;

    /// Initialize the plugin
    async fn initialize(&mut self) -> CliResult<()> {
        Ok(())
    }

    /// Shutdown the plugin
    async fn shutdown(&mut self) -> CliResult<()> {
        Ok(())
    }
}

/// Plugin metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginMetadata {
    pub name: String,
    pub version: String,
    pub description: String,
    pub author: String,
    pub commands: Vec<String>,
    pub dependencies: Vec<String>,
}

/// Mock plugin for demonstration
pub struct MockPlugin {
    metadata: PluginMetadata,
}

impl MockPlugin {
    pub fn new(name: &str) -> Self {
        Self {
            metadata: PluginMetadata {
                name: name.to_string(),
                version: "1.0.0".to_string(),
                description: format!("Mock plugin for {}", name),
                author: "CLI Tool".to_string(),
                commands: vec!["hello".to_string(), "echo".to_string()],
                dependencies: vec![],
            },
        }
    }
}

#[async_trait::async_trait]
impl Plugin for MockPlugin {
    fn metadata(&self) -> PluginMetadata {
        self.metadata.clone()
    }

    async fn execute(&self, command: &str, args: Vec<String>) -> CliResult<String> {
        match command {
            "hello" => Ok(format!("Hello from {} plugin!", self.metadata.name)),
            "echo" => Ok(args.join(" ")),
            _ => Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Unknown command: {}", command)
            ))),
        }
    }
}

impl Clone for Box<dyn Plugin> {
    fn clone(&self) -> Box<dyn Plugin> {
        // This is a simplified clone implementation
        // In practice, you'd need to implement proper cloning for each plugin type
        Box::new(MockPlugin::new("cloned"))
    }
}

/// Plugin registry for discovering and managing plugin repositories
pub struct PluginRegistry {
    registry_url: String,
    client: crate::networking::HttpClient,
}

impl PluginRegistry {
    /// Create a new plugin registry
    pub fn new(registry_url: String, config: crate::networking::HttpConfig) -> CliResult<Self> {
        let client = crate::networking::HttpClient::new(config)?;
        Ok(Self { registry_url, client })
    }

    /// Search for plugins in the registry
    pub async fn search_plugins(&self, query: &str) -> CliResult<Vec<PluginInfo>> {
        let url = format!("{}/search?q={}", self.registry_url, query);
        let response = self.client.get(&url, None).await?;
        let plugins: Vec<PluginInfo> = response.json()?;
        Ok(plugins)
    }

    /// Get plugin information
    pub async fn get_plugin_info(&self, name: &str) -> CliResult<PluginInfo> {
        let url = format!("{}/plugins/{}", self.registry_url, name);
        let response = self.client.get(&url, None).await?;
        let plugin: PluginInfo = response.json()?;
        Ok(plugin)
    }

    /// Download and install a plugin
    pub async fn install_plugin(&self, name: &str, install_path: &Path) -> CliResult<()> {
        let plugin_info = self.get_plugin_info(name).await?;
        let download_url = format!("{}/download/{}", self.registry_url, name);

        info!("Downloading plugin {} from {}", name, download_url);
        self.client.download_file(&download_url, install_path).await?;

        info!("Plugin {} installed successfully", name);
        Ok(())
    }

    /// Update a plugin to the latest version
    pub async fn update_plugin(&self, name: &str) -> CliResult<()> {
        info!("Updating plugin: {}", name);
        // Implementation would check for updates and install new version
        Ok(())
    }
}

/// Plugin information from registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginInfo {
    pub name: String,
    pub version: String,
    pub description: String,
    pub author: String,
    pub download_url: String,
    pub size: u64,
    pub checksum: String,
    pub dependencies: Vec<String>,
    pub tags: Vec<String>,
}

/// Plugin sandbox for safe plugin execution
pub struct PluginSandbox {
    memory_limit: usize,
    time_limit: std::time::Duration,
    allowed_syscalls: Vec<String>,
}

impl PluginSandbox {
    /// Create a new plugin sandbox
    pub fn new(memory_limit: usize, time_limit: std::time::Duration) -> Self {
        Self {
            memory_limit,
            time_limit,
            allowed_syscalls: vec![
                "read".to_string(),
                "write".to_string(),
                "open".to_string(),
                "close".to_string(),
            ],
        }
    }

    /// Execute a plugin in the sandbox
    pub async fn execute_sandboxed<F, Fut, T>(&self, plugin_fn: F) -> CliResult<T>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = CliResult<T>>,
    {
        // In a real implementation, this would use seccomp, namespaces, etc.
        // For now, we just execute the function with basic timeout

        match tokio::time::timeout(self.time_limit, plugin_fn()).await {
            Ok(result) => result,
            Err(_) => Err(CliError::Timeout("Plugin execution timed out".to_string())),
        }
    }
}

/// Plugin development kit for creating new plugins
pub struct PluginDevKit {
    template_path: PathBuf,
}

impl PluginDevKit {
    /// Create a new plugin development kit
    pub fn new(template_path: PathBuf) -> Self {
        Self { template_path }
    }

    /// Create a new plugin from template
    pub fn create_plugin(&self, name: &str, plugin_type: PluginType, output_path: &Path) -> CliResult<()> {
        info!("Creating new {} plugin: {}", plugin_type.as_str(), name);

        let template_dir = self.template_path.join(plugin_type.as_str());

        if !template_dir.exists() {
            return Err(CliError::Processing(crate::error::ProcessingError::InputError(
                format!("Template not found: {:?}", template_dir)
            )));
        }

        // Copy template to output path
        let file_ops = crate::file_ops::FileOps::new();
        file_ops.copy_directory(&template_dir, output_path)?;

        // Replace placeholders in template files
        self.replace_placeholders(output_path, name)?;

        info!("Plugin {} created successfully at {:?}", name, output_path);
        Ok(())
    }

    /// Replace placeholders in template files
    fn replace_placeholders(&self, plugin_path: &Path, name: &str) -> CliResult<()> {
        use walkdir::WalkDir;

        for entry in WalkDir::new(plugin_path) {
            let entry = entry?;
            if entry.file_type().is_file() {
                let path = entry.path();
                if let Some(ext) = path.extension() {
                    if ext == "rs" || ext == "toml" || ext == "md" {
                        self.replace_in_file(path, "{{PLUGIN_NAME}}", name)?;
                        self.replace_in_file(path, "{{PLUGIN_NAME_SNAKE}}", &name.to_lowercase().replace("-", "_"))?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Replace text in a file
    fn replace_in_file(&self, path: &Path, from: &str, to: &str) -> CliResult<()> {
        let content = std::fs::read_to_string(path)?;
        let new_content = content.replace(from, to);
        std::fs::write(path, new_content)?;
        Ok(())
    }

    /// Build a plugin
    pub fn build_plugin(&self, plugin_path: &Path) -> CliResult<()> {
        info!("Building plugin at {:?}", plugin_path);

        // Check if Cargo.toml exists
        let cargo_toml = plugin_path.join("Cargo.toml");
        if !cargo_toml.exists() {
            return Err(CliError::Processing(crate::error::ProcessingError::InputError(
                "Cargo.toml not found in plugin directory".to_string()
            )));
        }

        // Run cargo build
        let output = std::process::Command::new("cargo")
            .args(&["build", "--release"])
            .current_dir(plugin_path)
            .output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Plugin build failed: {}", stderr)
            )));
        }

        info!("Plugin built successfully");
        Ok(())
    }
}

/// Plugin types
#[derive(Debug, Clone, Copy)]
pub enum PluginType {
    Processor,
    Exporter,
    Importer,
    Transformer,
    Analyzer,
}

impl PluginType {
    pub fn as_str(&self) -> &'static str {
        match self {
            PluginType::Processor => "processor",
            PluginType::Exporter => "exporter",
            PluginType::Importer => "importer",
            PluginType::Transformer => "transformer",
            PluginType::Analyzer => "analyzer",
        }
    }
}

/// Plugin configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginConfig {
    pub enabled: bool,
    pub priority: i32,
    pub settings: HashMap<String, serde_json::Value>,
}

impl Default for PluginManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_plugin_manager() {
        let mut manager = PluginManager::new();
        let temp_dir = tempdir().unwrap();

        manager.add_plugin_path(temp_dir.path());

        // Should load without errors even with empty directory
        assert!(manager.load_plugins().await.is_ok());

        // Should have no plugins loaded
        assert_eq!(manager.list_plugins().await.len(), 0);
    }

    #[tokio::test]
    async fn test_mock_plugin() {
        let plugin = MockPlugin::new("test");

        let metadata = plugin.metadata();
        assert_eq!(metadata.name, "test");
        assert_eq!(metadata.version, "1.0.0");

        let result = plugin.execute("hello", vec![]).await.unwrap();
        assert_eq!(result, "Hello from test plugin!");

        let result = plugin.execute("echo", vec!["world".to_string()]).await.unwrap();
        assert_eq!(result, "world");
    }

    #[test]
    fn test_plugin_types() {
        assert_eq!(PluginType::Processor.as_str(), "processor");
        assert_eq!(PluginType::Exporter.as_str(), "exporter");
        assert_eq!(PluginType::Importer.as_str(), "importer");
        assert_eq!(PluginType::Transformer.as_str(), "transformer");
        assert_eq!(PluginType::Analyzer.as_str(), "analyzer");
    }

    #[test]
    fn test_plugin_config() {
        let config = PluginConfig {
            enabled: true,
            priority: 10,
            settings: HashMap::new(),
        };

        assert!(config.enabled);
        assert_eq!(config.priority, 10);
    }
}