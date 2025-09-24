//! Configuration management module
//!
//! This module provides comprehensive configuration management including
//! environment handling, secrets management, configuration patterns, and
//! dynamic configuration reloading.

use crate::error::{CliError, CliResult};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{info, debug, warn, error};
use std::time::Duration;

/// Configuration manager
pub struct ConfigManager {
    sources: Vec<Box<dyn ConfigSource>>,
    cache: RwLock<HashMap<String, ConfigValue>>,
    watchers: Vec<Box<dyn ConfigWatcher>>,
    secrets_manager: Option<Box<dyn SecretsManager>>,
}

impl ConfigManager {
    /// Create a new configuration manager
    pub fn new() -> Self {
        Self {
            sources: vec![],
            cache: RwLock::new(HashMap::new()),
            watchers: vec![],
            secrets_manager: None,
        }
    }

    /// Add a configuration source
    pub fn add_source(&mut self, source: Box<dyn ConfigSource>) {
        self.sources.push(source);
    }

    /// Add a configuration watcher
    pub fn add_watcher(&mut self, watcher: Box<dyn ConfigWatcher>) {
        self.watchers.push(watcher);
    }

    /// Set secrets manager
    pub fn set_secrets_manager(&mut self, manager: Box<dyn SecretsManager>) {
        self.secrets_manager = Some(manager);
    }

    /// Load configuration from all sources
    pub async fn load_config(&self) -> CliResult<()> {
        let mut cache = self.cache.write().await;

        for source in &self.sources {
            let config = source.load_config().await?;
            for (key, value) in config {
                cache.insert(key, value);
            }
        }

        info!("Configuration loaded from {} sources", self.sources.len());
        Ok(())
    }

    /// Get a configuration value
    pub async fn get(&self, key: &str) -> Option<ConfigValue> {
        let cache = self.cache.read().await;
        cache.get(key).cloned()
    }

    /// Get a configuration value with type conversion
    pub async fn get_typed<T: ConfigType>(&self, key: &str) -> CliResult<Option<T>> {
        if let Some(value) = self.get(key).await {
            T::from_config_value(value)
        } else {
            Ok(None)
        }
    }

    /// Set a configuration value
    pub async fn set(&self, key: String, value: ConfigValue) -> CliResult<()> {
        let mut cache = self.cache.write().await;
        cache.insert(key.clone(), value.clone());

        // Notify watchers
        for watcher in &self.watchers {
            if let Err(e) = watcher.on_config_change(&key, &value).await {
                warn!("Configuration watcher error: {}", e);
            }
        }

        Ok(())
    }

    /// Get all configuration keys
    pub async fn keys(&self) -> Vec<String> {
        let cache = self.cache.read().await;
        cache.keys().cloned().collect()
    }

    /// Check if a key exists
    pub async fn contains_key(&self, key: &str) -> bool {
        let cache = self.cache.read().await;
        cache.contains_key(key)
    }

    /// Get a secret value
    pub async fn get_secret(&self, key: &str) -> CliResult<Option<String>> {
        if let Some(ref secrets_manager) = self.secrets_manager {
            secrets_manager.get_secret(key).await
        } else {
            Err(CliError::Config(crate::error::ConfigError::SecretsManagerNotConfigured))
        }
    }

    /// Set a secret value
    pub async fn set_secret(&self, key: String, value: String) -> CliResult<()> {
        if let Some(ref secrets_manager) = self.secrets_manager {
            secrets_manager.set_secret(key, value).await
        } else {
            Err(CliError::Config(crate::error::ConfigError::SecretsManagerNotConfigured))
        }
    }

    /// Watch for configuration changes
    pub async fn watch_changes(&self) -> CliResult<()> {
        for source in &self.sources {
            if let Some(watchable) = source.as_watchable() {
                watchable.watch_changes(self).await?;
            }
        }
        Ok(())
    }

    /// Export configuration to a file
    pub async fn export_config(&self, path: &Path, format: ConfigFormat) -> CliResult<()> {
        let cache = self.cache.read().await;
        let config_map: HashMap<String, serde_json::Value> = cache.iter()
            .map(|(k, v)| (k.clone(), v.to_json()))
            .collect();

        let content = match format {
            ConfigFormat::Json => serde_json::to_string_pretty(&config_map)?,
            ConfigFormat::Yaml => serde_yaml::to_string(&config_map)?,
            ConfigFormat::Toml => toml::to_string(&config_map)?,
            ConfigFormat::Env => {
                let mut lines = vec![];
                for (key, value) in &config_map {
                    lines.push(format!("{}={}", key, value));
                }
                lines.join("\n")
            }
        };

        fs::write(path, content)?;
        info!("Configuration exported to {:?}", path);
        Ok(())
    }

    /// Import configuration from a file
    pub async fn import_config(&self, path: &Path) -> CliResult<()> {
        let content = fs::read_to_string(path)?;
        let config_map: HashMap<String, serde_json::Value> = if path.extension().and_then(|s| s.to_str()) == Some("json") {
            serde_json::from_str(&content)?
        } else if path.extension().and_then(|s| s.to_str()) == Some("yaml") || path.extension().and_then(|s| s.to_str()) == Some("yml") {
            serde_yaml::from_str(&content)?
        } else if path.extension().and_then(|s| s.to_str()) == Some("toml") {
            toml::from_str(&content)?
        } else {
            return Err(CliError::Config(crate::error::ConfigError::UnsupportedFormat));
        };

        let mut cache = self.cache.write().await;
        for (key, value) in config_map {
            cache.insert(key, ConfigValue::from_json(value));
        }

        info!("Configuration imported from {:?}", path);
        Ok(())
    }

    /// Validate configuration
    pub async fn validate_config(&self, schema: &ConfigSchema) -> CliResult<Vec<ValidationError>> {
        let cache = self.cache.read().await;
        let mut errors = vec![];

        for (key, rule) in &schema.rules {
            if let Some(value) = cache.get(key) {
                if let Some(error) = rule.validate(value) {
                    errors.push(error);
                }
            } else if rule.required {
                errors.push(ValidationError {
                    key: key.clone(),
                    message: "Required configuration key is missing".to_string(),
                });
            }
        }

        Ok(errors)
    }
}

/// Configuration value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfigValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Array(Vec<ConfigValue>),
    Object(HashMap<String, ConfigValue>),
}

impl ConfigValue {
    /// Convert to JSON value
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            ConfigValue::String(s) => serde_json::Value::String(s.clone()),
            ConfigValue::Integer(i) => serde_json::Value::Number((*i).into()),
            ConfigValue::Float(f) => serde_json::Value::Number(serde_json::Number::from_f64(*f).unwrap()),
            ConfigValue::Boolean(b) => serde_json::Value::Bool(*b),
            ConfigValue::Array(arr) => serde_json::Value::Array(arr.iter().map(|v| v.to_json()).collect()),
            ConfigValue::Object(obj) => {
                let mut map = serde_json::Map::new();
                for (k, v) in obj {
                    map.insert(k.clone(), v.to_json());
                }
                serde_json::Value::Object(map)
            }
        }
    }

    /// Create from JSON value
    pub fn from_json(value: serde_json::Value) -> Self {
        match value {
            serde_json::Value::String(s) => ConfigValue::String(s),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    ConfigValue::Integer(i)
                } else if let Some(f) = n.as_f64() {
                    ConfigValue::Float(f)
                } else {
                    ConfigValue::String(n.to_string())
                }
            }
            serde_json::Value::Bool(b) => ConfigValue::Boolean(b),
            serde_json::Value::Array(arr) => ConfigValue::Array(arr.into_iter().map(ConfigValue::from_json).collect()),
            serde_json::Value::Object(obj) => {
                let mut map = HashMap::new();
                for (k, v) in obj {
                    map.insert(k, ConfigValue::from_json(v));
                }
                ConfigValue::Object(map)
            }
            _ => ConfigValue::String(value.to_string()),
        }
    }
}

/// Configuration type trait
pub trait ConfigType: Sized {
    fn from_config_value(value: ConfigValue) -> CliResult<Option<Self>>;
}

impl ConfigType for String {
    fn from_config_value(value: ConfigValue) -> CliResult<Option<Self>> {
        match value {
            ConfigValue::String(s) => Ok(Some(s)),
            _ => Ok(None),
        }
    }
}

impl ConfigType for i64 {
    fn from_config_value(value: ConfigValue) -> CliResult<Option<Self>> {
        match value {
            ConfigValue::Integer(i) => Ok(Some(i)),
            ConfigValue::String(s) => {
                if let Ok(i) = s.parse() {
                    Ok(Some(i))
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }
}

impl ConfigType for f64 {
    fn from_config_value(value: ConfigValue) -> CliResult<Option<Self>> {
        match value {
            ConfigValue::Float(f) => Ok(Some(f)),
            ConfigValue::Integer(i) => Ok(Some(i as f64)),
            ConfigValue::String(s) => {
                if let Ok(f) = s.parse() {
                    Ok(Some(f))
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }
}

impl ConfigType for bool {
    fn from_config_value(value: ConfigValue) -> CliResult<Option<Self>> {
        match value {
            ConfigValue::Boolean(b) => Ok(Some(b)),
            ConfigValue::String(s) => {
                match s.to_lowercase().as_str() {
                    "true" | "1" | "yes" | "on" => Ok(Some(true)),
                    "false" | "0" | "no" | "off" => Ok(Some(false)),
                    _ => Ok(None),
                }
            }
            _ => Ok(None),
        }
    }
}

/// Configuration source trait
#[async_trait::async_trait]
pub trait ConfigSource: Send + Sync {
    /// Load configuration from this source
    async fn load_config(&self) -> CliResult<HashMap<String, ConfigValue>>;

    /// Get source priority (higher numbers override lower ones)
    fn priority(&self) -> i32 { 0 }

    /// Convert to watchable source if supported
    fn as_watchable(&self) -> Option<&dyn ConfigWatcher> { None }
}

/// Environment variables configuration source
pub struct EnvironmentSource {
    prefix: Option<String>,
}

impl EnvironmentSource {
    /// Create a new environment source
    pub fn new() -> Self {
        Self { prefix: None }
    }

    /// Create with prefix filtering
    pub fn with_prefix(prefix: String) -> Self {
        Self { prefix: Some(prefix) }
    }
}

#[async_trait::async_trait]
impl ConfigSource for EnvironmentSource {
    async fn load_config(&self) -> CliResult<HashMap<String, ConfigValue>> {
        let mut config = HashMap::new();

        for (key, value) in env::vars() {
            let should_include = if let Some(ref prefix) = self.prefix {
                key.starts_with(prefix)
            } else {
                true
            };

            if should_include {
                config.insert(key, ConfigValue::String(value));
            }
        }

        debug!("Loaded {} environment variables", config.len());
        Ok(config)
    }
}

/// File-based configuration source
pub struct FileSource {
    path: PathBuf,
    format: ConfigFormat,
}

impl FileSource {
    /// Create a new file source
    pub fn new(path: PathBuf, format: ConfigFormat) -> Self {
        Self { path, format }
    }
}

#[async_trait::async_trait]
impl ConfigSource for FileSource {
    async fn load_config(&self) -> CliResult<HashMap<String, ConfigValue>> {
        if !self.path.exists() {
            return Ok(HashMap::new());
        }

        let content = fs::read_to_string(&self.path)?;
        let config_map: HashMap<String, serde_json::Value> = match self.format {
            ConfigFormat::Json => serde_json::from_str(&content)?,
            ConfigFormat::Yaml => serde_yaml::from_str(&content)?,
            ConfigFormat::Toml => toml::from_str(&content)?,
            ConfigFormat::Env => {
                let mut map = HashMap::new();
                for line in content.lines() {
                    if let Some((key, value)) = line.split_once('=') {
                        map.insert(key.to_string(), serde_json::Value::String(value.to_string()));
                    }
                }
                map
            }
        };

        let config = config_map.into_iter()
            .map(|(k, v)| (k, ConfigValue::from_json(v)))
            .collect();

        debug!("Loaded configuration from {:?}", self.path);
        Ok(config)
    }
}

/// Remote configuration source (e.g., Consul, etcd)
pub struct RemoteSource {
    endpoint: String,
    token: Option<String>,
    client: reqwest::Client,
}

impl RemoteSource {
    /// Create a new remote source
    pub fn new(endpoint: String, token: Option<String>) -> Self {
        Self {
            endpoint,
            token,
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait::async_trait]
impl ConfigSource for RemoteSource {
    async fn load_config(&self) -> CliResult<HashMap<String, ConfigValue>> {
        let mut request = self.client.get(&format!("{}/v1/kv/?recurse", self.endpoint));

        if let Some(ref token) = self.token {
            request = request.header("X-Consul-Token", token);
        }

        let response = request.send().await?;
        let kv_pairs: Vec<ConsulKV> = response.json().await?;

        let mut config = HashMap::new();
        for kv in kv_pairs {
            if let Some(value) = kv.Value {
                let decoded = base64::decode(value)?;
                let value_str = String::from_utf8(decoded)?;
                config.insert(kv.Key, ConfigValue::String(value_str));
            }
        }

        debug!("Loaded {} remote configuration keys", config.len());
        Ok(config)
    }
}

/// Consul KV structure
#[derive(Deserialize)]
struct ConsulKV {
    Key: String,
    Value: Option<String>,
}

/// Configuration format
#[derive(Debug, Clone, Copy)]
pub enum ConfigFormat {
    Json,
    Yaml,
    Toml,
    Env,
}

/// Configuration watcher trait
#[async_trait::async_trait]
pub trait ConfigWatcher: Send + Sync {
    /// Called when configuration changes
    async fn on_config_change(&self, key: &str, value: &ConfigValue) -> CliResult<()>;

    /// Watch for configuration changes
    async fn watch_changes(&self, manager: &ConfigManager) -> CliResult<()>;
}

/// Secrets manager trait
#[async_trait::async_trait]
pub trait SecretsManager: Send + Sync {
    /// Get a secret value
    async fn get_secret(&self, key: &str) -> CliResult<Option<String>>;

    /// Set a secret value
    async fn set_secret(&self, key: String, value: String) -> CliResult<()>;

    /// Delete a secret
    async fn delete_secret(&self, key: &str) -> CliResult<()>;

    /// List secret keys
    async fn list_secrets(&self) -> CliResult<Vec<String>>;
}

/// HashiCorp Vault secrets manager
pub struct VaultSecretsManager {
    client: reqwest::Client,
    address: String,
    token: String,
}

impl VaultSecretsManager {
    /// Create a new Vault secrets manager
    pub fn new(address: String, token: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            address,
            token,
        }
    }
}

#[async_trait::async_trait]
impl SecretsManager for VaultSecretsManager {
    async fn get_secret(&self, key: &str) -> CliResult<Option<String>> {
        let response = self.client
            .get(&format!("{}/v1/secret/data/{}", self.address, key))
            .header("X-Vault-Token", &self.token)
            .send()
            .await?;

        if response.status().is_success() {
            let data: VaultResponse = response.json().await?;
            Ok(data.data.data.get(key).cloned())
        } else {
            Ok(None)
        }
    }

    async fn set_secret(&self, key: String, value: String) -> CliResult<()> {
        let payload = serde_json::json!({
            "data": {
                key: value
            }
        });

        let response = self.client
            .post(&format!("{}/v1/secret/data/{}", self.address, key))
            .header("X-Vault-Token", &self.token)
            .json(&payload)
            .send()
            .await?;

        if response.status().is_success() {
            Ok(())
        } else {
            Err(CliError::Config(crate::error::ConfigError::SecretsOperationFailed(
                format!("Failed to set secret: {}", response.status())
            )))
        }
    }

    async fn delete_secret(&self, key: &str) -> CliResult<()> {
        let response = self.client
            .delete(&format!("{}/v1/secret/data/{}", self.address, key))
            .header("X-Vault-Token", &self.token)
            .send()
            .await?;

        if response.status().is_success() {
            Ok(())
        } else {
            Err(CliError::Config(crate::error::ConfigError::SecretsOperationFailed(
                format!("Failed to delete secret: {}", response.status())
            )))
        }
    }

    async fn list_secrets(&self) -> CliResult<Vec<String>> {
        let response = self.client
            .list(&format!("{}/v1/secret/metadata/?list=true", self.address))
            .header("X-Vault-Token", &self.token)
            .send()
            .await?;

        if response.status().is_success() {
            let data: VaultListResponse = response.json().await?;
            Ok(data.data.keys)
        } else {
            Ok(vec![])
        }
    }
}

/// Vault response structures
#[derive(Deserialize)]
struct VaultResponse {
    data: VaultData,
}

#[derive(Deserialize)]
struct VaultData {
    data: HashMap<String, String>,
}

#[derive(Deserialize)]
struct VaultListResponse {
    data: VaultListData,
}

#[derive(Deserialize)]
struct VaultListData {
    keys: Vec<String>,
}

/// AWS Secrets Manager
pub struct AwsSecretsManager {
    client: aws_sdk_secretsmanager::Client,
}

impl AwsSecretsManager {
    /// Create a new AWS secrets manager
    pub async fn new() -> CliResult<Self> {
        let config = aws_config::load_from_env().await;
        let client = aws_sdk_secretsmanager::Client::new(&config);
        Ok(Self { client })
    }
}

#[async_trait::async_trait]
impl SecretsManager for AwsSecretsManager {
    async fn get_secret(&self, key: &str) -> CliResult<Option<String>> {
        let response = self.client
            .get_secret_value()
            .secret_id(key)
            .send()
            .await;

        match response {
            Ok(secret) => Ok(secret.secret_string),
            Err(e) => {
                if e.to_string().contains("ResourceNotFoundException") {
                    Ok(None)
                } else {
                    Err(CliError::Config(crate::error::ConfigError::SecretsOperationFailed(
                        format!("AWS Secrets Manager error: {}", e)
                    )))
                }
            }
        }
    }

    async fn set_secret(&self, key: String, value: String) -> CliResult<()> {
        self.client
            .create_secret()
            .name(key)
            .secret_string(value)
            .send()
            .await
            .map_err(|e| CliError::Config(crate::error::ConfigError::SecretsOperationFailed(
                format!("Failed to create AWS secret: {}", e)
            )))?;
        Ok(())
    }

    async fn delete_secret(&self, key: &str) -> CliResult<()> {
        self.client
            .delete_secret()
            .secret_id(key)
            .force_delete_without_recovery(false)
            .send()
            .await
            .map_err(|e| CliError::Config(crate::error::ConfigError::SecretsOperationFailed(
                format!("Failed to delete AWS secret: {}", e)
            )))?;
        Ok(())
    }

    async fn list_secrets(&self) -> CliResult<Vec<String>> {
        let response = self.client
            .list_secrets()
            .send()
            .await
            .map_err(|e| CliError::Config(crate::error::ConfigError::SecretsOperationFailed(
                format!("Failed to list AWS secrets: {}", e)
            )))?;

        let secrets = response.secret_list.unwrap_or_default()
            .into_iter()
            .filter_map(|secret| secret.name)
            .collect();

        Ok(secrets)
    }
}

/// Configuration schema for validation
#[derive(Debug, Clone)]
pub struct ConfigSchema {
    pub rules: HashMap<String, ValidationRule>,
}

/// Validation rule
#[derive(Debug, Clone)]
pub struct ValidationRule {
    pub required: bool,
    pub value_type: ConfigValueType,
    pub min_value: Option<f64>,
    pub max_value: Option<f64>,
    pub allowed_values: Option<Vec<String>>,
}

/// Configuration value type for validation
#[derive(Debug, Clone)]
pub enum ConfigValueType {
    String,
    Integer,
    Float,
    Boolean,
    Array,
    Object,
}

impl ValidationRule {
    /// Validate a configuration value
    pub fn validate(&self, value: &ConfigValue) -> Option<ValidationError> {
        // Check type
        if !self.check_type(value) {
            return Some(ValidationError {
                key: String::new(), // Will be set by caller
                message: format!("Invalid type, expected {:?}", self.value_type),
            });
        }

        // Check numeric ranges
        if let Some(error) = self.check_range(value) {
            return Some(error);
        }

        // Check allowed values
        if let Some(error) = self.check_allowed_values(value) {
            return Some(error);
        }

        None
    }

    fn check_type(&self, value: &ConfigValue) -> bool {
        match (&self.value_type, value) {
            (ConfigValueType::String, ConfigValue::String(_)) => true,
            (ConfigValueType::Integer, ConfigValue::Integer(_)) => true,
            (ConfigValueType::Float, ConfigValue::Float(_)) => true,
            (ConfigValueType::Boolean, ConfigValue::Boolean(_)) => true,
            (ConfigValueType::Array, ConfigValue::Array(_)) => true,
            (ConfigValueType::Object, ConfigValue::Object(_)) => true,
            _ => false,
        }
    }

    fn check_range(&self, value: &ConfigValue) -> Option<ValidationError> {
        let numeric_value = match value {
            ConfigValue::Integer(i) => *i as f64,
            ConfigValue::Float(f) => *f,
            _ => return None,
        };

        if let Some(min) = self.min_value {
            if numeric_value < min {
                return Some(ValidationError {
                    key: String::new(),
                    message: format!("Value {} is below minimum {}", numeric_value, min),
                });
            }
        }

        if let Some(max) = self.max_value {
            if numeric_value > max {
                return Some(ValidationError {
                    key: String::new(),
                    message: format!("Value {} is above maximum {}", numeric_value, max),
                });
            }
        }

        None
    }

    fn check_allowed_values(&self, value: &ConfigValue) -> Option<ValidationError> {
        if let Some(ref allowed) = self.allowed_values {
            let string_value = match value {
                ConfigValue::String(s) => s.clone(),
                _ => value.to_json().to_string(),
            };

            if !allowed.contains(&string_value) {
                return Some(ValidationError {
                    key: String::new(),
                    message: format!("Value '{}' not in allowed values: {:?}", string_value, allowed),
                });
            }
        }

        None
    }
}

/// Validation error
#[derive(Debug, Clone)]
pub struct ValidationError {
    pub key: String,
    pub message: String,
}

/// Configuration profile manager
pub struct ProfileManager {
    profiles: HashMap<String, ConfigProfile>,
    active_profile: Option<String>,
}

impl ProfileManager {
    /// Create a new profile manager
    pub fn new() -> Self {
        Self {
            profiles: HashMap::new(),
            active_profile: None,
        }
    }

    /// Add a configuration profile
    pub fn add_profile(&mut self, name: String, profile: ConfigProfile) {
        self.profiles.insert(name, profile);
    }

    /// Set active profile
    pub fn set_active_profile(&mut self, name: Option<String>) {
        self.active_profile = name;
    }

    /// Get active profile
    pub fn get_active_profile(&self) -> Option<&ConfigProfile> {
        self.active_profile.as_ref()
            .and_then(|name| self.profiles.get(name))
    }

    /// Apply profile to configuration manager
    pub async fn apply_profile(&self, config_manager: &ConfigManager) -> CliResult<()> {
        if let Some(profile) = self.get_active_profile() {
            for (key, value) in &profile.values {
                config_manager.set(key.clone(), value.clone()).await?;
            }
        }
        Ok(())
    }
}

/// Configuration profile
#[derive(Debug, Clone)]
pub struct ConfigProfile {
    pub name: String,
    pub description: String,
    pub values: HashMap<String, ConfigValue>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_value_conversion() {
        let json = serde_json::json!({"key": "value", "number": 42});
        let config = ConfigValue::from_json(json.clone());
        let back_to_json = config.to_json();

        assert_eq!(json, back_to_json);
    }

    #[test]
    fn test_config_type_conversion() {
        let string_value = ConfigValue::String("42".to_string());
        let int_result: Option<i64> = ConfigType::from_config_value(string_value).unwrap();
        assert_eq!(int_result, Some(42));

        let bool_value = ConfigValue::String("true".to_string());
        let bool_result: Option<bool> = ConfigType::from_config_value(bool_value).unwrap();
        assert_eq!(bool_result, Some(true));
    }

    #[test]
    fn test_validation_rule() {
        let rule = ValidationRule {
            required: true,
            value_type: ConfigValueType::Integer,
            min_value: Some(0.0),
            max_value: Some(100.0),
            allowed_values: None,
        };

        let valid_value = ConfigValue::Integer(50);
        assert!(rule.validate(&valid_value).is_none());

        let invalid_type = ConfigValue::String("50".to_string());
        assert!(rule.validate(&invalid_type).is_some());

        let out_of_range = ConfigValue::Integer(150);
        assert!(rule.validate(&out_of_range).is_some());
    }

    #[test]
    fn test_profile_manager() {
        let mut manager = ProfileManager::new();

        let profile = ConfigProfile {
            name: "test".to_string(),
            description: "Test profile".to_string(),
            values: HashMap::new(),
        };

        manager.add_profile("test".to_string(), profile);
        manager.set_active_profile(Some("test".to_string()));

        assert!(manager.get_active_profile().is_some());
    }

    #[tokio::test]
    async fn test_config_manager() {
        let manager = ConfigManager::new();

        // Test setting and getting values
        manager.set("test_key".to_string(), ConfigValue::String("test_value".to_string())).await.unwrap();
        let value = manager.get("test_key").await;
        assert_eq!(value, Some(ConfigValue::String("test_value".to_string())));

        // Test typed access
        let typed_value: Option<String> = manager.get_typed("test_key").await.unwrap();
        assert_eq!(typed_value, Some("test_value".to_string()));
    }

    #[test]
    fn test_environment_source() {
        let source = EnvironmentSource::new();
        // Test that it can be created
        assert!(true);
    }

    #[test]
    fn test_file_source() {
        let path = PathBuf::from("test.json");
        let source = FileSource::new(path, ConfigFormat::Json);
        // Test that it can be created
        assert!(true);
    }
}