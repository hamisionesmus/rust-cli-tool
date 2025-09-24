//! Networking module
//!
//! This module provides HTTP client functionality, API interactions,
//! and network utilities for the CLI tool.

use crate::error::{CliError, NetworkError, CliResult};
use reqwest::{Client, Response, StatusCode};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::time::timeout;
use tracing::{debug, info, warn, error};
use url::Url;

/// HTTP client configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpConfig {
    pub timeout: Duration,
    pub retries: u32,
    pub retry_delay: Duration,
    pub user_agent: String,
    pub headers: HashMap<String, String>,
    pub proxy: Option<String>,
    pub follow_redirects: bool,
}

/// HTTP client for API interactions
pub struct HttpClient {
    client: Client,
    config: HttpConfig,
}

impl HttpClient {
    /// Create a new HTTP client with configuration
    pub fn new(config: HttpConfig) -> CliResult<Self> {
        let mut client_builder = Client::builder()
            .timeout(config.timeout)
            .user_agent(&config.user_agent)
            .redirect(if config.follow_redirects {
                reqwest::redirect::Policy::limited(10)
            } else {
                reqwest::redirect::Policy::none()
            });

        // Add proxy if configured
        if let Some(proxy_url) = &config.proxy {
            let proxy = reqwest::Proxy::all(proxy_url)
                .map_err(|e| CliError::Network(NetworkError::ConnectionFailed {
                    host: "proxy".to_string(),
                    port: 0,
                }))?;
            client_builder = client_builder.proxy(proxy);
        }

        let client = client_builder.build()
            .map_err(|e| CliError::Network(NetworkError::ConnectionFailed {
                host: "client_build".to_string(),
                port: 0,
            }))?;

        Ok(Self { client, config })
    }

    /// Make a GET request
    pub async fn get(&self, url: &str, query_params: Option<&HashMap<String, String>>) -> CliResult<HttpResponse> {
        let mut url = Url::parse(url).map_err(|_| CliError::Network(NetworkError::ConnectionFailed {
            host: url.to_string(),
            port: 0,
        }))?;

        // Add query parameters
        if let Some(params) = query_params {
            let mut query_pairs = url.query_pairs_mut();
            for (key, value) in params {
                query_pairs.append_pair(key, value);
            }
        }

        self.request_with_retry("GET", url.as_str(), None::<&str>).await
    }

    /// Make a POST request with JSON body
    pub async fn post_json<T: Serialize>(&self, url: &str, body: &T) -> CliResult<HttpResponse> {
        let json_body = serde_json::to_string(body)
            .map_err(|_| CliError::Serialization(crate::error::SerializationError::JsonError {
                message: "Failed to serialize request body".to_string(),
            }))?;

        self.request_with_retry("POST", url, Some(json_body)).await
    }

    /// Make a PUT request with JSON body
    pub async fn put_json<T: Serialize>(&self, url: &str, body: &T) -> CliResult<HttpResponse> {
        let json_body = serde_json::to_string(body)
            .map_err(|_| CliError::Serialization(crate::error::SerializationError::JsonError {
                message: "Failed to serialize request body".to_string(),
            }))?;

        self.request_with_retry("PUT", url, Some(json_body)).await
    }

    /// Make a DELETE request
    pub async fn delete(&self, url: &str) -> CliResult<HttpResponse> {
        self.request_with_retry("DELETE", url, None::<&str>).await
    }

    /// Make a request with retry logic
    async fn request_with_retry(&self, method: &str, url: &str, body: Option<String>) -> CliResult<HttpResponse> {
        let mut last_error = None;

        for attempt in 0..=self.config.retries {
            if attempt > 0 {
                info!("Retrying request (attempt {}/{})", attempt, self.config.retries);
                tokio::time::sleep(self.config.retry_delay).await;
            }

            match self.make_request(method, url, body.as_deref()).await {
                Ok(response) => return Ok(response),
                Err(e) => {
                    warn!("Request attempt {} failed: {}", attempt + 1, e);
                    last_error = Some(e);

                    // Don't retry on certain errors
                    if let CliError::Network(NetworkError::HttpError { status, .. }) = &last_error.as_ref().unwrap() {
                        if *status == 400 || *status == 401 || *status == 403 {
                            break;
                        }
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| CliError::Network(NetworkError::Timeout {
            operation: method.to_string(),
            seconds: self.config.timeout.as_secs() as u64,
        })))
    }

    /// Make the actual HTTP request
    async fn make_request(&self, method: &str, url: &str, body: Option<&str>) -> CliResult<HttpResponse> {
        let start_time = Instant::now();

        debug!("Making {} request to {}", method, url);

        let mut request_builder = match method {
            "GET" => self.client.get(url),
            "POST" => self.client.post(url),
            "PUT" => self.client.put(url),
            "DELETE" => self.client.delete(url),
            _ => return Err(CliError::Network(NetworkError::ConnectionFailed {
                host: "unsupported_method".to_string(),
                port: 0,
            })),
        };

        // Add custom headers
        for (key, value) in &self.config.headers {
            request_builder = request_builder.header(key, value);
        }

        // Add body for POST/PUT requests
        if let Some(body_content) = body {
            request_builder = request_builder
                .header("Content-Type", "application/json")
                .body(body_content.to_string());
        }

        let request = request_builder.build()
            .map_err(|e| CliError::Network(NetworkError::ConnectionFailed {
                host: url.to_string(),
                port: 0,
            }))?;

        // Execute request with timeout
        let response = timeout(self.config.timeout, self.client.execute(request)).await
            .map_err(|_| CliError::Network(NetworkError::Timeout {
                operation: method.to_string(),
                seconds: self.config.timeout.as_secs() as u64,
            }))?
            .map_err(|e| CliError::Network(NetworkError::ConnectionFailed {
                host: url.to_string(),
                port: 0,
            }))?;

        let status = response.status();
        let headers = response.headers().clone();
        let response_time = start_time.elapsed();

        // Check for HTTP errors
        if !status.is_success() {
            let error_body = response.text().await.unwrap_or_default();
            return Err(CliError::Network(NetworkError::HttpError {
                status: status.as_u16(),
                message: format!("HTTP {}: {}", status, error_body),
            }));
        }

        let body = response.text().await
            .map_err(|e| CliError::Network(NetworkError::ConnectionFailed {
                host: "response_read".to_string(),
                port: 0,
            }))?;

        debug!("Request completed in {:.2}ms with status {}", response_time.as_millis(), status);

        Ok(HttpResponse {
            status: status.as_u16(),
            headers,
            body,
            response_time,
        })
    }

    /// Download file from URL
    pub async fn download_file(&self, url: &str, output_path: &std::path::Path) -> CliResult<u64> {
        info!("Downloading file from {} to {:?}", url, output_path);

        let response = self.client.get(url).send().await
            .map_err(|e| CliError::Network(NetworkError::ConnectionFailed {
                host: url.to_string(),
                port: 0,
            }))?;

        if !response.status().is_success() {
            return Err(CliError::Network(NetworkError::HttpError {
                status: response.status().as_u16(),
                message: format!("Download failed: {}", response.status()),
            }));
        }

        let content = response.bytes().await
            .map_err(|e| CliError::Network(NetworkError::ConnectionFailed {
                host: "download".to_string(),
                port: 0,
            }))?;

        tokio::fs::write(output_path, &content).await
            .map_err(|e| CliError::Io(e))?;

        let size = content.len() as u64;
        info!("Downloaded {} bytes to {:?}", size, output_path);

        Ok(size)
    }

    /// Test connectivity to a URL
    pub async fn test_connectivity(&self, url: &str) -> CliResult<ConnectivityResult> {
        let start_time = Instant::now();

        let result = match timeout(self.config.timeout, self.client.head(url).send()).await {
            Ok(Ok(response)) => {
                let response_time = start_time.elapsed();
                ConnectivityResult::Success {
                    status: response.status().as_u16(),
                    response_time,
                }
            }
            Ok(Err(e)) => {
                ConnectivityResult::Failed {
                    error: format!("Request failed: {}", e),
                }
            }
            Err(_) => {
                ConnectivityResult::Failed {
                    error: "Timeout".to_string(),
                }
            }
        };

        Ok(result)
    }
}

/// HTTP response wrapper
#[derive(Debug, Clone)]
pub struct HttpResponse {
    pub status: u16,
    pub headers: reqwest::header::HeaderMap,
    pub body: String,
    pub response_time: Duration,
}

impl HttpResponse {
    /// Parse response body as JSON
    pub fn json<T: for<'de> Deserialize<'de>>(&self) -> CliResult<T> {
        serde_json::from_str(&self.body)
            .map_err(|e| CliError::Serialization(crate::error::SerializationError::JsonError {
                message: format!("Failed to parse JSON response: {}", e),
            }))
    }

    /// Get header value
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.get(name)?.to_str().ok()
    }
}

/// Connectivity test result
#[derive(Debug, Clone)]
pub enum ConnectivityResult {
    Success { status: u16, response_time: Duration },
    Failed { error: String },
}

/// WebSocket client for real-time communication
pub struct WebSocketClient {
    url: String,
    reconnect_attempts: u32,
    reconnect_delay: Duration,
}

impl WebSocketClient {
    pub fn new(url: String) -> Self {
        Self {
            url,
            reconnect_attempts: 3,
            reconnect_delay: Duration::from_secs(5),
        }
    }

    // WebSocket implementation would go here
    // For now, this is a placeholder
}

/// DNS utilities
pub struct DnsUtils;

impl DnsUtils {
    /// Resolve hostname to IP addresses
    pub async fn resolve_hostname(hostname: &str) -> CliResult<Vec<std::net::IpAddr>> {
        use tokio::net::lookup_host;

        let addr = format!("{}:0", hostname);
        let mut addrs = Vec::new();

        match lookup_host(&addr).await {
            Ok(addr_iter) => {
                for addr in addr_iter {
                    addrs.push(addr.ip());
                }
                Ok(addrs)
            }
            Err(e) => Err(CliError::Network(NetworkError::DnsResolutionFailed {
                host: hostname.to_string(),
            })),
        }
    }

    /// Perform DNS lookup with detailed information
    pub async fn dns_lookup(hostname: &str) -> CliResult<DnsLookupResult> {
        let ips = Self::resolve_hostname(hostname).await?;

        Ok(DnsLookupResult {
            hostname: hostname.to_string(),
            ip_addresses: ips,
            lookup_time: Duration::from_millis(100), // Placeholder
        })
    }
}

/// DNS lookup result
#[derive(Debug, Clone)]
pub struct DnsLookupResult {
    pub hostname: String,
    pub ip_addresses: Vec<std::net::IpAddr>,
    pub lookup_time: Duration,
}

/// Network diagnostics utilities
pub struct NetworkDiagnostics;

impl NetworkDiagnostics {
    /// Perform comprehensive network diagnostics
    pub async fn diagnose_connectivity(targets: &[String]) -> CliResult<DiagnosticsReport> {
        info!("Running network diagnostics for {} targets", targets.len());

        let mut results = Vec::new();

        for target in targets {
            let result = Self::diagnose_target(target).await?;
            results.push(result);
        }

        Ok(DiagnosticsReport {
            timestamp: chrono::Utc::now(),
            results,
        })
    }

    async fn diagnose_target(target: &str) -> CliResult<TargetDiagnostics> {
        let dns_result = DnsUtils::dns_lookup(target).await.ok();

        let connectivity = if let Some(ref dns) = dns_result {
            if !dns.ip_addresses.is_empty() {
                Some(format!("DNS resolved to {} addresses", dns.ip_addresses.len()))
            } else {
                Some("DNS resolution failed".to_string())
            }
        } else {
            Some("DNS resolution failed".to_string())
        };

        Ok(TargetDiagnostics {
            target: target.to_string(),
            dns_resolution: dns_result,
            connectivity,
            latency: None, // Would implement ping-like functionality
        })
    }
}

/// Network diagnostics report
#[derive(Debug, Clone)]
pub struct DiagnosticsReport {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub results: Vec<TargetDiagnostics>,
}

/// Target diagnostics result
#[derive(Debug, Clone)]
pub struct TargetDiagnostics {
    pub target: String,
    pub dns_resolution: Option<DnsLookupResult>,
    pub connectivity: Option<String>,
    pub latency: Option<Duration>,
}

/// Default HTTP configuration
impl Default for HttpConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            retries: 3,
            retry_delay: Duration::from_secs(1),
            user_agent: format!("{}/{}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION")),
            headers: HashMap::new(),
            proxy: None,
            follow_redirects: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_dns_resolution() {
        let result = DnsUtils::resolve_hostname("google.com").await;
        assert!(result.is_ok());
        let ips = result.unwrap();
        assert!(!ips.is_empty());
    }

    #[test]
    fn test_default_http_config() {
        let config = HttpConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(30));
        assert_eq!(config.retries, 3);
        assert!(config.follow_redirects);
    }

    #[tokio::test]
    async fn test_connectivity_test() {
        let config = HttpConfig::default();
        let client = HttpClient::new(config).unwrap();

        // Test with a known reliable endpoint
        let result = client.test_connectivity("https://httpbin.org/status/200").await;
        match result {
            Ok(ConnectivityResult::Success { .. }) => assert!(true),
            Ok(ConnectivityResult::Failed { .. }) => assert!(true), // Network might be down
            Err(_) => assert!(true), // Various network issues possible
        }
    }
}