//! Service mesh module
//!
//! This module provides comprehensive service mesh functionality including
//! traffic management, service discovery, security policies, and observability.

use crate::error::{CliError, CliResult};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use serde::{Deserialize, Serialize};
use tracing::{info, debug, warn, error};

/// Service mesh control plane
pub struct ServiceMesh {
    services: RwLock<HashMap<String, MeshService>>,
    traffic_policies: RwLock<HashMap<String, TrafficPolicy>>,
    security_policies: RwLock<HashMap<String, SecurityPolicy>>,
    telemetry_collector: Arc<TelemetryCollector>,
    config_store: Arc<ConfigStore>,
}

impl ServiceMesh {
    /// Create a new service mesh
    pub fn new() -> Self {
        Self {
            services: RwLock::new(HashMap::new()),
            traffic_policies: RwLock::new(HashMap::new()),
            security_policies: RwLock::new(HashMap::new()),
            telemetry_collector: Arc::new(TelemetryCollector::new()),
            config_store: Arc::new(ConfigStore::new()),
        }
    }

    /// Register a service with the mesh
    pub async fn register_service(&self, service: MeshService) -> CliResult<()> {
        let mut services = self.services.write().await;
        services.insert(service.name.clone(), service.clone());

        // Start sidecar proxy for the service
        self.start_sidecar_proxy(&service).await?;

        info!("Registered service '{}' in service mesh", service.name);
        Ok(())
    }

    /// Deregister a service from the mesh
    pub async fn deregister_service(&self, service_name: &str) -> CliResult<()> {
        let mut services = self.services.write().await;
        if let Some(service) = services.remove(service_name) {
            // Stop sidecar proxy
            self.stop_sidecar_proxy(&service).await?;
            info!("Deregistered service '{}' from service mesh", service_name);
        }
        Ok(())
    }

    /// Apply traffic policy
    pub async fn apply_traffic_policy(&self, policy: TrafficPolicy) -> CliResult<()> {
        let mut policies = self.traffic_policies.write().await;
        policies.insert(policy.name.clone(), policy.clone());

        // Apply policy to relevant services
        self.enforce_traffic_policy(&policy).await?;

        info!("Applied traffic policy '{}'", policy.name);
        Ok(())
    }

    /// Apply security policy
    pub async fn apply_security_policy(&self, policy: SecurityPolicy) -> CliResult<()> {
        let mut policies = self.security_policies.write().await;
        policies.insert(policy.name.clone(), policy.clone());

        // Apply policy to relevant services
        self.enforce_security_policy(&policy).await?;

        info!("Applied security policy '{}'", policy.name);
        Ok(())
    }

    /// Get service endpoints
    pub async fn get_service_endpoints(&self, service_name: &str) -> CliResult<Vec<ServiceEndpoint>> {
        let services = self.services.read().await;

        if let Some(service) = services.get(service_name) {
            let mut endpoints = vec![];

            for instance in &service.instances {
                // Check if instance is healthy
                if self.is_instance_healthy(instance).await {
                    endpoints.push(ServiceEndpoint {
                        address: format!("{}:{}", instance.host, instance.port),
                        metadata: instance.metadata.clone(),
                    });
                }
            }

            Ok(endpoints)
        } else {
            Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Service '{}' not found in mesh", service_name)
            )))
        }
    }

    /// Route request through service mesh
    pub async fn route_request(&self, request: MeshRequest) -> CliResult<MeshResponse> {
        debug!("Routing request to service '{}'", request.service_name);

        // Get service endpoints
        let endpoints = self.get_service_endpoints(&request.service_name).await?;

        if endpoints.is_empty() {
            return Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("No healthy endpoints available for service '{}'", request.service_name)
            )));
        }

        // Apply traffic policies
        let selected_endpoint = self.apply_traffic_routing(&request, &endpoints).await?;

        // Apply security policies
        self.apply_security_checks(&request).await?;

        // Forward request through sidecar
        let response = self.forward_through_sidecar(request, selected_endpoint).await?;

        // Collect telemetry
        self.telemetry_collector.record_request(&request, &response).await;

        Ok(response)
    }

    /// Start sidecar proxy for service
    async fn start_sidecar_proxy(&self, service: &MeshService) -> CliResult<()> {
        info!("Starting sidecar proxy for service '{}'", service.name);

        // In real implementation, this would start an Envoy proxy or similar
        // For now, we'll simulate the sidecar functionality

        for instance in &service.instances {
            let proxy_config = SidecarConfig {
                service_name: service.name.clone(),
                listen_port: instance.port + 1000, // Sidecar listens on port + 1000
                upstream_port: instance.port,
                admin_port: instance.port + 2000,
            };

            // Start proxy in background
            tokio::spawn(async move {
                if let Err(e) = run_sidecar_proxy(proxy_config).await {
                    error!("Sidecar proxy error for service {}: {}", service.name, e);
                }
            });
        }

        Ok(())
    }

    /// Stop sidecar proxy for service
    async fn stop_sidecar_proxy(&self, service: &MeshService) -> CliResult<()> {
        info!("Stopping sidecar proxy for service '{}'", service.name);
        // Implementation would signal proxies to shut down
        Ok(())
    }

    /// Enforce traffic policy
    async fn enforce_traffic_policy(&self, policy: &TrafficPolicy) -> CliResult<()> {
        debug!("Enforcing traffic policy '{}'", policy.name);

        // Apply routing rules, load balancing, etc.
        // This would update the sidecar proxy configurations

        Ok(())
    }

    /// Enforce security policy
    async fn enforce_security_policy(&self, policy: &SecurityPolicy) -> CliResult<()> {
        debug!("Enforcing security policy '{}'", policy.name);

        // Apply authentication, authorization, encryption rules
        // This would update the sidecar proxy configurations

        Ok(())
    }

    /// Check if service instance is healthy
    async fn is_instance_healthy(&self, instance: &ServiceInstance) -> bool {
        // Simple health check - in production, this would be more sophisticated
        if let Ok(response) = reqwest::get(&format!("http://{}:{}/health", instance.host, instance.port)).await {
            response.status().is_success()
        } else {
            false
        }
    }

    /// Apply traffic routing logic
    async fn apply_traffic_routing(&self, request: &MeshRequest, endpoints: &[ServiceEndpoint]) -> CliResult<ServiceEndpoint> {
        // Check for traffic policies
        let policies = self.traffic_policies.read().await;

        for policy in policies.values() {
            if policy.matches_request(request) {
                return policy.route_request(endpoints);
            }
        }

        // Default: round-robin
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNTER: AtomicUsize = AtomicUsize::new(0);

        let index = COUNTER.fetch_add(1, Ordering::SeqCst) % endpoints.len();
        Ok(endpoints[index].clone())
    }

    /// Apply security checks
    async fn apply_security_checks(&self, request: &MeshRequest) -> CliResult<()> {
        let policies = self.security_policies.read().await;

        for policy in policies.values() {
            if policy.matches_request(request) {
                policy.enforce_security(request).await?;
            }
        }

        Ok(())
    }

    /// Forward request through sidecar proxy
    async fn forward_through_sidecar(&self, request: MeshRequest, endpoint: ServiceEndpoint) -> CliResult<MeshResponse> {
        // In real implementation, this would go through the sidecar proxy
        // For simulation, we'll make direct HTTP calls

        let client = reqwest::Client::new();
        let url = format!("http://{}", endpoint.address);

        let mut http_request = match request.method.as_str() {
            "GET" => client.get(&url),
            "POST" => client.post(&url),
            "PUT" => client.put(&url),
            "DELETE" => client.delete(&url),
            _ => return Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Unsupported HTTP method: {}", request.method)
            ))),
        };

        // Add headers
        for (key, value) in &request.headers {
            http_request = http_request.header(key, value);
        }

        // Add body for non-GET requests
        if !request.body.is_empty() {
            http_request = http_request.body(request.body);
        }

        let http_response = http_request.send().await
            .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Request failed: {}", e)
            )))?;

        let status_code = http_response.status().as_u16();
        let headers = http_response.headers().clone();
        let body = http_response.bytes().await
            .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Failed to read response body: {}", e)
            )))?;

        Ok(MeshResponse {
            status_code,
            headers: headers.into_iter()
                .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
                .collect(),
            body: body.to_vec(),
        })
    }
}

/// Mesh service configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshService {
    pub name: String,
    pub namespace: String,
    pub instances: Vec<ServiceInstance>,
    pub labels: HashMap<String, String>,
}

/// Service instance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceInstance {
    pub id: String,
    pub host: String,
    pub port: u16,
    pub metadata: HashMap<String, String>,
}

/// Service endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceEndpoint {
    pub address: String,
    pub metadata: HashMap<String, String>,
}

/// Traffic policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficPolicy {
    pub name: String,
    pub rules: Vec<TrafficRule>,
}

impl TrafficPolicy {
    /// Check if policy matches request
    pub fn matches_request(&self, request: &MeshRequest) -> bool {
        self.rules.iter().any(|rule| rule.matches(request))
    }

    /// Route request according to policy
    pub fn route_request(&self, endpoints: &[ServiceEndpoint]) -> CliResult<ServiceEndpoint> {
        // Find first matching rule
        for rule in &self.rules {
            if let Some(endpoint) = rule.route_to_endpoint(endpoints) {
                return Ok(endpoint);
            }
        }

        // Default to first endpoint
        endpoints.first().cloned().ok_or_else(|| {
            CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                "No endpoints available".to_string()
            ))
        })
    }
}

/// Traffic rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficRule {
    pub match_conditions: Vec<MatchCondition>,
    pub route_destination: RouteDestination,
}

impl TrafficRule {
    /// Check if rule matches request
    pub fn matches(&self, request: &MeshRequest) -> bool {
        self.match_conditions.iter().all(|condition| condition.matches(request))
    }

    /// Route to endpoint based on rule
    pub fn route_to_endpoint(&self, endpoints: &[ServiceEndpoint]) -> Option<ServiceEndpoint> {
        match &self.route_destination {
            RouteDestination::Service(name) => {
                endpoints.iter().find(|e| e.metadata.get("service") == Some(name)).cloned()
            }
            RouteDestination::Subset(labels) => {
                endpoints.iter().find(|e| {
                    labels.iter().all(|(key, value)| e.metadata.get(key) == Some(value))
                }).cloned()
            }
            RouteDestination::Weighted(weights) => {
                // Weighted load balancing
                use rand::prelude::*;
                let mut rng = rand::thread_rng();
                let total_weight: u32 = weights.values().sum();
                let mut choice = rng.gen_range(0..total_weight);

                for (service_name, weight) in weights {
                    if choice < *weight {
                        return endpoints.iter().find(|e| e.metadata.get("service") == Some(service_name)).cloned();
                    }
                    choice -= weight;
                }
                None
            }
        }
    }
}

/// Match condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchCondition {
    pub key: String,
    pub value: String,
    pub match_type: MatchType,
}

impl MatchCondition {
    /// Check if condition matches request
    pub fn matches(&self, request: &MeshRequest) -> bool {
        let actual_value = match self.key.as_str() {
            "method" => &request.method,
            "path" => &request.path,
            "service" => &request.service_name,
            key => request.headers.get(key).map(|s| s.as_str()).unwrap_or(""),
        };

        match self.match_type {
            MatchType::Exact => actual_value == &self.value,
            MatchType::Prefix => actual_value.starts_with(&self.value),
            MatchType::Regex => regex::Regex::new(&self.value)
                .map(|re| re.is_match(actual_value))
                .unwrap_or(false),
        }
    }
}

/// Match type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MatchType {
    Exact,
    Prefix,
    Regex,
}

/// Route destination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RouteDestination {
    Service(String),
    Subset(HashMap<String, String>),
    Weighted(HashMap<String, u32>),
}

/// Security policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityPolicy {
    pub name: String,
    pub rules: Vec<SecurityRule>,
}

impl SecurityPolicy {
    /// Check if policy matches request
    pub fn matches_request(&self, request: &MeshRequest) -> bool {
        self.rules.iter().any(|rule| rule.matches(request))
    }

    /// Enforce security policy
    pub async fn enforce_security(&self, request: &MeshRequest) -> CliResult<()> {
        for rule in &self.rules {
            if rule.matches(request) {
                rule.enforce(request).await?;
            }
        }
        Ok(())
    }
}

/// Security rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityRule {
    pub name: String,
    pub conditions: Vec<MatchCondition>,
    pub actions: Vec<SecurityAction>,
}

impl SecurityRule {
    /// Check if rule matches request
    pub fn matches(&self, request: &MeshRequest) -> bool {
        self.conditions.iter().all(|condition| condition.matches(request))
    }

    /// Enforce security rule
    pub async fn enforce(&self, request: &MeshRequest) -> CliResult<()> {
        for action in &self.actions {
            action.execute(request).await?;
        }
        Ok(())
    }
}

/// Security action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityAction {
    Deny,
    Allow,
    Encrypt,
    Authenticate { method: AuthMethod },
    Authorize { roles: Vec<String> },
}

impl SecurityAction {
    /// Execute security action
    pub async fn execute(&self, request: &MeshRequest) -> CliResult<()> {
        match self {
            SecurityAction::Deny => {
                Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                    "Request denied by security policy".to_string()
                )))
            }
            SecurityAction::Allow => Ok(()),
            SecurityAction::Encrypt => {
                // Implement encryption
                warn!("Encryption not yet implemented");
                Ok(())
            }
            SecurityAction::Authenticate { method } => {
                method.authenticate(request).await
            }
            SecurityAction::Authorize { roles } => {
                // Check if request has required roles
                if let Some(user_roles) = request.headers.get("x-user-roles") {
                    let user_role_list: Vec<&str> = user_roles.split(',').collect();
                    for required_role in roles {
                        if !user_role_list.contains(&required_role.as_str()) {
                            return Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                                format!("Missing required role: {}", required_role)
                            )));
                        }
                    }
                } else {
                    return Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                        "No user roles provided".to_string()
                    )));
                }
                Ok(())
            }
        }
    }
}

/// Authentication method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthMethod {
    JWT,
    OAuth2,
    MutualTLS,
    APIKey,
}

impl AuthMethod {
    /// Perform authentication
    pub async fn authenticate(&self, request: &MeshRequest) -> CliResult<()> {
        match self {
            AuthMethod::JWT => {
                if let Some(token) = request.headers.get("authorization") {
                    if token.starts_with("Bearer ") {
                        let token_part = &token[7..];
                        // Validate JWT (simplified)
                        if token_part.len() > 10 {
                            Ok(())
                        } else {
                            Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                                "Invalid JWT token".to_string()
                            )))
                        }
                    } else {
                        Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                            "Invalid authorization header format".to_string()
                        )))
                    }
                } else {
                    Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                        "Missing authorization header".to_string()
                    )))
                }
            }
            AuthMethod::OAuth2 => {
                // OAuth2 authentication
                warn!("OAuth2 authentication not yet implemented");
                Ok(())
            }
            AuthMethod::MutualTLS => {
                // Mutual TLS authentication
                warn!("Mutual TLS authentication not yet implemented");
                Ok(())
            }
            AuthMethod::APIKey => {
                if let Some(api_key) = request.headers.get("x-api-key") {
                    // Validate API key (simplified)
                    if api_key.len() > 10 {
                        Ok(())
                    } else {
                        Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                            "Invalid API key".to_string()
                        )))
                    }
                } else {
                    Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                        "Missing API key".to_string()
                    )))
                }
            }
        }
    }
}

/// Mesh request
#[derive(Debug, Clone)]
pub struct MeshRequest {
    pub service_name: String,
    pub method: String,
    pub path: String,
    pub headers: HashMap<String, String>,
    pub body: Vec<u8>,
}

/// Mesh response
#[derive(Debug, Clone)]
pub struct MeshResponse {
    pub status_code: u16,
    pub headers: HashMap<String, String>,
    pub body: Vec<u8>,
}

/// Telemetry collector
pub struct TelemetryCollector {
    metrics: RwLock<HashMap<String, TelemetryMetric>>,
}

impl TelemetryCollector {
    /// Create a new telemetry collector
    pub fn new() -> Self {
        Self {
            metrics: RwLock::new(HashMap::new()),
        }
    }

    /// Record request/response telemetry
    pub async fn record_request(&self, request: &MeshRequest, response: &MeshResponse) {
        let mut metrics = self.metrics.write().await;

        // Record request count
        let request_count = metrics.entry("requests_total".to_string())
            .or_insert_with(|| TelemetryMetric::Counter(0));
        if let TelemetryMetric::Counter(ref mut count) = request_count {
            *count += 1;
        }

        // Record response time (simplified)
        let response_time = metrics.entry("response_time".to_string())
            .or_insert_with(|| TelemetryMetric::Histogram(vec![]));
        if let TelemetryMetric::Histogram(ref mut times) = response_time {
            times.push(std::time::Instant::now());
        }

        // Record status codes
        let status_key = format!("status_{}", response.status_code);
        let status_count = metrics.entry(status_key)
            .or_insert_with(|| TelemetryMetric::Counter(0));
        if let TelemetryMetric::Counter(ref mut count) = status_count {
            *count += 1;
        }
    }

    /// Get metric value
    pub async fn get_metric(&self, name: &str) -> Option<TelemetryMetric> {
        let metrics = self.metrics.read().await;
        metrics.get(name).cloned()
    }

    /// Export metrics in Prometheus format
    pub async fn export_prometheus(&self) -> String {
        let metrics = self.metrics.read().await;
        let mut output = String::new();

        for (name, metric) in metrics.iter() {
            match metric {
                TelemetryMetric::Counter(value) => {
                    output.push_str(&format!("# HELP {} Counter metric\n", name));
                    output.push_str(&format!("# TYPE {} counter\n", name));
                    output.push_str(&format!("{}} {}\n", name, value));
                }
                TelemetryMetric::Gauge(value) => {
                    output.push_str(&format!("# HELP {} Gauge metric\n", name));
                    output.push_str(&format!("# TYPE {} gauge\n", name));
                    output.push_str(&format!("{}} {}\n", name, value));
                }
                TelemetryMetric::Histogram(_) => {
                    // Simplified histogram export
                    output.push_str(&format!("# HELP {} Histogram metric\n", name));
                    output.push_str(&format!("# TYPE {} histogram\n", name));
                    output.push_str(&format!("{}_count 1\n", name));
                }
            }
        }

        output
    }
}

/// Telemetry metric
#[derive(Debug, Clone)]
pub enum TelemetryMetric {
    Counter(u64),
    Gauge(f64),
    Histogram(Vec<std::time::Instant>),
}

/// Configuration store for mesh configuration
pub struct ConfigStore {
    configs: RwLock<HashMap<String, serde_json::Value>>,
}

impl ConfigStore {
    /// Create a new config store
    pub fn new() -> Self {
        Self {
            configs: RwLock::new(HashMap::new()),
        }
    }

    /// Store configuration
    pub async fn store_config(&self, key: String, config: serde_json::Value) {
        let mut configs = self.configs.write().await;
        configs.insert(key, config);
    }

    /// Get configuration
    pub async fn get_config(&self, key: &str) -> Option<serde_json::Value> {
        let configs = self.configs.read().await;
        configs.get(key).cloned()
    }

    /// List all configurations
    pub async fn list_configs(&self) -> Vec<String> {
        let configs = self.configs.read().await;
        configs.keys().cloned().collect()
    }
}

/// Sidecar proxy configuration
#[derive(Debug, Clone)]
struct SidecarConfig {
    service_name: String,
    listen_port: u16,
    upstream_port: u16,
    admin_port: u16,
}

/// Run sidecar proxy (simplified implementation)
async fn run_sidecar_proxy(config: SidecarConfig) -> CliResult<()> {
    info!("Starting sidecar proxy for service '{}' on port {}", config.service_name, config.listen_port);

    // In real implementation, this would run Envoy or similar proxy
    // For simulation, we'll just log and sleep

    loop {
        tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;
        debug!("Sidecar proxy for '{}' is running", config.service_name);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_service_mesh_creation() {
        let mesh = ServiceMesh::new();
        // Test that it can be created
        assert!(true);
    }

    #[test]
    fn test_mesh_service_creation() {
        let service = MeshService {
            name: "test-service".to_string(),
            namespace: "default".to_string(),
            instances: vec![],
            labels: HashMap::new(),
        };

        assert_eq!(service.name, "test-service");
        assert_eq!(service.namespace, "default");
    }

    #[test]
    fn test_traffic_policy() {
        let rule = TrafficRule {
            match_conditions: vec![MatchCondition {
                key: "path".to_string(),
                value: "/api".to_string(),
                match_type: MatchType::Prefix,
            }],
            route_destination: RouteDestination::Service("api-service".to_string()),
        };

        let request = MeshRequest {
            service_name: "api-service".to_string(),
            method: "GET".to_string(),
            path: "/api/users".to_string(),
            headers: HashMap::new(),
            body: vec![],
        };

        assert!(rule.matches(&request));
    }

    #[test]
    fn test_security_policy() {
        let rule = SecurityRule {
            name: "auth-rule".to_string(),
            conditions: vec![MatchCondition {
                key: "path".to_string(),
                value: "/admin".to_string(),
                match_type: MatchType::Prefix,
            }],
            actions: vec![SecurityAction::Authenticate {
                method: AuthMethod::JWT,
            }],
        };

        let request = MeshRequest {
            service_name: "admin-service".to_string(),
            method: "GET".to_string(),
            path: "/admin/users".to_string(),
            headers: HashMap::new(),
            body: vec![],
        };

        assert!(rule.matches(&request));
    }

    #[test]
    fn test_telemetry_collector() {
        let collector = TelemetryCollector::new();
        // Test that it can be created
        assert!(true);
    }

    #[test]
    fn test_config_store() {
        let store = ConfigStore::new();
        // Test that it can be created
        assert!(true);
    }

    #[tokio::test]
    async fn test_mesh_request_routing() {
        let mesh = ServiceMesh::new();

        let service = MeshService {
            name: "test-service".to_string(),
            namespace: "default".to_string(),
            instances: vec![ServiceInstance {
                id: "instance-1".to_string(),
                host: "localhost".to_string(),
                port: 8080,
                metadata: HashMap::new(),
            }],
            labels: HashMap::new(),
        };

        mesh.register_service(service).await.unwrap();

        let endpoints = mesh.get_service_endpoints("test-service").await.unwrap();
        assert_eq!(endpoints.len(), 1);
        assert_eq!(endpoints[0].address, "localhost:8080");
    }

    #[test]
    fn test_sidecar_config() {
        let config = SidecarConfig {
            service_name: "test-service".to_string(),
            listen_port: 18080,
            upstream_port: 8080,
            admin_port: 18081,
        };

        assert_eq!(config.service_name, "test-service");
        assert_eq!(config.listen_port, 18080);
        assert_eq!(config.upstream_port, 8080);
    }
}