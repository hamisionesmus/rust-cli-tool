//! API Gateway module
//!
//! This module provides comprehensive API gateway functionality including
//! load balancing, rate limiting, service discovery, and traffic management.

use crate::error::{CliError, CliResult};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use serde::{Deserialize, Serialize};
use tracing::{info, debug, warn, error};
use std::net::SocketAddr;

/// API Gateway server
pub struct ApiGateway {
    routes: RwLock<HashMap<String, RouteConfig>>,
    services: RwLock<HashMap<String, ServiceConfig>>,
    load_balancers: HashMap<String, Box<dyn LoadBalancer>>,
    rate_limiters: HashMap<String, Box<dyn RateLimiter>>,
    middlewares: Vec<Box<dyn GatewayMiddleware>>,
    port: u16,
    host: String,
}

impl ApiGateway {
    /// Create a new API gateway
    pub fn new(host: String, port: u16) -> Self {
        Self {
            routes: RwLock::new(HashMap::new()),
            services: RwLock::new(HashMap::new()),
            load_balancers: HashMap::new(),
            rate_limiters: HashMap::new(),
            middlewares: vec![],
            port,
            host,
        }
    }

    /// Add a route configuration
    pub async fn add_route(&self, route: RouteConfig) -> CliResult<()> {
        let mut routes = self.routes.write().await;
        routes.insert(route.path.clone(), route);
        info!("Added route: {}", route.path);
        Ok(())
    }

    /// Add a service configuration
    pub async fn add_service(&self, service: ServiceConfig) -> CliResult<()> {
        let mut services = self.services.write().await;
        services.insert(service.name.clone(), service);
        info!("Added service: {}", service.name);
        Ok(())
    }

    /// Add a load balancer
    pub fn add_load_balancer(&mut self, name: &str, balancer: Box<dyn LoadBalancer>) {
        self.load_balancers.insert(name.to_string(), balancer);
    }

    /// Add a rate limiter
    pub fn add_rate_limiter(&mut self, name: &str, limiter: Box<dyn RateLimiter>) {
        self.rate_limiters.insert(name.to_string(), limiter);
    }

    /// Add middleware
    pub fn add_middleware(&mut self, middleware: Box<dyn GatewayMiddleware>) {
        self.middlewares.push(middleware);
    }

    /// Start the API gateway
    pub async fn start(self) -> CliResult<()> {
        info!("Starting API Gateway on {}:{}", self.host, self.port);

        let addr: SocketAddr = format!("{}:{}", self.host, self.port)
            .parse()
            .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Invalid address: {}", e)
            )))?;

        let routes = self.routes;
        let services = self.services;
        let load_balancers = Arc::new(self.load_balancers);
        let rate_limiters = Arc::new(self.rate_limiters);
        let middlewares = Arc::new(self.middlewares);

        let app = warp::path::full()
            .and(warp::method())
            .and(warp::header::headers_cloned())
            .and(warp::body::bytes())
            .and_then(move |path: warp::path::FullPath, method: http::Method, headers: http::HeaderMap, body: bytes::Bytes| {
                let routes = routes.clone();
                let services = services.clone();
                let load_balancers = load_balancers.clone();
                let rate_limiters = rate_limiters.clone();
                let middlewares = middlewares.clone();

                async move {
                    Self::handle_request(
                        path, method, headers, body,
                        routes, services, load_balancers, rate_limiters, middlewares
                    ).await
                }
            });

        info!("API Gateway listening on {}", addr);
        warp::serve(app).run(addr).await;

        Ok(())
    }

    /// Handle incoming request
    async fn handle_request(
        path: warp::path::FullPath,
        method: http::Method,
        headers: http::HeaderMap,
        body: bytes::Bytes,
        routes: Arc<RwLock<HashMap<String, RouteConfig>>>,
        services: Arc<RwLock<HashMap<String, ServiceConfig>>>,
        load_balancers: Arc<HashMap<String, Box<dyn LoadBalancer>>>,
        rate_limiters: Arc<HashMap<String, Box<dyn RateLimiter>>>,
        middlewares: Arc<Vec<Box<dyn GatewayMiddleware>>>,
    ) -> Result<Box<dyn warp::Reply>, warp::Rejection> {
        let path_str = path.as_str();

        debug!("Handling request: {} {}", method, path_str);

        // Find matching route
        let routes_read = routes.read().await;
        let route = routes_read.values()
            .find(|r| Self::matches_route(path_str, &r.path))
            .cloned();

        let route = match route {
            Some(r) => r,
            None => {
                return Ok(Box::new(warp::reply::with_status(
                    "Route not found",
                    warp::http::StatusCode::NOT_FOUND,
                )));
            }
        };

        // Apply rate limiting
        if let Some(ref limiter_name) = route.rate_limiter {
            if let Some(limiter) = rate_limiters.get(limiter_name) {
                if !limiter.allow().await {
                    return Ok(Box::new(warp::reply::with_status(
                        "Rate limit exceeded",
                        warp::http::StatusCode::TOO_MANY_REQUESTS,
                    )));
                }
            }
        }

        // Apply middlewares
        let mut request = GatewayRequest {
            path: path_str.to_string(),
            method: method.clone(),
            headers: headers.clone(),
            body: body.clone(),
        };

        for middleware in middlewares.iter() {
            request = match middleware.process_request(request).await {
                Ok(req) => req,
                Err(e) => {
                    error!("Middleware error: {}", e);
                    return Ok(Box::new(warp::reply::with_status(
                        "Internal server error",
                        warp::http::StatusCode::INTERNAL_SERVER_ERROR,
                    )));
                }
            };
        }

        // Get service configuration
        let services_read = services.read().await;
        let service = match services_read.get(&route.service) {
            Some(s) => s.clone(),
            None => {
                return Ok(Box::new(warp::reply::with_status(
                    "Service not found",
                    warp::http::StatusCode::NOT_FOUND,
                )));
            }
        };

        // Load balancing
        let backend_url = if let Some(ref balancer_name) = service.load_balancer {
            if let Some(balancer) = load_balancers.get(balancer_name) {
                balancer.next_backend(&service.backends).await
            } else {
                service.backends.first().cloned()
            }
        } else {
            service.backends.first().cloned()
        };

        let backend_url = match backend_url {
            Some(url) => url,
            None => {
                return Ok(Box::new(warp::reply::with_status(
                    "No backend available",
                    warp::http::StatusCode::SERVICE_UNAVAILABLE,
                )));
            }
        };

        // Forward request to backend
        match Self::forward_request(&backend_url, &method, &request.headers, &request.body).await {
            Ok(response) => {
                debug!("Request forwarded successfully to {}", backend_url);
                Ok(Box::new(response))
            }
            Err(e) => {
                error!("Failed to forward request: {}", e);
                Ok(Box::new(warp::reply::with_status(
                    "Backend service error",
                    warp::http::StatusCode::BAD_GATEWAY,
                )))
            }
        }
    }

    /// Check if path matches route pattern
    fn matches_route(path: &str, route_pattern: &str) -> bool {
        if route_pattern.contains('*') {
            // Simple wildcard matching
            let pattern = route_pattern.replace('*', ".*");
            regex::Regex::new(&format!("^{}$", pattern))
                .map(|re| re.is_match(path))
                .unwrap_or(false)
        } else {
            path.starts_with(route_pattern)
        }
    }

    /// Forward request to backend service
    async fn forward_request(
        backend_url: &str,
        method: &http::Method,
        headers: &http::HeaderMap,
        body: &bytes::Bytes,
    ) -> CliResult<warp::reply::Response> {
        let client = reqwest::Client::new();

        let mut request = match *method {
            http::Method::GET => client.get(backend_url),
            http::Method::POST => client.post(backend_url),
            http::Method::PUT => client.put(backend_url),
            http::Method::DELETE => client.delete(backend_url),
            http::Method::PATCH => client.patch(backend_url),
            _ => return Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Unsupported HTTP method: {}", method)
            ))),
        };

        // Forward headers (excluding hop-by-hop headers)
        for (key, value) in headers.iter() {
            if !Self::is_hop_by_hop_header(key) {
                if let Ok(value_str) = value.to_str() {
                    request = request.header(key, value_str);
                }
            }
        }

        // Add body for non-GET requests
        if *method != http::Method::GET && !body.is_empty() {
            request = request.body(body.clone());
        }

        let response = request.send().await
            .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Backend request failed: {}", e)
            )))?;

        let status = response.status();
        let headers = response.headers().clone();
        let body = response.bytes().await
            .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Failed to read response body: {}", e)
            )))?;

        let mut reply = warp::reply::Response::new(body.into());
        *reply.status_mut() = status;

        // Copy response headers
        for (key, value) in headers.iter() {
            reply.headers_mut().insert(key, value.clone());
        }

        Ok(reply)
    }

    /// Check if header is hop-by-hop
    fn is_hop_by_hop_header(name: &http::HeaderName) -> bool {
        matches!(name.as_str(), "connection" | "keep-alive" | "proxy-authenticate" |
                               "proxy-authorization" | "te" | "trailers" | "transfer-encoding" | "upgrade")
    }
}

/// Route configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteConfig {
    pub path: String,
    pub service: String,
    pub methods: Vec<String>,
    pub rate_limiter: Option<String>,
    pub middlewares: Vec<String>,
}

/// Service configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceConfig {
    pub name: String,
    pub backends: Vec<String>,
    pub load_balancer: Option<String>,
    pub health_check: Option<HealthCheckConfig>,
    pub timeout: Option<std::time::Duration>,
}

/// Health check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    pub path: String,
    pub interval: std::time::Duration,
    pub timeout: std::time::Duration,
    pub healthy_threshold: u32,
    pub unhealthy_threshold: u32,
}

/// Load balancer trait
#[async_trait::async_trait]
pub trait LoadBalancer: Send + Sync {
    /// Get next backend URL
    async fn next_backend(&self, backends: &[String]) -> Option<String>;
}

/// Round-robin load balancer
pub struct RoundRobinBalancer {
    current: std::sync::atomic::AtomicUsize,
}

impl RoundRobinBalancer {
    pub fn new() -> Self {
        Self {
            current: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

#[async_trait::async_trait]
impl LoadBalancer for RoundRobinBalancer {
    async fn next_backend(&self, backends: &[String]) -> Option<String> {
        if backends.is_empty() {
            return None;
        }

        let current = self.current.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let index = current % backends.len();
        Some(backends[index].clone())
    }
}

/// Least connections load balancer
pub struct LeastConnectionsBalancer {
    connections: RwLock<HashMap<String, usize>>,
}

impl LeastConnectionsBalancer {
    pub fn new() -> Self {
        Self {
            connections: RwLock::new(HashMap::new()),
        }
    }

    /// Record connection start
    pub async fn connection_start(&self, backend: &str) {
        let mut connections = self.connections.write().await;
        *connections.entry(backend.to_string()).or_insert(0) += 1;
    }

    /// Record connection end
    pub async fn connection_end(&self, backend: &str) {
        let mut connections = self.connections.write().await;
        if let Some(count) = connections.get_mut(backend) {
            if *count > 0 {
                *count -= 1;
            }
        }
    }
}

#[async_trait::async_trait]
impl LoadBalancer for LeastConnectionsBalancer {
    async fn next_backend(&self, backends: &[String]) -> Option<String> {
        if backends.is_empty() {
            return None;
        }

        let connections = self.connections.read().await;
        let mut min_connections = usize::MAX;
        let mut selected_backend = None;

        for backend in backends {
            let conn_count = connections.get(backend).copied().unwrap_or(0);
            if conn_count < min_connections {
                min_connections = conn_count;
                selected_backend = Some(backend.clone());
            }
        }

        selected_backend
    }
}

/// Rate limiter trait
#[async_trait::async_trait]
pub trait RateLimiter: Send + Sync {
    /// Check if request is allowed
    async fn allow(&self) -> bool;

    /// Get current rate limit status
    async fn status(&self) -> RateLimitStatus;
}

/// Token bucket rate limiter
pub struct TokenBucketLimiter {
    capacity: usize,
    refill_rate: usize,
    tokens: RwLock<usize>,
    last_refill: RwLock<std::time::Instant>,
}

impl TokenBucketLimiter {
    pub fn new(capacity: usize, refill_rate: usize) -> Self {
        Self {
            capacity,
            refill_rate,
            tokens: RwLock::new(capacity),
            last_refill: RwLock::new(std::time::Instant::now()),
        }
    }
}

#[async_trait::async_trait]
impl RateLimiter for TokenBucketLimiter {
    async fn allow(&self) -> bool {
        let mut tokens = self.tokens.write().await;
        let mut last_refill = self.last_refill.write().await;

        // Refill tokens
        let now = std::time::Instant::now();
        let elapsed = now.duration_since(*last_refill);
        let tokens_to_add = (elapsed.as_secs() as usize) * self.refill_rate;

        if tokens_to_add > 0 {
            *tokens = (*tokens + tokens_to_add).min(self.capacity);
            *last_refill = now;
        }

        // Check if we can allow the request
        if *tokens > 0 {
            *tokens -= 1;
            true
        } else {
            false
        }
    }

    async fn status(&self) -> RateLimitStatus {
        let tokens = self.tokens.read().await;
        RateLimitStatus {
            current_tokens: *tokens,
            capacity: self.capacity,
        }
    }
}

/// Rate limit status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitStatus {
    pub current_tokens: usize,
    pub capacity: usize,
}

/// Gateway middleware trait
#[async_trait::async_trait]
pub trait GatewayMiddleware: Send + Sync {
    /// Process incoming request
    async fn process_request(&self, request: GatewayRequest) -> CliResult<GatewayRequest>;

    /// Process outgoing response
    async fn process_response(&self, response: GatewayResponse) -> CliResult<GatewayResponse>;
}

/// Gateway request
#[derive(Debug, Clone)]
pub struct GatewayRequest {
    pub path: String,
    pub method: http::Method,
    pub headers: http::HeaderMap,
    pub body: bytes::Bytes,
}

/// Gateway response
#[derive(Debug, Clone)]
pub struct GatewayResponse {
    pub status: u16,
    pub headers: http::HeaderMap,
    pub body: bytes::Bytes,
}

/// Authentication middleware
pub struct AuthMiddleware {
    jwt_secret: String,
}

impl AuthMiddleware {
    pub fn new(jwt_secret: String) -> Self {
        Self { jwt_secret }
    }
}

#[async_trait::async_trait]
impl GatewayMiddleware for AuthMiddleware {
    async fn process_request(&self, request: GatewayRequest) -> CliResult<GatewayRequest> {
        // Check for Authorization header
        if let Some(auth_header) = request.headers.get("authorization") {
            if let Ok(auth_str) = auth_header.to_str() {
                if auth_str.starts_with("Bearer ") {
                    let token = &auth_str[7..];
                    // In real implementation, validate JWT token
                    if self.validate_jwt(token) {
                        return Ok(request);
                    }
                }
            }
        }

        Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
            "Invalid or missing authentication token".to_string()
        )))
    }

    async fn process_response(&self, response: GatewayResponse) -> CliResult<GatewayResponse> {
        Ok(response)
    }
}

impl AuthMiddleware {
    fn validate_jwt(&self, token: &str) -> bool {
        // Simplified JWT validation
        // In production, use a proper JWT library
        token.len() > 10 && token.contains(".")
    }
}

/// Logging middleware
pub struct LoggingMiddleware;

impl LoggingMiddleware {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl GatewayMiddleware for LoggingMiddleware {
    async fn process_request(&self, request: GatewayRequest) -> CliResult<GatewayRequest> {
        info!("Gateway request: {} {} from {:?}", request.method, request.path,
              request.headers.get("x-forwarded-for"));
        Ok(request)
    }

    async fn process_response(&self, response: GatewayResponse) -> CliResult<GatewayResponse> {
        info!("Gateway response: status {}", response.status);
        Ok(response)
    }
}

/// Service discovery
pub struct ServiceDiscovery {
    services: RwLock<HashMap<String, Vec<ServiceInstance>>>,
    registry_client: Option<Box<dyn RegistryClient>>,
}

impl ServiceDiscovery {
    pub fn new() -> Self {
        Self {
            services: RwLock::new(HashMap::new()),
            registry_client: None,
        }
    }

    pub fn set_registry_client(&mut self, client: Box<dyn RegistryClient>) {
        self.registry_client = Some(client);
    }

    /// Register a service instance
    pub async fn register_service(&self, service: ServiceInstance) -> CliResult<()> {
        let mut services = self.services.write().await;
        services.entry(service.name.clone()).or_insert_with(Vec::new).push(service);
        Ok(())
    }

    /// Discover service instances
    pub async fn discover_service(&self, service_name: &str) -> CliResult<Vec<ServiceInstance>> {
        // First check local registry
        let services = self.services.read().await;
        if let Some(instances) = services.get(service_name) {
            return Ok(instances.clone());
        }

        // Then check external registry
        if let Some(ref client) = self.registry_client {
            client.discover_service(service_name).await
        } else {
            Ok(vec![])
        }
    }

    /// Health check services
    pub async fn health_check(&self) -> CliResult<()> {
        let services = self.services.read().await;
        let mut unhealthy = vec![];

        for (service_name, instances) in services.iter() {
            for instance in instances {
                if !self.check_instance_health(instance).await {
                    unhealthy.push(instance.id.clone());
                }
            }
        }

        if !unhealthy.is_empty() {
            warn!("Unhealthy service instances: {:?}", unhealthy);
        }

        Ok(())
    }

    async fn check_instance_health(&self, instance: &ServiceInstance) -> bool {
        // Simple health check - in production, this would be more sophisticated
        if let Ok(response) = reqwest::get(&format!("{}/health", instance.address)).await {
            response.status().is_success()
        } else {
            false
        }
    }
}

/// Service instance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceInstance {
    pub id: String,
    pub name: String,
    pub address: String,
    pub port: u16,
    pub metadata: HashMap<String, String>,
}

/// Registry client trait
#[async_trait::async_trait]
pub trait RegistryClient: Send + Sync {
    /// Discover service instances
    async fn discover_service(&self, service_name: &str) -> CliResult<Vec<ServiceInstance>>;

    /// Register service instance
    async fn register_instance(&self, instance: ServiceInstance) -> CliResult<()>;

    /// Deregister service instance
    async fn deregister_instance(&self, instance_id: &str) -> CliResult<()>;
}

/// Circuit breaker for fault tolerance
pub struct CircuitBreaker {
    failure_threshold: usize,
    recovery_timeout: std::time::Duration,
    failures: RwLock<usize>,
    last_failure_time: RwLock<Option<std::time::Instant>>,
    state: RwLock<CircuitState>,
}

impl CircuitBreaker {
    pub fn new(failure_threshold: usize, recovery_timeout: std::time::Duration) -> Self {
        Self {
            failure_threshold,
            recovery_timeout,
            failures: RwLock::new(0),
            last_failure_time: RwLock::new(None),
            state: RwLock::new(CircuitState::Closed),
        }
    }

    /// Check if request should be allowed
    pub async fn allow_request(&self) -> bool {
        let state = self.state.read().await;

        match *state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                if let Some(last_failure) = *self.last_failure_time.read().await {
                    if last_failure.elapsed() > self.recovery_timeout {
                        // Try to close circuit
                        drop(state);
                        let mut state = self.state.write().await;
                        *state = CircuitState::HalfOpen;
                        return true;
                    }
                }
                false
            }
            CircuitState::HalfOpen => true,
        }
    }

    /// Record successful request
    pub async fn record_success(&self) {
        let mut failures = self.failures.write().await;
        *failures = 0;

        let mut state = self.state.write().await;
        *state = CircuitState::Closed;
    }

    /// Record failed request
    pub async fn record_failure(&self) {
        let mut failures = self.failures.write().await;
        *failures += 1;

        let mut last_failure = self.last_failure_time.write().await;
        *last_failure = Some(std::time::Instant::now());

        if *failures >= self.failure_threshold {
            let mut state = self.state.write().await;
            *state = CircuitState::Open;
        }
    }

    /// Get current state
    pub async fn get_state(&self) -> CircuitState {
        *self.state.read().await
    }
}

/// Circuit breaker states
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_robin_balancer() {
        let balancer = RoundRobinBalancer::new();
        let backends = vec!["http://service1".to_string(), "http://service2".to_string()];

        // Test round-robin behavior
        let first = tokio::runtime::Runtime::new().unwrap().block_on(balancer.next_backend(&backends));
        assert_eq!(first, Some("http://service1".to_string()));

        let second = tokio::runtime::Runtime::new().unwrap().block_on(balancer.next_backend(&backends));
        assert_eq!(second, Some("http://service2".to_string()));
    }

    #[test]
    fn test_token_bucket_limiter() {
        let limiter = TokenBucketLimiter::new(10, 1);

        let rt = tokio::runtime::Runtime::new().unwrap();

        // Should allow initial requests
        for _ in 0..10 {
            assert!(rt.block_on(limiter.allow()));
        }

        // Should deny additional requests
        assert!(!rt.block_on(limiter.allow()));
    }

    #[test]
    fn test_circuit_breaker() {
        let breaker = CircuitBreaker::new(3, std::time::Duration::from_secs(60));

        let rt = tokio::runtime::Runtime::new().unwrap();

        // Initially closed
        assert_eq!(rt.block_on(breaker.get_state()), CircuitState::Closed);
        assert!(rt.block_on(breaker.allow_request()));

        // Record failures
        for _ in 0..3 {
            rt.block_on(breaker.record_failure());
        }

        // Should be open
        assert_eq!(rt.block_on(breaker.get_state()), CircuitState::Open);
        assert!(!rt.block_on(breaker.allow_request()));

        // Record success to close
        rt.block_on(breaker.record_success());
        assert_eq!(rt.block_on(breaker.get_state()), CircuitState::Closed);
        assert!(rt.block_on(breaker.allow_request()));
    }

    #[test]
    fn test_service_discovery() {
        let discovery = ServiceDiscovery::new();

        let instance = ServiceInstance {
            id: "test-1".to_string(),
            name: "test-service".to_string(),
            address: "localhost".to_string(),
            port: 8080,
            metadata: HashMap::new(),
        };

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(discovery.register_service(instance)).unwrap();

        let services = rt.block_on(discovery.discover_service("test-service")).unwrap();
        assert_eq!(services.len(), 1);
        assert_eq!(services[0].name, "test-service");
    }

    #[test]
    fn test_route_config() {
        let route = RouteConfig {
            path: "/api/users".to_string(),
            service: "user-service".to_string(),
            methods: vec!["GET".to_string(), "POST".to_string()],
            rate_limiter: Some("default".to_string()),
            middlewares: vec!["auth".to_string(), "logging".to_string()],
        };

        assert_eq!(route.path, "/api/users");
        assert_eq!(route.service, "user-service");
        assert_eq!(route.methods.len(), 2);
    }

    #[test]
    fn test_service_config() {
        let service = ServiceConfig {
            name: "user-service".to_string(),
            backends: vec!["http://localhost:8080".to_string()],
            load_balancer: Some("round-robin".to_string()),
            health_check: None,
            timeout: Some(std::time::Duration::from_secs(30)),
        };

        assert_eq!(service.name, "user-service");
        assert_eq!(service.backends.len(), 1);
    }
}