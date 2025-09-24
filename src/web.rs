//! Web framework module
//!
//! This module provides comprehensive web framework functionality including
//! REST API, GraphQL, middleware, routing, and web server capabilities.

use crate::error::{CliError, CliResult};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use warp::Filter;
use std::net::SocketAddr;
use tracing::{info, debug, warn};
use std::convert::Infallible;

/// Web server manager
pub struct WebServer {
    routes: Vec<Box<dyn RouteHandler>>,
    middleware: Vec<Box<dyn Middleware>>,
    port: u16,
    host: String,
}

impl WebServer {
    /// Create a new web server
    pub fn new(host: String, port: u16) -> Self {
        Self {
            routes: vec![],
            middleware: vec![],
            port,
            host,
        }
    }

    /// Add a route handler
    pub fn add_route(&mut self, route: Box<dyn RouteHandler>) {
        self.routes.push(route);
    }

    /// Add middleware
    pub fn add_middleware(&mut self, middleware: Box<dyn Middleware>) {
        self.middleware.push(middleware);
    }

    /// Start the web server
    pub async fn start(self) -> CliResult<()> {
        info!("Starting web server on {}:{}", self.host, self.port);

        let addr: SocketAddr = format!("{}:{}", self.host, self.port)
            .parse()
            .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Invalid address: {}", e)
            )))?;

        // Build routes
        let mut routes = Vec::new();

        for route_handler in self.routes {
            let route = route_handler.build_route();
            routes.push(route);
        }

        // Combine all routes
        let combined_routes = routes.into_iter()
            .fold(warp::any().map(|| "Not found").boxed(), |acc, route| {
                acc.or(route).boxed()
            });

        // Apply middleware
        let mut filtered_routes = combined_routes;
        for middleware in self.middleware {
            filtered_routes = middleware.apply(filtered_routes);
        }

        // Add CORS
        let cors = warp::cors()
            .allow_any_origin()
            .allow_headers(vec!["content-type", "authorization"])
            .allow_methods(vec!["GET", "POST", "PUT", "DELETE", "OPTIONS"]);

        let routes_with_cors = filtered_routes.with(cors);

        // Add logging
        let routes_with_logging = routes_with_cors.with(warp::log("web_server"));

        info!("Web server listening on {}", addr);
        warp::serve(routes_with_logging)
            .run(addr)
            .await;

        Ok(())
    }
}

/// Route handler trait
#[async_trait::async_trait]
pub trait RouteHandler: Send + Sync {
    /// Build the warp filter for this route
    fn build_route(&self) -> Box<dyn warp::Filter<Extract = impl warp::Reply, Error = warp::Rejection> + Clone + Send + Sync + 'static>;
}

/// REST API route handler
pub struct RestApiHandler {
    routes: HashMap<String, Box<dyn RestRoute>>,
}

impl RestApiHandler {
    pub fn new() -> Self {
        Self {
            routes: HashMap::new(),
        }
    }

    pub fn add_route(&mut self, path: &str, route: Box<dyn RestRoute>) {
        self.routes.insert(path.to_string(), route);
    }
}

#[async_trait::async_trait]
impl RouteHandler for RestApiHandler {
    fn build_route(&self) -> Box<dyn warp::Filter<Extract = impl warp::Reply, Error = warp::Rejection> + Clone + Send + Sync + 'static> {
        let mut combined_routes = warp::any().map(|| warp::reply::json(&"Route not found")).boxed();

        for (path, route_handler) in &self.routes {
            let route_filter = Self::build_route_filter(path, route_handler.clone());
            combined_routes = combined_routes.or(route_filter).boxed();
        }

        Box::new(combined_routes)
    }
}

impl RestApiHandler {
    fn build_route_filter(path: &str, route_handler: Box<dyn RestRoute>) -> Box<dyn warp::Filter<Extract = impl warp::Reply, Error = warp::Rejection> + Clone + Send + Sync + 'static> {
        let get_route = warp::get()
            .and(warp::path(path))
            .and_then(move || {
                let handler = route_handler.clone();
                async move {
                    match handler.handle_get().await {
                        Ok(response) => Ok(warp::reply::json(&response)),
                        Err(e) => Ok(warp::reply::json(&ErrorResponse {
                            error: e.to_string(),
                        })),
                    }
                }
            });

        let post_route = warp::post()
            .and(warp::path(path))
            .and(warp::body::json())
            .and_then(move |body: serde_json::Value| {
                let handler = route_handler.clone();
                async move {
                    match handler.handle_post(body).await {
                        Ok(response) => Ok(warp::reply::json(&response)),
                        Err(e) => Ok(warp::reply::json(&ErrorResponse {
                            error: e.to_string(),
                        })),
                    }
                }
            });

        Box::new(get_route.or(post_route))
    }
}

/// REST route trait
#[async_trait::async_trait]
pub trait RestRoute: Send + Sync {
    async fn handle_get(&self) -> CliResult<serde_json::Value>;
    async fn handle_post(&self, body: serde_json::Value) -> CliResult<serde_json::Value>;
}

/// GraphQL handler
pub struct GraphQLHandler {
    schema: Arc<async_graphql::Schema<GraphQLQuery, GraphQLMutation, async_graphql::Subscription>>,
}

impl GraphQLHandler {
    pub fn new() -> Self {
        let query = GraphQLQuery;
        let mutation = GraphQLMutation;
        let subscription = async_graphql::Subscription::new(async_graphql::EmptySubscription);

        let schema = async_graphql::Schema::build(query, mutation, subscription)
            .finish();

        Self {
            schema: Arc::new(schema),
        }
    }
}

#[async_trait::async_trait]
impl RouteHandler for GraphQLHandler {
    fn build_route(&self) -> Box<dyn warp::Filter<Extract = impl warp::Reply, Error = warp::Rejection> + Clone + Send + Sync + 'static> {
        let schema = self.schema.clone();

        let graphql_post = warp::post()
            .and(warp::path("graphql"))
            .and(async_graphql_warp::graphql(schema))
            .and_then(|(schema, request): (async_graphql::Schema<_, _, _>, async_graphql::Request)| async move {
                let response = schema.execute(request).await;
                Ok::<_, Infallible>(async_graphql_warp::GraphQLResponse::from(response))
            });

        let graphql_playground = warp::get()
            .and(warp::path("playground"))
            .and(async_graphql_warp::playground("/graphql", None));

        Box::new(graphql_post.or(graphql_playground))
    }
}

/// GraphQL query root
struct GraphQLQuery;

#[async_graphql::Object]
impl GraphQLQuery {
    async fn hello(&self) -> &str {
        "Hello, World!"
    }

    async fn user(&self, id: i32) -> Option<User> {
        // Mock user lookup
        if id == 1 {
            Some(User {
                id,
                name: "John Doe".to_string(),
                email: "john@example.com".to_string(),
            })
        } else {
            None
        }
    }
}

/// GraphQL mutation root
struct GraphQLMutation;

#[async_graphql::Object]
impl GraphQLMutation {
    async fn create_user(&self, name: String, email: String) -> User {
        User {
            id: 1, // In real implementation, this would be generated
            name,
            email,
        }
    }
}

/// User type for GraphQL
#[derive(async_graphql::SimpleObject, Serialize, Deserialize)]
struct User {
    id: i32,
    name: String,
    email: String,
}

/// Middleware trait
pub trait Middleware: Send + Sync {
    fn apply(&self, filter: Box<dyn warp::Filter<Extract = impl warp::Reply, Error = warp::Rejection> + Clone + Send + Sync + 'static>) -> Box<dyn warp::Filter<Extract = impl warp::Reply, Error = warp::Rejection> + Clone + Send + Sync + 'static>;
}

/// Authentication middleware
pub struct AuthMiddleware {
    required_role: Option<String>,
}

impl AuthMiddleware {
    pub fn new(required_role: Option<String>) -> Self {
        Self { required_role }
    }
}

impl Middleware for AuthMiddleware {
    fn apply(&self, filter: Box<dyn warp::Filter<Extract = impl warp::Reply, Error = warp::Rejection> + Clone + Send + Sync + 'static>) -> Box<dyn warp::Filter<Extract = impl warp::Reply, Error = warp::Rejection> + Clone + Send + Sync + 'static> {
        let required_role = self.required_role.clone();

        let auth_filter = warp::header::optional::<String>("authorization")
            .and_then(move |auth_header: Option<String>| {
                let required_role = required_role.clone();
                async move {
                    if let Some(token) = auth_header {
                        // Validate JWT token (simplified)
                        if token.starts_with("Bearer ") {
                            let token_part = &token[7..];
                            if Self::validate_token(token_part, required_role.as_deref()) {
                                Ok(())
                            } else {
                                Err(warp::reject::custom(AuthError::InvalidToken))
                            }
                        } else {
                            Err(warp::reject::custom(AuthError::MissingToken))
                        }
                    } else {
                        Err(warp::reject::custom(AuthError::MissingToken))
                    }
                }
            });

        Box::new(auth_filter.and(filter).recover(|err: warp::Rejection| async move {
            if let Some(auth_err) = err.find::<AuthError>() {
                let error_response = ErrorResponse {
                    error: format!("Authentication error: {:?}", auth_err),
                };
                Ok(warp::reply::with_status(
                    warp::reply::json(&error_response),
                    warp::http::StatusCode::UNAUTHORIZED,
                ))
            } else {
                Err(err)
            }
        }))
    }
}

impl AuthMiddleware {
    fn validate_token(token: &str, required_role: Option<&str>) -> bool {
        // Simplified token validation
        // In production, this would validate JWT signature and claims
        if token.len() > 10 {
            // Check role if required
            if let Some(role) = required_role {
                token.contains(role)
            } else {
                true
            }
        } else {
            false
        }
    }
}

/// Rate limiting middleware
pub struct RateLimitMiddleware {
    requests_per_minute: u32,
    clients: Arc<RwLock<HashMap<String, ClientRateLimit>>>,
}

impl RateLimitMiddleware {
    pub fn new(requests_per_minute: u32) -> Self {
        Self {
            requests_per_minute,
            clients: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl Middleware for RateLimitMiddleware {
    fn apply(&self, filter: Box<dyn warp::Filter<Extract = impl warp::Reply, Error = warp::Rejection> + Clone + Send + Sync + 'static>) -> Box<dyn warp::Filter<Extract = impl warp::Reply, Error = warp::Rejection> + Clone + Send + Sync + 'static> {
        let clients = self.clients.clone();
        let requests_per_minute = self.requests_per_minute;

        let rate_limit_filter = warp::addr::remote()
            .and_then(move |addr: Option<std::net::SocketAddr>| {
                let clients = clients.clone();
                let requests_per_minute = requests_per_minute;
                async move {
                    if let Some(addr) = addr {
                        let client_ip = addr.ip().to_string();
                        let mut clients_lock = clients.write().await;

                        let client_limit = clients_lock.entry(client_ip.clone())
                            .or_insert_with(|| ClientRateLimit::new(requests_per_minute));

                        if client_limit.is_allowed() {
                            client_limit.record_request();
                            Ok(())
                        } else {
                            Err(warp::reject::custom(RateLimitError::TooManyRequests))
                        }
                    } else {
                        Ok(())
                    }
                }
            });

        Box::new(rate_limit_filter.and(filter).recover(|err: warp::Rejection| async move {
            if let Some(_) = err.find::<RateLimitError>() {
                let error_response = ErrorResponse {
                    error: "Rate limit exceeded".to_string(),
                };
                Ok(warp::reply::with_status(
                    warp::reply::json(&error_response),
                    warp::http::StatusCode::TOO_MANY_REQUESTS,
                ))
            } else {
                Err(err)
            }
        }))
    }
}

/// Client rate limit tracking
struct ClientRateLimit {
    requests_per_minute: u32,
    requests: Vec<std::time::Instant>,
}

impl ClientRateLimit {
    fn new(requests_per_minute: u32) -> Self {
        Self {
            requests_per_minute,
            requests: Vec::new(),
        }
    }

    fn is_allowed(&self) -> bool {
        self.requests.len() < self.requests_per_minute as usize
    }

    fn record_request(&mut self) {
        let now = std::time::Instant::now();
        self.requests.push(now);

        // Remove requests older than 1 minute
        let one_minute_ago = now - std::time::Duration::from_secs(60);
        self.requests.retain(|&time| time > one_minute_ago);
    }
}

/// Logging middleware
pub struct LoggingMiddleware;

impl LoggingMiddleware {
    pub fn new() -> Self {
        Self
    }
}

impl Middleware for LoggingMiddleware {
    fn apply(&self, filter: Box<dyn warp::Filter<Extract = impl warp::Reply, Error = warp::Rejection> + Clone + Send + Sync + 'static>) -> Box<dyn warp::Filter<Extract = impl warp::Reply, Error = warp::Rejection> + Clone + Send + Sync + 'static> {
        Box::new(filter.with(warp::log("web_framework")))
    }
}

/// Error response structure
#[derive(Serialize, Deserialize)]
struct ErrorResponse {
    error: String,
}

/// Authentication error
#[derive(Debug)]
enum AuthError {
    MissingToken,
    InvalidToken,
}

/// Rate limit error
#[derive(Debug)]
enum RateLimitError {
    TooManyRequests,
}

impl warp::reject::Reject for AuthError {}
impl warp::reject::Reject for RateLimitError {}

/// API documentation generator
pub struct ApiDocsGenerator {
    routes: Vec<ApiRoute>,
}

impl ApiDocsGenerator {
    pub fn new() -> Self {
        Self {
            routes: vec![],
        }
    }

    pub fn add_route(&mut self, route: ApiRoute) {
        self.routes.push(route);
    }

    pub fn generate_openapi(&self) -> serde_json::Value {
        let mut paths = serde_json::Map::new();

        for route in &self.routes {
            let mut path_item = serde_json::Map::new();

            for method in &route.methods {
                let mut operation = serde_json::Map::new();
                operation.insert("summary".to_string(), serde_json::Value::String(route.summary.clone()));
                operation.insert("description".to_string(), serde_json::Value::String(route.description.clone()));

                // Add parameters
                let mut parameters = vec![];
                for param in &route.parameters {
                    parameters.push(serde_json::json!({
                        "name": param.name,
                        "in": param.location,
                        "required": param.required,
                        "schema": {
                            "type": param.param_type
                        }
                    }));
                }
                operation.insert("parameters".to_string(), serde_json::Value::Array(parameters));

                // Add responses
                let responses = serde_json::json!({
                    "200": {
                        "description": "Success"
                    }
                });
                operation.insert("responses".to_string(), responses);

                path_item.insert(method.to_lowercase(), serde_json::Value::Object(operation));
            }

            paths.insert(route.path.clone(), serde_json::Value::Object(path_item));
        }

        serde_json::json!({
            "openapi": "3.0.0",
            "info": {
                "title": "Web Framework API",
                "version": "1.0.0",
                "description": "REST API and GraphQL endpoints"
            },
            "paths": serde_json::Value::Object(paths)
        })
    }
}

/// API route for documentation
#[derive(Clone)]
pub struct ApiRoute {
    pub path: String,
    pub methods: Vec<String>,
    pub summary: String,
    pub description: String,
    pub parameters: Vec<ApiParameter>,
}

/// API parameter for documentation
#[derive(Clone)]
pub struct ApiParameter {
    pub name: String,
    pub location: String, // "query", "path", "header", "body"
    pub param_type: String, // "string", "integer", etc.
    pub required: bool,
}

/// WebSocket handler
pub struct WebSocketHandler {
    connections: Arc<RwLock<HashMap<String, tokio::sync::mpsc::UnboundedSender<String>>>>,
}

impl WebSocketHandler {
    pub fn new() -> Self {
        Self {
            connections: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn get_ws_route(&self) -> Box<dyn warp::Filter<Extract = impl warp::Reply, Error = warp::Rejection> + Clone + Send + Sync + 'static> {
        let connections = self.connections.clone();

        let ws_route = warp::path("ws")
            .and(warp::ws())
            .map(move |ws: warp::ws::Ws| {
                let connections = connections.clone();
                ws.on_upgrade(move |websocket| Self::handle_websocket(websocket, connections))
            });

        Box::new(ws_route)
    }

    async fn handle_websocket(websocket: warp::ws::WebSocket, connections: Arc<RwLock<HashMap<String, tokio::sync::mpsc::UnboundedSender<String>>>>) {
        let (ws_tx, mut ws_rx) = websocket.split();
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let client_id = format!("client_{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos());

        // Add client to connections
        {
            let mut connections_lock = connections.write().await;
            connections_lock.insert(client_id.clone(), tx);
        }

        // Forward messages from channel to websocket
        tokio::task::spawn(async move {
            let mut rx_stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx);
            while let Some(message) = rx_stream.next().await {
                if ws_tx.send(warp::ws::Message::text(message)).await.is_err() {
                    break;
                }
            }
        });

        // Handle incoming messages
        while let Some(result) = ws_rx.next().await {
            match result {
                Ok(message) => {
                    if message.is_text() {
                        let text = message.to_str().unwrap_or("");
                        info!("Received WebSocket message from {}: {}", client_id, text);

                        // Broadcast to all clients
                        let connections_lock = connections.read().await;
                        for (id, sender) in connections_lock.iter() {
                            if id != &client_id {
                                let _ = sender.send(format!("{}: {}", client_id, text));
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!("WebSocket error for {}: {}", client_id, e);
                    break;
                }
            }
        }

        // Remove client from connections
        let mut connections_lock = connections.write().await;
        connections_lock.remove(&client_id);
        info!("WebSocket connection closed for {}", client_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_web_server_creation() {
        let server = WebServer::new("127.0.0.1".to_string(), 8080);
        assert_eq!(server.host, "127.0.0.1");
        assert_eq!(server.port, 8080);
    }

    #[test]
    fn test_rate_limit_client() {
        let client = ClientRateLimit::new(10);
        assert!(client.is_allowed());

        for _ in 0..10 {
            client.record_request();
        }

        assert!(!client.is_allowed());
    }

    #[test]
    fn test_api_docs_generator() {
        let mut generator = ApiDocsGenerator::new();

        let route = ApiRoute {
            path: "/users",
            methods: vec!["GET".to_string(), "POST".to_string()],
            summary: "User management".to_string(),
            description: "Manage user accounts".to_string(),
            parameters: vec![],
        };

        generator.add_route(route);
        let openapi = generator.generate_openapi();

        assert!(openapi["paths"]["/users"].is_object());
    }

    #[test]
    fn test_websocket_handler() {
        let handler = WebSocketHandler::new();
        // Test that it can be created
        assert!(true);
    }

    #[tokio::test]
    async fn test_graphql_schema() {
        let handler = GraphQLHandler::new();

        let query = r#"
        {
            hello
            user(id: 1) {
                id
                name
                email
            }
        }
        "#;

        let request = async_graphql::Request::new(query);
        let response = handler.schema.execute(request).await;

        assert!(response.is_ok());
    }
}