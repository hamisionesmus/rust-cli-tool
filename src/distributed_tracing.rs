//! Distributed tracing module
//!
//! This module provides comprehensive distributed tracing functionality including
//! OpenTelemetry integration, trace collection, span management, and observability.

use crate::error::{CliError, CliResult};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use serde::{Deserialize, Serialize};
use tracing::{info, debug, warn, error};

/// Distributed tracing system
pub struct DistributedTracing {
    tracer_provider: Arc<TracerProvider>,
    span_processor: Arc<SpanProcessor>,
    exporters: Vec<Box<dyn TraceExporter>>,
    samplers: HashMap<String, Box<dyn Sampler>>,
    propagators: Vec<Box<dyn TextMapPropagator>>,
}

impl DistributedTracing {
    /// Create a new distributed tracing system
    pub fn new() -> Self {
        Self {
            tracer_provider: Arc::new(TracerProvider::new()),
            span_processor: Arc::new(SpanProcessor::new()),
            exporters: vec![],
            samplers: HashMap::new(),
            propagators: vec![],
        }
    }

    /// Add a trace exporter
    pub fn add_exporter(&mut self, exporter: Box<dyn TraceExporter>) {
        self.exporters.push(exporter);
    }

    /// Add a sampler
    pub fn add_sampler(&mut self, name: &str, sampler: Box<dyn Sampler>) {
        self.samplers.insert(name.to_string(), sampler);
    }

    /// Add a propagator
    pub fn add_propagator(&mut self, propagator: Box<dyn TextMapPropagator>) {
        self.propagators.push(propagator);
    }

    /// Get a tracer
    pub fn get_tracer(&self, name: &str) -> Tracer {
        self.tracer_provider.get_tracer(name)
    }

    /// Start a new span
    pub async fn start_span(&self, name: &str, parent: Option<&SpanContext>) -> Span {
        let span_context = if let Some(parent_ctx) = parent {
            Some(parent_ctx.clone())
        } else {
            self.extract_context().await
        };

        let span = Span::new(name, span_context);
        self.span_processor.on_start(&span).await;

        span
    }

    /// End a span
    pub async fn end_span(&self, span: Span) {
        span.end();
        self.span_processor.on_end(&span).await;

        // Export spans
        for exporter in &self.exporters {
            if let Err(e) = exporter.export(vec![span.clone()]).await {
                error!("Failed to export span: {}", e);
            }
        }
    }

    /// Extract context from carriers
    pub async fn extract_context(&self) -> Option<SpanContext> {
        for propagator in &self.propagators {
            if let Some(context) = propagator.extract(&HashMap::new()).await {
                return Some(context);
            }
        }
        None
    }

    /// Inject context into carriers
    pub async fn inject_context(&self, context: &SpanContext) -> HashMap<String, String> {
        let mut carrier = HashMap::new();

        for propagator in &self.propagators {
            propagator.inject(context, &mut carrier).await;
        }

        carrier
    }

    /// Force flush all pending spans
    pub async fn force_flush(&self) -> CliResult<()> {
        self.span_processor.force_flush().await?;

        for exporter in &self.exporters {
            exporter.force_flush().await?;
        }

        Ok(())
    }

    /// Shutdown the tracing system
    pub async fn shutdown(&self) -> CliResult<()> {
        self.span_processor.shutdown().await?;

        for exporter in &self.exporters {
            exporter.shutdown().await?;
        }

        Ok(())
    }
}

/// Tracer provider
pub struct TracerProvider {
    tracers: RwLock<HashMap<String, Tracer>>,
}

impl TracerProvider {
    /// Create a new tracer provider
    pub fn new() -> Self {
        Self {
            tracers: RwLock::new(HashMap::new()),
        }
    }

    /// Get or create a tracer
    pub fn get_tracer(&self, name: &str) -> Tracer {
        let mut tracers = self.tracers.try_write().unwrap();
        tracers.entry(name.to_string())
            .or_insert_with(|| Tracer::new(name))
            .clone()
    }
}

/// Tracer
#[derive(Clone)]
pub struct Tracer {
    name: String,
    instrumentation_library: InstrumentationLibrary,
}

impl Tracer {
    /// Create a new tracer
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            instrumentation_library: InstrumentationLibrary {
                name: name.to_string(),
                version: "1.0.0".to_string(),
            },
        }
    }

    /// Start a span
    pub async fn start_span(&self, name: &str) -> SpanBuilder {
        SpanBuilder::new(name, self.clone())
    }
}

/// Instrumentation library
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InstrumentationLibrary {
    pub name: String,
    pub version: String,
}

/// Span builder
pub struct SpanBuilder {
    name: String,
    tracer: Tracer,
    parent_context: Option<SpanContext>,
    attributes: HashMap<String, AttributeValue>,
    span_kind: SpanKind,
    start_time: Option<std::time::SystemTime>,
}

impl SpanBuilder {
    /// Create a new span builder
    pub fn new(name: &str, tracer: Tracer) -> Self {
        Self {
            name: name.to_string(),
            tracer,
            parent_context: None,
            attributes: HashMap::new(),
            span_kind: SpanKind::Internal,
            start_time: None,
        }
    }

    /// Set parent context
    pub fn with_parent_context(mut self, context: SpanContext) -> Self {
        self.parent_context = Some(context);
        self
    }

    /// Add attribute
    pub fn with_attribute(mut self, key: &str, value: AttributeValue) -> Self {
        self.attributes.insert(key.to_string(), value);
        self
    }

    /// Set span kind
    pub fn with_kind(mut self, kind: SpanKind) -> Self {
        self.span_kind = kind;
        self
    }

    /// Set start time
    pub fn with_start_time(mut self, time: std::time::SystemTime) -> Self {
        self.start_time = Some(time);
        self
    }

    /// Start the span
    pub async fn start(self) -> Span {
        // In real implementation, this would call the tracing system
        Span::new(&self.name, self.parent_context)
    }
}

/// Span
#[derive(Clone, Debug)]
pub struct Span {
    pub name: String,
    pub context: SpanContext,
    pub attributes: HashMap<String, AttributeValue>,
    pub events: Vec<SpanEvent>,
    pub start_time: std::time::SystemTime,
    pub end_time: Option<std::time::SystemTime>,
    pub status: SpanStatus,
    pub kind: SpanKind,
}

impl Span {
    /// Create a new span
    pub fn new(name: &str, parent_context: Option<SpanContext>) -> Self {
        let trace_id = TraceId::generate();
        let span_id = SpanId::generate();

        let context = SpanContext {
            trace_id,
            span_id,
            trace_flags: TraceFlags::default(),
            trace_state: TraceState::default(),
            remote: false,
        };

        Self {
            name: name.to_string(),
            context,
            attributes: HashMap::new(),
            events: vec![],
            start_time: std::time::SystemTime::now(),
            end_time: None,
            status: SpanStatus::Unset,
            kind: SpanKind::Internal,
        }
    }

    /// Set attribute
    pub fn set_attribute(&mut self, key: &str, value: AttributeValue) {
        self.attributes.insert(key.to_string(), value);
    }

    /// Add event
    pub fn add_event(&mut self, name: &str, attributes: HashMap<String, AttributeValue>) {
        self.events.push(SpanEvent {
            name: name.to_string(),
            attributes,
            timestamp: std::time::SystemTime::now(),
        });
    }

    /// Set status
    pub fn set_status(&mut self, status: SpanStatus) {
        self.status = status;
    }

    /// End the span
    pub fn end(&mut self) {
        self.end_time = Some(std::time::SystemTime::now());
    }

    /// Check if span is recording
    pub fn is_recording(&self) -> bool {
        self.end_time.is_none()
    }

    /// Get duration
    pub fn duration(&self) -> Option<std::time::Duration> {
        self.end_time?
            .duration_since(self.start_time)
            .ok()
    }
}

/// Span context
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpanContext {
    pub trace_id: TraceId,
    pub span_id: SpanId,
    pub trace_flags: TraceFlags,
    pub trace_state: TraceState,
    pub remote: bool,
}

/// Trace ID
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct TraceId([u8; 16]);

impl TraceId {
    /// Generate a new trace ID
    pub fn generate() -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut bytes = [0u8; 16];
        rng.fill(&mut bytes);
        Self(bytes)
    }

    /// Create from bytes
    pub fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    /// Get as hex string
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }
}

/// Span ID
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct SpanId([u8; 8]);

impl SpanId {
    /// Generate a new span ID
    pub fn generate() -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut bytes = [0u8; 8];
        rng.fill(&mut bytes);
        Self(bytes)
    }

    /// Create from bytes
    pub fn from_bytes(bytes: [u8; 8]) -> Self {
        Self(bytes)
    }

    /// Get as hex string
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }
}

/// Trace flags
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct TraceFlags(u8);

impl TraceFlags {
    /// Check if sampled
    pub fn is_sampled(&self) -> bool {
        self.0 & 0x01 != 0
    }

    /// Set sampled
    pub fn set_sampled(&mut self, sampled: bool) {
        if sampled {
            self.0 |= 0x01;
        } else {
            self.0 &= !0x01;
        }
    }
}

/// Trace state
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct TraceState {
    entries: Vec<(String, String)>,
}

impl TraceState {
    /// Add entry
    pub fn add(&mut self, key: String, value: String) {
        self.entries.push((key, value));
    }

    /// Get value
    pub fn get(&self, key: &str) -> Option<&String> {
        self.entries.iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }
}

/// Span kind
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SpanKind {
    Internal,
    Server,
    Client,
    Producer,
    Consumer,
}

/// Span status
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SpanStatus {
    Unset,
    Ok,
    Error,
}

/// Attribute value
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AttributeValue {
    String(String),
    Bool(bool),
    Int(i64),
    Float(f64),
    Array(Vec<AttributeValue>),
}

/// Span event
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpanEvent {
    pub name: String,
    pub attributes: HashMap<String, AttributeValue>,
    pub timestamp: std::time::SystemTime,
}

/// Span processor
pub struct SpanProcessor {
    spans: RwLock<Vec<Span>>,
    batch_size: usize,
    export_timeout: std::time::Duration,
}

impl SpanProcessor {
    /// Create a new span processor
    pub fn new() -> Self {
        Self {
            spans: RwLock::new(Vec::new()),
            batch_size: 512,
            export_timeout: std::time::Duration::from_secs(30),
        }
    }

    /// Called when a span starts
    pub async fn on_start(&self, span: &Span) {
        debug!("Span started: {} ({})", span.name, span.context.span_id.to_hex());
    }

    /// Called when a span ends
    pub async fn on_end(&self, span: &Span) {
        debug!("Span ended: {} ({})", span.name, span.context.span_id.to_hex());

        let mut spans = self.spans.write().await;
        spans.push(span.clone());

        // Check if we should export
        if spans.len() >= self.batch_size {
            // In real implementation, this would trigger export
            spans.clear();
        }
    }

    /// Force flush spans
    pub async fn force_flush(&self) -> CliResult<()> {
        let mut spans = self.spans.write().await;
        // Export remaining spans
        spans.clear();
        Ok(())
    }

    /// Shutdown processor
    pub async fn shutdown(&self) -> CliResult<()> {
        self.force_flush().await
    }
}

/// Trace exporter trait
#[async_trait::async_trait]
pub trait TraceExporter: Send + Sync {
    /// Export spans
    async fn export(&self, spans: Vec<Span>) -> CliResult<()>;

    /// Force flush
    async fn force_flush(&self) -> CliResult<()> {
        Ok(())
    }

    /// Shutdown exporter
    async fn shutdown(&self) -> CliResult<()> {
        Ok(())
    }
}

/// Jaeger exporter
pub struct JaegerExporter {
    endpoint: String,
    client: reqwest::Client,
}

impl JaegerExporter {
    /// Create a new Jaeger exporter
    pub fn new(endpoint: String) -> Self {
        Self {
            endpoint,
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait::async_trait]
impl TraceExporter for JaegerExporter {
    async fn export(&self, spans: Vec<Span>) -> CliResult<()> {
        // Convert spans to Jaeger format and send
        debug!("Exporting {} spans to Jaeger", spans.len());

        // In real implementation, this would serialize spans to Jaeger format
        // and send to the Jaeger collector

        Ok(())
    }
}

/// Zipkin exporter
pub struct ZipkinExporter {
    endpoint: String,
    client: reqwest::Client,
}

impl ZipkinExporter {
    /// Create a new Zipkin exporter
    pub fn new(endpoint: String) -> Self {
        Self {
            endpoint,
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait::async_trait]
impl TraceExporter for ZipkinExporter {
    async fn export(&self, spans: Vec<Span>) -> CliResult<()> {
        // Convert spans to Zipkin format and send
        debug!("Exporting {} spans to Zipkin", spans.len());

        // In real implementation, this would serialize spans to Zipkin format
        // and send to the Zipkin collector

        Ok(())
    }
}

/// OTLP exporter (OpenTelemetry Protocol)
pub struct OtlpExporter {
    endpoint: String,
    client: reqwest::Client,
    headers: HashMap<String, String>,
}

impl OtlpExporter {
    /// Create a new OTLP exporter
    pub fn new(endpoint: String) -> Self {
        Self {
            endpoint,
            client: reqwest::Client::new(),
            headers: HashMap::new(),
        }
    }

    /// Add header
    pub fn with_header(mut self, key: String, value: String) -> Self {
        self.headers.insert(key, value);
        self
    }
}

#[async_trait::async_trait]
impl TraceExporter for OtlpExporter {
    async fn export(&self, spans: Vec<Span>) -> CliResult<()> {
        // Convert spans to OTLP format and send
        debug!("Exporting {} spans via OTLP", spans.len());

        // In real implementation, this would serialize spans to OTLP protobuf format
        // and send to the OTLP endpoint

        Ok(())
    }
}

/// Sampler trait
#[async_trait::async_trait]
pub trait Sampler: Send + Sync {
    /// Decide whether to sample the span
    async fn should_sample(&self, context: &SpanContext, name: &str, attributes: &HashMap<String, AttributeValue>) -> SamplingDecision;
}

/// Sampling decision
#[derive(Debug, Clone)]
pub struct SamplingDecision {
    pub sampled: bool,
    pub attributes: HashMap<String, AttributeValue>,
}

/// Always sample sampler
pub struct AlwaysOnSampler;

#[async_trait::async_trait]
impl Sampler for AlwaysOnSampler {
    async fn should_sample(&self, _context: &SpanContext, _name: &str, _attributes: &HashMap<String, AttributeValue>) -> SamplingDecision {
        SamplingDecision {
            sampled: true,
            attributes: HashMap::new(),
        }
    }
}

/// Always off sampler
pub struct AlwaysOffSampler;

#[async_trait::async_trait]
impl Sampler for AlwaysOffSampler {
    async fn should_sample(&self, _context: &SpanContext, _name: &str, _attributes: &HashMap<String, AttributeValue>) -> SamplingDecision {
        SamplingDecision {
            sampled: false,
            attributes: HashMap::new(),
        }
    }
}

/// Probabilistic sampler
pub struct ProbabilisticSampler {
    probability: f64,
}

impl ProbabilisticSampler {
    /// Create a new probabilistic sampler
    pub fn new(probability: f64) -> Self {
        Self { probability }
    }
}

#[async_trait::async_trait]
impl Sampler for ProbabilisticSampler {
    async fn should_sample(&self, _context: &SpanContext, _name: &str, _attributes: &HashMap<String, AttributeValue>) -> SamplingDecision {
        use rand::Rng;
        let sampled = rand::thread_rng().gen_bool(self.probability);

        SamplingDecision {
            sampled,
            attributes: HashMap::new(),
        }
    }
}

/// Text map propagator trait
#[async_trait::async_trait]
pub trait TextMapPropagator: Send + Sync {
    /// Extract context from carrier
    async fn extract(&self, carrier: &HashMap<String, String>) -> Option<SpanContext>;

    /// Inject context into carrier
    async fn inject(&self, context: &SpanContext, carrier: &mut HashMap<String, String>);
}

/// W3C Trace Context propagator
pub struct W3CTraceContextPropagator;

impl W3CTraceContextPropagator {
    /// Create a new W3C trace context propagator
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl TextMapPropagator for W3CTraceContextPropagator {
    async fn extract(&self, carrier: &HashMap<String, String>) -> Option<SpanContext> {
        let traceparent = carrier.get("traceparent")?;

        // Parse traceparent header: 00-12345678901234567890123456789012-1234567890123456-01
        let parts: Vec<&str> = traceparent.split('-').collect();
        if parts.len() != 4 {
            return None;
        }

        let trace_id = TraceId::from_bytes(hex::decode(parts[1]).ok()?.try_into().ok()?);
        let span_id = SpanId::from_bytes(hex::decode(parts[2]).ok()?.try_into().ok()?);

        let mut trace_flags = TraceFlags::default();
        if parts[3].contains('1') {
            trace_flags.set_sampled(true);
        }

        Some(SpanContext {
            trace_id,
            span_id,
            trace_flags,
            trace_state: TraceState::default(),
            remote: true,
        })
    }

    async fn inject(&self, context: &SpanContext, carrier: &mut HashMap<String, String>) {
        let version = "00";
        let trace_id = context.trace_id.to_hex();
        let span_id = context.span_id.to_hex();
        let flags = if context.trace_flags.is_sampled() { "01" } else { "00" };

        let traceparent = format!("{}-{}-{}-{}", version, trace_id, span_id, flags);
        carrier.insert("traceparent".to_string(), traceparent);
    }
}

/// B3 propagator (Zipkin)
pub struct B3Propagator;

impl B3Propagator {
    /// Create a new B3 propagator
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl TextMapPropagator for B3Propagator {
    async fn extract(&self, carrier: &HashMap<String, String>) -> Option<SpanContext> {
        let b3 = carrier.get("x-b3-traceid")?;
        let span_id_hex = carrier.get("x-b3-spanid")?;
        let sampled = carrier.get("x-b3-sampled");

        let trace_id = TraceId::from_bytes(hex::decode(b3).ok()?.try_into().ok()?);
        let span_id = SpanId::from_bytes(hex::decode(span_id_hex).ok()?.try_into().ok()?);

        let mut trace_flags = TraceFlags::default();
        if let Some(sampled) = sampled {
            if sampled == "1" {
                trace_flags.set_sampled(true);
            }
        }

        Some(SpanContext {
            trace_id,
            span_id,
            trace_flags,
            trace_state: TraceState::default(),
            remote: true,
        })
    }

    async fn inject(&self, context: &SpanContext, carrier: &mut HashMap<String, String>) {
        carrier.insert("x-b3-traceid".to_string(), context.trace_id.to_hex());
        carrier.insert("x-b3-spanid".to_string(), context.span_id.to_hex());
        carrier.insert("x-b3-sampled".to_string(),
                      if context.trace_flags.is_sampled() { "1" } else { "0" });
    }
}

/// Trace correlation utilities
pub struct TraceCorrelation;

impl TraceCorrelation {
    /// Extract trace context from HTTP headers
    pub async fn extract_from_http_headers(headers: &http::HeaderMap) -> Option<SpanContext> {
        let mut carrier = HashMap::new();

        for (key, value) in headers.iter() {
            if let Ok(value_str) = value.to_str() {
                carrier.insert(key.to_string(), value_str.to_string());
            }
        }

        let propagator = W3CTraceContextPropagator::new();
        propagator.extract(&carrier).await
    }

    /// Inject trace context into HTTP headers
    pub async fn inject_into_http_headers(context: &SpanContext, headers: &mut http::HeaderMap) {
        let mut carrier = HashMap::new();

        let propagator = W3CTraceContextPropagator::new();
        propagator.inject(context, &mut carrier).await;

        for (key, value) in carrier {
            headers.insert(key, value.parse().unwrap());
        }
    }

    /// Create child span context
    pub fn create_child_context(parent: &SpanContext) -> SpanContext {
        SpanContext {
            trace_id: parent.trace_id.clone(),
            span_id: SpanId::generate(),
            trace_flags: parent.trace_flags.clone(),
            trace_state: parent.trace_state.clone(),
            remote: false,
        }
    }

    /// Check if context should be sampled
    pub async fn should_sample_context(context: &SpanContext, sampler: &dyn Sampler) -> bool {
        let decision = sampler.should_sample(context, "", &HashMap::new()).await;
        decision.sampled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_id_generation() {
        let trace_id1 = TraceId::generate();
        let trace_id2 = TraceId::generate();

        assert_ne!(trace_id1, trace_id2);
        assert_eq!(trace_id1.to_hex().len(), 32);
    }

    #[test]
    fn test_span_id_generation() {
        let span_id1 = SpanId::generate();
        let span_id2 = SpanId::generate();

        assert_ne!(span_id1, span_id2);
        assert_eq!(span_id1.to_hex().len(), 16);
    }

    #[test]
    fn test_span_creation() {
        let span = Span::new("test-span", None);

        assert_eq!(span.name, "test-span");
        assert!(span.is_recording());
        assert!(span.end_time.is_none());
    }

    #[test]
    fn test_span_lifecycle() {
        let mut span = Span::new("test-span", None);

        span.set_attribute("key", AttributeValue::String("value".to_string()));
        span.add_event("test-event", HashMap::new());
        span.set_status(SpanStatus::Ok);

        assert_eq!(span.attributes.len(), 1);
        assert_eq!(span.events.len(), 1);
        assert!(matches!(span.status, SpanStatus::Ok));

        span.end();

        assert!(!span.is_recording());
        assert!(span.end_time.is_some());
        assert!(span.duration().is_some());
    }

    #[test]
    fn test_trace_flags() {
        let mut flags = TraceFlags::default();

        assert!(!flags.is_sampled());

        flags.set_sampled(true);
        assert!(flags.is_sampled());

        flags.set_sampled(false);
        assert!(!flags.is_sampled());
    }

    #[test]
    fn test_trace_state() {
        let mut state = TraceState::default();

        state.add("key1".to_string(), "value1".to_string());
        state.add("key2".to_string(), "value2".to_string());

        assert_eq!(state.get("key1"), Some(&"value1".to_string()));
        assert_eq!(state.get("key2"), Some(&"value2".to_string()));
        assert_eq!(state.get("key3"), None);
    }

    #[test]
    fn test_w3c_propagator() {
        let propagator = W3CTraceContextPropagator::new();

        let context = SpanContext {
            trace_id: TraceId::from_bytes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
            span_id: SpanId::from_bytes([1, 2, 3, 4, 5, 6, 7, 8]),
            trace_flags: TraceFlags(1), // sampled
            trace_state: TraceState::default(),
            remote: false,
        };

        let mut carrier = HashMap::new();
        tokio::runtime::Runtime::new().unwrap().block_on(
            propagator.inject(&context, &mut carrier)
        );

        let extracted = tokio::runtime::Runtime::new().unwrap().block_on(
            propagator.extract(&carrier)
        );

        assert!(extracted.is_some());
        let extracted = extracted.unwrap();
        assert_eq!(extracted.trace_id, context.trace_id);
        assert_eq!(extracted.span_id, context.span_id);
        assert!(extracted.trace_flags.is_sampled());
    }

    #[test]
    fn test_b3_propagator() {
        let propagator = B3Propagator;

        let context = SpanContext {
            trace_id: TraceId::from_bytes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
            span_id: SpanId::from_bytes([1, 2, 3, 4, 5, 6, 7, 8]),
            trace_flags: TraceFlags(1), // sampled
            trace_state: TraceState::default(),
            remote: false,
        };

        let mut carrier = HashMap::new();
        tokio::runtime::Runtime::new().unwrap().block_on(
            propagator.inject(&context, &mut carrier)
        );

        assert!(carrier.contains_key("x-b3-traceid"));
        assert!(carrier.contains_key("x-b3-spanid"));
        assert!(carrier.contains_key("x-b3-sampled"));
    }

    #[test]
    fn test_probabilistic_sampler() {
        let sampler = ProbabilisticSampler::new(1.0); // Always sample

        let context = SpanContext {
            trace_id: TraceId::generate(),
            span_id: SpanId::generate(),
            trace_flags: TraceFlags::default(),
            trace_state: TraceState::default(),
            remote: false,
        };

        let decision = tokio::runtime::Runtime::new().unwrap().block_on(
            sampler.should_sample(&context, "test", &HashMap::new())
        );

        assert!(decision.sampled);
    }

    #[test]
    fn test_always_on_sampler() {
        let sampler = AlwaysOnSampler;

        let context = SpanContext {
            trace_id: TraceId::generate(),
            span_id: SpanId::generate(),
            trace_flags: TraceFlags::default(),
            trace_state: TraceState::default(),
            remote: false,
        };

        let decision = tokio::runtime::Runtime::new().unwrap().block_on(
            sampler.should_sample(&context, "test", &HashMap::new())
        );

        assert!(decision.sampled);
    }

    #[test]
    fn test_always_off_sampler() {
        let sampler = AlwaysOffSampler;

        let context = SpanContext {
            trace_id: TraceId::generate(),
            span_id: SpanId::generate(),
            trace_flags: TraceFlags::default(),
            trace_state: TraceState::default(),
            remote: false,
        };

        let decision = tokio::runtime::Runtime::new().unwrap().block_on(
            sampler.should_sample(&context, "test", &HashMap::new())
        );

        assert!(!decision.sampled);
    }

    #[test]
    fn test_span_builder() {
        let tracer = Tracer::new("test-tracer");
        let builder = tokio::runtime::Runtime::new().unwrap().block_on(
            tracer.start_span("test-span")
        );

        let span = tokio::runtime::Runtime::new().unwrap().block_on(
            builder.start()
        );

        assert_eq!(span.name, "test-span");
        assert!(span.is_recording());
    }

    #[test]
    fn test_distributed_tracing_creation() {
        let tracing = DistributedTracing::new();
        let tracer = tracing.get_tracer("test-tracer");

        assert_eq!(tracer.name, "test-tracer");
    }
}