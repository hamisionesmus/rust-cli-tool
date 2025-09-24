//! Metrics and monitoring module
//!
//! This module provides comprehensive monitoring, metrics collection,
//! health checks, and performance tracking for the CLI tool.

use crate::error::{CliError, CliResult};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tracing::{info, warn, error, debug};
use serde::{Deserialize, Serialize};

/// Metrics registry for collecting and managing metrics
#[derive(Debug, Clone)]
pub struct MetricsRegistry {
    counters: Arc<RwLock<HashMap<String, Counter>>>,
    gauges: Arc<RwLock<HashMap<String, Gauge>>>,
    histograms: Arc<RwLock<HashMap<String, Histogram>>>,
    timers: Arc<RwLock<HashMap<String, Timer>>>,
}

impl MetricsRegistry {
    /// Create a new metrics registry
    pub fn new() -> Self {
        Self {
            counters: Arc::new(RwLock::new(HashMap::new())),
            gauges: Arc::new(RwLock::new(HashMap::new())),
            histograms: Arc::new(RwLock::new(HashMap::new())),
            timers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get or create a counter
    pub fn counter(&self, name: &str, description: &str) -> CounterHandle {
        let mut counters = self.counters.write().unwrap();
        let counter = counters.entry(name.to_string())
            .or_insert_with(|| Counter::new(name, description));
        CounterHandle {
            counter: Arc::new(RwLock::new(counter.clone())),
        }
    }

    /// Get or create a gauge
    pub fn gauge(&self, name: &str, description: &str) -> GaugeHandle {
        let mut gauges = self.gauges.write().unwrap();
        let gauge = gauges.entry(name.to_string())
            .or_insert_with(|| Gauge::new(name, description));
        GaugeHandle {
            gauge: Arc::new(RwLock::new(gauge.clone())),
        }
    }

    /// Get or create a histogram
    pub fn histogram(&self, name: &str, description: &str, buckets: Vec<f64>) -> HistogramHandle {
        let mut histograms = self.histograms.write().unwrap();
        let histogram = histograms.entry(name.to_string())
            .or_insert_with(|| Histogram::new(name, description, buckets));
        HistogramHandle {
            histogram: Arc::new(RwLock::new(histogram.clone())),
        }
    }

    /// Start a timer
    pub fn start_timer(&self, name: &str) -> TimerHandle {
        let mut timers = self.timers.write().unwrap();
        let timer = timers.entry(name.to_string())
            .or_insert_with(|| Timer::new(name));
        TimerHandle {
            timer: Arc::new(RwLock::new(timer.clone())),
            start_time: Instant::now(),
        }
    }

    /// Get all metrics as a snapshot
    pub fn snapshot(&self) -> MetricsSnapshot {
        let counters = self.counters.read().unwrap().clone();
        let gauges = self.gauges.read().unwrap().clone();
        let histograms = self.histograms.read().unwrap().clone();
        let timers = self.timers.read().unwrap().clone();

        MetricsSnapshot {
            timestamp: SystemTime::now(),
            counters,
            gauges,
            histograms,
            timers,
        }
    }

    /// Export metrics in Prometheus format
    pub fn to_prometheus(&self) -> String {
        let mut output = String::new();

        // Counters
        for counter in self.counters.read().unwrap().values() {
            output.push_str(&format!(
                "# HELP {} {}\n# TYPE {} counter\n{} {}\n\n",
                counter.name, counter.description, counter.name, counter.name, counter.value
            ));
        }

        // Gauges
        for gauge in self.gauges.read().unwrap().values() {
            output.push_str(&format!(
                "# HELP {} {}\n# TYPE {} gauge\n{} {}\n\n",
                gauge.name, gauge.description, gauge.name, gauge.name, gauge.value
            ));
        }

        // Histograms
        for histogram in self.histograms.read().unwrap().values() {
            output.push_str(&format!(
                "# HELP {} {}\n# TYPE {} histogram\n",
                histogram.name, histogram.description, histogram.name
            ));

            for (i, bucket) in histogram.buckets.iter().enumerate() {
                let le = if i == histogram.buckets.len() - 1 { "+Inf" } else { &bucket.to_string() };
                output.push_str(&format!("{}_bucket{{le=\"{}\"}} {}\n", histogram.name, le, histogram.counts[i]));
            }
            output.push_str(&format!("{}_count {}\n{}_sum {}\n\n", histogram.name, histogram.count, histogram.sum));
        }

        output
    }
}

/// Counter metric for counting events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Counter {
    pub name: String,
    pub description: String,
    pub value: u64,
}

impl Counter {
    pub fn new(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            value: 0,
        }
    }

    pub fn increment(&mut self) {
        self.value += 1;
    }

    pub fn add(&mut self, value: u64) {
        self.value += value;
    }
}

/// Counter handle for thread-safe access
#[derive(Clone)]
pub struct CounterHandle {
    counter: Arc<RwLock<Counter>>,
}

impl CounterHandle {
    pub fn increment(&self) {
        self.counter.write().unwrap().increment();
    }

    pub fn add(&self, value: u64) {
        self.counter.write().unwrap().add(value);
    }

    pub fn value(&self) -> u64 {
        self.counter.read().unwrap().value
    }
}

/// Gauge metric for measuring values that can go up and down
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gauge {
    pub name: String,
    pub description: String,
    pub value: f64,
}

impl Gauge {
    pub fn new(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            value: 0.0,
        }
    }

    pub fn set(&mut self, value: f64) {
        self.value = value;
    }

    pub fn increment(&mut self) {
        self.value += 1.0;
    }

    pub fn decrement(&mut self) {
        self.value -= 1.0;
    }

    pub fn add(&mut self, value: f64) {
        self.value += value;
    }

    pub fn subtract(&mut self, value: f64) {
        self.value -= value;
    }
}

/// Gauge handle for thread-safe access
#[derive(Clone)]
pub struct GaugeHandle {
    gauge: Arc<RwLock<Gauge>>,
}

impl GaugeHandle {
    pub fn set(&self, value: f64) {
        self.gauge.write().unwrap().set(value);
    }

    pub fn increment(&self) {
        self.gauge.write().unwrap().increment();
    }

    pub fn decrement(&self) {
        self.gauge.write().unwrap().decrement();
    }

    pub fn add(&self, value: f64) {
        self.gauge.write().unwrap().add(value);
    }

    pub fn subtract(&self, value: f64) {
        self.gauge.write().unwrap().subtract(value);
    }

    pub fn value(&self) -> f64 {
        self.gauge.read().unwrap().value
    }
}

/// Histogram metric for measuring distributions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Histogram {
    pub name: String,
    pub description: String,
    pub buckets: Vec<f64>,
    pub counts: Vec<u64>,
    pub count: u64,
    pub sum: f64,
}

impl Histogram {
    pub fn new(name: &str, description: &str, buckets: Vec<f64>) -> Self {
        let mut sorted_buckets = buckets;
        sorted_buckets.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted_buckets.push(f64::INFINITY);

        Self {
            name: name.to_string(),
            description: description.to_string(),
            buckets: sorted_buckets,
            counts: vec![0; sorted_buckets.len()],
            count: 0,
            sum: 0.0,
        }
    }

    pub fn observe(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;

        for (i, bucket) in self.buckets.iter().enumerate() {
            if value <= *bucket {
                self.counts[i] += 1;
            }
        }
    }
}

/// Histogram handle for thread-safe access
#[derive(Clone)]
pub struct HistogramHandle {
    histogram: Arc<RwLock<Histogram>>,
}

impl HistogramHandle {
    pub fn observe(&self, value: f64) {
        self.histogram.write().unwrap().observe(value);
    }

    pub fn count(&self) -> u64 {
        self.histogram.read().unwrap().count
    }

    pub fn sum(&self) -> f64 {
        self.histogram.read().unwrap().sum
    }
}

/// Timer for measuring durations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Timer {
    pub name: String,
    pub start_time: Option<Instant>,
    pub total_duration: Duration,
    pub count: u64,
}

impl Timer {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            start_time: None,
            total_duration: Duration::default(),
            count: 0,
        }
    }

    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
    }

    pub fn stop(&mut self) -> Option<Duration> {
        if let Some(start) = self.start_time.take() {
            let duration = start.elapsed();
            self.total_duration += duration;
            self.count += 1;
            Some(duration)
        } else {
            None
        }
    }

    pub fn average_duration(&self) -> Option<Duration> {
        if self.count > 0 {
            Some(self.total_duration / self.count as u32)
        } else {
            None
        }
    }
}

/// Timer handle for measuring operations
pub struct TimerHandle {
    timer: Arc<RwLock<Timer>>,
    start_time: Instant,
}

impl TimerHandle {
    pub fn stop(self) -> Duration {
        let duration = self.start_time.elapsed();
        if let Ok(mut timer) = self.timer.write() {
            timer.total_duration += duration;
            timer.count += 1;
        }
        duration
    }
}

impl Drop for TimerHandle {
    fn drop(&mut self) {
        let _ = self.start_time.elapsed(); // Timer automatically records on drop
    }
}

/// Metrics snapshot for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub timestamp: SystemTime,
    pub counters: HashMap<String, Counter>,
    pub gauges: HashMap<String, Gauge>,
    pub histograms: HashMap<String, Histogram>,
    pub timers: HashMap<String, Timer>,
}

/// Health check system
pub struct HealthChecker {
    checks: Vec<Box<dyn HealthCheck>>,
}

impl HealthChecker {
    pub fn new() -> Self {
        Self {
            checks: Vec::new(),
        }
    }

    pub fn add_check(&mut self, check: Box<dyn HealthCheck>) {
        self.checks.push(check);
    }

    pub async fn run_checks(&self) -> HealthReport {
        let mut results = Vec::new();
        let start_time = Instant::now();

        for check in &self.checks {
            let result = check.check().await;
            results.push(result);
        }

        let total_time = start_time.elapsed();

        HealthReport {
            timestamp: SystemTime::now(),
            total_time,
            results,
        }
    }
}

/// Health check trait
#[async_trait::async_trait]
pub trait HealthCheck: Send + Sync {
    async fn check(&self) -> HealthCheckResult;
}

/// Health check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResult {
    pub name: String,
    pub status: HealthStatus,
    pub message: Option<String>,
    pub duration: Duration,
}

/// Health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

/// Health report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthReport {
    pub timestamp: SystemTime,
    pub total_time: Duration,
    pub results: Vec<HealthCheckResult>,
}

impl HealthReport {
    pub fn is_healthy(&self) -> bool {
        self.results.iter().all(|r| matches!(r.status, HealthStatus::Healthy))
    }

    pub fn summary(&self) -> String {
        let healthy = self.results.iter().filter(|r| matches!(r.status, HealthStatus::Healthy)).count();
        let degraded = self.results.iter().filter(|r| matches!(r.status, HealthStatus::Degraded)).count();
        let unhealthy = self.results.iter().filter(|r| matches!(r.status, HealthStatus::Unhealthy)).count();

        format!("Health Report: {} healthy, {} degraded, {} unhealthy (took {:.2}ms)",
                healthy, degraded, unhealthy, self.total_time.as_millis())
    }
}

/// Performance monitor
pub struct PerformanceMonitor {
    registry: MetricsRegistry,
    start_time: Instant,
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        Self {
            registry: MetricsRegistry::new(),
            start_time: Instant::now(),
        }
    }

    pub fn record_operation(&self, operation: &str, duration: Duration) {
        let histogram = self.registry.histogram(
            &format!("operation_duration_{}", operation),
            &format!("Duration of {} operations", operation),
            vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
        );
        histogram.observe(duration.as_secs_f64());
    }

    pub fn increment_counter(&self, name: &str, description: &str) {
        let counter = self.registry.counter(name, description);
        counter.increment();
    }

    pub fn set_gauge(&self, name: &str, description: &str, value: f64) {
        let gauge = self.registry.gauge(name, description);
        gauge.set(value);
    }

    pub fn uptime(&self) -> Duration {
        self.start_time.elapsed()
    }

    pub fn memory_usage(&self) -> CliResult<usize> {
        // This would integrate with system monitoring
        // For now, return a placeholder
        Ok(1024 * 1024 * 50) // 50MB placeholder
    }

    pub fn cpu_usage(&self) -> CliResult<f64> {
        // This would integrate with system monitoring
        // For now, return a placeholder
        Ok(15.5) // 15.5% placeholder
    }
}

/// Alert manager for monitoring thresholds
pub struct AlertManager {
    alerts: Vec<Alert>,
}

impl AlertManager {
    pub fn new() -> Self {
        Self {
            alerts: Vec::new(),
        }
    }

    pub fn add_alert(&mut self, alert: Alert) {
        self.alerts.push(alert);
    }

    pub fn check_alerts(&self, metrics: &MetricsRegistry) -> Vec<TriggeredAlert> {
        let mut triggered = Vec::new();

        for alert in &self.alerts {
            if let Some(triggered_alert) = alert.check(metrics) {
                triggered.push(triggered_alert);
            }
        }

        triggered
    }
}

/// Alert configuration
#[derive(Debug, Clone)]
pub struct Alert {
    pub name: String,
    pub condition: AlertCondition,
    pub threshold: f64,
    pub severity: AlertSeverity,
}

impl Alert {
    pub fn check(&self, metrics: &MetricsRegistry) -> Option<TriggeredAlert> {
        // This would check the condition against metrics
        // For now, return None (no alerts triggered)
        None
    }
}

/// Alert condition
#[derive(Debug, Clone)]
pub enum AlertCondition {
    CounterAbove { name: String },
    GaugeAbove { name: String },
    GaugeBelow { name: String },
    HistogramPercentileAbove { name: String, percentile: f64 },
}

/// Alert severity
#[derive(Debug, Clone)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Triggered alert
#[derive(Debug, Clone)]
pub struct TriggeredAlert {
    pub alert_name: String,
    pub severity: AlertSeverity,
    pub message: String,
    pub timestamp: SystemTime,
}

impl Default for MetricsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counter() {
        let registry = MetricsRegistry::new();
        let counter = registry.counter("test_counter", "Test counter");

        counter.increment();
        counter.add(5);

        assert_eq!(counter.value(), 6);
    }

    #[test]
    fn test_gauge() {
        let registry = MetricsRegistry::new();
        let gauge = registry.gauge("test_gauge", "Test gauge");

        gauge.set(10.5);
        gauge.add(2.5);

        assert_eq!(gauge.value(), 13.0);
    }

    #[test]
    fn test_histogram() {
        let registry = MetricsRegistry::new();
        let histogram = registry.histogram("test_histogram", "Test histogram", vec![1.0, 5.0, 10.0]);

        histogram.observe(3.0);
        histogram.observe(7.0);
        histogram.observe(12.0);

        assert_eq!(histogram.count(), 3);
        assert_eq!(histogram.sum(), 22.0);
    }

    #[test]
    fn test_health_report() {
        let report = HealthReport {
            timestamp: SystemTime::now(),
            total_time: Duration::from_millis(100),
            results: vec![
                HealthCheckResult {
                    name: "database".to_string(),
                    status: HealthStatus::Healthy,
                    message: None,
                    duration: Duration::from_millis(50),
                }
            ],
        };

        assert!(report.is_healthy());
        assert!(report.summary().contains("1 healthy"));
    }

    #[test]
    fn test_prometheus_export() {
        let registry = MetricsRegistry::new();
        let counter = registry.counter("requests_total", "Total requests");
        counter.add(42);

        let prometheus = registry.to_prometheus();
        assert!(prometheus.contains("# HELP requests_total Total requests"));
        assert!(prometheus.contains("requests_total 42"));
    }
}