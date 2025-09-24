//! Data processing module
//!
//! This module handles all data processing operations including
//! parallel processing, streaming, batch processing, and various
//! data transformation operations.

use crate::error::{CliError, ProcessingError, CliResult};
use crate::config::Config;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tokio::task;
use futures::future::join_all;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tracing::{info, warn, error, debug};
use rayon::prelude::*;
use crossbeam::channel::{bounded, Receiver, Sender};
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};

/// Processing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingStats {
    pub total_records: usize,
    pub processed_records: usize,
    pub failed_records: usize,
    pub processing_time: Duration,
    pub throughput: f64, // records per second
    pub memory_usage: usize, // bytes
    pub cpu_usage: f64, // percentage
}

/// Data processor with parallel processing capabilities
pub struct DataProcessor {
    config: Arc<Config>,
    stats: Arc<std::sync::Mutex<ProcessingStats>>,
    semaphore: Arc<Semaphore>,
}

impl DataProcessor {
    /// Create a new data processor
    pub fn new(config: Arc<Config>) -> Self {
        let stats = Arc::new(std::sync::Mutex::new(ProcessingStats {
            total_records: 0,
            processed_records: 0,
            failed_records: 0,
            processing_time: Duration::default(),
            throughput: 0.0,
            memory_usage: 0,
            cpu_usage: 0.0,
        }));

        let semaphore = Arc::new(Semaphore::new(config.workers));

        Self {
            config,
            stats,
            semaphore,
        }
    }

    /// Process data with parallel execution
    pub async fn process_parallel<T, F, Fut>(
        &self,
        data: Vec<T>,
        processor: F,
    ) -> CliResult<Vec<Fut::Output>>
    where
        T: Send + 'static,
        F: Fn(T) -> Fut + Send + Sync + Clone + 'static,
        Fut: std::future::Future + Send + 'static,
        Fut::Output: Send + 'static,
    {
        let start_time = Instant::now();
        let total_records = data.len();

        // Update stats
        {
            let mut stats = self.stats.lock().unwrap();
            stats.total_records = total_records;
        }

        info!("Starting parallel processing of {} records with {} workers", total_records, self.config.workers);

        // Create progress bar
        let pb = if self.config.progress {
            Some(ProgressBar::new(total_records as u64))
        } else {
            None
        };

        if let Some(ref pb) = pb {
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>7}/{len:7} {msg}")
                    .progress_chars("#>-")
            );
            pb.set_message("Processing...");
        }

        // Process in batches
        let mut results = Vec::with_capacity(total_records);
        let batch_size = self.config.batch_size;

        for chunk in data.chunks(batch_size) {
            let batch_results = self.process_batch(chunk, &processor).await?;
            results.extend(batch_results);

            if let Some(ref pb) = pb {
                pb.inc(chunk.len() as u64);
            }
        }

        let processing_time = start_time.elapsed();

        // Update final stats
        {
            let mut stats = self.stats.lock().unwrap();
            stats.processing_time = processing_time;
            stats.processed_records = results.len();
            stats.throughput = results.len() as f64 / processing_time.as_secs_f64();
        }

        if let Some(ref pb) = pb {
            pb.finish_with_message("✅ Processing completed!");
        }

        info!("Processing completed in {:.2}s, throughput: {:.0} records/sec",
              processing_time.as_secs_f64(),
              results.len() as f64 / processing_time.as_secs_f64());

        Ok(results)
    }

    /// Process a batch of data
    async fn process_batch<T, F, Fut>(
        &self,
        batch: &[T],
        processor: &F,
    ) -> CliResult<Vec<Fut::Output>>
    where
        T: Clone + Send + 'static,
        F: Fn(T) -> Fut + Send + Sync + Clone + 'static,
        Fut: std::future::Future + Send + 'static,
        Fut::Output: Send + 'static,
    {
        let mut handles = Vec::with_capacity(batch.len());

        for item in batch.iter().cloned() {
            let processor = processor.clone();
            let permit = self.semaphore.clone().acquire_owned().await
                .map_err(|_| CliError::Processing(ProcessingError::ProcessingFailed {
                    message: "Failed to acquire processing permit".to_string()
                }))?;

            let handle = task::spawn(async move {
                let _permit = permit; // Hold permit for duration of task
                processor(item).await
            });

            handles.push(handle);
        }

        // Wait for all tasks to complete
        let results = join_all(handles).await;

        // Collect results and handle errors
        let mut batch_results = Vec::with_capacity(results.len());
        let mut failed_count = 0;

        for result in results {
            match result {
                Ok(output) => batch_results.push(output),
                Err(e) => {
                    error!("Task panicked: {:?}", e);
                    failed_count += 1;
                }
            }
        }

        // Update failed records count
        if failed_count > 0 {
            let mut stats = self.stats.lock().unwrap();
            stats.failed_records += failed_count;
            warn!("{} records failed in batch", failed_count);
        }

        Ok(batch_results)
    }

    /// Process data using Rayon for CPU-bound tasks
    pub fn process_rayon<T, F, R>(&self, data: Vec<T>, processor: F) -> CliResult<Vec<R>>
    where
        T: Send + 'static,
        F: Fn(T) -> R + Send + Sync + 'static,
        R: Send + 'static,
    {
        let start_time = Instant::now();
        let total_records = data.len();

        info!("Starting Rayon processing of {} records", total_records);

        // Create progress bar
        let pb = if self.config.progress {
            Some(ProgressBar::new(total_records as u64))
        } else {
            None
        };

        if let Some(ref pb) = pb {
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.yellow} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>7}/{len:7} {msg}")
                    .progress_chars("#>-")
            );
            pb.set_message("Processing with Rayon...");
        }

        // Process with Rayon
        let results: Vec<R> = data.into_par_iter()
            .map(|item| {
                let result = processor(item);
                if let Some(ref pb) = pb {
                    pb.inc(1);
                }
                result
            })
            .collect();

        let processing_time = start_time.elapsed();

        // Update stats
        {
            let mut stats = self.stats.lock().unwrap();
            stats.total_records = total_records;
            stats.processed_records = results.len();
            stats.processing_time = processing_time;
            stats.throughput = results.len() as f64 / processing_time.as_secs_f64();
        }

        if let Some(ref pb) = pb {
            pb.finish_with_message("✅ Rayon processing completed!");
        }

        info!("Rayon processing completed in {:.2}s", processing_time.as_secs_f64());

        Ok(results)
    }

    /// Stream processing for large datasets
    pub async fn process_stream<T, F, Fut>(
        &self,
        receiver: Receiver<T>,
        processor: F,
        buffer_size: usize,
    ) -> CliResult<()>
    where
        T: Send + 'static,
        F: Fn(T) -> Fut + Send + Sync + Clone + 'static,
        Fut: std::future::Future<Output = CliResult<()>> + Send + 'static,
    {
        info!("Starting stream processing with buffer size {}", buffer_size);

        let (tx, rx) = bounded(buffer_size);
        let processor_clone = processor.clone();

        // Spawn consumer task
        let consumer_handle = task::spawn(async move {
            let mut buffer = Vec::with_capacity(buffer_size);
            let mut processed_count = 0;

            while let Ok(item) = rx.recv() {
                buffer.push(item);

                if buffer.len() >= buffer_size {
                    // Process batch
                    let batch = std::mem::take(&mut buffer);
                    let processor = processor_clone.clone();

                    for item in batch {
                        if let Err(e) = processor(item).await {
                            error!("Stream processing error: {}", e);
                        }
                        processed_count += 1;
                    }
                }
            }

            // Process remaining items
            let processor = processor_clone.clone();
            for item in buffer {
                if let Err(e) = processor(item).await {
                    error!("Stream processing error: {}", e);
                }
                processed_count += 1;
            }

            processed_count
        });

        // Feed items to consumer
        let mut sent_count = 0;
        while let Ok(item) = receiver.recv() {
            if tx.send(item).is_err() {
                break; // Consumer disconnected
            }
            sent_count += 1;
        }

        drop(tx); // Signal end of stream

        // Wait for consumer to finish
        let processed_count = consumer_handle.await
            .map_err(|e| CliError::Processing(ProcessingError::ProcessingFailed {
                message: format!("Consumer task failed: {}", e)
            }))?;

        info!("Stream processing completed: {} sent, {} processed", sent_count, processed_count);

        Ok(())
    }

    /// Get current processing statistics
    pub fn get_stats(&self) -> ProcessingStats {
        self.stats.lock().unwrap().clone()
    }

    /// Reset processing statistics
    pub fn reset_stats(&self) {
        let mut stats = self.stats.lock().unwrap();
        *stats = ProcessingStats {
            total_records: 0,
            processed_records: 0,
            failed_records: 0,
            processing_time: Duration::default(),
            throughput: 0.0,
            memory_usage: 0,
            cpu_usage: 0.0,
        };
    }
}

/// Data transformation utilities
pub struct DataTransformer;

impl DataTransformer {
    /// Transform data using a mapping function
    pub fn map<T, U, F>(data: Vec<T>, mapper: F) -> Vec<U>
    where
        F: Fn(T) -> U,
    {
        data.into_iter().map(mapper).collect()
    }

    /// Filter data using a predicate function
    pub fn filter<T, F>(data: Vec<T>, predicate: F) -> Vec<T>
    where
        F: Fn(&T) -> bool,
    {
        data.into_iter().filter(predicate).collect()
    }

    /// Group data by a key function
    pub fn group_by<T, K, F>(data: Vec<T>, key_fn: F) -> HashMap<K, Vec<T>>
    where
        K: Eq + std::hash::Hash,
        F: Fn(&T) -> K,
    {
        let mut groups: HashMap<K, Vec<T>> = HashMap::new();

        for item in data {
            let key = key_fn(&item);
            groups.entry(key).or_insert_with(Vec::new).push(item);
        }

        groups
    }

    /// Aggregate data using custom aggregation functions
    pub fn aggregate<T, A, F, G>(
        data: Vec<T>,
        init_accumulator: F,
        accumulator_fn: G,
    ) -> A
    where
        F: Fn() -> A,
        G: Fn(A, T) -> A,
    {
        let mut accumulator = init_accumulator();

        for item in data {
            accumulator = accumulator_fn(accumulator, item);
        }

        accumulator
    }

    /// Sort data using a comparison function
    pub fn sort_by<T, F, K>(mut data: Vec<T>, key_fn: F) -> Vec<T>
    where
        F: Fn(&T) -> K,
        K: Ord,
    {
        data.sort_by_key(key_fn);
        data
    }

    /// Deduplicate data
    pub fn deduplicate<T, F, K>(data: Vec<T>, key_fn: F) -> Vec<T>
    where
        T: Clone,
        F: Fn(&T) -> K,
        K: Eq + std::hash::Hash,
    {
        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::new();

        for item in data {
            let key = key_fn(&item);
            if seen.insert(key) {
                result.push(item);
            }
        }

        result
    }
}

/// Data validation utilities
pub struct DataValidator;

impl DataValidator {
    /// Validate data against a schema
    pub fn validate_schema<T: serde::Serialize>(
        data: &T,
        schema: &serde_json::Value,
    ) -> CliResult<()> {
        // Basic JSON schema validation
        let data_value = serde_json::to_value(data)
            .map_err(|e| CliError::Validation(crate::error::ValidationError::InvalidFormat {
                field: "data".to_string(),
                value: format!("Serialization error: {}", e),
            }))?;

        Self::validate_json_against_schema(&data_value, schema)
    }

    /// Validate JSON against schema
    fn validate_json_against_schema(
        data: &serde_json::Value,
        schema: &serde_json::Value,
    ) -> CliResult<()> {
        // Simple validation - check required fields
        if let Some(required) = schema.get("required") {
            if let Some(required_array) = required.as_array() {
                for field in required_array {
                    if let Some(field_name) = field.as_str() {
                        if !data.get(field_name).is_some() {
                            return Err(CliError::Validation(
                                crate::error::ValidationError::RequiredFieldMissing {
                                    field: field_name.to_string(),
                                }
                            ));
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Validate data types
    pub fn validate_types<T>(data: &[T], type_checks: &[Box<dyn Fn(&T) -> bool>]) -> CliResult<()> {
        for (i, item) in data.iter().enumerate() {
            for check in type_checks {
                if !check(item) {
                    return Err(CliError::Validation(
                        crate::error::ValidationError::InvalidFieldValue {
                            field: format!("item[{}]", i),
                            value: format!("{:?}", item),
                        }
                    ));
                }
            }
        }

        Ok(())
    }

    /// Check data integrity
    pub fn check_integrity<T: std::hash::Hash>(
        data: &[T],
        expected_hash: &str,
    ) -> CliResult<()> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;

        let mut hasher = DefaultHasher::new();
        for item in data {
            item.hash(&mut hasher);
        }

        let actual_hash = format!("{:x}", hasher.finish());
        if actual_hash != expected_hash {
            return Err(CliError::Validation(
                crate::error::ValidationError::InvalidFieldValue {
                    field: "data_integrity".to_string(),
                    value: format!("Expected {}, got {}", expected_hash, actual_hash),
                }
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_parallel_processing() {
        let config = Arc::new(Config::default());
        let processor = DataProcessor::new(config);

        let data = vec![1, 2, 3, 4, 5];
        let result = processor.process_parallel(
            data,
            |x| async move { x * 2 }
        ).await.unwrap();

        assert_eq!(result, vec![2, 4, 6, 8, 10]);
    }

    #[test]
    fn test_data_transformation() {
        let data = vec![1, 2, 3, 4, 5];

        // Test map
        let mapped = DataTransformer::map(data.clone(), |x| x * 2);
        assert_eq!(mapped, vec![2, 4, 6, 8, 10]);

        // Test filter
        let filtered = DataTransformer::filter(data, |x| x % 2 == 0);
        assert_eq!(filtered, vec![2, 4]);
    }

    #[test]
    fn test_data_validation() {
        let data = vec![1, 2, 3, 4, 5];
        let type_checks = vec![Box::new(|x: &i32| *x > 0) as Box<dyn Fn(&i32) -> bool>];

        assert!(DataValidator::validate_types(&data, &type_checks).is_ok());

        let invalid_data = vec![1, -2, 3];
        assert!(DataValidator::validate_types(&invalid_data, &type_checks).is_err());
    }
}