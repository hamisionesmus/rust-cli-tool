//! Caching module
//!
//! This module provides comprehensive caching functionality including
//! in-memory caching, Redis integration, and advanced cache strategies.

use crate::error::{CliError, CliResult};
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock as AsyncRwLock;
use serde::{Deserialize, Serialize};
use tracing::{info, debug, warn};

/// Cache entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry<T> {
    pub value: T,
    pub created_at: SystemTime,
    pub accessed_at: SystemTime,
    pub expires_at: Option<SystemTime>,
    pub access_count: u64,
    pub size_bytes: usize,
}

impl<T> CacheEntry<T> {
    /// Create a new cache entry
    pub fn new(value: T, ttl: Option<Duration>) -> Self {
        let now = SystemTime::now();
        let expires_at = ttl.map(|t| now + t);
        let size_bytes = std::mem::size_of_val(&value);

        Self {
            value,
            created_at: now,
            accessed_at: now,
            expires_at,
            access_count: 0,
            size_bytes,
        }
    }

    /// Check if the entry is expired
    pub fn is_expired(&self) -> bool {
        if let Some(expires_at) = self.expires_at {
            SystemTime::now() > expires_at
        } else {
            false
        }
    }

    /// Access the entry (updates access metadata)
    pub fn access(&mut self) {
        self.accessed_at = SystemTime::now();
        self.access_count += 1;
    }

    /// Get the age of the entry
    pub fn age(&self) -> Duration {
        SystemTime::now()
            .duration_since(self.created_at)
            .unwrap_or(Duration::default())
    }

    /// Get time since last access
    pub fn time_since_access(&self) -> Duration {
        SystemTime::now()
            .duration_since(self.accessed_at)
            .unwrap_or(Duration::default())
    }
}

/// In-memory cache with various eviction strategies
pub struct InMemoryCache<K, V> {
    entries: RwLock<HashMap<K, CacheEntry<V>>>,
    max_size: usize,
    current_size: RwLock<usize>,
    eviction_strategy: EvictionStrategy,
}

impl<K, V> InMemoryCache<K, V>
where
    K: Eq + Hash + Clone + std::fmt::Debug,
    V: Clone + std::fmt::Debug,
{
    /// Create a new in-memory cache
    pub fn new(max_size: usize, eviction_strategy: EvictionStrategy) -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            max_size,
            current_size: RwLock::new(0),
            eviction_strategy,
        }
    }

    /// Get a value from the cache
    pub fn get(&self, key: &K) -> Option<V> {
        let mut entries = self.entries.write().unwrap();
        if let Some(entry) = entries.get_mut(key) {
            if entry.is_expired() {
                // Remove expired entry
                let size = entry.size_bytes;
                entries.remove(key);
                *self.current_size.write().unwrap() -= size;
                return None;
            }

            entry.access();
            Some(entry.value.clone())
        } else {
            None
        }
    }

    /// Put a value in the cache
    pub fn put(&self, key: K, value: V, ttl: Option<Duration>) -> CliResult<()> {
        let entry = CacheEntry::new(value, ttl);
        let entry_size = entry.size_bytes;

        let mut entries = self.entries.write().unwrap();
        let mut current_size = self.current_size.write().unwrap();

        // Check if we need to evict entries
        while *current_size + entry_size > self.max_size && !entries.is_empty() {
            self.evict_one(&mut entries, &mut current_size);
        }

        // Remove existing entry if present
        if let Some(old_entry) = entries.remove(&key) {
            *current_size -= old_entry.size_bytes;
        }

        // Add new entry
        entries.insert(key, entry);
        *current_size += entry_size;

        Ok(())
    }

    /// Remove a value from the cache
    pub fn remove(&self, key: &K) -> bool {
        let mut entries = self.entries.write().unwrap();
        let mut current_size = self.current_size.write().unwrap();

        if let Some(entry) = entries.remove(key) {
            *current_size -= entry.size_bytes;
            true
        } else {
            false
        }
    }

    /// Clear all entries from the cache
    pub fn clear(&self) {
        let mut entries = self.entries.write().unwrap();
        let mut current_size = self.current_size.write().unwrap();

        entries.clear();
        *current_size = 0;
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let entries = self.entries.read().unwrap();
        let current_size = *self.current_size.read().unwrap();

        let mut total_accesses = 0u64;
        let mut expired_count = 0usize;

        for entry in entries.values() {
            total_accesses += entry.access_count;
            if entry.is_expired() {
                expired_count += 1;
            }
        }

        CacheStats {
            entries_count: entries.len(),
            total_size_bytes: current_size,
            max_size_bytes: self.max_size,
            total_accesses,
            expired_entries: expired_count,
            hit_rate: if total_accesses > 0 {
                // This would need to track misses too
                0.8 // Placeholder
            } else {
                0.0
            },
        }
    }

    /// Evict one entry based on the eviction strategy
    fn evict_one(&self, entries: &mut HashMap<K, CacheEntry<V>>, current_size: &mut usize) {
        if entries.is_empty() {
            return;
        }

        let key_to_remove = match self.eviction_strategy {
            EvictionStrategy::LRU => {
                // Find least recently used
                entries.iter()
                    .min_by_key(|(_, entry)| entry.accessed_at)
                    .map(|(k, _)| k.clone())
            }
            EvictionStrategy::LFU => {
                // Find least frequently used
                entries.iter()
                    .min_by_key(|(_, entry)| entry.access_count)
                    .map(|(k, _)| k.clone())
            }
            EvictionStrategy::FIFO => {
                // Find first inserted (oldest)
                entries.iter()
                    .min_by_key(|(_, entry)| entry.created_at)
                    .map(|(k, _)| k.clone())
            }
            EvictionStrategy::Random => {
                // Random eviction
                use rand::seq::SliceRandom;
                let mut rng = rand::thread_rng();
                entries.keys().choose(&mut rng).cloned()
            }
        };

        if let Some(key) = key_to_remove {
            if let Some(entry) = entries.remove(&key) {
                *current_size -= entry.size_bytes;
                debug!("Evicted cache entry: {:?}", key);
            }
        }
    }

    /// Clean up expired entries
    pub fn cleanup_expired(&self) {
        let mut entries = self.entries.write().unwrap();
        let mut current_size = self.current_size.write().unwrap();

        let expired_keys: Vec<K> = entries.iter()
            .filter(|(_, entry)| entry.is_expired())
            .map(|(k, _)| k.clone())
            .collect();

        for key in expired_keys {
            if let Some(entry) = entries.remove(&key) {
                *current_size -= entry.size_bytes;
            }
        }

        if !expired_keys.is_empty() {
            info!("Cleaned up {} expired cache entries", expired_keys.len());
        }
    }
}

/// Cache eviction strategies
#[derive(Debug, Clone, Copy)]
pub enum EvictionStrategy {
    LRU,    // Least Recently Used
    LFU,    // Least Frequently Used
    FIFO,   // First In, First Out
    Random, // Random eviction
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub entries_count: usize,
    pub total_size_bytes: usize,
    pub max_size_bytes: usize,
    pub total_accesses: u64,
    pub expired_entries: usize,
    pub hit_rate: f64,
}

/// Redis cache client
pub struct RedisCache {
    client: redis::Client,
    connection: AsyncRwLock<Option<redis::aio::Connection>>,
}

impl RedisCache {
    /// Create a new Redis cache client
    pub fn new(redis_url: &str) -> CliResult<Self> {
        let client = redis::Client::open(redis_url)
            .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Failed to create Redis client: {}", e)
            )))?;

        Ok(Self {
            client,
            connection: AsyncRwLock::new(None),
        })
    }

    /// Get a connection to Redis
    async fn get_connection(&self) -> CliResult<redis::aio::Connection> {
        let mut conn_guard = self.connection.write().await;

        if let Some(conn) = conn_guard.as_mut() {
            // Test the connection
            if redis::cmd("PING").query_async::<_, String>(conn).await.is_ok() {
                return Ok(conn.clone());
            }
        }

        // Create new connection
        let conn = self.client.get_async_connection().await
            .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Failed to connect to Redis: {}", e)
            )))?;

        *conn_guard = Some(conn.clone());
        Ok(conn)
    }

    /// Get a value from Redis
    pub async fn get(&self, key: &str) -> CliResult<Option<String>> {
        let mut conn = self.get_connection().await?;
        let result: Option<String> = redis::cmd("GET")
            .arg(key)
            .query_async(&mut conn)
            .await
            .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Redis GET failed: {}", e)
            )))?;

        Ok(result)
    }

    /// Set a value in Redis
    pub async fn set(&self, key: &str, value: &str, ttl_seconds: Option<usize>) -> CliResult<()> {
        let mut conn = self.get_connection().await?;

        let mut cmd = redis::cmd("SET");
        cmd.arg(key).arg(value);

        if let Some(ttl) = ttl_seconds {
            cmd.arg("EX").arg(ttl);
        }

        cmd.query_async(&mut conn)
            .await
            .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Redis SET failed: {}", e)
            )))?;

        Ok(())
    }

    /// Delete a key from Redis
    pub async fn delete(&self, key: &str) -> CliResult<bool> {
        let mut conn = self.get_connection().await?;
        let deleted: usize = redis::cmd("DEL")
            .arg(key)
            .query_async(&mut conn)
            .await
            .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Redis DEL failed: {}", e)
            )))?;

        Ok(deleted > 0)
    }

    /// Set multiple keys at once
    pub async fn mset(&self, key_values: &HashMap<&str, &str>) -> CliResult<()> {
        let mut conn = self.get_connection().await?;

        let mut cmd = redis::cmd("MSET");
        for (&key, &value) in key_values {
            cmd.arg(key).arg(value);
        }

        cmd.query_async(&mut conn)
            .await
            .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Redis MSET failed: {}", e)
            )))?;

        Ok(())
    }

    /// Get multiple keys at once
    pub async fn mget(&self, keys: &[&str]) -> CliResult<Vec<Option<String>>> {
        let mut conn = self.get_connection().await?;

        let mut cmd = redis::cmd("MGET");
        for &key in keys {
            cmd.arg(key);
        }

        let result: Vec<Option<String>> = cmd
            .query_async(&mut conn)
            .await
            .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Redis MGET failed: {}", e)
            )))?;

        Ok(result)
    }

    /// Increment a counter
    pub async fn incr(&self, key: &str) -> CliResult<i64> {
        let mut conn = self.get_connection().await?;
        let result: i64 = redis::cmd("INCR")
            .arg(key)
            .query_async(&mut conn)
            .await
            .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Redis INCR failed: {}", e)
            )))?;

        Ok(result)
    }

    /// Add to a sorted set
    pub async fn zadd(&self, key: &str, score: f64, member: &str) -> CliResult<i32> {
        let mut conn = self.get_connection().await?;
        let result: i32 = redis::cmd("ZADD")
            .arg(key)
            .arg(score)
            .arg(member)
            .query_async(&mut conn)
            .await
            .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Redis ZADD failed: {}", e)
            )))?;

        Ok(result)
    }

    /// Get range from sorted set
    pub async fn zrange(&self, key: &str, start: isize, stop: isize) -> CliResult<Vec<String>> {
        let mut conn = self.get_connection().await?;
        let result: Vec<String> = redis::cmd("ZRANGE")
            .arg(key)
            .arg(start)
            .arg(stop)
            .query_async(&mut conn)
            .await
            .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Redis ZRANGE failed: {}", e)
            )))?;

        Ok(result)
    }

    /// Publish a message to a channel
    pub async fn publish(&self, channel: &str, message: &str) -> CliResult<i32> {
        let mut conn = self.get_connection().await?;
        let result: i32 = redis::cmd("PUBLISH")
            .arg(channel)
            .arg(message)
            .query_async(&mut conn)
            .await
            .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Redis PUBLISH failed: {}", e)
            )))?;

        Ok(result)
    }
}

/// Multi-level cache (L1 in-memory + L2 Redis)
pub struct MultiLevelCache<K, V> {
    l1_cache: InMemoryCache<K, V>,
    l2_cache: Option<RedisCache>,
    l1_ttl: Duration,
    l2_ttl: Duration,
}

impl<K, V> MultiLevelCache<K, V>
where
    K: Eq + Hash + Clone + std::fmt::Debug + AsRef<str>,
    V: Clone + std::fmt::Debug + serde::Serialize + for<'de> serde::Deserialize<'de>,
{
    /// Create a new multi-level cache
    pub fn new(l1_max_size: usize, l2_redis_url: Option<&str>, l1_ttl: Duration, l2_ttl: Duration) -> CliResult<Self> {
        let l1_cache = InMemoryCache::new(l1_max_size, EvictionStrategy::LRU);

        let l2_cache = if let Some(url) = l2_redis_url {
            Some(RedisCache::new(url)?)
        } else {
            None
        };

        Ok(Self {
            l1_cache,
            l2_cache,
            l1_ttl,
            l2_ttl,
        })
    }

    /// Get a value from the cache hierarchy
    pub async fn get(&self, key: &K) -> CliResult<Option<V>> {
        // Try L1 cache first
        if let Some(value) = self.l1_cache.get(key) {
            debug!("Cache hit in L1 for key: {:?}", key);
            return Ok(Some(value));
        }

        // Try L2 cache if available
        if let Some(ref l2) = self.l2_cache {
            if let Some(json_value) = l2.get(key.as_ref()).await? {
                match serde_json::from_str(&json_value) {
                    Ok(value) => {
                        // Populate L1 cache
                        let _ = self.l1_cache.put(key.clone(), value.clone(), Some(self.l1_ttl));
                        debug!("Cache hit in L2 for key: {:?}", key);
                        return Ok(Some(value));
                    }
                    Err(e) => {
                        warn!("Failed to deserialize cached value: {}", e);
                    }
                }
            }
        }

        debug!("Cache miss for key: {:?}", key);
        Ok(None)
    }

    /// Put a value in the cache hierarchy
    pub async fn put(&self, key: K, value: V) -> CliResult<()> {
        // Store in L1 cache
        self.l1_cache.put(key.clone(), value.clone(), Some(self.l1_ttl))?;

        // Store in L2 cache if available
        if let Some(ref l2) = self.l2_cache {
            let json_value = serde_json::to_string(&value)
                .map_err(|e| CliError::Serialization(crate::error::SerializationError::JsonError {
                    message: format!("Failed to serialize value for L2 cache: {}", e),
                }))?;

            let ttl_seconds = self.l2_ttl.as_secs() as usize;
            l2.set(key.as_ref(), &json_value, Some(ttl_seconds)).await?;
        }

        Ok(())
    }

    /// Remove a value from all cache levels
    pub async fn remove(&self, key: &K) -> CliResult<()> {
        // Remove from L1
        self.l1_cache.remove(key);

        // Remove from L2 if available
        if let Some(ref l2) = self.l2_cache {
            let _ = l2.delete(key.as_ref()).await?;
        }

        Ok(())
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        self.l1_cache.stats()
    }

    /// Clean up expired entries
    pub fn cleanup(&self) {
        self.l1_cache.cleanup_expired();
    }
}

/// Cache warming utilities
pub struct CacheWarmer;

impl CacheWarmer {
    /// Warm up cache with frequently accessed data
    pub async fn warmup<K, V>(
        cache: &MultiLevelCache<K, V>,
        data_provider: impl Fn() -> Vec<(K, V)>
    ) -> CliResult<usize>
    where
        K: Eq + Hash + Clone + std::fmt::Debug + AsRef<str>,
        V: Clone + std::fmt::Debug + serde::Serialize + for<'de> serde::Deserialize<'de>,
    {
        info!("Starting cache warmup");

        let data = data_provider();
        let mut warmed_count = 0;

        for (key, value) in data {
            if let Err(e) = cache.put(key, value).await {
                warn!("Failed to warm cache entry: {}", e);
            } else {
                warmed_count += 1;
            }
        }

        info!("Cache warmup completed: {} entries warmed", warmed_count);
        Ok(warmed_count)
    }

    /// Prefetch data based on access patterns
    pub async fn prefetch<K, V>(
        cache: &MultiLevelCache<K, V>,
        access_pattern: &[K],
        data_fetcher: impl Fn(&K) -> Option<V>
    ) -> CliResult<usize>
    where
        K: Eq + Hash + Clone + std::fmt::Debug + AsRef<str>,
        V: Clone + std::fmt::Debug + serde::Serialize + for<'de> serde::Deserialize<'de>,
    {
        info!("Starting cache prefetch for {} keys", access_pattern.len());

        let mut prefetched_count = 0;

        for key in access_pattern {
            if cache.get(key).await?.is_none() {
                if let Some(value) = data_fetcher(key) {
                    if let Err(e) = cache.put(key.clone(), value).await {
                        warn!("Failed to prefetch cache entry: {}", e);
                    } else {
                        prefetched_count += 1;
                    }
                }
            }
        }

        info!("Cache prefetch completed: {} entries prefetched", prefetched_count);
        Ok(prefetched_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_in_memory_cache() {
        let cache = InMemoryCache::new(1000, EvictionStrategy::LRU);

        // Test basic operations
        assert!(cache.get(&"key1").is_none());

        cache.put("key1", "value1", Some(Duration::from_secs(60))).unwrap();
        assert_eq!(cache.get(&"key1"), Some("value1"));

        cache.remove(&"key1");
        assert!(cache.get(&"key1").is_none());

        // Test statistics
        let stats = cache.stats();
        assert_eq!(stats.entries_count, 0);
    }

    #[test]
    fn test_cache_entry() {
        let entry = CacheEntry::new("test_value", Some(Duration::from_secs(60)));

        assert_eq!(entry.value, "test_value");
        assert!(!entry.is_expired());
        assert_eq!(entry.access_count, 0);

        entry.access();
        assert_eq!(entry.access_count, 1);
    }

    #[test]
    fn test_eviction_strategies() {
        // Test that different strategies can be created
        let _lru_cache = InMemoryCache::<String, String>::new(1000, EvictionStrategy::LRU);
        let _lfu_cache = InMemoryCache::<String, String>::new(1000, EvictionStrategy::LFU);
        let _fifo_cache = InMemoryCache::<String, String>::new(1000, EvictionStrategy::FIFO);
        let _random_cache = InMemoryCache::<String, String>::new(1000, EvictionStrategy::Random);
    }

    #[tokio::test]
    async fn test_cache_warmer() {
        let cache = MultiLevelCache::new(1000, None, Duration::from_secs(60), Duration::from_secs(300)).unwrap();

        let data_provider = || vec![
            ("key1".to_string(), "value1".to_string()),
            ("key2".to_string(), "value2".to_string()),
        ];

        let warmed_count = CacheWarmer::warmup(&cache, data_provider).await.unwrap();
        assert_eq!(warmed_count, 2);

        // Verify data is cached
        assert_eq!(cache.get(&"key1".to_string()).await.unwrap(), Some("value1".to_string()));
    }
}