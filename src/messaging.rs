//! Messaging module
//!
//! This module provides comprehensive message queue functionality including
//! Kafka integration, RabbitMQ support, and event-driven architecture.

use crate::error::{CliError, CliResult};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use serde::{Deserialize, Serialize};
use tracing::{info, debug, warn, error};

/// Message queue manager
pub struct MessageQueueManager {
    producers: HashMap<String, Box<dyn MessageProducer>>,
    consumers: HashMap<String, Box<dyn MessageConsumer>>,
    topics: HashMap<String, TopicConfig>,
}

impl MessageQueueManager {
    /// Create a new message queue manager
    pub fn new() -> Self {
        Self {
            producers: HashMap::new(),
            consumers: HashMap::new(),
            topics: HashMap::new(),
        }
    }

    /// Add a message producer
    pub fn add_producer(&mut self, name: &str, producer: Box<dyn MessageProducer>) {
        self.producers.insert(name.to_string(), producer);
    }

    /// Add a message consumer
    pub fn add_consumer(&mut self, name: &str, consumer: Box<dyn MessageConsumer>) {
        self.consumers.insert(name.to_string(), consumer);
    }

    /// Configure a topic
    pub fn configure_topic(&mut self, name: &str, config: TopicConfig) {
        self.topics.insert(name.to_string(), config);
    }

    /// Send a message
    pub async fn send_message(&self, producer_name: &str, topic: &str, message: Message) -> CliResult<String> {
        if let Some(producer) = self.producers.get(producer_name) {
            producer.send_message(topic, message).await
        } else {
            Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Producer '{}' not found", producer_name)
            )))
        }
    }

    /// Start consuming messages
    pub async fn start_consuming(&self, consumer_name: &str, topic: &str, handler: Box<dyn MessageHandler>) -> CliResult<()> {
        if let Some(consumer) = self.consumers.get(consumer_name) {
            consumer.start_consuming(topic, handler).await
        } else {
            Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Consumer '{}' not found", consumer_name)
            )))
        }
    }

    /// Get topic information
    pub fn get_topic_info(&self, topic: &str) -> Option<&TopicConfig> {
        self.topics.get(topic)
    }

    /// List all topics
    pub fn list_topics(&self) -> Vec<String> {
        self.topics.keys().cloned().collect()
    }
}

/// Message structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    pub topic: String,
    pub payload: serde_json::Value,
    pub headers: HashMap<String, String>,
    pub timestamp: std::time::SystemTime,
    pub key: Option<String>,
}

impl Message {
    /// Create a new message
    pub fn new(topic: &str, payload: serde_json::Value) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            topic: topic.to_string(),
            payload,
            headers: HashMap::new(),
            timestamp: std::time::SystemTime::now(),
            key: None,
        }
    }

    /// Add a header
    pub fn with_header(mut self, key: &str, value: &str) -> Self {
        self.headers.insert(key.to_string(), value.to_string());
        self
    }

    /// Set message key
    pub fn with_key(mut self, key: String) -> Self {
        self.key = Some(key);
        self
    }
}

/// Topic configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicConfig {
    pub name: String,
    pub partitions: i32,
    pub replication_factor: i16,
    pub retention_ms: Option<i64>,
    pub max_message_size: Option<i32>,
}

/// Message producer trait
#[async_trait::async_trait]
pub trait MessageProducer: Send + Sync {
    /// Send a message to a topic
    async fn send_message(&self, topic: &str, message: Message) -> CliResult<String>;

    /// Send multiple messages in batch
    async fn send_batch(&self, topic: &str, messages: Vec<Message>) -> CliResult<Vec<String>>;

    /// Flush pending messages
    async fn flush(&self) -> CliResult<()>;

    /// Close the producer
    async fn close(&self) -> CliResult<()>;
}

/// Message consumer trait
#[async_trait::async_trait]
pub trait MessageConsumer: Send + Sync {
    /// Start consuming messages from a topic
    async fn start_consuming(&self, topic: &str, handler: Box<dyn MessageHandler>) -> CliResult<()>;

    /// Stop consuming messages
    async fn stop_consuming(&self) -> CliResult<()>;

    /// Subscribe to additional topics
    async fn subscribe(&self, topics: Vec<String>) -> CliResult<()>;

    /// Unsubscribe from topics
    async fn unsubscribe(&self, topics: Vec<String>) -> CliResult<()>;

    /// Commit offsets
    async fn commit_offsets(&self) -> CliResult<()>;
}

/// Message handler trait
#[async_trait::async_trait]
pub trait MessageHandler: Send + Sync {
    /// Handle a received message
    async fn handle_message(&self, message: Message) -> CliResult<()>;

    /// Handle batch of messages
    async fn handle_batch(&self, messages: Vec<Message>) -> CliResult<()>;
}

/// Kafka producer implementation
pub struct KafkaProducer {
    producer: rdkafka::producer::FutureProducer,
    config: KafkaConfig,
}

impl KafkaProducer {
    /// Create a new Kafka producer
    pub fn new(config: KafkaConfig) -> CliResult<Self> {
        let mut producer_config = rdkafka::config::ClientConfig::new();

        for (key, value) in &config.properties {
            producer_config.set(key, value);
        }

        let producer: rdkafka::producer::FutureProducer = producer_config
            .create()
            .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Failed to create Kafka producer: {}", e)
            )))?;

        Ok(Self { producer, config })
    }
}

#[async_trait::async_trait]
impl MessageProducer for KafkaProducer {
    async fn send_message(&self, topic: &str, message: Message) -> CliResult<String> {
        let payload = serde_json::to_string(&message.payload)
            .map_err(|e| CliError::Serialization(crate::error::SerializationError::JsonError {
                message: format!("Failed to serialize message payload: {}", e),
            }))?;

        let record = rdkafka::producer::FutureRecord::to(topic)
            .payload(&payload)
            .key(message.key.as_deref());

        // Add headers
        let mut record = record;
        for (key, value) in &message.headers {
            record = record.header(key, value);
        }

        let delivery_future = self.producer.send(record, std::time::Duration::from_secs(0));

        match delivery_future.await {
            Ok((partition, offset)) => {
                let message_id = format!("{}-{}", partition, offset);
                debug!("Message sent to Kafka: topic={}, partition={}, offset={}", topic, partition, offset);
                Ok(message_id)
            }
            Err((e, _)) => Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Failed to send message to Kafka: {}", e)
            ))),
        }
    }

    async fn send_batch(&self, topic: &str, messages: Vec<Message>) -> CliResult<Vec<String>> {
        let mut message_ids = Vec::new();

        for message in messages {
            let message_id = self.send_message(topic, message).await?;
            message_ids.push(message_id);
        }

        Ok(message_ids)
    }

    async fn flush(&self) -> CliResult<()> {
        self.producer.flush(std::time::Duration::from_secs(30));
        Ok(())
    }

    async fn close(&self) -> CliResult<()> {
        // Producer will be automatically closed when dropped
        Ok(())
    }
}

/// Kafka consumer implementation
pub struct KafkaConsumer {
    consumer: rdkafka::consumer::StreamConsumer,
    config: KafkaConfig,
}

impl KafkaConsumer {
    /// Create a new Kafka consumer
    pub fn new(config: KafkaConfig, group_id: &str) -> CliResult<Self> {
        let mut consumer_config = rdkafka::config::ClientConfig::new();

        consumer_config.set("group.id", group_id);
        consumer_config.set("auto.offset.reset", "earliest");

        for (key, value) in &config.properties {
            consumer_config.set(key, value);
        }

        let consumer: rdkafka::consumer::StreamConsumer = consumer_config
            .create()
            .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Failed to create Kafka consumer: {}", e)
            )))?;

        Ok(Self { consumer, config })
    }
}

#[async_trait::async_trait]
impl MessageConsumer for KafkaConsumer {
    async fn start_consuming(&self, topic: &str, handler: Box<dyn MessageHandler>) -> CliResult<()> {
        self.consumer.subscribe(&[topic])
            .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Failed to subscribe to topic {}: {}", topic, e)
            )))?;

        info!("Started consuming from Kafka topic: {}", topic);

        let consumer = self.consumer.clone();
        tokio::spawn(async move {
            loop {
                match consumer.recv().await {
                    Ok(message) => {
                        let payload = match std::str::from_utf8(message.payload().unwrap_or(&[])) {
                            Ok(p) => p,
                            Err(e) => {
                                error!("Failed to decode message payload: {}", e);
                                continue;
                            }
                        };

                        let message_data: serde_json::Value = match serde_json::from_str(payload) {
                            Ok(m) => m,
                            Err(e) => {
                                error!("Failed to parse message JSON: {}", e);
                                continue;
                            }
                        };

                        let kafka_message = Message {
                            id: message.offset().to_string(),
                            topic: topic.to_string(),
                            payload: message_data,
                            headers: HashMap::new(), // Headers would be extracted from Kafka headers
                            timestamp: std::time::SystemTime::now(),
                            key: message.key().map(|k| String::from_utf8_lossy(k).to_string()),
                        };

                        if let Err(e) = handler.handle_message(kafka_message).await {
                            error!("Error handling message: {}", e);
                        }
                    }
                    Err(e) => {
                        error!("Error receiving message from Kafka: {}", e);
                        break;
                    }
                }
            }
        });

        Ok(())
    }

    async fn stop_consuming(&self) -> CliResult<()> {
        // Consumer will be stopped when dropped
        Ok(())
    }

    async fn subscribe(&self, topics: Vec<String>) -> CliResult<()> {
        let topic_refs: Vec<&str> = topics.iter().map(|s| s.as_str()).collect();
        self.consumer.subscribe(&topic_refs)
            .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Failed to subscribe to topics: {}", e)
            )))?;
        Ok(())
    }

    async fn unsubscribe(&self, topics: Vec<String>) -> CliResult<()> {
        // Kafka consumers unsubscribe from all topics
        self.consumer.unsubscribe();
        Ok(())
    }

    async fn commit_offsets(&self) -> CliResult<()> {
        self.consumer.commit_consumer_state(rdkafka::consumer::CommitMode::Async)
            .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Failed to commit offsets: {}", e)
            )))?;
        Ok(())
    }
}

/// RabbitMQ producer implementation
pub struct RabbitMQProducer {
    channel: lapin::Channel,
    connection: lapin::Connection,
}

impl RabbitMQProducer {
    /// Create a new RabbitMQ producer
    pub async fn new(connection_string: &str) -> CliResult<Self> {
        let connection = lapin::Connection::connect(
            connection_string,
            lapin::ConnectionProperties::default(),
        ).await
        .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
            format!("Failed to connect to RabbitMQ: {}", e)
        )))?;

        let channel = connection.create_channel().await
            .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Failed to create RabbitMQ channel: {}", e)
            )))?;

        Ok(Self { channel, connection })
    }
}

#[async_trait::async_trait]
impl MessageProducer for RabbitMQProducer {
    async fn send_message(&self, topic: &str, message: Message) -> CliResult<String> {
        let payload = serde_json::to_vec(&message.payload)
            .map_err(|e| CliError::Serialization(crate::error::SerializationError::JsonError {
                message: format!("Failed to serialize message payload: {}", e),
            }))?;

        let publish_result = self.channel
            .basic_publish(
                "", // default exchange
                topic,
                lapin::options::BasicPublishOptions::default(),
                &payload,
                lapin::types::AMQPProperties::default(),
            )
            .await
            .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Failed to publish message to RabbitMQ: {}", e)
            )))?;

        Ok(format!("{:?}", publish_result))
    }

    async fn send_batch(&self, topic: &str, messages: Vec<Message>) -> CliResult<Vec<String>> {
        let mut message_ids = Vec::new();

        for message in messages {
            let message_id = self.send_message(topic, message).await?;
            message_ids.push(message_id);
        }

        Ok(message_ids)
    }

    async fn flush(&self) -> CliResult<()> {
        // RabbitMQ doesn't require explicit flushing
        Ok(())
    }

    async fn close(&self) -> CliResult<()> {
        self.connection.close(lapin::options::ConnectionCloseOptions::default()).await
            .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Failed to close RabbitMQ connection: {}", e)
            )))?;
        Ok(())
    }
}

/// RabbitMQ consumer implementation
pub struct RabbitMQConsumer {
    channel: lapin::Channel,
    connection: lapin::Connection,
}

impl RabbitMQConsumer {
    /// Create a new RabbitMQ consumer
    pub async fn new(connection_string: &str) -> CliResult<Self> {
        let connection = lapin::Connection::connect(
            connection_string,
            lapin::ConnectionProperties::default(),
        ).await
        .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
            format!("Failed to connect to RabbitMQ: {}", e)
        )))?;

        let channel = connection.create_channel().await
            .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Failed to create RabbitMQ channel: {}", e)
            )))?;

        Ok(Self { channel, connection })
    }
}

#[async_trait::async_trait]
impl MessageConsumer for RabbitMQConsumer {
    async fn start_consuming(&self, topic: &str, handler: Box<dyn MessageHandler>) -> CliResult<()> {
        // Declare queue
        self.channel.queue_declare(
            topic,
            lapin::options::QueueDeclareOptions::default(),
            lapin::types::FieldTable::default(),
        ).await
        .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
            format!("Failed to declare RabbitMQ queue: {}", e)
        )))?;

        info!("Started consuming from RabbitMQ queue: {}", topic);

        let consumer = self.channel
            .basic_consume(
                topic,
                "rust-cli-consumer",
                lapin::options::BasicConsumeOptions::default(),
                lapin::types::FieldTable::default(),
            )
            .await
            .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Failed to create RabbitMQ consumer: {}", e)
            )))?;

        let handler = Arc::new(handler);
        tokio::spawn(async move {
            consumer.set_delegate(move |delivery: lapin::message::DeliveryResult| {
                let handler = Arc::clone(&handler);
                async move {
                    if let Ok(Some(delivery)) = delivery {
                        let payload = delivery.data;
                        let message_data: serde_json::Value = match serde_json::from_slice(&payload) {
                            Ok(m) => m,
                            Err(e) => {
                                error!("Failed to parse message JSON: {}", e);
                                return;
                            }
                        };

                        let rabbit_message = Message {
                            id: format!("{:?}", delivery.delivery_tag),
                            topic: topic.to_string(),
                            payload: message_data,
                            headers: HashMap::new(),
                            timestamp: std::time::SystemTime::now(),
                            key: None,
                        };

                        if let Err(e) = handler.handle_message(rabbit_message).await {
                            error!("Error handling message: {}", e);
                        }

                        // Acknowledge message
                        if let Err(e) = delivery.acker.ack(lapin::options::BasicAckOptions::default()).await {
                            error!("Failed to acknowledge message: {}", e);
                        }
                    }
                }
            });
        });

        Ok(())
    }

    async fn stop_consuming(&self) -> CliResult<()> {
        // Consumer will be stopped when channel/connection is closed
        Ok(())
    }

    async fn subscribe(&self, _topics: Vec<String>) -> CliResult<()> {
        // RabbitMQ consumers are queue-specific
        Ok(())
    }

    async fn unsubscribe(&self, _topics: Vec<String>) -> CliResult<()> {
        // RabbitMQ consumers are queue-specific
        Ok(())
    }

    async fn commit_offsets(&self) -> CliResult<()> {
        // RabbitMQ uses acknowledgments instead of offset commits
        Ok(())
    }
}

/// Kafka configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KafkaConfig {
    pub bootstrap_servers: Vec<String>,
    pub properties: HashMap<String, String>,
}

impl Default for KafkaConfig {
    fn default() -> Self {
        let mut properties = HashMap::new();
        properties.insert("bootstrap.servers".to_string(), "localhost:9092".to_string());

        Self {
            bootstrap_servers: vec!["localhost:9092".to_string()],
            properties,
        }
    }
}

/// Message router for event-driven architecture
pub struct MessageRouter {
    routes: HashMap<String, Vec<Box<dyn MessageHandler>>>,
}

impl MessageRouter {
    /// Create a new message router
    pub fn new() -> Self {
        Self {
            routes: HashMap::new(),
        }
    }

    /// Register a handler for a topic
    pub fn register_handler(&mut self, topic: &str, handler: Box<dyn MessageHandler>) {
        self.routes.entry(topic.to_string()).or_insert_with(Vec::new).push(handler);
    }

    /// Route a message to registered handlers
    pub async fn route_message(&self, message: Message) -> CliResult<()> {
        if let Some(handlers) = self.routes.get(&message.topic) {
            for handler in handlers {
                if let Err(e) = handler.handle_message(message.clone()).await {
                    error!("Handler error for topic {}: {}", message.topic, e);
                }
            }
        } else {
            debug!("No handlers registered for topic: {}", message.topic);
        }

        Ok(())
    }

    /// Get registered topics
    pub fn get_topics(&self) -> Vec<String> {
        self.routes.keys().cloned().collect()
    }
}

/// Event bus for in-process messaging
pub struct EventBus {
    subscribers: HashMap<String, Vec<mpsc::UnboundedSender<Message>>>,
}

impl EventBus {
    /// Create a new event bus
    pub fn new() -> Self {
        Self {
            subscribers: HashMap::new(),
        }
    }

    /// Subscribe to an event type
    pub fn subscribe(&mut self, event_type: &str) -> mpsc::UnboundedReceiver<Message> {
        let (tx, rx) = mpsc::unbounded_channel();
        self.subscribers.entry(event_type.to_string()).or_insert_with(Vec::new).push(tx);
        rx
    }

    /// Publish an event
    pub async fn publish(&self, event_type: &str, message: Message) -> CliResult<()> {
        if let Some(subscribers) = self.subscribers.get(event_type) {
            for subscriber in subscribers {
                if let Err(e) = subscriber.send(message.clone()) {
                    warn!("Failed to send event to subscriber: {}", e);
                }
            }
        }

        Ok(())
    }

    /// Get subscriber count for an event type
    pub fn subscriber_count(&self, event_type: &str) -> usize {
        self.subscribers.get(event_type).map(|subs| subs.len()).unwrap_or(0)
    }
}

/// Dead letter queue for failed messages
pub struct DeadLetterQueue {
    queue: RwLock<Vec<Message>>,
    max_size: usize,
}

impl DeadLetterQueue {
    /// Create a new dead letter queue
    pub fn new(max_size: usize) -> Self {
        Self {
            queue: RwLock::new(Vec::new()),
            max_size,
        }
    }

    /// Add a failed message to the queue
    pub async fn enqueue(&self, message: Message) -> CliResult<()> {
        let mut queue = self.queue.write().await;

        if queue.len() >= self.max_size {
            // Remove oldest message
            queue.remove(0);
        }

        queue.push(message);
        Ok(())
    }

    /// Get messages from the queue
    pub async fn dequeue_batch(&self, batch_size: usize) -> Vec<Message> {
        let mut queue = self.queue.write().await;
        let drain_size = batch_size.min(queue.len());
        queue.drain(0..drain_size).collect()
    }

    /// Get queue size
    pub async fn size(&self) -> usize {
        self.queue.read().await.len()
    }

    /// Clear the queue
    pub async fn clear(&self) {
        self.queue.write().await.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_creation() {
        let payload = serde_json::json!({"key": "value"});
        let message = Message::new("test-topic", payload);

        assert_eq!(message.topic, "test-topic");
        assert!(message.id.len() > 0);
        assert!(message.key.is_none());
    }

    #[test]
    fn test_message_with_headers() {
        let message = Message::new("test", serde_json::json!({"data": "test"}))
            .with_header("content-type", "application/json")
            .with_key("test-key".to_string());

        assert_eq!(message.headers.get("content-type"), Some(&"application/json".to_string()));
        assert_eq!(message.key, Some("test-key".to_string()));
    }

    #[test]
    fn test_message_router() {
        let mut router = MessageRouter::new();

        struct TestHandler;
        #[async_trait::async_trait]
        impl MessageHandler for TestHandler {
            async fn handle_message(&self, _message: Message) -> CliResult<()> {
                Ok(())
            }
        }

        router.register_handler("test-topic", Box::new(TestHandler));
        assert_eq!(router.get_topics(), vec!["test-topic"]);
    }

    #[test]
    fn test_event_bus() {
        let mut bus = EventBus::new();
        let mut rx = bus.subscribe("test-event");

        assert_eq!(bus.subscriber_count("test-event"), 1);
        assert_eq!(bus.subscriber_count("nonexistent"), 0);
    }

    #[tokio::test]
    async fn test_dead_letter_queue() {
        let dlq = DeadLetterQueue::new(10);

        let message = Message::new("test", serde_json::json!({"test": "data"}));
        dlq.enqueue(message).await.unwrap();

        assert_eq!(dlq.size().await, 1);

        let messages = dlq.dequeue_batch(5).await;
        assert_eq!(messages.len(), 1);

        assert_eq!(dlq.size().await, 0);
    }

    #[test]
    fn test_kafka_config_default() {
        let config = KafkaConfig::default();
        assert_eq!(config.bootstrap_servers, vec!["localhost:9092"]);
        assert!(config.properties.contains_key("bootstrap.servers"));
    }

    #[test]
    fn test_topic_config() {
        let config = TopicConfig {
            name: "test-topic".to_string(),
            partitions: 3,
            replication_factor: 2,
            retention_ms: Some(86400000),
            max_message_size: Some(1048576),
        };

        assert_eq!(config.name, "test-topic");
        assert_eq!(config.partitions, 3);
        assert_eq!(config.replication_factor, 2);
    }
}