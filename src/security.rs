//! Security module
//!
//! This module provides comprehensive security features including
//! encryption, authentication, authorization, and secure communication.

use crate::error::{CliError, CliResult};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use ring::aead::{Aad, LessSafeKey, Nonce, UnboundKey, AES_256_GCM};
use ring::pbkdf2;
use ring::rand::{SecureRandom, SystemRandom};
use ring::signature::{EcdsaKeyPair, ECDSA_P256_SHA256_FIXED_SIGNING};
use serde::{Deserialize, Serialize};
use std::num::NonZeroU32;
use tokio::sync::RwLock;
use tracing::{info, warn, error, debug};
use base64::{Engine as _, engine::general_purpose};

/// Encryption manager for data protection
pub struct EncryptionManager {
    key: LessSafeKey,
    rng: SystemRandom,
}

impl EncryptionManager {
    /// Create a new encryption manager with a key
    pub fn new(key_bytes: &[u8]) -> CliResult<Self> {
        let unbound_key = UnboundKey::new(&AES_256_GCM, key_bytes)
            .map_err(|_| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                "Invalid encryption key".to_string()
            )))?;

        let key = LessSafeKey::new(unbound_key);
        let rng = SystemRandom::new();

        Ok(Self { key, rng })
    }

    /// Generate a random encryption key
    pub fn generate_key() -> CliResult<Vec<u8>> {
        let rng = SystemRandom::new();
        let mut key_bytes = [0u8; 32]; // 256 bits

        rng.fill(&mut key_bytes)
            .map_err(|_| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                "Failed to generate encryption key".to_string()
            )))?;

        Ok(key_bytes.to_vec())
    }

    /// Derive a key from a password using PBKDF2
    pub fn derive_key_from_password(password: &str, salt: &[u8], iterations: u32) -> CliResult<Vec<u8>> {
        let mut key = [0u8; 32];
        let iterations = NonZeroU32::new(iterations)
            .ok_or_else(|| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                "Invalid PBKDF2 iterations".to_string()
            )))?;

        pbkdf2::derive(
            pbkdf2::PBKDF2_HMAC_SHA256,
            iterations,
            salt,
            password.as_bytes(),
            &mut key,
        );

        Ok(key.to_vec())
    }

    /// Encrypt data
    pub fn encrypt(&self, plaintext: &[u8], associated_data: &[u8]) -> CliResult<Vec<u8>> {
        let mut nonce_bytes = [0u8; 12];
        self.rng.fill(&mut nonce_bytes)
            .map_err(|_| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                "Failed to generate nonce".to_string()
            )))?;

        let nonce = Nonce::assume_unique_for_key(nonce_bytes);
        let aad = Aad::from(associated_data);

        let mut in_out = plaintext.to_vec();
        self.key.seal_in_place_append_tag(nonce, aad, &mut in_out)
            .map_err(|_| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                "Encryption failed".to_string()
            )))?;

        // Prepend nonce to ciphertext
        let mut result = nonce_bytes.to_vec();
        result.extend_from_slice(&in_out);

        Ok(result)
    }

    /// Decrypt data
    pub fn decrypt(&self, ciphertext: &[u8], associated_data: &[u8]) -> CliResult<Vec<u8>> {
        if ciphertext.len() < 12 {
            return Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                "Invalid ciphertext".to_string()
            )));
        }

        let nonce_bytes = &ciphertext[..12];
        let encrypted_data = &ciphertext[12..];

        let nonce = Nonce::try_assume_unique_for_key(nonce_bytes)
            .map_err(|_| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                "Invalid nonce".to_string()
            )))?;

        let aad = Aad::from(associated_data);

        let mut in_out = encrypted_data.to_vec();
        let plaintext = self.key.open_in_place(nonce, aad, &mut in_out)
            .map_err(|_| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                "Decryption failed".to_string()
            )))?;

        Ok(plaintext.to_vec())
    }

    /// Encrypt a file
    pub fn encrypt_file(&self, input_path: &Path, output_path: &Path, associated_data: &[u8]) -> CliResult<()> {
        info!("Encrypting file: {:?} -> {:?}", input_path, output_path);

        let plaintext = fs::read(input_path)?;
        let ciphertext = self.encrypt(&plaintext, associated_data)?;

        // Ensure output directory exists
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)?;
        }

        fs::write(output_path, ciphertext)?;
        debug!("File encrypted successfully");

        Ok(())
    }

    /// Decrypt a file
    pub fn decrypt_file(&self, input_path: &Path, output_path: &Path, associated_data: &[u8]) -> CliResult<()> {
        info!("Decrypting file: {:?} -> {:?}", input_path, output_path);

        let ciphertext = fs::read(input_path)?;
        let plaintext = self.decrypt(&ciphertext, associated_data)?;

        // Ensure output directory exists
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)?;
        }

        fs::write(output_path, plaintext)?;
        debug!("File decrypted successfully");

        Ok(())
    }
}

/// Authentication manager
pub struct AuthManager {
    users: RwLock<HashMap<String, User>>,
    sessions: RwLock<HashMap<String, Session>>,
    jwt_secret: Vec<u8>,
}

impl AuthManager {
    /// Create a new authentication manager
    pub fn new(jwt_secret: Vec<u8>) -> Self {
        Self {
            users: RwLock::new(HashMap::new()),
            sessions: RwLock::new(HashMap::new()),
            jwt_secret,
        }
    }

    /// Register a new user
    pub async fn register_user(&self, username: &str, password: &str, role: UserRole) -> CliResult<()> {
        let mut users = self.users.write().await;

        if users.contains_key(username) {
            return Err(CliError::Auth(crate::error::AuthError::AuthenticationFailed(
                "User already exists".to_string()
            )));
        }

        let hashed_password = self.hash_password(password)?;
        let user = User {
            username: username.to_string(),
            password_hash: hashed_password,
            role,
            created_at: std::time::SystemTime::now(),
            last_login: None,
        };

        users.insert(username.to_string(), user);
        info!("User registered: {}", username);

        Ok(())
    }

    /// Authenticate a user
    pub async fn authenticate(&self, username: &str, password: &str) -> CliResult<String> {
        let users = self.users.read().await;

        let user = users.get(username)
            .ok_or_else(|| CliError::Auth(crate::error::AuthError::AuthenticationFailed(
                "User not found".to_string()
            )))?;

        if !self.verify_password(password, &user.password_hash)? {
            return Err(CliError::Auth(crate::error::AuthError::AuthenticationFailed(
                "Invalid password".to_string()
            )));
        }

        // Create session
        let session_id = self.generate_session_id();
        let session = Session {
            user_id: username.to_string(),
            created_at: std::time::SystemTime::now(),
            expires_at: std::time::SystemTime::now() + std::time::Duration::from_secs(3600), // 1 hour
        };

        let mut sessions = self.sessions.write().await;
        sessions.insert(session_id.clone(), session);

        // Update last login
        drop(users);
        let mut users = self.users.write().await;
        if let Some(user) = users.get_mut(username) {
            user.last_login = Some(std::time::SystemTime::now());
        }

        info!("User authenticated: {}", username);
        Ok(session_id)
    }

    /// Authorize a session for a specific action
    pub async fn authorize(&self, session_id: &str, required_role: UserRole) -> CliResult<String> {
        let sessions = self.sessions.read().await;

        let session = sessions.get(session_id)
            .ok_or_else(|| CliError::Auth(crate::error::AuthError::InvalidToken(
                "Session not found".to_string()
            )))?;

        // Check if session is expired
        if std::time::SystemTime::now() > session.expires_at {
            return Err(CliError::Auth(crate::error::AuthError::TokenExpired));
        }

        let users = self.users.read().await;
        let user = users.get(&session.user_id)
            .ok_or_else(|| CliError::Auth(crate::error::AuthError::InvalidToken(
                "User not found".to_string()
            )))?;

        // Check role hierarchy
        if !self.check_role_access(&user.role, &required_role) {
            return Err(CliError::Auth(crate::error::AuthError::AuthorizationFailed));
        }

        Ok(session.user_id.clone())
    }

    /// Logout a user
    pub async fn logout(&self, session_id: &str) -> CliResult<()> {
        let mut sessions = self.sessions.write().await;
        sessions.remove(session_id);
        info!("User logged out: {}", session_id);
        Ok(())
    }

    /// Hash a password
    fn hash_password(&self, password: &str) -> CliResult<String> {
        // Use a proper password hashing library in production
        // This is a simplified version for demonstration
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(password.as_bytes());
        let result = hasher.finalize();
        Ok(general_purpose::STANDARD.encode(&result))
    }

    /// Verify a password against its hash
    fn verify_password(&self, password: &str, hash: &str) -> CliResult<bool> {
        let expected_hash = self.hash_password(password)?;
        Ok(expected_hash == hash)
    }

    /// Generate a random session ID
    fn generate_session_id(&self) -> String {
        use ring::rand::SecureRandom;
        let rng = SystemRandom::new();
        let mut bytes = [0u8; 32];
        rng.fill(&mut bytes).unwrap();
        general_purpose::STANDARD.encode(&bytes)
    }

    /// Check if a role has access to a required role
    fn check_role_access(&self, user_role: &UserRole, required_role: &UserRole) -> bool {
        match (user_role, required_role) {
            (UserRole::Admin, _) => true,
            (UserRole::Moderator, UserRole::User) => true,
            (UserRole::Moderator, UserRole::Moderator) => true,
            (UserRole::User, UserRole::User) => true,
            _ => false,
        }
    }
}

/// User role enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum UserRole {
    Admin,
    Moderator,
    User,
}

/// User structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub username: String,
    pub password_hash: String,
    pub role: UserRole,
    pub created_at: std::time::SystemTime,
    pub last_login: Option<std::time::SystemTime>,
}

/// Session structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub user_id: String,
    pub created_at: std::time::SystemTime,
    pub expires_at: std::time::SystemTime,
}

/// Access control list (ACL) for fine-grained permissions
pub struct ACL {
    rules: HashMap<String, Vec<PermissionRule>>,
}

impl ACL {
    /// Create a new ACL
    pub fn new() -> Self {
        Self {
            rules: HashMap::new(),
        }
    }

    /// Add a permission rule
    pub fn add_rule(&mut self, resource: &str, rule: PermissionRule) {
        self.rules.entry(resource.to_string())
            .or_insert_with(Vec::new)
            .push(rule);
    }

    /// Check if a user has permission for a resource
    pub fn check_permission(&self, user: &str, resource: &str, action: &str, user_role: &UserRole) -> bool {
        if let Some(rules) = self.rules.get(resource) {
            for rule in rules {
                if rule.matches(user, action, user_role) {
                    return rule.allow;
                }
            }
        }
        false
    }
}

/// Permission rule
#[derive(Debug, Clone)]
pub struct PermissionRule {
    pub user_pattern: String,
    pub action: String,
    pub role: Option<UserRole>,
    pub allow: bool,
}

impl PermissionRule {
    /// Check if this rule matches the given parameters
    pub fn matches(&self, user: &str, action: &str, user_role: &UserRole) -> bool {
        // Simple pattern matching (use regex in production)
        if !user.contains(&self.user_pattern) && self.user_pattern != "*" {
            return false;
        }

        if self.action != "*" && self.action != action {
            return false;
        }

        if let Some(required_role) = &self.role {
            if required_role != user_role {
                return false;
            }
        }

        true
    }
}

/// Secure communication manager
pub struct SecureCommunicator {
    encryption: EncryptionManager,
    auth: AuthManager,
}

impl SecureCommunicator {
    /// Create a new secure communicator
    pub fn new(encryption_key: &[u8], jwt_secret: Vec<u8>) -> CliResult<Self> {
        let encryption = EncryptionManager::new(encryption_key)?;
        let auth = AuthManager::new(jwt_secret);

        Ok(Self { encryption, auth })
    }

    /// Send an encrypted message
    pub async fn send_secure_message(&self, recipient: &str, message: &[u8]) -> CliResult<Vec<u8>> {
        // Encrypt the message
        let associated_data = recipient.as_bytes();
        let encrypted = self.encryption.encrypt(message, associated_data)?;

        // Add authentication header
        let auth_token = self.generate_auth_token(recipient)?;
        let mut secure_message = auth_token.into_bytes();
        secure_message.extend_from_slice(&encrypted);

        Ok(secure_message)
    }

    /// Receive and decrypt a secure message
    pub async fn receive_secure_message(&self, sender: &str, secure_message: &[u8]) -> CliResult<Vec<u8>> {
        if secure_message.len() < 32 { // Minimum auth token length
            return Err(CliError::Auth(crate::error::AuthError::InvalidToken(
                "Message too short".to_string()
            )));
        }

        // Extract auth token and encrypted data
        let auth_token = &secure_message[..32];
        let encrypted_data = &secure_message[32..];

        // Verify auth token
        self.verify_auth_token(sender, std::str::from_utf8(auth_token)?)?;

        // Decrypt the message
        let associated_data = sender.as_bytes();
        let decrypted = self.encryption.decrypt(encrypted_data, associated_data)?;

        Ok(decrypted)
    }

    /// Generate an authentication token
    fn generate_auth_token(&self, user: &str) -> CliResult<String> {
        // Simplified token generation (use JWT in production)
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let token_data = format!("{}:{}", user, timestamp);
        let token_bytes = self.encryption.encrypt(token_data.as_bytes(), b"auth")?;
        Ok(general_purpose::STANDARD.encode(&token_bytes))
    }

    /// Verify an authentication token
    fn verify_auth_token(&self, expected_user: &str, token: &str) -> CliResult<()> {
        let token_bytes = general_purpose::STANDARD.decode(token)
            .map_err(|_| CliError::Auth(crate::error::AuthError::InvalidToken(
                "Invalid token format".to_string()
            )))?;

        let decrypted = self.encryption.decrypt(&token_bytes, b"auth")?;
        let token_data = std::str::from_utf8(&decrypted)
            .map_err(|_| CliError::Auth(crate::error::AuthError::InvalidToken(
                "Invalid token data".to_string()
            )))?;

        let parts: Vec<&str> = token_data.split(':').collect();
        if parts.len() != 2 || parts[0] != expected_user {
            return Err(CliError::Auth(crate::error::AuthError::InvalidToken(
                "Token validation failed".to_string()
            )));
        }

        Ok(())
    }
}

/// Security audit logger
pub struct SecurityAuditor {
    log_file: PathBuf,
}

impl SecurityAuditor {
    /// Create a new security auditor
    pub fn new(log_file: PathBuf) -> Self {
        Self { log_file }
    }

    /// Log a security event
    pub fn log_event(&self, event: SecurityEvent) -> CliResult<()> {
        let log_entry = serde_json::to_string(&event)?;
        let timestamp = chrono::Utc::now().to_rfc3339();
        let full_entry = format!("{}: {}\n", timestamp, log_entry);

        // Ensure log directory exists
        if let Some(parent) = self.log_file.parent() {
            fs::create_dir_all(parent)?;
        }

        use std::io::Write;
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_file)?;

        file.write_all(full_entry.as_bytes())?;
        file.flush()?;

        Ok(())
    }

    /// Get security events within a time range
    pub fn get_events(&self, start_time: std::time::SystemTime, end_time: std::time::SystemTime) -> CliResult<Vec<SecurityEvent>> {
        let content = fs::read_to_string(&self.log_file)?;
        let mut events = Vec::new();

        for line in content.lines() {
            if let Some(json_part) = line.split(": ").nth(1) {
                if let Ok(event) = serde_json::from_str::<SecurityEvent>(json_part) {
                    // Check if event is within time range
                    if event.timestamp >= start_time && event.timestamp <= end_time {
                        events.push(event);
                    }
                }
            }
        }

        Ok(events)
    }
}

/// Security event structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityEvent {
    pub timestamp: std::time::SystemTime,
    pub event_type: SecurityEventType,
    pub user: Option<String>,
    pub resource: String,
    pub action: String,
    pub success: bool,
    pub details: Option<String>,
}

/// Security event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityEventType {
    Authentication,
    Authorization,
    DataAccess,
    ConfigurationChange,
    SecurityAlert,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encryption_decryption() {
        let key = EncryptionManager::generate_key().unwrap();
        let manager = EncryptionManager::new(&key).unwrap();

        let plaintext = b"Hello, World!";
        let associated_data = b"test";

        let ciphertext = manager.encrypt(plaintext, associated_data).unwrap();
        let decrypted = manager.decrypt(&ciphertext, associated_data).unwrap();

        assert_eq!(plaintext, decrypted.as_slice());
    }

    #[test]
    fn test_password_hashing() {
        let auth = AuthManager::new(vec![1, 2, 3, 4]);

        let hash1 = auth.hash_password("password123").unwrap();
        let hash2 = auth.hash_password("password123").unwrap();

        assert_eq!(hash1, hash2);
        assert!(auth.verify_password("password123", &hash1).unwrap());
        assert!(!auth.verify_password("wrongpassword", &hash1).unwrap());
    }

    #[test]
    fn test_acl_permissions() {
        let mut acl = ACL::new();

        // Add rule allowing users to read their own data
        acl.add_rule("user_data", PermissionRule {
            user_pattern: "*", // Any user
            action: "read",
            role: Some(UserRole::User),
            allow: true,
        });

        // Add rule denying users from writing admin data
        acl.add_rule("admin_data", PermissionRule {
            user_pattern: "*",
            action: "write",
            role: Some(UserRole::User),
            allow: false,
        });

        assert!(acl.check_permission("alice", "user_data", "read", &UserRole::User));
        assert!(!acl.check_permission("alice", "admin_data", "write", &UserRole::User));
        assert!(acl.check_permission("admin", "admin_data", "write", &UserRole::Admin));
    }

    #[tokio::test]
    async fn test_user_registration_and_auth() {
        let auth = AuthManager::new(vec![1, 2, 3, 4]);

        // Register user
        auth.register_user("alice", "password123", UserRole::User).await.unwrap();

        // Authenticate user
        let session_id = auth.authenticate("alice", "password123").await.unwrap();
        assert!(!session_id.is_empty());

        // Authorize session
        let user_id = auth.authorize(&session_id, UserRole::User).await.unwrap();
        assert_eq!(user_id, "alice");

        // Logout
        auth.logout(&session_id).await.unwrap();
    }
}