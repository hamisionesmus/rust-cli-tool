//! Containerization module
//!
//! This module provides comprehensive containerization support including
//! Docker integration, Kubernetes orchestration, and container management.

use crate::error::{CliError, CliResult};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use serde::{Deserialize, Serialize};
use tokio::process::Command as TokioCommand;
use tracing::{info, warn, error, debug};

/// Docker manager for container operations
pub struct DockerManager {
    docker_available: bool,
    registry_config: Option<DockerRegistryConfig>,
}

impl DockerManager {
    /// Create a new Docker manager
    pub fn new() -> Self {
        let docker_available = Self::check_docker_availability();
        Self {
            docker_available,
            registry_config: None,
        }
    }

    /// Check if Docker is available
    fn check_docker_availability() -> bool {
        Command::new("docker")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
    }

    /// Set registry configuration
    pub fn set_registry_config(&mut self, config: DockerRegistryConfig) {
        self.registry_config = Some(config);
    }

    /// Build a Docker image
    pub async fn build_image(&self, context_path: &Path, image_name: &str, dockerfile_path: Option<&Path>, build_args: HashMap<String, String>) -> CliResult<String> {
        if !self.docker_available {
            return Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                "Docker is not available".to_string()
            )));
        }

        info!("Building Docker image: {}", image_name);

        let mut cmd = TokioCommand::new("docker");
        cmd.arg("build");

        // Add build arguments
        for (key, value) in build_args {
            cmd.arg("--build-arg").arg(format!("{}={}", key, value));
        }

        // Specify Dockerfile if provided
        if let Some(dockerfile) = dockerfile_path {
            cmd.arg("-f").arg(dockerfile);
        }

        // Set context and tag
        cmd.arg("-t").arg(image_name).arg(context_path);

        let output = cmd.output().await
            .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Failed to execute docker build: {}", e)
            )))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Docker build failed: {}", stderr)
            )));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        debug!("Docker build output: {}", stdout);

        // Extract image ID from output
        let image_id = self.extract_image_id(&stdout)?;

        info!("Successfully built image: {} ({})", image_name, image_id);
        Ok(image_id)
    }

    /// Push an image to registry
    pub async fn push_image(&self, image_name: &str) -> CliResult<()> {
        if !self.docker_available {
            return Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                "Docker is not available".to_string()
            )));
        }

        info!("Pushing Docker image: {}", image_name);

        // Login to registry if configured
        if let Some(ref config) = self.registry_config {
            self.login_to_registry(config).await?;
        }

        let output = TokioCommand::new("docker")
            .arg("push")
            .arg(image_name)
            .output()
            .await
            .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Failed to execute docker push: {}", e)
            )))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Docker push failed: {}", stderr)
            )));
        }

        info!("Successfully pushed image: {}", image_name);
        Ok(())
    }

    /// Pull an image from registry
    pub async fn pull_image(&self, image_name: &str) -> CliResult<()> {
        if !self.docker_available {
            return Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                "Docker is not available".to_string()
            )));
        }

        info!("Pulling Docker image: {}", image_name);

        let output = TokioCommand::new("docker")
            .arg("pull")
            .arg(image_name)
            .output()
            .await
            .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Failed to execute docker pull: {}", e)
            )))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Docker pull failed: {}", stderr)
            )));
        }

        info!("Successfully pulled image: {}", image_name);
        Ok(())
    }

    /// Run a container
    pub async fn run_container(&self, image_name: &str, container_config: ContainerConfig) -> CliResult<String> {
        if !self.docker_available {
            return Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                "Docker is not available".to_string()
            )));
        }

        info!("Running container from image: {}", image_name);

        let mut cmd = TokioCommand::new("docker");
        cmd.arg("run");

        if container_config.detached {
            cmd.arg("-d");
        }

        if let Some(name) = container_config.name {
            cmd.arg("--name").arg(name);
        }

        // Add environment variables
        for (key, value) in container_config.environment {
            cmd.arg("-e").arg(format!("{}={}", key, value));
        }

        // Add port mappings
        for port_mapping in container_config.ports {
            cmd.arg("-p").arg(format!("{}:{}", port_mapping.host_port, port_mapping.container_port));
        }

        // Add volume mounts
        for volume in container_config.volumes {
            cmd.arg("-v").arg(format!("{}:{}", volume.host_path, volume.container_path));
        }

        // Add network
        if let Some(network) = container_config.network {
            cmd.arg("--network").arg(network);
        }

        // Add restart policy
        if let Some(restart) = container_config.restart_policy {
            cmd.arg("--restart").arg(restart.as_str());
        }

        cmd.arg(image_name);

        // Add command arguments
        for arg in container_config.command_args {
            cmd.arg(arg);
        }

        let output = cmd.output().await
            .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Failed to execute docker run: {}", e)
            )))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Docker run failed: {}", stderr)
            )));
        }

        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let container_id = stdout.lines().last().unwrap_or(&stdout).to_string();

        info!("Successfully started container: {}", container_id);
        Ok(container_id)
    }

    /// Stop a container
    pub async fn stop_container(&self, container_id: &str, timeout_seconds: Option<u32>) -> CliResult<()> {
        if !self.docker_available {
            return Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                "Docker is not available".to_string()
            )));
        }

        info!("Stopping container: {}", container_id);

        let mut cmd = TokioCommand::new("docker");
        cmd.arg("stop");

        if let Some(timeout) = timeout_seconds {
            cmd.arg("-t").arg(timeout.to_string());
        }

        cmd.arg(container_id);

        let output = cmd.output().await
            .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Failed to execute docker stop: {}", e)
            )))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Docker stop failed: {}", stderr)
            )));
        }

        info!("Successfully stopped container: {}", container_id);
        Ok(())
    }

    /// Remove a container
    pub async fn remove_container(&self, container_id: &str, force: bool) -> CliResult<()> {
        if !self.docker_available {
            return Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                "Docker is not available".to_string()
            )));
        }

        info!("Removing container: {}", container_id);

        let mut cmd = TokioCommand::new("docker");
        cmd.arg("rm");

        if force {
            cmd.arg("-f");
        }

        cmd.arg(container_id);

        let output = cmd.output().await
            .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Failed to execute docker rm: {}", e)
            )))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Docker rm failed: {}", stderr)
            )));
        }

        info!("Successfully removed container: {}", container_id);
        Ok(())
    }

    /// Get container logs
    pub async fn get_container_logs(&self, container_id: &str, follow: bool, tail: Option<usize>) -> CliResult<String> {
        if !self.docker_available {
            return Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                "Docker is not available".to_string()
            )));
        }

        let mut cmd = TokioCommand::new("docker");
        cmd.arg("logs");

        if follow {
            cmd.arg("-f");
        }

        if let Some(lines) = tail {
            cmd.arg("--tail").arg(lines.to_string());
        }

        cmd.arg(container_id);

        let output = cmd.output().await
            .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Failed to execute docker logs: {}", e)
            )))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Docker logs failed: {}", stderr)
            )));
        }

        let logs = String::from_utf8_lossy(&output.stdout).to_string();
        Ok(logs)
    }

    /// List containers
    pub async fn list_containers(&self, all: bool) -> CliResult<Vec<ContainerInfo>> {
        if !self.docker_available {
            return Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                "Docker is not available".to_string()
            )));
        }

        let mut cmd = TokioCommand::new("docker");
        cmd.arg("ps");

        if all {
            cmd.arg("-a");
        }

        cmd.arg("--format").arg("json");

        let output = cmd.output().await
            .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Failed to execute docker ps: {}", e)
            )))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Docker ps failed: {}", stderr)
            )));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let containers: Vec<ContainerInfo> = stdout.lines()
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect();

        Ok(containers)
    }

    /// Login to Docker registry
    async fn login_to_registry(&self, config: &DockerRegistryConfig) -> CliResult<()> {
        info!("Logging in to Docker registry: {}", config.registry);

        let mut cmd = TokioCommand::new("docker");
        cmd.arg("login")
            .arg("-u").arg(&config.username)
            .arg("-p").arg(&config.password)
            .arg(&config.registry);

        let output = cmd.output().await
            .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Failed to execute docker login: {}", e)
            )))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Docker login failed: {}", stderr)
            )));
        }

        info!("Successfully logged in to registry: {}", config.registry);
        Ok(())
    }

    /// Extract image ID from build output
    fn extract_image_id(&self, output: &str) -> CliResult<String> {
        for line in output.lines() {
            if line.contains("Successfully built") {
                if let Some(id) = line.split_whitespace().last() {
                    return Ok(id.to_string());
                }
            }
        }

        // Fallback: try to get image ID using docker images
        let output = Command::new("docker")
            .args(&["images", "--format", "{{.ID}}", "--no-trunc"])
            .output()
            .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Failed to get image ID: {}", e)
            )))?;

        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if let Some(first_line) = stdout.lines().next() {
                return Ok(first_line.to_string());
            }
        }

        Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
            "Could not extract image ID".to_string()
        )))
    }
}

/// Docker registry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockerRegistryConfig {
    pub registry: String,
    pub username: String,
    pub password: String,
}

/// Container configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerConfig {
    pub name: Option<String>,
    pub detached: bool,
    pub environment: HashMap<String, String>,
    pub ports: Vec<PortMapping>,
    pub volumes: Vec<VolumeMount>,
    pub network: Option<String>,
    pub restart_policy: Option<RestartPolicy>,
    pub command_args: Vec<String>,
}

/// Port mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortMapping {
    pub host_port: u16,
    pub container_port: u16,
    pub protocol: Option<String>,
}

/// Volume mount
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeMount {
    pub host_path: String,
    pub container_path: String,
    pub read_only: bool,
}

/// Restart policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RestartPolicy {
    No,
    Always,
    OnFailure,
    UnlessStopped,
}

impl RestartPolicy {
    pub fn as_str(&self) -> &'static str {
        match self {
            RestartPolicy::No => "no",
            RestartPolicy::Always => "always",
            RestartPolicy::OnFailure => "on-failure",
            RestartPolicy::UnlessStopped => "unless-stopped",
        }
    }
}

/// Container information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerInfo {
    pub id: String,
    pub names: Vec<String>,
    pub image: String,
    pub status: String,
    pub ports: Vec<String>,
}

/// Kubernetes manager for orchestration
pub struct KubernetesManager {
    kubectl_available: bool,
    config_path: Option<PathBuf>,
}

impl KubernetesManager {
    /// Create a new Kubernetes manager
    pub fn new() -> Self {
        let kubectl_available = Self::check_kubectl_availability();
        Self {
            kubectl_available,
            config_path: None,
        }
    }

    /// Check if kubectl is available
    fn check_kubectl_availability() -> bool {
        Command::new("kubectl")
            .arg("version")
            .arg("--client")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
    }

    /// Set kubeconfig path
    pub fn set_config_path(&mut self, path: PathBuf) {
        self.config_path = Some(path);
    }

    /// Apply Kubernetes manifests
    pub async fn apply_manifests(&self, manifest_paths: Vec<PathBuf>) -> CliResult<()> {
        if !self.kubectl_available {
            return Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                "kubectl is not available".to_string()
            )));
        }

        for manifest_path in manifest_paths {
            info!("Applying Kubernetes manifest: {:?}", manifest_path);

            let mut cmd = TokioCommand::new("kubectl");
            cmd.arg("apply").arg("-f").arg(&manifest_path);

            if let Some(ref config) = self.config_path {
                cmd.env("KUBECONFIG", config);
            }

            let output = cmd.output().await
                .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                    format!("Failed to execute kubectl apply: {}", e)
                )))?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                    format!("kubectl apply failed: {}", stderr)
                )));
            }

            info!("Successfully applied manifest: {:?}", manifest_path);
        }

        Ok(())
    }

    /// Delete Kubernetes resources
    pub async fn delete_resources(&self, resource_type: &str, names: Vec<String>, namespace: Option<&str>) -> CliResult<()> {
        if !self.kubectl_available {
            return Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                "kubectl is not available".to_string()
            )));
        }

        info!("Deleting Kubernetes resources: {} {:?}", resource_type, names);

        let mut cmd = TokioCommand::new("kubectl");
        cmd.arg("delete").arg(resource_type);

        if let Some(ns) = namespace {
            cmd.arg("-n").arg(ns);
        }

        for name in names {
            cmd.arg(name);
        }

        let output = cmd.output().await
            .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Failed to execute kubectl delete: {}", e)
            )))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("kubectl delete failed: {}", stderr)
            )));
        }

        info!("Successfully deleted resources");
        Ok(())
    }

    /// Get cluster information
    pub async fn get_cluster_info(&self) -> CliResult<ClusterInfo> {
        if !self.kubectl_available {
            return Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                "kubectl is not available".to_string()
            )));
        }

        // Get nodes
        let nodes_output = TokioCommand::new("kubectl")
            .arg("get")
            .arg("nodes")
            .arg("-o")
            .arg("json")
            .output()
            .await?;

        let nodes: serde_json::Value = serde_json::from_slice(&nodes_output.stdout)?;

        // Get namespaces
        let namespaces_output = TokioCommand::new("kubectl")
            .arg("get")
            .arg("namespaces")
            .arg("-o")
            .arg("json")
            .output()
            .await?;

        let namespaces: serde_json::Value = serde_json::from_slice(&namespaces_output.stdout)?;

        Ok(ClusterInfo {
            nodes: nodes["items"].as_array().map(|arr| arr.len()).unwrap_or(0),
            namespaces: namespaces["items"].as_array().map(|arr| arr.len()).unwrap_or(0),
        })
    }

    /// Create a namespace
    pub async fn create_namespace(&self, name: &str) -> CliResult<()> {
        if !self.kubectl_available {
            return Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                "kubectl is not available".to_string()
            )));
        }

        info!("Creating namespace: {}", name);

        let output = TokioCommand::new("kubectl")
            .arg("create")
            .arg("namespace")
            .arg(name)
            .output()
            .await
            .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Failed to execute kubectl create namespace: {}", e)
            )))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            // Ignore "already exists" errors
            if !stderr.contains("AlreadyExists") {
                return Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                    format!("kubectl create namespace failed: {}", stderr)
                )));
            }
        }

        info!("Successfully created namespace: {}", name);
        Ok(())
    }

    /// Get pod logs
    pub async fn get_pod_logs(&self, pod_name: &str, namespace: Option<&str>, follow: bool) -> CliResult<String> {
        if !self.kubectl_available {
            return Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                "kubectl is not available".to_string()
            )));
        }

        let mut cmd = TokioCommand::new("kubectl");
        cmd.arg("logs");

        if let Some(ns) = namespace {
            cmd.arg("-n").arg(ns);
        }

        if follow {
            cmd.arg("-f");
        }

        cmd.arg(pod_name);

        let output = cmd.output().await
            .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Failed to execute kubectl logs: {}", e)
            )))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("kubectl logs failed: {}", stderr)
            )));
        }

        let logs = String::from_utf8_lossy(&output.stdout).to_string();
        Ok(logs)
    }
}

/// Cluster information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterInfo {
    pub nodes: usize,
    pub namespaces: usize,
}

/// Container orchestration manager
pub struct OrchestrationManager {
    docker_manager: DockerManager,
    k8s_manager: KubernetesManager,
    compose_files: Vec<PathBuf>,
}

impl OrchestrationManager {
    /// Create a new orchestration manager
    pub fn new() -> Self {
        Self {
            docker_manager: DockerManager::new(),
            k8s_manager: KubernetesManager::new(),
            compose_files: vec![],
        }
    }

    /// Add Docker Compose file
    pub fn add_compose_file(&mut self, path: PathBuf) {
        self.compose_files.push(path);
    }

    /// Deploy using Docker Compose
    pub async fn deploy_with_compose(&self, project_name: Option<&str>) -> CliResult<()> {
        if self.compose_files.is_empty() {
            return Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                "No Docker Compose files configured".to_string()
            )));
        }

        info!("Deploying with Docker Compose");

        let mut cmd = TokioCommand::new("docker-compose");

        for compose_file in &self.compose_files {
            cmd.arg("-f").arg(compose_file);
        }

        cmd.arg("up").arg("-d");

        if let Some(name) = project_name {
            cmd.arg("-p").arg(name);
        }

        let output = cmd.output().await
            .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Failed to execute docker-compose: {}", e)
            )))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Docker Compose deployment failed: {}", stderr)
            )));
        }

        info!("Successfully deployed with Docker Compose");
        Ok(())
    }

    /// Deploy to Kubernetes
    pub async fn deploy_to_kubernetes(&self, manifest_paths: Vec<PathBuf>) -> CliResult<()> {
        self.k8s_manager.apply_manifests(manifest_paths).await
    }

    /// Scale a service
    pub async fn scale_service(&self, service_name: &str, replicas: u32, namespace: Option<&str>) -> CliResult<()> {
        info!("Scaling service {} to {} replicas", service_name, replicas);

        let mut cmd = TokioCommand::new("kubectl");
        cmd.arg("scale");

        if let Some(ns) = namespace {
            cmd.arg("-n").arg(ns);
        }

        cmd.arg("deployment").arg(service_name).arg(format!("--replicas={}", replicas));

        let output = cmd.output().await
            .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Failed to execute kubectl scale: {}", e)
            )))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("kubectl scale failed: {}", stderr)
            )));
        }

        info!("Successfully scaled service {} to {} replicas", service_name, replicas);
        Ok(())
    }

    /// Get deployment status
    pub async fn get_deployment_status(&self, deployment_name: &str, namespace: Option<&str>) -> CliResult<DeploymentStatus> {
        let mut cmd = TokioCommand::new("kubectl");
        cmd.arg("get").arg("deployment").arg(deployment_name);

        if let Some(ns) = namespace {
            cmd.arg("-n").arg(ns);
        }

        cmd.arg("-o").arg("json");

        let output = cmd.output().await
            .map_err(|e| CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("Failed to execute kubectl get deployment: {}", e)
            )))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(CliError::Processing(crate::error::ProcessingError::ProcessingFailed(
                format!("kubectl get deployment failed: {}", stderr)
            )));
        }

        let deployment: serde_json::Value = serde_json::from_slice(&output.stdout)?;
        let status = &deployment["status"];

        Ok(DeploymentStatus {
            replicas: status["replicas"].as_u64().unwrap_or(0) as u32,
            available_replicas: status["availableReplicas"].as_u64().unwrap_or(0) as u32,
            ready_replicas: status["readyReplicas"].as_u64().unwrap_or(0) as u32,
            updated_replicas: status["updatedReplicas"].as_u64().unwrap_or(0) as u32,
        })
    }
}

/// Deployment status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentStatus {
    pub replicas: u32,
    pub available_replicas: u32,
    pub ready_replicas: u32,
    pub updated_replicas: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_docker_manager_creation() {
        let manager = DockerManager::new();
        // Just test that it can be created
        assert!(true);
    }

    #[test]
    fn test_kubernetes_manager_creation() {
        let manager = KubernetesManager::new();
        // Just test that it can be created
        assert!(true);
    }

    #[test]
    fn test_container_config() {
        let config = ContainerConfig {
            name: Some("test-container".to_string()),
            detached: true,
            environment: HashMap::new(),
            ports: vec![],
            volumes: vec![],
            network: None,
            restart_policy: Some(RestartPolicy::Always),
            command_args: vec![],
        };

        assert_eq!(config.name, Some("test-container".to_string()));
        assert!(config.detached);
    }

    #[test]
    fn test_restart_policy() {
        assert_eq!(RestartPolicy::Always.as_str(), "always");
        assert_eq!(RestartPolicy::No.as_str(), "no");
        assert_eq!(RestartPolicy::OnFailure.as_str(), "on-failure");
    }

    #[test]
    fn test_orchestration_manager() {
        let manager = OrchestrationManager::new();
        // Just test that it can be created
        assert!(true);
    }
}