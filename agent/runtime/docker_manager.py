"""Docker container manager for MCP servers and LLM providers.

Manages Docker containers for:
- MCP servers: Pulls images, ensures availability
- LLM providers (Ollama): Starts/stops containers, manages lifecycle
"""

import asyncio
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.observability import get_logger


class DockerManager:
    """Manages Docker containers for MCP servers."""

    def __init__(self):
        """Initialize Docker manager."""
        self.logger = get_logger("docker.manager", "docker")
        self._images_available: Dict[str, bool] = {}  # image -> available
        self._llm_containers: Dict[str, str] = {}  # provider -> container_id
        self._running = False

    async def ensure_images(self, mcp_configs: Dict[str, Any]) -> Dict[str, bool]:
        """Ensure Docker images are available for enabled MCP servers.

        Args:
            mcp_configs: MCP server configurations from config

        Returns:
            Dictionary mapping MCP server name to success status
        """
        if self._running:
            self.logger.warning("Docker manager already started")
            return {}

        results = {}

        for mcp_name, mcp_config in mcp_configs.items():
            # Only manage Docker-type MCP servers
            if mcp_config.get("type") != "docker":
                continue

            # Skip if disabled
            if mcp_config.get("enabled") is False:
                self.logger.info(f"MCP server {mcp_name} is disabled, skipping Docker image check")
                continue

            image = mcp_config.get("image")
            if not image:
                self.logger.error(f"MCP server {mcp_name} missing 'image' in Docker config")
                results[mcp_name] = False
                continue

            try:
                await self._ensure_image(image)
                self._images_available[image] = True
                results[mcp_name] = True
            except Exception as e:
                self.logger.error(f"Failed to ensure image for MCP server {mcp_name}: {e}", exc_info=True)
                results[mcp_name] = False

        self._running = True
        return results

    async def _start_container(self, mcp_name: str, mcp_config: Dict[str, Any], workspace_path: str) -> bool:
        """Start a single Docker container for an MCP server.

        Args:
            mcp_name: MCP server name
            mcp_config: MCP server configuration
            workspace_path: Worksolved workspace path

        Returns:
            True if container started successfully, False otherwise
        """
        image = mcp_config.get("image")
        if not image:
            self.logger.error(f"MCP server {mcp_name} missing 'image' in Docker config")
            return False

        # Check if container already exists and is running
        container_name = f"forge-agent-mcp-{mcp_name}"
        existing_container = await self._get_container_id(container_name)
        
        if existing_container:
            # Check if it's running
            is_running = await self._is_container_running(existing_container)
            if is_running:
                self.logger.info(f"Container {container_name} already running")
                self._containers[mcp_name] = existing_container
                return True
            else:
                # Start existing container
                self.logger.info(f"Starting existing container {container_name}")
                await self._docker_command(["start", existing_container])
                self._containers[mcp_name] = existing_container
                return True

        # Pull image if not available
        await self._ensure_image(image)

        # Build docker run command
        volumes = mcp_config.get("volumes", [])
        args = mcp_config.get("args", [])
        
        # Resolve workspace path in volumes
        resolved_volumes = []
        for volume in volumes:
            if "{{workspace.base_path}}" in volume:
                volume = volume.replace("{{workspace.base_path}}", workspace_path)
            resolved_volumes.append(volume)

        # Create container (don't start yet - we'll use docker exec for stdio)
        # For MCP stdio, we need to run with -i (interactive) and --rm (remove on exit)
        # But we want to keep it running, so we'll use a different approach
        # Only args, _docker_command adds "docker"
        docker_cmd = [
            "run", "-d",  # Detached mode
            "--name", container_name,
            "--rm",  # Remove on exit (but we'll manage lifecycle)
        ]

        # Add volumes
        for volume in resolved_volumes:
            docker_cmd.extend(["-v", volume])

        # Add image and args
        docker_cmd.append(image)
        docker_cmd.extend(args)

        # Create container template (not started - used as reference)
        self.logger.info(f"Creating container template {container_name} with image {image}")
        container_id = await self._docker_command(docker_cmd, capture_output=True)
        
        if container_id:
            container_id = container_id.strip()
            self._containers[mcp_name] = container_id
            self.logger.info(f"Container template {container_name} created with ID {container_id[:12]}")
            # Store image and config for later use in connections
            return True
        else:
            self.logger.error(f"Failed to create container template {container_name}")
            return False

    async def start_ollama(self, port: int = 11434, gpu: bool = True) -> bool:
        """Start Ollama Docker container.

        Args:
            port: Port to expose Ollama API (default: 11434)
            gpu: Whether to enable GPU support (default: True)

        Returns:
            True if started successfully, False otherwise
        """
        container_name = "forge-agent-ollama"
        image = "ollama/ollama:latest"

        # Check if container already exists and is running
        existing_container = await self._get_container_id(container_name)
        if existing_container:
            is_running = await self._is_container_running(existing_container)
            if is_running:
                self.logger.info(f"Ollama container {container_name} already running")
                self._llm_containers["ollama"] = existing_container
                return True
            else:
                # Start existing container
                self.logger.info(f"Starting existing Ollama container {container_name}")
                await self._docker_command(["start", existing_container])
                self._llm_containers["ollama"] = existing_container
                return True

        # Ensure image is available
        await self._ensure_image(image)

        # Build docker run command (only args, _docker_command adds "docker")
        docker_cmd = [
            "run", "-d",
            "--name", container_name,
            "-p", f"{port}:11434",
            "--restart", "unless-stopped",
        ]

        # Add GPU support if requested
        if gpu:
            docker_cmd.extend(["--gpus", "all"])
            # Add environment variables for GPU support
            docker_cmd.extend(["-e", "OLLAMA_NUM_GPU=-1"])  # Use all available GPUs
            docker_cmd.extend(["-e", "CUDA_VISIBLE_DEVICES=0"])  # Use first GPU

        # Add volume for model storage
        docker_cmd.extend(["-v", "ollama-data:/root/.ollama"])

        # Add image
        docker_cmd.append(image)

        # Create and start container
        self.logger.info(f"Creating Ollama container {container_name} on port {port}")
        container_id = await self._docker_command(docker_cmd, capture_output=True)

        if container_id:
            container_id = container_id.strip()
            self._llm_containers["ollama"] = container_id
            self.logger.info(f"Ollama container {container_name} started with ID {container_id[:12]}")
            
            # Wait for Ollama to be ready (check health endpoint)
            max_retries = 30  # 30 seconds max
            retry_count = 0
            import httpx
            
            while retry_count < max_retries:
                try:
                    async with httpx.AsyncClient(timeout=2.0) as client:
                        response = await client.get(f"http://localhost:{port}/api/tags")
                        if response.status_code == 200:
                            self.logger.info(f"Ollama container {container_name} is ready")
                            return True
                except Exception:
                    pass
                
                retry_count += 1
                await asyncio.sleep(1)
            
            self.logger.warning(
                f"Ollama container {container_name} started but health check timed out. "
                "It may still be initializing."
            )
            return True
        else:
            self.logger.error(f"Failed to create Ollama container {container_name}")
            return False

    async def stop_ollama(self):
        """Stop Ollama Docker container."""
        if "ollama" not in self._llm_containers:
            return

        container_id = self._llm_containers["ollama"]
        try:
            self.logger.info("Stopping Ollama container")
            await self._docker_command(["stop", container_id])
            del self._llm_containers["ollama"]
        except Exception as e:
            self.logger.error(f"Error stopping Ollama container: {e}")

    async def restart_ollama(self, port: int = 11434, gpu: bool = True) -> bool:
        """Restart Ollama Docker container.
        
        This method stops the existing Ollama container and starts a new one,
        effectively refreshing the model state.
        
        Args:
            port: Port to expose Ollama on (default: 11434)
            gpu: Whether to enable GPU support (default: True)
            
        Returns:
            True if restart successful, False otherwise
        """
        try:
            self.logger.info("Restarting Ollama container...")
            
            # Get container name
            container_name = "forge-agent-ollama"
            
            # Try to stop container by name (works even if not in _llm_containers)
            try:
                container_id = await self._get_container_id(container_name)
                if container_id:
                    self.logger.info(f"Stopping Ollama container {container_id}")
                    await self._docker_command(["stop", container_id])
                    # Remove from tracking if present
                    if "ollama" in self._llm_containers:
                        del self._llm_containers["ollama"]
            except Exception as e:
                self.logger.warning(f"Error stopping Ollama container: {e}")
            
            # Wait a moment for container to fully stop
            await asyncio.sleep(2)
            
            # Start Ollama again
            success = await self.start_ollama(port=port, gpu=gpu)
            
            if success:
                self.logger.info("Ollama container restarted successfully")
            else:
                self.logger.error("Failed to restart Ollama container")
            
            return success
        except Exception as e:
            self.logger.error(f"Error restarting Ollama container: {e}")
            return False

    async def stop_containers(self):
        """Cleanup Docker manager (stop LLM containers, reset state)."""
        if not self._running:
            return

        # Stop LLM containers
        await self.stop_ollama()

        self._images_available.clear()
        self._running = False
        self.logger.info("Docker manager stopped")

    async def _get_container_id(self, container_name: str) -> Optional[str]:
        """Get container ID by name.

        Args:
            container_name: Container name

        Returns:
            Container ID or None if not found
        """
        result = await self._docker_command(
            ["ps", "-a", "--filter", f"name={container_name}", "--format", "{{.ID}}"],
            capture_output=True
        )
        if result and result.strip():
            return result.strip().split("\n")[0]
        return None

    async def _is_container_running(self, container_id: str) -> bool:
        """Check if container is running.

        Args:
            container_id: Container ID

        Returns:
            True if running, False otherwise
        """
        result = await self._docker_command(
            ["ps", "--filter", f"id={container_id}", "--format", "{{.ID}}"],
            capture_output=True
        )
        return bool(result and result.strip())


    async def _ensure_image(self, image: str):
        """Ensure Docker image is available, pull if not.

        Args:
            image: Docker image name
        """
        # Check if image exists
        result = await self._docker_command(["images", "-q", image], capture_output=True)
        
        if not result or not result.strip():
            self.logger.info(f"Pulling Docker image {image}")
            await self._docker_command(["pull", image])
        else:
            self.logger.debug(f"Docker image {image} already available")


    async def _docker_command(self, args: List[str], capture_output: bool = False) -> Optional[str]:
        """Execute Docker command.

        Args:
            args: Docker command arguments
            capture_output: Whether to capture and return output

        Returns:
            Command output if capture_output=True, None otherwise
        """
        cmd = ["docker"] + args
        
        try:
            if capture_output:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await process.communicate()
                
                if process.returncode != 0:
                    error_msg = stderr.decode() if stderr else "Unknown error"
                    self.logger.error(f"Docker command failed: {' '.join(cmd)} - {error_msg}")
                    return None
                
                return stdout.decode() if stdout else None
            else:
                process = await asyncio.create_subprocess_exec(*cmd)
                await process.communicate()
                
                if process.returncode != 0:
                    self.logger.error(f"Docker command failed: {' '.join(cmd)}")
                    return None
                
                return ""
        except Exception as e:
            self.logger.error(f"Error executing Docker command: {' '.join(cmd)} - {e}")
            return None


# Global Docker manager instance
_docker_manager: Optional[DockerManager] = None


def get_docker_manager() -> DockerManager:
    """Get global Docker manager instance.

    Returns:
        DockerManager instance
    """
    global _docker_manager
    if _docker_manager is None:
        _docker_manager = DockerManager()
    return _docker_manager
