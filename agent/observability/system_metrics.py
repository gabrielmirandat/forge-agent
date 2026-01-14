"""System metrics collector for real-time observability.

Collects system metrics including:
- CPU usage and temperature
- Memory usage
- GPU usage, temperature, and power (if available)
- Disk I/O
- Network I/O

This module provides a passive collector that can be queried on-demand.
"""

import subprocess
import time
from typing import Any, Dict, Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class SystemMetricsCollector:
    """Collects system metrics for observability.
    
    Provides read-only access to system metrics without side effects.
    """

    def __init__(self):
        """Initialize metrics collector."""
        self._gpu_available = self._check_gpu_available()

    def _check_gpu_available(self) -> bool:
        """Check if GPU monitoring is available (nvidia-smi).
        
        Returns:
            True if nvidia-smi is available, False otherwise
        """
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                timeout=2,
                text=True
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def collect_all(self) -> Dict[str, Any]:
        """Collect all available system metrics.
        
        Returns:
            Dictionary with all collected metrics
        """
        metrics = {
            "timestamp": time.time(),
            "cpu": self.collect_cpu(),
            "memory": self.collect_memory(),
            "disk": self.collect_disk(),
            "network": self.collect_network(),
        }
        
        if self._gpu_available:
            metrics["gpu"] = self.collect_gpu()
        else:
            metrics["gpu"] = {"available": False}
        
        return metrics

    def collect_cpu(self) -> Dict[str, Any]:
        """Collect CPU metrics.
        
        Returns:
            CPU metrics (usage, temperature if available)
        """
        if not PSUTIL_AVAILABLE:
            return {"available": False, "error": "psutil not available"}
        
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Try to get CPU temperature (platform-dependent)
            cpu_temp = None
            try:
                if hasattr(psutil, "sensors_temperatures"):
                    temps = psutil.sensors_temperatures()
                    if temps:
                        # Get first available temperature sensor
                        for sensor_name, sensor_list in temps.items():
                            if sensor_list:
                                cpu_temp = sensor_list[0].current
                                break
            except Exception:
                pass
            
            return {
                "available": True,
                "usage_percent": round(cpu_percent, 2),
                "cores": cpu_count,
                "frequency_mhz": round(cpu_freq.current, 2) if cpu_freq else None,
                "temperature_celsius": round(cpu_temp, 2) if cpu_temp else None,
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

    def collect_memory(self) -> Dict[str, Any]:
        """Collect memory metrics.
        
        Returns:
            Memory metrics (usage, available, total)
        """
        if not PSUTIL_AVAILABLE:
            return {"available": False, "error": "psutil not available"}
        
        try:
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return {
                "available": True,
                "total_gb": round(mem.total / (1024**3), 2),
                "used_gb": round(mem.used / (1024**3), 2),
                "available_gb": round(mem.available / (1024**3), 2),
                "usage_percent": round(mem.percent, 2),
                "swap_total_gb": round(swap.total / (1024**3), 2),
                "swap_used_gb": round(swap.used / (1024**3), 2),
                "swap_usage_percent": round(swap.percent, 2),
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

    def collect_disk(self) -> Dict[str, Any]:
        """Collect disk I/O metrics.
        
        Returns:
            Disk metrics (usage, I/O)
        """
        if not PSUTIL_AVAILABLE:
            return {"available": False, "error": "psutil not available"}
        
        try:
            disk = psutil.disk_usage("/")
            disk_io = psutil.disk_io_counters()
            
            result = {
                "available": True,
                "total_gb": round(disk.total / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "usage_percent": round(disk.percent, 2),
            }
            
            if disk_io:
                result["io"] = {
                    "read_mb": round(disk_io.read_bytes / (1024**2), 2),
                    "write_mb": round(disk_io.write_bytes / (1024**2), 2),
                    "read_count": disk_io.read_count,
                    "write_count": disk_io.write_count,
                }
            
            return result
        except Exception as e:
            return {"available": False, "error": str(e)}

    def collect_network(self) -> Dict[str, Any]:
        """Collect network I/O metrics.
        
        Returns:
            Network metrics (bytes sent/received)
        """
        if not PSUTIL_AVAILABLE:
            return {"available": False, "error": "psutil not available"}
        
        try:
            net_io = psutil.net_io_counters()
            
            if not net_io:
                return {"available": False, "error": "No network data available"}
            
            return {
                "available": True,
                "bytes_sent_mb": round(net_io.bytes_sent / (1024**2), 2),
                "bytes_recv_mb": round(net_io.bytes_recv / (1024**2), 2),
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv,
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

    def collect_gpu(self) -> Dict[str, Any]:
        """Collect GPU metrics using nvidia-smi.
        
        Returns:
            GPU metrics (usage, temperature, power, memory)
        """
        if not self._gpu_available:
            return {"available": False}
        
        try:
            # Query GPU information
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                timeout=2,
                text=True
            )
            
            if result.returncode != 0:
                return {"available": False, "error": "nvidia-smi query failed"}
            
            lines = result.stdout.strip().split("\n")
            if not lines or not lines[0]:
                return {"available": False, "error": "No GPU data"}
            
            # Parse first GPU (assuming single GPU for now)
            gpu_data = [x.strip() for x in lines[0].split(",")]
            
            if len(gpu_data) < 8:
                return {"available": False, "error": "Incomplete GPU data"}
            
            return {
                "available": True,
                "name": gpu_data[0],
                "temperature_celsius": float(gpu_data[1]) if gpu_data[1] else None,
                "utilization_percent": float(gpu_data[2]) if gpu_data[2] else None,
                "memory_utilization_percent": float(gpu_data[3]) if gpu_data[3] else None,
                "memory_used_mb": float(gpu_data[4]) if gpu_data[4] else None,
                "memory_total_mb": float(gpu_data[5]) if gpu_data[5] else None,
                "power_draw_watts": float(gpu_data[6]) if gpu_data[6] else None,
                "power_limit_watts": float(gpu_data[7]) if gpu_data[7] else None,
            }
        except subprocess.TimeoutExpired:
            return {"available": False, "error": "nvidia-smi timeout"}
        except Exception as e:
            return {"available": False, "error": str(e)}


# Global singleton instance
_metrics_collector: Optional[SystemMetricsCollector] = None


def get_metrics_collector() -> SystemMetricsCollector:
    """Get global metrics collector instance.
    
    Returns:
        SystemMetricsCollector singleton
    """
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = SystemMetricsCollector()
    return _metrics_collector
