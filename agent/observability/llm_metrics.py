"""LLM usage metrics tracking.

Tracks LLM usage per session including:
- Model name
- Tokens used (prompt + completion)
- Number of calls
- Total cost (if applicable)
"""

from collections import defaultdict
from typing import Dict, Optional
import time


class LLMUsageMetrics:
    """Tracks LLM usage metrics per session."""

    def __init__(self):
        """Initialize metrics tracker."""
        self._session_metrics: Dict[str, Dict] = defaultdict(lambda: {
            "model": None,
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "calls": 0,
            "last_used_at": None,
            "response_times": [],  # List of response times in seconds
        })

    def record_usage(
        self,
        session_id: str,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: Optional[int] = None,
        response_time: Optional[float] = None,
    ):
        """Record LLM usage for a session.
        
        Args:
            session_id: Session identifier
            model: Model name used
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            total_tokens: Total tokens (if provided, overrides prompt + completion)
            response_time: Response time in seconds (optional)
        """
        metrics = self._session_metrics[session_id]
        metrics["model"] = model
        metrics["calls"] += 1
        metrics["last_used_at"] = time.time()
        
        if total_tokens is not None:
            metrics["total_tokens"] += total_tokens
        else:
            metrics["prompt_tokens"] += prompt_tokens
            metrics["completion_tokens"] += completion_tokens
            metrics["total_tokens"] += (prompt_tokens + completion_tokens)
        
        # Track response time if provided
        if response_time is not None and response_time > 0:
            metrics["response_times"].append(response_time)
            # Keep only last 100 response times to avoid memory issues
            if len(metrics["response_times"]) > 100:
                metrics["response_times"] = metrics["response_times"][-100:]

    def get_session_metrics(self, session_id: str) -> Dict:
        """Get metrics for a specific session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with session metrics
        """
        return self._session_metrics.get(session_id, {
            "model": None,
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "calls": 0,
            "last_used_at": None,
            "response_times": [],
        })

    def get_all_sessions_metrics(self) -> Dict[str, Dict]:
        """Get metrics for all sessions.
        
        Returns:
            Dictionary mapping session_id to metrics
        """
        return dict(self._session_metrics)

    def get_global_metrics(self) -> Dict:
        """Get aggregated global LLM metrics.
        
        Returns:
            Dictionary with global metrics including per-model breakdown
        """
        all_metrics = self._session_metrics.items()  # Get (session_id, metrics) pairs
        
        # Aggregate across all sessions
        total_tokens = sum(m["total_tokens"] for _, m in all_metrics)
        total_calls = sum(m["calls"] for _, m in all_metrics)
        
        # Count unique models
        models_used = set(m["model"] for _, m in all_metrics if m["model"])
        
        # Get most recent usage
        last_used = max((m["last_used_at"] for _, m in all_metrics if m["last_used_at"]), default=None)
        
        # Aggregate metrics per model
        model_metrics: Dict[str, Dict] = defaultdict(lambda: {
            "total_tokens": 0,
            "total_calls": 0,
            "active_sessions": 0,
            "response_times": [],
            "last_used_at": None,
        })
        
        # Track which sessions are active per model
        model_sessions: Dict[str, set] = defaultdict(set)
        
        for session_id, m in all_metrics:
            if m["model"]:
                model = m["model"]
                model_metrics[model]["total_tokens"] += m["total_tokens"]
                model_metrics[model]["total_calls"] += m["calls"]
                if m["calls"] > 0:
                    model_sessions[model].add(session_id)
                if m.get("response_times"):
                    model_metrics[model]["response_times"].extend(m["response_times"])
                # Track last used timestamp per model
                if m.get("last_used_at"):
                    if model_metrics[model]["last_used_at"] is None:
                        model_metrics[model]["last_used_at"] = m["last_used_at"]
                    else:
                        model_metrics[model]["last_used_at"] = max(
                            model_metrics[model]["last_used_at"],
                            m["last_used_at"]
                        )
        
        # Calculate per-model metrics
        per_model_metrics = {}
        for model, metrics in model_metrics.items():
            avg_response_time = None
            if metrics["response_times"]:
                avg_response_time = round(sum(metrics["response_times"]) / len(metrics["response_times"]), 3)
            
            per_model_metrics[model] = {
                "total_tokens": metrics["total_tokens"],
                "total_calls": metrics["total_calls"],
                "active_sessions": len(model_sessions[model]),
                "avg_response_time": avg_response_time,
                "last_used_at": metrics["last_used_at"],
            }
        
        # Calculate average response time per model (for backward compatibility)
        model_avg_response_times = {}
        for model, metrics in per_model_metrics.items():
            if metrics["avg_response_time"] is not None:
                model_avg_response_times[model] = metrics["avg_response_time"]
        
        return {
            "total_tokens": total_tokens,
            "total_calls": total_calls,
            "models_used": list(models_used),
            "active_sessions": len([m for _, m in all_metrics if m["calls"] > 0]),
            "last_used_at": last_used,
            "model_avg_response_times": model_avg_response_times,
            "per_model": per_model_metrics,  # New: per-model breakdown
        }

    def clear_session(self, session_id: str):
        """Clear metrics for a specific session.
        
        Args:
            session_id: Session identifier
        """
        if session_id in self._session_metrics:
            del self._session_metrics[session_id]


# Global singleton instance
_llm_metrics: Optional[LLMUsageMetrics] = None


def get_llm_metrics() -> LLMUsageMetrics:
    """Get global LLM metrics tracker.
    
    Returns:
        LLMUsageMetrics singleton
    """
    global _llm_metrics
    if _llm_metrics is None:
        _llm_metrics = LLMUsageMetrics()
    return _llm_metrics
