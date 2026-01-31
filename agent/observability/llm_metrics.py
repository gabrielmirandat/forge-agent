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
        """Initialize metrics tracker.
        
        Structure: _session_metrics[session_id][model] = metrics
        This allows tracking multiple models per session.
        """
        # Changed structure: session_id -> model -> metrics
        # This allows a session to use multiple models and track them separately
        self._session_metrics: Dict[str, Dict[str, Dict]] = defaultdict(lambda: defaultdict(lambda: {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "calls": 0,
            "last_used_at": None,
            "response_times": [],  # List of response times in seconds
        }))

    def record_usage(
        self,
        session_id: str,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: Optional[int] = None,
        response_time: Optional[float] = None,
    ):
        """Record LLM usage for a session and model.
        
        Args:
            session_id: Session identifier
            model: Model name used (each session can use multiple models)
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            total_tokens: Total tokens (if provided, overrides prompt + completion)
            response_time: Response time in seconds (optional)
        """
        # Track metrics per model within each session
        # This allows a session to use multiple models and track them separately
        metrics = self._session_metrics[session_id][model]
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
            Dictionary with session metrics (aggregated across all models used in this session)
        """
        session_models = self._session_metrics.get(session_id, {})
        
        # Aggregate across all models used in this session
        total_tokens = sum(m["total_tokens"] for m in session_models.values())
        prompt_tokens = sum(m["prompt_tokens"] for m in session_models.values())
        completion_tokens = sum(m["completion_tokens"] for m in session_models.values())
        calls = sum(m["calls"] for m in session_models.values())
        last_used = max((m["last_used_at"] for m in session_models.values() if m["last_used_at"]), default=None)
        response_times = []
        for m in session_models.values():
            response_times.extend(m["response_times"])
        
        # Get most recently used model
        most_recent_model = None
        most_recent_time = None
        for model, m in session_models.items():
            if m["last_used_at"] and (most_recent_time is None or m["last_used_at"] > most_recent_time):
                most_recent_time = m["last_used_at"]
                most_recent_model = model
        
        return {
            "model": most_recent_model,  # Most recently used model
            "total_tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "calls": calls,
            "last_used_at": last_used,
            "response_times": response_times,
        }

    def get_all_sessions_metrics(self) -> Dict[str, Dict]:
        """Get metrics for all sessions.
        
        Returns:
            Dictionary mapping session_id to aggregated metrics (across all models used in that session)
            For backward compatibility, returns the same format as before but aggregated from per-model data
        """
        result = {}
        for session_id, models_dict in self._session_metrics.items():
            # Aggregate metrics across all models for this session
            total_tokens = sum(m["total_tokens"] for m in models_dict.values())
            prompt_tokens = sum(m["prompt_tokens"] for m in models_dict.values())
            completion_tokens = sum(m["completion_tokens"] for m in models_dict.values())
            calls = sum(m["calls"] for m in models_dict.values())
            last_used = max((m["last_used_at"] for m in models_dict.values() if m["last_used_at"]), default=None)
            response_times = []
            for m in models_dict.values():
                response_times.extend(m["response_times"])
            
            # Get most recently used model
            most_recent_model = None
            most_recent_time = None
            for model, m in models_dict.items():
                if m["last_used_at"] and (most_recent_time is None or m["last_used_at"] > most_recent_time):
                    most_recent_time = m["last_used_at"]
                    most_recent_model = model
            
            result[session_id] = {
                "model": most_recent_model,  # Most recently used model
                "total_tokens": total_tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "calls": calls,
                "last_used_at": last_used,
                "response_times": response_times,
            }
        return result

    def get_global_metrics(self) -> Dict:
        """Get aggregated global LLM metrics.
        
        Returns:
            Dictionary with global metrics including per-model breakdown
        """
        # New structure: _session_metrics[session_id][model] = metrics
        # Iterate through all sessions and their models
        all_sessions = self._session_metrics.items()  # Get (session_id, models_dict) pairs
        
        # Aggregate metrics per model across all sessions
        model_metrics: Dict[str, Dict] = defaultdict(lambda: {
            "total_tokens": 0,
            "total_calls": 0,
            "active_sessions": 0,
            "response_times": [],
            "last_used_at": None,
        })
        
        # Track which sessions are active per model
        model_sessions: Dict[str, set] = defaultdict(set)
        
        # Aggregate totals across all sessions and models
        total_tokens = 0
        total_calls = 0
        models_used = set()
        last_used = None
        
        for session_id, models_dict in all_sessions:
            for model, m in models_dict.items():
                if model:  # Only process if model name exists
                    models_used.add(model)
                    model_metrics[model]["total_tokens"] += m["total_tokens"]
                    model_metrics[model]["total_calls"] += m["calls"]
                    total_tokens += m["total_tokens"]
                    total_calls += m["calls"]
                    
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
                        
                        # Track global last used
                        if last_used is None or m["last_used_at"] > last_used:
                            last_used = m["last_used_at"]
        
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
        
        # Count active sessions (sessions with at least one call)
        active_sessions_count = 0
        active_session_ids = set()
        for session_id, models_dict in all_sessions:
            for model, m in models_dict.items():
                if m["calls"] > 0:
                    active_session_ids.add(session_id)
        active_sessions_count = len(active_session_ids)
        
        return {
            "total_tokens": total_tokens,
            "total_calls": total_calls,
            "models_used": list(models_used),
            "active_sessions": active_sessions_count,
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
