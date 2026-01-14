"""AirLLM provider implementation.

AirLLM allows running large models (up to 70B) on small GPUs (4GB) by using
layer-wise loading and optional compression (4bit/8bit).

See: https://github.com/lyogavin/airllm
"""

import logging
from typing import Any, Dict, List

from agent.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class AirLLMProvider(LLMProvider):
    """AirLLM provider for local inference with large models on small GPUs."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize AirLLM provider.

        Args:
            config: Configuration dict with:
                - model: HuggingFace model name (e.g., "Qwen/Qwen-7B")
                - compression: Optional compression ("4bit", "8bit", or None)
                - temperature: Temperature (default: 0.1)
                - max_tokens: Max tokens to generate (default: 4096)
                - timeout: Request timeout in seconds (default: 300)
                - hf_token: Optional HuggingFace token for gated models
                - profiling_mode: Enable profiling (default: False)
                - layer_shards_saving_path: Optional path for layer shards
                - delete_original: Delete original model after splitting (default: False)
        """
        super().__init__(config)
        
        try:
            from airllm import AutoModel
        except ImportError:
            raise ImportError(
                "airllm is not installed. Install it with: pip install airllm"
            )
        
        self.AutoModel = AutoModel
        self.model_name = config.get("model", "Qwen/Qwen-7B")
        self.compression = config.get("compression", None)  # "4bit", "8bit", or None
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 4096)
        self.timeout = config.get("timeout", 300)
        self.hf_token = config.get("hf_token", None)
        self.profiling_mode = config.get("profiling_mode", False)
        self.layer_shards_saving_path = config.get("layer_shards_saving_path", None)
        self.delete_original = config.get("delete_original", False)
        
        # Model will be loaded lazily on first use
        self._model = None
        self._tokenizer = None

    def _ensure_model_loaded(self):
        """Load model if not already loaded."""
        if self._model is None:
            logger.info(f"Loading AirLLM model: {self.model_name} (compression={self.compression})")
            
            model_kwargs = {
                "profiling_mode": self.profiling_mode,
            }
            
            if self.compression:
                model_kwargs["compression"] = self.compression
            
            if self.layer_shards_saving_path:
                model_kwargs["layer_shards_saving_path"] = self.layer_shards_saving_path
            
            if self.delete_original:
                model_kwargs["delete_original"] = self.delete_original
            
            if self.hf_token:
                model_kwargs["hf_token"] = self.hf_token
            
            self._model = self.AutoModel.from_pretrained(self.model_name, **model_kwargs)
            self._tokenizer = self._model.tokenizer
            logger.info(f"AirLLM model loaded successfully: {self.model_name}")

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from prompt.

        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        messages = [{"role": "user", "content": prompt}]
        return await self.chat(messages, **kwargs)

    async def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Generate response from chat messages.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional generation parameters:
                - temperature: Override default temperature
                - max_tokens: Max tokens to generate
                - session_id: Optional session ID for metrics tracking

        Returns:
            Generated response text
        """
        import asyncio
        import time
        
        # Ensure model is loaded
        self._ensure_model_loaded()
        
        # Use kwargs temperature if provided, otherwise use instance default
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        session_id = kwargs.get("session_id")
        
        # Convert messages to prompt format
        # AirLLM expects a single prompt string or tokenized input
        # For chat format, we'll concatenate messages
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_text = "\n".join(prompt_parts) + "\nAssistant:"
        
        # Tokenize input
        input_tokens = self._tokenizer(
            [prompt_text],
            return_tensors="pt",
            return_attention_mask=False,
            truncation=True,
            max_length=2048,  # Max input length
            padding=False,
        )
        
        # Move to GPU if available
        try:
            import torch
            if torch.cuda.is_available():
                input_tokens["input_ids"] = input_tokens["input_ids"].cuda()
        except ImportError:
            # torch not available, use CPU
            pass
        
        # Generate in executor to avoid blocking
        loop = asyncio.get_event_loop()
        request_start = time.time()
        
        def _generate():
            """Synchronous generation function."""
            generation_output = self._model.generate(
                input_tokens["input_ids"],
                max_new_tokens=min(max_tokens, 2048),  # AirLLM may have limits
                use_cache=True,
                return_dict_in_generate=True,
                temperature=temperature,
            )
            return generation_output
        
        try:
            generation_output = await asyncio.wait_for(
                loop.run_in_executor(None, _generate),
                timeout=self.timeout
            )
            
            request_duration = time.time() - request_start
            logger.info(
                f"AirLLM request completed in {request_duration:.2f}s "
                f"(model={self.model_name}, messages={len(messages)})"
            )
            
            # Decode output
            output_text = self._tokenizer.decode(generation_output.sequences[0])
            
            # Extract only the generated part (remove input prompt)
            # The output includes the input, so we need to extract just the new tokens
            input_length = input_tokens["input_ids"].shape[1]
            generated_sequences = generation_output.sequences[0][input_length:]
            generated_text = self._tokenizer.decode(generated_sequences, skip_special_tokens=True)
            
            # Track usage metrics if session_id provided
            if session_id:
                try:
                    from agent.observability.llm_metrics import get_llm_metrics
                    
                    # Estimate token usage (AirLLM doesn't provide exact counts)
                    # Use input length + generated length as approximation
                    prompt_tokens = input_length
                    completion_tokens = len(generated_sequences)
                    total_tokens = prompt_tokens + completion_tokens
                    
                    metrics = get_llm_metrics()
                    metrics.record_usage(
                        session_id=session_id,
                        model=self.model_name,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                    )
                    
                    logger.debug(
                        f"Recorded LLM usage for session {session_id}: "
                        f"model={self.model_name}, tokens={total_tokens}, calls=1"
                    )
                except Exception as e:
                    logger.warning(f"Failed to track LLM metrics: {e}", exc_info=True)
            
            return generated_text.strip()
            
        except asyncio.TimeoutError:
            raise TimeoutError(f"AirLLM generation timed out after {self.timeout}s")
        except Exception as e:
            logger.error(f"AirLLM generation failed: {e}", exc_info=True)
            raise RuntimeError(f"AirLLM generation error: {e}") from e
