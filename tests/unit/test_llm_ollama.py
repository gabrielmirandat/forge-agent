"""Unit tests for Ollama LLM provider (with mocked HTTP)."""

import pytest
from unittest.mock import AsyncMock, patch

from agent.llm.ollama import OllamaProvider
import httpx


class TestOllamaProvider:
    """Test OllamaProvider with mocked HTTP."""

    @pytest.fixture
    def provider(self):
        """Create Ollama provider."""
        return OllamaProvider({
            "base_url": "http://localhost:11434",
            "model": "test-model",
            "temperature": 0.1,
            "timeout": 30,
        })

    def test_provider_initialization(self, provider):
        """Test provider initialization."""
        assert provider.base_url == "http://localhost:11434"
        assert provider.model == "test-model"
        assert provider.temperature == 0.1
        assert provider.timeout == 30

    def test_provider_defaults(self):
        """Test provider with default values."""
        provider = OllamaProvider({})
        assert provider.base_url == "http://localhost:11434"
        assert provider.model == "qwen2.5-coder:7b"
        assert provider.temperature == 0.1
        assert provider.timeout == 300

    @pytest.mark.asyncio
    async def test_chat_success(self, provider):
        """Test successful chat."""
        mock_response = {
            "message": {
                "content": "Test response"
            }
        }
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_client_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            mock_response_obj = AsyncMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status = AsyncMock()
            mock_client_instance.post.return_value = mock_response_obj
            
            result = await provider.chat([{"role": "user", "content": "test"}])
            
            assert result == "Test response"
            mock_client_instance.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_http_error(self, provider):
        """Test chat with HTTP error."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            mock_client_instance.post.side_effect = httpx.HTTPStatusError(
                "Error", request=AsyncMock(), response=AsyncMock()
            )
            
            with pytest.raises(httpx.HTTPError):
                await provider.chat([{"role": "user", "content": "test"}])

    @pytest.mark.asyncio
    async def test_chat_invalid_response(self, provider):
        """Test chat with invalid response format."""
        mock_response = {"invalid": "format"}
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_client_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            mock_response_obj = AsyncMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status = AsyncMock()
            mock_client_instance.post.return_value = mock_response_obj
            
            with pytest.raises(ValueError, match="Unexpected response format"):
                await provider.chat([{"role": "user", "content": "test"}])

    @pytest.mark.asyncio
    async def test_generate(self, provider):
        """Test generate method."""
        mock_response = {
            "message": {
                "content": "Generated text"
            }
        }
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_client_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            mock_response_obj = AsyncMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status = AsyncMock()
            mock_client_instance.post.return_value = mock_response_obj
            
            result = await provider.generate("test prompt")
            
            assert result == "Generated text"

    @pytest.mark.asyncio
    async def test_chat_with_custom_temperature(self, provider):
        """Test chat with custom temperature override."""
        mock_response = {
            "message": {
                "content": "Test"
            }
        }
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_client_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            mock_response_obj = AsyncMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status = AsyncMock()
            mock_client_instance.post.return_value = mock_response_obj
            
            await provider.chat([{"role": "user", "content": "test"}], temperature=0.5)
            
            # Verify temperature was passed in payload
            call_args = mock_client_instance.post.call_args
            payload = call_args[1]["json"]
            assert payload["options"]["temperature"] == 0.5
