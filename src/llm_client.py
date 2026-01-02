"""
Unified LLM client that supports both OpenAI and Anthropic Claude.
Provides a consistent interface for streaming chat completions.
"""
import os
from typing import Iterator, Dict, Any, Optional


class LLMClient:
    """Unified interface for OpenAI and Anthropic LLM APIs."""

    def __init__(self, provider: str = "openai", model: Optional[str] = None):
        """
        Initialize LLM client.

        Args:
            provider: Either "openai" or "anthropic" (default: "openai")
            model: Model name to use (optional, uses defaults if not specified)
        """
        self.provider = provider.lower()

        if self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = model or "gpt-5-mini"  # Default to GPT-5 Mini
        elif self.provider == "anthropic":
            from anthropic import Anthropic
            self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.model = model or "claude-sonnet-4-5-20250929"  # Default to Claude Sonnet 4.5
        else:
            raise ValueError(f"Unknown provider: {provider}. Must be 'openai' or 'anthropic'")

    def stream_chat(self, messages: list, max_tokens: int = 2000) -> Iterator[str]:
        """
        Stream chat completions from the LLM.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum tokens to generate

        Yields:
            Text chunks as they arrive from the LLM
        """
        if self.provider == "openai":
            yield from self._stream_openai(messages, max_tokens)
        elif self.provider == "anthropic":
            yield from self._stream_anthropic(messages, max_tokens)

    def _stream_openai(self, messages: list, max_tokens: int) -> Iterator[str]:
        """Stream from OpenAI API."""
        print(f"[OpenAI API] Model: {self.model}, Max tokens: {max_tokens}")
        print(f"[OpenAI API] Messages: {len(messages)} messages")

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=max_tokens,
            stream=True
        )

        chunk_count = 0
        content_chars = 0
        for chunk in stream:
            chunk_count += 1

            # Debug: show finish reason
            if chunk.choices[0].finish_reason:
                print(f"[OpenAI API] Finish reason: {chunk.choices[0].finish_reason}")

            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                content_chars += len(content)
                yield content

        print(f"[OpenAI API] Stream complete: {chunk_count} chunks, {content_chars} characters")

    def _stream_anthropic(self, messages: list, max_tokens: int) -> Iterator[str]:
        """Stream from Anthropic API."""
        stream = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=messages,
            stream=True
        )

        for chunk in stream:
            if chunk.type == "content_block_delta":
                yield chunk.delta.text


def get_llm_client(provider: Optional[str] = None, model: Optional[str] = None) -> LLMClient:
    """
    Factory function to create an LLM client.

    Args:
        provider: LLM provider ("openai" or "anthropic").
                 Defaults to "openai" or env var LLM_PROVIDER
        model: Model name (optional, uses provider defaults)

    Returns:
        Configured LLMClient instance
    """
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "openai")

    return LLMClient(provider=provider, model=model)
