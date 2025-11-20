"""Unified LLM analyzer with multiple backend support.

This module provides a unified interface for LLM analysis that can
automatically select and use the best available backend (local llama.cpp or llama.cpp server).
"""

from typing import Iterator, Optional, Dict, Any, Literal
import warnings
import requests
import json

from .llama_cpp_analyzer import LlamaCppAnalyzer, is_llama_cpp_available

Backend = Literal["auto", "llama_cpp", "llama_server", "none"]


class UnifiedLLMAnalyzer:
    """Unified interface for LLM analysis with multiple backends.

    Automatically selects the best available backend or allows manual selection.
    Supports both llama.cpp (for GGUF models) and llama.cpp server (OpenAI-compatible).

    Example:
        >>> # Auto-select backend
        >>> analyzer = UnifiedLLMAnalyzer(backend="auto")
        >>> for token in analyzer.stream_response("Analyze this"):
        ...     print(token, end="")

        >>> # Use specific backend
        >>> analyzer = UnifiedLLMAnalyzer(
        ...     backend="llama_cpp",
        ...     llama_cpp_model_path="/path/to/model.gguf"
        ... )
    """

    def __init__(
        self,
        backend: Backend = "auto",
        llama_cpp_model_path: Optional[str] = None,
        llama_cpp_n_ctx: int = 4096,
        llama_cpp_n_threads: int = 4,
        llama_cpp_n_gpu_layers: int = 0,
        llama_server_url: str = "http://localhost:8080",
        llama_server_model: str = "local",
    ):
        """Initialize unified LLM analyzer.

        Args:
            backend: Backend to use ("auto", "llama_cpp", "llama_server", "none")
            llama_cpp_model_path: Path to GGUF model (for llama.cpp backend)
            llama_cpp_n_ctx: Context window size for llama.cpp
            llama_cpp_n_threads: Number of threads for llama.cpp
            llama_cpp_n_gpu_layers: GPU layers for llama.cpp
            llama_server_url: Base URL for llama.cpp server (OpenAI-compatible)
            llama_server_model: Model name configured in llama-server
        """
        self.backend_type = backend
        self.llama_cpp_model_path = llama_cpp_model_path
        self.llama_server_url = llama_server_url.rstrip("/")
        self.llama_server_model = llama_server_model

        # Auto-select backend if requested
        if backend == "auto":
            self.backend_type = self._auto_select_backend()

        # Initialize the selected backend
        self.backend = None
        if self.backend_type == "llama_cpp":
            self._init_llama_cpp(
                llama_cpp_model_path,
                llama_cpp_n_ctx,
                llama_cpp_n_threads,
                llama_cpp_n_gpu_layers
            )
        elif self.backend_type == "llama_server":
            self._init_llama_server()
        elif self.backend_type == "none":
            warnings.warn("No LLM backend available")

    def _auto_select_backend(self) -> Backend:
        """Automatically select the best available backend.

        Priority:
        1. llama.cpp (if model path provided and library available)
        2. llama.cpp server (if reachable)
        3. none (no backend available)

        Returns:
            Selected backend type
        """
        # Try llama.cpp first if model path is provided
        if self.llama_cpp_model_path and is_llama_cpp_available():
            try:
                from pathlib import Path
                if Path(self.llama_cpp_model_path).exists():
                    return "llama_cpp"
            except Exception:
                pass

        # Fall back to llama.cpp server if reachable
        if self._is_llama_server_available():
            return "llama_server"

        # No backend available
        return "none"

    def _init_llama_cpp(
        self,
        model_path: Optional[str],
        n_ctx: int,
        n_threads: int,
        n_gpu_layers: int
    ) -> None:
        """Initialize llama.cpp backend."""
        if not model_path:
            raise ValueError("llama_cpp_model_path must be provided for llama.cpp backend")

        try:
            self.backend = LlamaCppAnalyzer(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers
            )
        except Exception as e:
            warnings.warn(f"Failed to initialize llama.cpp: {e}")
            self.backend_type = "none"

    def _init_llama_server(self) -> None:
        """Initialize llama.cpp server backend (verify it is available)."""
        if not self._is_llama_server_available():
            warnings.warn(
                f"llama-server not available at {self.llama_server_url}. "
                "Start it with: llama-server -m <model.gguf> --port 8080"
            )
            self.backend_type = "none"

    def _is_llama_server_available(self) -> bool:
        """Check if llama.cpp server is available."""
        try:
            response = requests.get(
                f"{self.llama_server_url}/v1/models",
                timeout=2
            )
            return response.status_code == 200
        except Exception:
            return False

    def stream_response(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        *,
        image_base64: Optional[str] = None,
    ) -> Iterator[str]:
        """Stream response from the configured backend.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Yields:
            str: Generated text tokens
        """
        if self.backend_type == "llama_cpp" and self.backend:
            yield from self.backend.stream_response(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
        elif self.backend_type == "llama_server":
            yield from self._stream_llama_server_response(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                image_base64=image_base64,
            )
        else:
            yield (
                "LLM analysis not available. Please:\n"
                "1. Install llama-cpp-python and provide a model path, OR\n"
                "2. Run llama-server with a GGUF model (OpenAI-compatible endpoint)\n"
                "\n"
                "See documentation for setup instructions."
            )

    def _stream_llama_server_response(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        image_base64: Optional[str] = None,
    ) -> Iterator[str]:
        """Stream response from llama.cpp server (OpenAI-compatible, vision-aware)."""
        url = f"{self.llama_server_url}/v1/chat/completions"
        content = [{"type": "text", "text": prompt}]
        if image_base64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                }
            )

        payload = {
            "model": self.llama_server_model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        try:
            with requests.post(url, json=payload, stream=True, timeout=180) as r:
                for raw_line in r.iter_lines():
                    if not raw_line:
                        continue

                    line = raw_line
                    if raw_line.startswith(b"data:"):
                        line = raw_line[len(b"data:") :].strip()

                    try:
                        data = json.loads(line.decode("utf-8"))
                    except Exception:
                        continue

                    choice = None
                    if isinstance(data, dict) and data.get("choices"):
                        choice = data["choices"][0]

                    text = ""
                    if choice:
                        if isinstance(choice, dict):
                            if "delta" in choice and choice["delta"].get("content"):
                                delta = choice["delta"]["content"]
                                if isinstance(delta, list):
                                    text = "".join(
                                        part.get("text", "")
                                        for part in delta
                                        if isinstance(part, dict)
                                    )
                                elif isinstance(delta, str):
                                    text = delta
                            elif "message" in choice and choice["message"].get("content"):
                                msg = choice["message"]["content"]
                                if isinstance(msg, list):
                                    text = "".join(
                                        part.get("text", "")
                                        for part in msg
                                        if isinstance(part, dict)
                                    )
                                elif isinstance(msg, str):
                                    text = msg
                    if text:
                        yield text
        except Exception as exc:
            yield f"\n\nError connecting to llama-server: {exc}"

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> str:
        """Generate complete response (non-streaming).

        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            str: Complete generated text
        """
        if self.backend_type == "llama_cpp" and self.backend:
            return self.backend.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
        elif self.backend_type == "llama_server":
            # Collect streaming response
            return "".join(
                self._stream_llama_server_response(prompt, max_tokens, temperature)
            )
        else:
            return "LLM analysis not available."

    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the current backend.

        Returns:
            Dict containing backend information
        """
        info = {
            'backend_type': self.backend_type,
            'available': self.backend_type != "none"
        }

        if self.backend_type == "llama_cpp" and self.backend:
            info.update({
                'model_path': self.llama_cpp_model_path,
                'model_info': self.backend.get_model_info()
            })
        elif self.backend_type == "llama_server":
            info.update({
                'llama_server_url': self.llama_server_url,
                'llama_server_model': self.llama_server_model,
            })

        return info

    def __repr__(self) -> str:
        """String representation of the analyzer."""
        if self.backend_type == "llama_cpp":
            return f"UnifiedLLMAnalyzer(backend=llama.cpp, model={self.llama_cpp_model_path})"
        elif self.backend_type == "llama_server":
            return f"UnifiedLLMAnalyzer(backend=llama.cpp server, url={self.llama_server_url})"
        else:
            return "UnifiedLLMAnalyzer(backend=none)"


def create_analyzer(
    backend: Backend = "auto",
    **kwargs
) -> UnifiedLLMAnalyzer:
    """Factory function to create a UnifiedLLMAnalyzer.

    Args:
        backend: Backend to use ("auto", "llama_cpp", "llama_server", "none")
        **kwargs: Additional arguments passed to UnifiedLLMAnalyzer

    Returns:
        UnifiedLLMAnalyzer instance
    """
    return UnifiedLLMAnalyzer(backend=backend, **kwargs)
