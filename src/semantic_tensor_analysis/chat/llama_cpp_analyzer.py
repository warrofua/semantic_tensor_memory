"""LLM analysis using llama.cpp for local GGUF models.

This module provides an alternative LLM backend using llama.cpp,
which offers faster inference and lower memory footprint compared to Ollama.
"""

from typing import Iterator, Optional, Dict, Any
from pathlib import Path
import warnings

# Try to import llama-cpp-python, but make it optional
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True

    # Defensive patch: older llama_cpp versions can raise AttributeError in LlamaModel.__del__
    # when initialization fails mid-way. Wrap the destructor to ignore missing attrs.
    try:  # pragma: no cover - runtime safeguard
        from llama_cpp import LlamaModel as _LlamaModel  # type: ignore

        _orig_del = getattr(_LlamaModel, "__del__", None)

        def _safe_del(self) -> None:  # type: ignore
            try:
                if _orig_del:
                    _orig_del(self)
            except AttributeError:
                # Ignore partially constructed objects
                return

        if _orig_del:
            _LlamaModel.__del__ = _safe_del  # type: ignore
    except Exception:
        pass
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    warnings.warn(
        "llama-cpp-python not available. Install with: pip install llama-cpp-python"
    )


class LlamaCppAnalyzer:
    """LLM analysis using llama.cpp for local GGUF models.

    This provides a drop-in replacement for Ollama with better performance
    for local model inference.

    Example:
        >>> analyzer = LlamaCppAnalyzer("/path/to/model.gguf")
        >>> for token in analyzer.stream_response("Analyze this data"):
        ...     print(token, end="")
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_threads: int = 4,
        n_gpu_layers: int = 0,
        verbose: bool = False
    ):
        """Initialize llama.cpp model.

        Args:
            model_path: Path to GGUF model file
            n_ctx: Context window size (default: 4096)
            n_threads: Number of CPU threads (default: 4)
            n_gpu_layers: Number of layers to offload to GPU (default: 0)
            verbose: Enable verbose logging (default: False)

        Raises:
            ImportError: If llama-cpp-python is not installed
            FileNotFoundError: If model file doesn't exist
        """
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python is not installed. "
                "Install with: pip install llama-cpp-python"
            )

        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model_path = str(model_path_obj)
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers

        # Initialize the model
        try:
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                verbose=verbose,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load llama.cpp model: {e}") from e

    def stream_response(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: Optional[list] = None
    ) -> Iterator[str]:
        """Stream response from llama.cpp model.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate (default: 1024)
            temperature: Sampling temperature (default: 0.7)
            top_p: Nucleus sampling parameter (default: 0.95)
            stop: Stop sequences (default: None)

        Yields:
            str: Generated text tokens
        """
        try:
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                stream=True,
                echo=False,
            )

            for output in response:
                if 'choices' in output and len(output['choices']) > 0:
                    choice = output['choices'][0]
                    if 'text' in choice:
                        yield choice['text']
        except Exception as e:
            yield f"\n\nError during generation: {e}"

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: Optional[list] = None
    ) -> str:
        """Generate complete response (non-streaming).

        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate (default: 1024)
            temperature: Sampling temperature (default: 0.7)
            top_p: Nucleus sampling parameter (default: 0.95)
            stop: Stop sequences (default: None)

        Returns:
            str: Complete generated text
        """
        try:
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                stream=False,
                echo=False,
            )

            if 'choices' in response and len(response['choices']) > 0:
                return response['choices'][0]['text']
            return ""
        except Exception as e:
            return f"Error during generation: {e}"

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.

        Returns:
            Dict containing model metadata
        """
        return {
            'model_path': self.model_path,
            'n_ctx': self.n_ctx,
            'n_threads': self.n_threads,
            'n_gpu_layers': self.n_gpu_layers,
        }

    def __repr__(self) -> str:
        """String representation of the analyzer."""
        return (
            f"LlamaCppAnalyzer(model_path={Path(self.model_path).name}, "
            f"n_ctx={self.n_ctx}, n_threads={self.n_threads})"
        )


def is_llama_cpp_available() -> bool:
    """Check if llama-cpp-python is available.

    Returns:
        bool: True if llama-cpp-python is installed
    """
    return LLAMA_CPP_AVAILABLE


def get_recommended_models() -> list:
    """Get list of recommended GGUF models for semantic analysis.

    Returns:
        List of dictionaries with model information
    """
    return [
        {
            'name': 'Mistral-7B-Instruct',
            'size': '~4.1GB',
            'url': 'https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF',
            'recommended_file': 'mistral-7b-instruct-v0.2.Q4_K_M.gguf',
            'description': 'Excellent for analysis tasks, good reasoning'
        },
        {
            'name': 'Llama-3-8B-Instruct',
            'size': '~4.7GB',
            'url': 'https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF',
            'recommended_file': 'Meta-Llama-3-8B-Instruct-Q4_K_M.gguf',
            'description': 'Latest Llama model, strong performance'
        },
        {
            'name': 'Qwen2-7B-Instruct',
            'size': '~4.4GB',
            'url': 'https://huggingface.co/Qwen/Qwen2-7B-Instruct-GGUF',
            'recommended_file': 'qwen2-7b-instruct-q4_k_m.gguf',
            'description': 'Good for detailed analysis and explanation'
        },
        {
            'name': 'Phi-3-Mini-Instruct',
            'size': '~2.3GB',
            'url': 'https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf',
            'recommended_file': 'Phi-3-mini-4k-instruct-q4.gguf',
            'description': 'Smaller model, faster inference, good quality'
        }
    ]
