from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RuntimeReport:
    python_version: str
    platform: str
    torch_version: str | None
    cuda_available: bool | None
    cuda_version: str | None
    gpu_name: str | None
    numpy_version: str | None


def _safe_import_version(module_name: str) -> str | None:
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", None)
    except Exception:
        return None


def get_runtime_report() -> RuntimeReport:
    python_version = platform.python_version()
    plat = platform.platform()

    torch_version = None
    cuda_available = None
    cuda_version = None
    gpu_name = None
    try:
        import torch

        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    numpy_version = _safe_import_version("numpy")

    return RuntimeReport(
        python_version=python_version,
        platform=plat,
        torch_version=torch_version,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        gpu_name=gpu_name,
        numpy_version=numpy_version,
    )


def print_runtime_report(extra: dict[str, Any] | None = None) -> RuntimeReport:
    r = get_runtime_report()
    print("============================================================")
    print("Runtime Report")
    print("============================================================")
    print(f"Python: {r.python_version}")
    print(f"Platform: {r.platform}")
    print(f"Torch: {r.torch_version}")
    print(f"CUDA available: {r.cuda_available}")
    print(f"CUDA version: {r.cuda_version}")
    print(f"GPU: {r.gpu_name}")
    print(f"Numpy: {r.numpy_version}")
    if extra:
        for k, v in extra.items():
            print(f"{k}: {v}")
    return r


def assert_supported_colab_runtime() -> None:
    """
    Lightweight guardrail for Colab runtime versions.

    Per the Colab runtime FAQ, recent runtimes commonly use:
    - Python 3.11 (e.g., 2025.07)
    - Python 3.12 (e.g., 2025.10)
    and modern PyTorch.
    """
    r = get_runtime_report()
    major_minor = tuple(int(x) for x in r.python_version.split(".")[:2])
    if major_minor < (3, 11):
        raise RuntimeError(
            "Python < 3.11 detected. Please switch to a newer Colab runtime "
            "(Runtime → Change runtime type → Runtime Version) before running."
        )

    # Torch is optional for import-time usage, but required for training.
    if not r.torch_version:
        raise RuntimeError("PyTorch not found. Ensure your runtime has torch installed.")


def read_secret(name: str) -> str | None:
    """
    Read a secret from environment variables.

    In Colab, Secrets are typically exposed as env vars.
    Also tries to load from .env file if python-dotenv is available.
    """
    # Try to load from .env file if available (for local dev or Colab)
    try:
        from dotenv import load_dotenv
        # Try to find .env in common locations
        import sys
        from pathlib import Path
        
        # Check current directory and parent directories
        current = Path.cwd()
        for path in [current, current.parent, Path("/content/LLM4Rec")]:
            env_file = path / ".env"
            if env_file.exists():
                load_dotenv(env_file, override=False)  # Don't override existing env vars
                break
    except ImportError:
        pass  # python-dotenv not installed, skip
    
    v = os.environ.get(name)
    return v.strip() if isinstance(v, str) and v.strip() else None


