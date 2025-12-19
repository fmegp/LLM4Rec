from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .runtime import get_runtime_report, read_secret


@dataclass(frozen=True)
class WandbHandle:
    run: Any  # wandb.sdk.wandb_run.Run
    enabled: bool


def _try_import_wandb():
    try:
        import wandb  # type: ignore

        return wandb
    except Exception:
        return None


def init_wandb_run(
    *,
    project: str = "cllm4rec",
    name: str | None = None,
    tags: list[str] | None = None,
    notes: str | None = None,
    config: dict[str, Any] | None = None,
) -> WandbHandle:
    wandb = _try_import_wandb()
    if wandb is None:
        print("wandb not installed; skipping W&B logging.")
        return WandbHandle(run=None, enabled=False)

    api_key = read_secret("WANDB_API_KEY")
    if api_key:
        # Do NOT hardcode keys in code/notebooks; rely on Secrets/env.
        wandb.login(key=api_key, relogin=True)
    else:
        # This still works if already logged in (e.g., local dev), or can run offline.
        print("WANDB_API_KEY not found in env; attempting wandb.init() without explicit login.")

    runtime = get_runtime_report()
    base_config: dict[str, Any] = {
        "runtime_python": runtime.python_version,
        "runtime_torch": runtime.torch_version,
        "runtime_cuda_available": runtime.cuda_available,
        "runtime_cuda_version": runtime.cuda_version,
        "runtime_gpu_name": runtime.gpu_name,
        "runtime_numpy": runtime.numpy_version,
    }
    if config:
        base_config.update(config)

    run = wandb.init(
        project=project,
        name=name,
        tags=tags,
        notes=notes,
        config=base_config,
    )
    return WandbHandle(run=run, enabled=True)


def log_dataset_manifest(handle: WandbHandle, *, manifest: dict[str, Any], source: str, dataset_name: str) -> None:
    if not handle.enabled:
        return
    wandb = _try_import_wandb()
    assert wandb is not None

    # Log minimal provenance into config for easy filtering.
    handle.run.config.update(
        {
            "dataset_name": dataset_name,
            "dataset_source": source,
            "dataset_manifest_sha256": manifest.get("manifest_sha256"),
            "dataset_files_count": len(manifest.get("files", [])),
        },
        allow_val_change=True,
    )

    # Save the manifest as a small run file (manifest-only choice).
    out_dir = Path(wandb.run.dir) if wandb.run is not None else Path(".")
    manifest_path = out_dir / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    wandb.save(str(manifest_path), base_path=str(out_dir))


def log_metrics(handle: WandbHandle, metrics: dict[str, Any], *, step: int | None = None) -> None:
    if not handle.enabled:
        return
    wandb = _try_import_wandb()
    assert wandb is not None
    wandb.log(metrics, step=step)


def log_gpu_memory(handle: WandbHandle, *, prefix: str = "gpu") -> None:
    if not handle.enabled:
        return
    try:
        import torch

        if not torch.cuda.is_available():
            return
        metrics = {
            f"{prefix}/mem_allocated_bytes": int(torch.cuda.memory_allocated()),
            f"{prefix}/mem_reserved_bytes": int(torch.cuda.memory_reserved()),
            f"{prefix}/max_mem_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            f"{prefix}/max_mem_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
        log_metrics(handle, metrics)
    except Exception:
        return


class StepTimer:
    def __init__(self):
        self._t0 = time.time()

    def reset(self) -> None:
        self._t0 = time.time()

    @property
    def elapsed_s(self) -> float:
        return float(time.time() - self._t0)


