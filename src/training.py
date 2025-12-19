"""
Colab-ready CLI wrapper for the training stage.

For the full end-to-end workflow, prefer the notebook in `colab/`.
"""

from __future__ import annotations

import argparse

from llm4rec.stages.training_stage import run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to extracted dataset directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to write artifacts.")
    parser.add_argument("--lambda_V", type=float, required=True)
    parser.add_argument("--hf_model_name", type=str, default="openai-community/gpt2")
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--mixed_precision", type=str, default="bf16")
    args = parser.parse_args()

    run_training(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        lambda_V=args.lambda_V,
        hf_model_name=args.hf_model_name,
        hf_cache_dir=args.hf_cache_dir,
        mixed_precision=args.mixed_precision,
        log_to_wandb=True,
    )


if __name__ == "__main__":
    main()
