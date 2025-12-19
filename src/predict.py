"""
Colab-ready CLI wrapper for the evaluation stage.

For the full end-to-end workflow, prefer the notebook in `colab/`.
"""

from __future__ import annotations

import argparse

from llm4rec.stages.eval_stage import run_eval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to extracted dataset directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to write eval artifacts.")
    parser.add_argument(
        "--rec_embeddings_dir",
        type=str,
        required=True,
        help="Path to finetuned rec embeddings (usually: <output_dir>/finetuning/rec).",
    )
    parser.add_argument("--lambda_V", type=float, required=True)
    parser.add_argument("--hf_model_name", type=str, default="openai-community/gpt2")
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    args = parser.parse_args()

    run_eval(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        rec_embeddings_dir=args.rec_embeddings_dir,
        lambda_V=args.lambda_V,
        hf_model_name=args.hf_model_name,
        hf_cache_dir=args.hf_cache_dir,
        log_to_wandb=True,
    )


if __name__ == "__main__":
    main()


