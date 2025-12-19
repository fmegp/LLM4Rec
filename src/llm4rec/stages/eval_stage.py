from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import torch
from scipy.sparse import load_npz
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..data import RecommendationGPTTestGeneratorBatch
from ..hf import load_gpt2_base_model, load_tokenizer_with_user_item_tokens
from ..io import build_dataset_manifest, ensure_dir, save_json, validate_dataset_layout
from ..logging_wandb import WandbHandle, init_wandb_run, log_dataset_manifest, log_metrics
from ..metrics import NDCG_at_k, Recall_at_k
from ..model import CollaborativeGPTwithItemRecommendHead, GPT4RecommendationBaseModel


def _load_embeddings(base_model: GPT4RecommendationBaseModel, *, user_path: str, item_path: str, device: str) -> None:
    base_model.user_embeddings.load_state_dict(torch.load(user_path, map_location=device))
    base_model.item_embeddings.load_state_dict(torch.load(item_path, map_location=device))


def run_eval(
    *,
    dataset_dir: str,
    output_dir: str,
    lambda_V: float,
    rec_embeddings_dir: str,
    hf_model_name: str = "openai-community/gpt2",
    hf_cache_dir: str | None = None,
    hf_token: str | None = None,
    batch_size: int = 256,
    mixed_precision: str = "bf16",
    log_to_wandb: bool = True,
    wandb_project: str = "cllm4rec",
    wandb_handle: WandbHandle | None = None,
) -> dict[str, Any]:
    dataset_dir = str(Path(dataset_dir).resolve())
    output_dir = str(Path(output_dir).resolve())
    rec_embeddings_dir = str(Path(rec_embeddings_dir).resolve())
    ensure_dir(output_dir)

    layout = validate_dataset_layout(dataset_dir)
    if not layout.test_matrix_path:
        raise FileNotFoundError("test_matrix.npz not found; eval requires test split.")

    with open(layout.meta_path, "rb") as f:
        meta = pickle.load(f)
    num_users = int(meta["num_users"])
    num_items = int(meta["num_items"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if wandb_handle is None and log_to_wandb:
        wandb_handle = init_wandb_run(
            project=wandb_project,
            name=f"eval_{Path(dataset_dir).name}_lambdaV{lambda_V}",
            tags=["eval"],
            config={
                "stage": "eval",
                "dataset_dir": dataset_dir,
                "lambda_V": lambda_V,
                "rec_embeddings_dir": rec_embeddings_dir,
                "hf_model_name": hf_model_name,
                "batch_size": batch_size,
            },
        )
    if wandb_handle is not None and wandb_handle.enabled:
        manifest = build_dataset_manifest(dataset_dir, include_optional=True)
        log_dataset_manifest(
            wandb_handle,
            manifest=manifest,
            source="google-drive (downloaded in notebook)",
            dataset_name=Path(dataset_dir).name,
        )

    gpt2model, base_config, _assets = load_gpt2_base_model(
        model_name_or_path=hf_model_name,
        cache_dir=hf_cache_dir,
        token=hf_token,
    )
    base_config.num_users = num_users
    base_config.num_items = num_items
    tokenizer = load_tokenizer_with_user_item_tokens(
        num_users=num_users,
        num_items=num_items,
        model_name_or_path=hf_model_name,
        cache_dir=hf_cache_dir,
        token=hf_token,
    )

    train_mat = load_npz(layout.train_matrix_path)
    test_mat = load_npz(layout.test_matrix_path)
    test_data = RecommendationGPTTestGeneratorBatch(tokenizer, train_mat, test_mat)
    test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=test_data.collate_fn)

    base_model = GPT4RecommendationBaseModel(base_config, gpt2model)
    user_path = os.path.join(rec_embeddings_dir, f"user_embeddings_{lambda_V}.pt")
    item_path = os.path.join(rec_embeddings_dir, f"item_embeddings_{lambda_V}.pt")
    if not (os.path.exists(user_path) and os.path.exists(item_path)):
        raise FileNotFoundError(f"Missing rec embeddings:\n- {user_path}\n- {item_path}")
    _load_embeddings(base_model, user_path=user_path, item_path=item_path, device=device)

    rec_model = CollaborativeGPTwithItemRecommendHead(base_config, base_model).to(device)
    rec_model.eval()

    cur_recall_20 = 0.0
    cur_recall_40 = 0.0
    cur_ndcg_100 = 0.0

    print("Evaluating on test set...")
    with torch.no_grad():
        for input_ids, train_mat_b, target_mat_b, attention_mask in tqdm(
            test_loader, desc="Eval", leave=True, position=0, miniters=1
        ):
            input_ids = input_ids.to(device)
            train_mat_b = train_mat_b.to(device)
            target_mat_b = target_mat_b.to(device)
            attention_mask = attention_mask.to(device)

            _loss, item_scores = rec_model(input_ids, target_mat_b, attention_mask)
            item_scores[train_mat_b > 0] = -float("inf")

            target_np = target_mat_b.cpu().numpy()
            scores_np = item_scores.cpu().numpy()
            cur_recall_20 += Recall_at_k(target_np, scores_np, k=20, agg="sum")
            cur_recall_40 += Recall_at_k(target_np, scores_np, k=40, agg="sum")
            cur_ndcg_100 += NDCG_at_k(target_np, scores_np, k=100, agg="sum")

    cur_recall_20 /= len(test_data)
    cur_recall_40 /= len(test_data)
    cur_ndcg_100 /= len(test_data)

    results = {
        "recall@20": cur_recall_20,
        "recall@40": cur_recall_40,
        "ndcg@100": cur_ndcg_100,
    }
    save_json(results, os.path.join(output_dir, "eval", f"results_{lambda_V}.json"))

    if wandb_handle is not None and wandb_handle.enabled:
        log_metrics(
            wandb_handle,
            {
                "eval/recall@20": cur_recall_20,
                "eval/recall@40": cur_recall_40,
                "eval/ndcg@100": cur_ndcg_100,
            },
        )
    return results


