from __future__ import annotations

import concurrent.futures
import os
import pickle
import time
from pathlib import Path
from typing import Any

import torch
import torch.optim as optim
from accelerate import Accelerator
from scipy.sparse import load_npz
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import GPT2Model


def _next_with_heartbeat(iterator, *, label: str, interval: float = 10.0):
    """Fetch next item from iterator with heartbeat messages while waiting."""
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(next, iterator)
        while True:
            try:
                return fut.result(timeout=interval)
            except concurrent.futures.TimeoutError:
                elapsed = int(time.time() - t0)
                print(f"[{label}] preparing batch... {elapsed}s elapsed", flush=True)

from ..data import CollaborativeGPTGeneratorBatch, UserItemContentGPTDatasetBatch
from ..hf import load_gpt2_base_model, load_tokenizer_with_user_item_tokens
from ..io import build_dataset_manifest, ensure_dir, save_json, validate_dataset_layout
from ..logging_wandb import WandbHandle, init_wandb_run, log_dataset_manifest, log_gpu_memory, log_metrics
from ..model import (
    CollaborativeGPTwithItemLMHeadBatch,
    ContentGPTForUserItemWithLMHeadBatch,
    GPT4RecommendationBaseModel,
)
from ..progress import RunningAverage, print_epoch_header


def _freeze_all_except_user_item_embeddings(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        if ("user_embeddings" not in name) and ("item_embeddings" not in name):
            param.requires_grad = False


def _save_embeddings(base_model: GPT4RecommendationBaseModel, out_dir: str, *, lambda_V: float) -> None:
    ensure_dir(out_dir)
    torch.save(base_model.user_embeddings.state_dict(), os.path.join(out_dir, f"user_embeddings_{lambda_V}.pt"))
    torch.save(base_model.item_embeddings.state_dict(), os.path.join(out_dir, f"item_embeddings_{lambda_V}.pt"))


def run_training(
    *,
    dataset_dir: str,
    output_dir: str,
    lambda_V: float,
    hf_model_name: str = "openai-community/gpt2",
    hf_cache_dir: str | None = None,
    hf_token: str | None = None,
    batch_size: int = 20,
    learning_rate: float = 1e-3,
    num_content_pretrain_epochs: int = 10,
    num_iter_epochs: int = 100,
    mixed_precision: str = "bf16",
    log_to_wandb: bool = True,
    wandb_project: str = "cllm4rec",
    wandb_handle: WandbHandle | None = None,
) -> dict[str, Any]:
    """
    Training stage: content pretrain + iterative collaborative/content mutual regularization.
    Saves best user/item embeddings for both content and collaborative models.
    """
    dataset_dir = str(Path(dataset_dir).resolve())
    output_dir = str(Path(output_dir).resolve())
    ensure_dir(output_dir)

    layout = validate_dataset_layout(dataset_dir)
    with open(layout.meta_path, "rb") as f:
        meta = pickle.load(f)
    num_users = int(meta["num_users"])
    num_items = int(meta["num_items"])

    # Accelerator (single GPU in Colab; also works on multi-GPU if user configures)
    accelerator = Accelerator(mixed_precision=mixed_precision)
    device = accelerator.device

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # W&B (either re-use provided handle, or create a new run per stage)
    if wandb_handle is None and log_to_wandb and accelerator.is_main_process:
        wandb_handle = init_wandb_run(
            project=wandb_project,
            name=f"training_{Path(dataset_dir).name}_lambdaV{lambda_V}",
            tags=["training"],
            config={
                "stage": "training",
                "dataset_dir": dataset_dir,
                "lambda_V": lambda_V,
                "hf_model_name": hf_model_name,
                "mixed_precision": mixed_precision,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_content_pretrain_epochs": num_content_pretrain_epochs,
                "num_iter_epochs": num_iter_epochs,
            },
        )
    if wandb_handle is not None and wandb_handle.enabled and accelerator.is_main_process:
        manifest = build_dataset_manifest(dataset_dir, include_optional=True)
        log_dataset_manifest(
            wandb_handle,
            manifest=manifest,
            source="google-drive (downloaded in notebook)",
            dataset_name=Path(dataset_dir).name,
        )
        save_json(manifest, os.path.join(output_dir, "dataset_manifest.json"))

    # HF assets
    gpt2model, base_config, _assets = load_gpt2_base_model(
        model_name_or_path=hf_model_name,
        cache_dir=hf_cache_dir,
        token=hf_token,
    )
    # Extend config with num users/items
    base_config.num_users = num_users
    base_config.num_items = num_items

    tokenizer = load_tokenizer_with_user_item_tokens(
        num_users=num_users,
        num_items=num_items,
        model_name_or_path=hf_model_name,
        cache_dir=hf_cache_dir,
        token=hf_token,
    )

    # Data
    review_data = UserItemContentGPTDatasetBatch(tokenizer, layout.review_path)
    train_mat = load_npz(layout.train_matrix_path)
    collab_data = CollaborativeGPTGeneratorBatch(tokenizer, train_mat)

    # num_workers=0 required because collate_fn does tokenization which isn't picklable
    # But we add prefetch_factor for slight improvement
    review_loader = DataLoader(
        review_data, 
        batch_size=batch_size, 
        collate_fn=review_data.collate_fn,
        num_workers=0,
    )
    collab_loader = DataLoader(
        collab_data, 
        batch_size=batch_size, 
        collate_fn=collab_data.collate_fn,
        num_workers=0,
    )

    # Models
    content_base = GPT4RecommendationBaseModel(base_config, gpt2model)
    content_model = ContentGPTForUserItemWithLMHeadBatch(base_config, content_base)

    collab_base = GPT4RecommendationBaseModel(base_config, gpt2model)
    collab_model = CollaborativeGPTwithItemLMHeadBatch(base_config, collab_base)

    _freeze_all_except_user_item_embeddings(content_model)
    _freeze_all_except_user_item_embeddings(collab_model)

    review_opt = optim.Adam(content_model.parameters(), lr=learning_rate)
    collab_opt = optim.Adam(collab_model.parameters(), lr=learning_rate)

    content_model, review_opt, review_loader = accelerator.prepare(content_model, review_opt, review_loader)
    collab_model, collab_opt, collab_loader = accelerator.prepare(collab_model, collab_opt, collab_loader)

    content_model.train()
    collab_model.train()

    # Output paths
    out_content = os.path.join(output_dir, "training", "content")
    out_collab = os.path.join(output_dir, "training", "collaborative")
    ensure_dir(out_content)
    ensure_dir(out_collab)

    metrics: dict[str, Any] = {"num_users": num_users, "num_items": num_items}

    # Content pretraining loop
    best_review_loss = float("inf")
    accelerator.print("-----Begin Content GPT Pretraining Loop-----")
    for epoch in range(num_content_pretrain_epochs):
        if accelerator.is_local_main_process:
            print_epoch_header("Content GPT Pretraining", epoch + 1, num_content_pretrain_epochs)

        running = RunningAverage()
        epoch_loss_sum = 0.0
        total_batches = len(review_loader)

        if accelerator.is_local_main_process:
            print(f"Starting epoch {epoch+1}, {total_batches} batches total...", flush=True)

        for batch_idx, (input_ids_prompt, input_ids_main, attention_mask) in enumerate(review_loader):
            review_opt.zero_grad()
            input_ids_prompt = input_ids_prompt.to(device)
            input_ids_main = input_ids_main.to(device)
            attention_mask = attention_mask.to(device)

            outputs = content_model(
                input_ids_prompt,
                input_ids_main,
                labels_main=input_ids_main,
                attention_mask=attention_mask,
            )
            loss = outputs[0]
            accelerator.backward(loss)
            review_opt.step()

            v = float(loss.item())
            epoch_loss_sum += v
            avg = running.update(v)
            
            # Print progress every 50 batches or on first batch
            if accelerator.is_local_main_process and (batch_idx == 0 or (batch_idx + 1) % 50 == 0):
                print(f"  batch {batch_idx+1}/{total_batches} loss={v:.4f} avg={avg:.4f}", flush=True)

        avg_loss = epoch_loss_sum / max(1, total_batches)
        if accelerator.is_local_main_process:
            print(f"Epoch {epoch + 1}/{num_content_pretrain_epochs} complete - Avg Loss: {avg_loss:.4f}", flush=True)

        if accelerator.is_main_process and wandb_handle is not None and wandb_handle.enabled:
            log_metrics(wandb_handle, {"content_pretrain/review_avg_loss": avg_loss, "epoch": epoch + 1})
            log_gpu_memory(wandb_handle, prefix="gpu")

        if avg_loss < best_review_loss and accelerator.is_main_process:
            best_review_loss = avg_loss
            _save_embeddings(accelerator.unwrap_model(content_model).base_model, out_content, lambda_V=lambda_V)

    # Iterative mutual training
    best_collab_loss = float("inf")
    accelerator.print("-----Begin Iterative Training Loop-----")
    for epoch in range(num_iter_epochs):
        if accelerator.is_local_main_process:
            print_epoch_header("Iterative Training (Collaborative GPT)", epoch + 1, num_iter_epochs)

        collab_running = RunningAverage()
        collab_sum = 0.0
        reg_sum = 0.0
        collab_total = len(collab_loader)

        if accelerator.is_local_main_process:
            print(f"  Collaborative training: {collab_total} batches...", flush=True)

        for batch_idx, (input_ids_prompt, input_ids_main, attention_mask) in enumerate(collab_loader):
            collab_opt.zero_grad()
            input_ids_prompt = input_ids_prompt.to(device)
            input_ids_main = input_ids_main.to(device)
            attention_mask = attention_mask.to(device)

            accelerator.wait_for_everyone()
            with torch.no_grad():
                content_embeds = torch.cat(
                    (
                        accelerator.unwrap_model(content_model).base_model.embed(input_ids_prompt),
                        accelerator.unwrap_model(content_model).base_model.embed(input_ids_main),
                    ),
                    axis=1,
                ).to(device)

            outputs = collab_model(
                input_ids_prompt,
                input_ids_main,
                labels_main=input_ids_main,
                attention_mask=attention_mask,
                regularize=True,
                lambda_V=lambda_V,
                content_embeds=content_embeds,
            )
            loss = outputs[0]
            reg_loss = outputs[1]
            accelerator.backward(loss)
            collab_opt.step()

            v = float(loss.item())
            collab_sum += v
            reg_sum += float(reg_loss.item())
            avg = collab_running.update(v)
            
            if accelerator.is_local_main_process and (batch_idx == 0 or (batch_idx + 1) % 50 == 0):
                print(f"    collab batch {batch_idx+1}/{collab_total} loss={v:.4f} avg={avg:.4f}", flush=True)

        collab_avg = collab_sum / max(1, collab_total)
        reg_avg = reg_sum / max(1, collab_total)
        if accelerator.is_local_main_process:
            print(f"  Collaborative done - Avg Loss: {collab_avg:.4f}, Reg Loss: {reg_avg:.4f}", flush=True)
        accelerator.print(f"Epoch {epoch + 1} - Average Regularize Loss: {reg_avg:.4f}")

        if accelerator.is_main_process and wandb_handle is not None and wandb_handle.enabled:
            log_metrics(
                wandb_handle,
                {
                    "collab/lm_avg_loss": collab_avg,
                    "collab/regularize_avg_loss": reg_avg,
                    "epoch": epoch + 1,
                },
            )

        if collab_avg < best_collab_loss and accelerator.is_main_process:
            best_collab_loss = collab_avg
            _save_embeddings(accelerator.unwrap_model(collab_model).base_model, out_collab, lambda_V=lambda_V)

        if accelerator.is_local_main_process:
            print_epoch_header("Iterative Training (Content GPT)", epoch + 1, num_iter_epochs)

        review_running = RunningAverage()
        review_sum = 0.0
        reg_sum = 0.0
        review_total = len(review_loader)

        if accelerator.is_local_main_process:
            print(f"  Content training: {review_total} batches...", flush=True)

        for batch_idx, (input_ids_prompt, input_ids_main, attention_mask) in enumerate(review_loader):
            review_opt.zero_grad()
            input_ids_prompt = input_ids_prompt.to(device)
            input_ids_main = input_ids_main.to(device)
            attention_mask = attention_mask.to(device)

            accelerator.wait_for_everyone()
            with torch.no_grad():
                collab_embeds = accelerator.unwrap_model(collab_model).base_model.embed(input_ids_prompt).to(device)

            outputs = content_model(
                input_ids_prompt,
                input_ids_main,
                labels_main=input_ids_main,
                attention_mask=attention_mask,
                regularize=True,
                lambda_V=lambda_V,
                collaborative_embeds=collab_embeds,
            )
            loss = outputs[0]
            reg_loss = outputs[1]

            accelerator.backward(loss)
            review_opt.step()

            v = float(loss.item())
            review_sum += v
            reg_sum += float(reg_loss.item())
            avg = review_running.update(v)
            
            if accelerator.is_local_main_process and (batch_idx == 0 or (batch_idx + 1) % 50 == 0):
                print(f"    content batch {batch_idx+1}/{review_total} loss={v:.4f} avg={avg:.4f}", flush=True)

        review_avg = review_sum / max(1, review_total)
        reg_avg2 = reg_sum / max(1, review_total)
        if accelerator.is_local_main_process:
            print(f"  Content done - Avg Loss: {review_avg:.4f}, Reg Loss: {reg_avg2:.4f}", flush=True)
        accelerator.print(f"Epoch {epoch + 1} - Average Regularize Loss: {reg_avg2:.4f}")

        if accelerator.is_main_process and wandb_handle is not None and wandb_handle.enabled:
            log_metrics(
                wandb_handle,
                {
                    "content/lm_avg_loss": review_avg,
                    "content/regularize_avg_loss": reg_avg2,
                    "epoch": epoch + 1,
                },
            )
            log_gpu_memory(wandb_handle, prefix="gpu")

        if review_avg < best_review_loss and accelerator.is_main_process:
            best_review_loss = review_avg
            _save_embeddings(accelerator.unwrap_model(content_model).base_model, out_content, lambda_V=lambda_V)

    metrics.update(
        {
            "best_content_pretrain_review_loss": best_review_loss,
            "best_collab_loss": best_collab_loss,
        }
    )
    save_json(metrics, os.path.join(output_dir, "training", "metrics.json"))
    return metrics


