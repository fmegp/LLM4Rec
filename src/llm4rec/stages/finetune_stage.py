from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import torch
import torch.optim as optim
from accelerate import Accelerator
from scipy.sparse import load_npz
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..data import RecommendationGPTTestGeneratorBatch, RecommendationGPTTrainGeneratorBatch, UserItemContentGPTDatasetBatch
from ..hf import load_gpt2_base_model, load_tokenizer_with_user_item_tokens
from ..io import build_dataset_manifest, ensure_dir, save_json, validate_dataset_layout
from ..logging_wandb import WandbHandle, init_wandb_run, log_dataset_manifest, log_gpu_memory, log_metrics
from ..metrics import NDCG_at_k, Recall_at_k
from ..model import (
    CollaborativeGPTwithItemRecommendHead,
    ContentGPTForUserItemWithLMHeadBatch,
    GPT4RecommendationBaseModel,
)
from ..progress import RunningAverage, print_epoch_header


def _load_embeddings(base_model: GPT4RecommendationBaseModel, *, user_path: str, item_path: str, device: str) -> None:
    base_model.user_embeddings.load_state_dict(torch.load(user_path, map_location=device))
    base_model.item_embeddings.load_state_dict(torch.load(item_path, map_location=device))


def _freeze_all_except_user_item_embeddings(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        if ("user_embeddings" not in name) and ("item_embeddings" not in name):
            param.requires_grad = False


def run_finetuning(
    *,
    dataset_dir: str,
    output_dir: str,
    lambda_V: float,
    pretrained_dir: str,
    hf_model_name: str = "openai-community/gpt2",
    hf_cache_dir: str | None = None,
    hf_token: str | None = None,
    batch_size: int = 20,
    val_batch_size: int = 256,
    learning_rate: float = 1e-4,
    num_epochs: int = 150,
    mixed_precision: str = "bf16",
    log_to_wandb: bool = True,
    wandb_project: str = "cllm4rec",
    wandb_handle: WandbHandle | None = None,
) -> dict[str, Any]:
    dataset_dir = str(Path(dataset_dir).resolve())
    output_dir = str(Path(output_dir).resolve())
    pretrained_dir = str(Path(pretrained_dir).resolve())
    ensure_dir(output_dir)

    layout = validate_dataset_layout(dataset_dir)
    if not layout.val_matrix_path:
        raise FileNotFoundError("val_matrix.npz not found; finetuning requires validation split.")

    with open(layout.meta_path, "rb") as f:
        meta = pickle.load(f)
    num_users = int(meta["num_users"])
    num_items = int(meta["num_items"])

    accelerator = Accelerator(mixed_precision=mixed_precision)
    device = accelerator.device

    if wandb_handle is None and log_to_wandb and accelerator.is_main_process:
        wandb_handle = init_wandb_run(
            project=wandb_project,
            name=f"finetuning_{Path(dataset_dir).name}_lambdaV{lambda_V}",
            tags=["finetuning"],
            config={
                "stage": "finetuning",
                "dataset_dir": dataset_dir,
                "lambda_V": lambda_V,
                "pretrained_dir": pretrained_dir,
                "hf_model_name": hf_model_name,
                "mixed_precision": mixed_precision,
                "batch_size": batch_size,
                "val_batch_size": val_batch_size,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
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

    # HF + tokenizer
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

    # Data
    review_data = UserItemContentGPTDatasetBatch(tokenizer, layout.review_path)
    train_mat = load_npz(layout.train_matrix_path)
    val_mat = load_npz(layout.val_matrix_path)
    train_data = RecommendationGPTTrainGeneratorBatch(tokenizer, train_mat)
    val_data = RecommendationGPTTestGeneratorBatch(tokenizer, train_mat, val_mat)

    train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=train_data.collate_fn)
    val_loader = DataLoader(val_data, batch_size=val_batch_size, collate_fn=val_data.collate_fn)
    review_loader = DataLoader(review_data, batch_size=batch_size, collate_fn=review_data.collate_fn)

    # Models: content model (pretrained from training/content)
    content_base = GPT4RecommendationBaseModel(base_config, gpt2model)
    content_user = os.path.join(pretrained_dir, "content", f"user_embeddings_{lambda_V}.pt")
    content_item = os.path.join(pretrained_dir, "content", f"item_embeddings_{lambda_V}.pt")
    if not (os.path.exists(content_user) and os.path.exists(content_item)):
        raise FileNotFoundError(
            "Missing content embeddings from training stage. Expected:\n"
            f"- {content_user}\n- {content_item}"
        )
    _load_embeddings(content_base, user_path=content_user, item_path=content_item, device=str(device))
    content_model = ContentGPTForUserItemWithLMHeadBatch(base_config, content_base)

    # Models: rec model (pretrained from training/collaborative)
    rec_base = GPT4RecommendationBaseModel(base_config, gpt2model)
    collab_user = os.path.join(pretrained_dir, "collaborative", f"user_embeddings_{lambda_V}.pt")
    collab_item = os.path.join(pretrained_dir, "collaborative", f"item_embeddings_{lambda_V}.pt")
    if not (os.path.exists(collab_user) and os.path.exists(collab_item)):
        raise FileNotFoundError(
            "Missing collaborative embeddings from training stage. Expected:\n"
            f"- {collab_user}\n- {collab_item}"
        )
    _load_embeddings(rec_base, user_path=collab_user, item_path=collab_item, device=str(device))
    rec_model = CollaborativeGPTwithItemRecommendHead(base_config, rec_base)

    _freeze_all_except_user_item_embeddings(rec_model)
    _freeze_all_except_user_item_embeddings(content_model)

    opt = optim.Adam(rec_model.parameters(), lr=learning_rate)
    review_opt = optim.Adam(content_model.parameters(), lr=learning_rate)

    rec_model, opt, train_loader = accelerator.prepare(rec_model, opt, train_loader)
    content_model, review_opt, review_loader = accelerator.prepare(content_model, review_opt, review_loader)

    out_rec = os.path.join(output_dir, "finetuning", "rec")
    out_content = os.path.join(output_dir, "finetuning", "content")
    ensure_dir(out_rec)
    ensure_dir(out_content)

    best_sum = -float("inf")
    best_val_loss = float("inf")

    accelerator.print("-----Begin Finetuning Loop-----")
    for epoch in range(num_epochs):
        if accelerator.is_local_main_process:
            print_epoch_header("Rec GPT Finetuning", epoch + 1, num_epochs)

        rec_model.train()
        train_running = RunningAverage()
        train_loss_sum = 0.0
        reg_sum = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            disable=not accelerator.is_local_main_process,
            leave=True,
            position=0,
            miniters=1,
        )
        for input_ids, target_mat, attention_mask, input_ids_main in pbar:
            opt.zero_grad()
            input_ids = input_ids.to(device)
            target_mat = target_mat.to(device)
            attention_mask = attention_mask.to(device)
            input_ids_main = input_ids_main.to(device)

            accelerator.wait_for_everyone()
            with torch.no_grad():
                content_embeds = torch.cat(
                    (
                        accelerator.unwrap_model(content_model).base_model.embed(input_ids),
                        accelerator.unwrap_model(content_model).base_model.embed(input_ids_main),
                    ),
                    axis=1,
                ).to(device)

            outputs = rec_model(
                input_ids,
                target_mat,
                attention_mask=attention_mask,
                regularize=True,
                lambda_V=lambda_V,
                main_ids=input_ids_main,
                content_embeds=content_embeds,
            )
            loss = outputs[0]
            reg_loss = outputs[1]
            accelerator.backward(loss)
            opt.step()

            v = float(loss.item())
            train_loss_sum += v
            reg_sum += float(reg_loss.item())
            avg = train_running.update(v)
            pbar.set_postfix({"Loss": f"{v:.4f}", "Avg Loss": f"{avg:.4f}"})

        train_loss_avg = train_loss_sum / max(1, len(train_loader))
        reg_loss_avg = reg_sum / max(1, len(train_loader))

        # Validation (only main process)
        rec_model.eval()
        val_loss_sum = 0.0
        cur_recall_20 = 0.0
        cur_recall_40 = 0.0
        cur_ndcg_100 = 0.0

        accelerator.wait_for_everyone()
        with torch.no_grad():
            for input_ids, train_mat_b, target_mat_b, attention_mask in val_loader:
                input_ids = input_ids.to(device)
                train_mat_b = train_mat_b.to(device)
                target_mat_b = target_mat_b.to(device)
                attention_mask = attention_mask.to(device)

                rec_loss, item_scores = rec_model(input_ids, target_mat_b, attention_mask)
                item_scores[train_mat_b > 0] = -float("inf")

                target_np = target_mat_b.cpu().numpy()
                scores_np = item_scores.cpu().numpy()
                val_loss_sum += float(rec_loss.item())
                cur_recall_20 += Recall_at_k(target_np, scores_np, k=20, agg="sum")
                cur_recall_40 += Recall_at_k(target_np, scores_np, k=40, agg="sum")
                cur_ndcg_100 += NDCG_at_k(target_np, scores_np, k=100, agg="sum")

        val_loss_avg = val_loss_sum / max(1, len(val_loader))
        cur_recall_20 /= len(val_data)
        cur_recall_40 /= len(val_data)
        cur_ndcg_100 /= len(val_data)
        cur_sum = cur_recall_20 + cur_recall_40 + cur_ndcg_100

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg

        accelerator.print(f"Epoch {epoch + 1} - Train Rec Loss: {train_loss_avg:.4f}")
        accelerator.print(f"Epoch {epoch + 1} - Val Rec Loss: {val_loss_avg:.4f} / Best: {best_val_loss:.4f}")
        accelerator.print(f"Epoch {epoch + 1} - Recall@20: {cur_recall_20:.4f}")
        accelerator.print(f"Epoch {epoch + 1} - Recall@40: {cur_recall_40:.4f}")
        accelerator.print(f"Epoch {epoch + 1} - NDCG@100: {cur_ndcg_100:.4f}")

        if accelerator.is_main_process and wandb_handle is not None and wandb_handle.enabled:
            log_metrics(
                wandb_handle,
                {
                    "rec/train_loss": train_loss_avg,
                    "rec/train_regularize_loss": reg_loss_avg,
                    "rec/val_loss": val_loss_avg,
                    "rec/recall@20": cur_recall_20,
                    "rec/recall@40": cur_recall_40,
                    "rec/ndcg@100": cur_ndcg_100,
                    "epoch": epoch + 1,
                },
            )
            log_gpu_memory(wandb_handle, prefix="gpu")

        # Save best rec embeddings by the sum metric
        if cur_sum > best_sum and accelerator.is_main_process:
            best_sum = cur_sum
            torch.save(
                accelerator.unwrap_model(rec_model).base_model.user_embeddings.state_dict(),
                os.path.join(out_rec, f"user_embeddings_{lambda_V}.pt"),
            )
            torch.save(
                accelerator.unwrap_model(rec_model).base_model.item_embeddings.state_dict(),
                os.path.join(out_rec, f"item_embeddings_{lambda_V}.pt"),
            )

        # Update content model with mutual regularization
        if accelerator.is_local_main_process:
            print_epoch_header("Content Model (Mutual Regularization)", epoch + 1, num_epochs)

        content_model.train()
        review_running = RunningAverage()
        review_sum = 0.0
        reg_sum2 = 0.0

        pbar = tqdm(
            review_loader,
            desc=f"Content Epoch {epoch + 1}/{num_epochs}",
            disable=not accelerator.is_local_main_process,
            leave=True,
            position=0,
            miniters=1,
        )
        for input_ids_prompt, input_ids_main, attention_mask in pbar:
            review_opt.zero_grad()
            input_ids_prompt = input_ids_prompt.to(device)
            input_ids_main = input_ids_main.to(device)
            attention_mask = attention_mask.to(device)

            accelerator.wait_for_everyone()
            with torch.no_grad():
                rec_embeds = accelerator.unwrap_model(rec_model).base_model.embed(input_ids_prompt).to(device)

            outputs = content_model(
                input_ids_prompt,
                input_ids_main,
                labels_main=input_ids_main,
                attention_mask=attention_mask,
                regularize=True,
                lambda_V=lambda_V,
                collaborative_embeds=rec_embeds,
            )
            loss = outputs[0]
            reg_loss = outputs[1]
            accelerator.backward(loss)
            review_opt.step()

            v = float(loss.item())
            review_sum += v
            reg_sum2 += float(reg_loss.item())
            avg = review_running.update(v)
            pbar.set_postfix({"Loss": f"{v:.4f}", "Avg Loss": f"{avg:.4f}"})

        content_loss_avg = review_sum / max(1, len(review_loader))
        content_reg_avg = reg_sum2 / max(1, len(review_loader))
        accelerator.print(f"Epoch {epoch + 1} - Content Avg Loss: {content_loss_avg:.4f}")

        if accelerator.is_main_process and wandb_handle is not None and wandb_handle.enabled:
            log_metrics(
                wandb_handle,
                {
                    "content/lm_avg_loss": content_loss_avg,
                    "content/regularize_avg_loss": content_reg_avg,
                    "epoch": epoch + 1,
                },
            )

        # Save best content embeddings by content loss
        if accelerator.is_main_process:
            torch.save(
                accelerator.unwrap_model(content_model).base_model.user_embeddings.state_dict(),
                os.path.join(out_content, f"user_embeddings_{lambda_V}.pt"),
            )
            torch.save(
                accelerator.unwrap_model(content_model).base_model.item_embeddings.state_dict(),
                os.path.join(out_content, f"item_embeddings_{lambda_V}.pt"),
            )

    metrics = {
        "best_sum_metric": best_sum,
        "best_val_loss": best_val_loss,
    }
    save_json(metrics, os.path.join(output_dir, "finetuning", "metrics.json"))
    return metrics


