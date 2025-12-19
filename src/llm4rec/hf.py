from __future__ import annotations

from dataclasses import dataclass

from transformers import GPT2Model

from .runtime import read_secret
from .tokenizer import TokenizerWithUserItemIDTokensBatch


@dataclass(frozen=True)
class HFAssets:
    model_name_or_path: str
    cache_dir: str | None
    token: str | None


def resolve_hf_assets(model_name_or_path: str, *, cache_dir: str | None = None, token: str | None = None) -> HFAssets:
    if token is None:
        token = read_secret("HF_TOKEN") or read_secret("HUGGINGFACE_HUB_TOKEN")
    return HFAssets(model_name_or_path=model_name_or_path, cache_dir=cache_dir, token=token)


def load_gpt2_base_model(
    *,
    model_name_or_path: str = "openai-community/gpt2",
    cache_dir: str | None = None,
    token: str | None = None,
):
    assets = resolve_hf_assets(model_name_or_path, cache_dir=cache_dir, token=token)
    model = GPT2Model.from_pretrained(
        assets.model_name_or_path,
        cache_dir=assets.cache_dir,
        token=assets.token,
    )
    return model, model.config, assets


def load_tokenizer_with_user_item_tokens(
    *,
    num_users: int,
    num_items: int,
    model_name_or_path: str = "openai-community/gpt2",
    cache_dir: str | None = None,
    token: str | None = None,
) -> TokenizerWithUserItemIDTokensBatch:
    assets = resolve_hf_assets(model_name_or_path, cache_dir=cache_dir, token=token)
    tok = TokenizerWithUserItemIDTokensBatch.from_pretrained_hf(
        assets.model_name_or_path,
        num_users=num_users,
        num_items=num_items,
        cache_dir=assets.cache_dir,
        token=assets.token,
    )
    return tok


