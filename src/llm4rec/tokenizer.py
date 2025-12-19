"""
Tokenizer with user/item ID tokens.

This is adapted from `src/libs/tokenizer.py` in the original repo.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from transformers import GPT2Tokenizer


class TokenizerWithUserItemIDTokens(GPT2Tokenizer):
    def __init__(
        self,
        vocab_file: str,
        merges_file: str,
        num_users: int,
        num_items: int,
        **kwargs: Any,
    ):
        super().__init__(vocab_file=vocab_file, merges_file=merges_file, **kwargs)
        self.num_users = num_users
        self.num_items = num_items
        self.user_token_encoder = self._add_user_token_encoder()
        self.item_token_encoder = self._add_item_token_encoder()

        # Add the user/item token encoders to the original vocab encoder
        self.encoder.update(self.user_token_encoder)
        self.encoder.update(self.item_token_encoder)

        # Add the corresponding decoders to the original vocab decoder
        self.user_token_decoder = {v: k for k, v in self.user_token_encoder.items()}
        self.item_token_decoder = {v: k for k, v in self.item_token_encoder.items()}
        self.decoder.update(self.user_token_decoder)
        self.decoder.update(self.item_token_decoder)

    def _add_user_token_encoder(self) -> dict[str, int]:
        return {f"user_{i}": (i + self.vocab_size) for i in range(self.num_users)}

    def _add_item_token_encoder(self) -> dict[str, int]:
        return {
            f"item_{j}": (j + self.vocab_size + self.num_users) for j in range(self.num_items)
        }

    def _pre_tokenize(self, text: str) -> list[str]:
        """
        Break a sentence into pieces where `user_i` / `item_j` remain atomic tokens.
        """
        pattern = r"(user_\d+|item_\d+)"
        _matches = re.findall(pattern, text)
        pieces = re.split(pattern, text)
        pieces = [piece.rstrip() for piece in pieces if piece.rstrip()]
        return pieces

    def _tokenize(self, text: str) -> list[str]:
        split_tokens: list[str] = []
        pieces = self._pre_tokenize(text)
        for piece in pieces:
            if piece in self.user_token_encoder:
                split_tokens.append(piece)
            elif piece in self.item_token_encoder:
                split_tokens.append(piece)
            else:
                split_tokens += super()._tokenize(piece)
        return split_tokens

    @classmethod
    def from_pretrained_hf(
        cls,
        model_name_or_path: str,
        *,
        num_users: int,
        num_items: int,
        token: str | None = None,
        cache_dir: str | None = None,
        **kwargs: Any,
    ) -> "TokenizerWithUserItemIDTokens":
        """
        Create the tokenizer using Hugging Face Hub assets for GPT‑2 (vocab + merges).
        """
        from pathlib import Path
        import os
        import tempfile
        import shutil
        
        base = GPT2Tokenizer.from_pretrained(
            model_name_or_path,
            token=token,
            cache_dir=cache_dir,
        )
        
        # Get vocab and merges file paths
        # In newer transformers, these aren't attributes, so we save the tokenizer
        # to a temp location to get the file paths, then use those paths
        with tempfile.TemporaryDirectory() as tmpdir:
            base.save_pretrained(tmpdir)
            vocab_file = str(Path(tmpdir) / "vocab.json")
            merges_file = str(Path(tmpdir) / "merges.txt")
            
            if not os.path.exists(vocab_file) or not os.path.exists(merges_file):
                raise RuntimeError(
                    f"Could not find vocab.json and merges.txt after saving tokenizer. "
                    f"vocab.json exists: {os.path.exists(vocab_file)}, "
                    f"merges.txt exists: {os.path.exists(merges_file)}"
                )
            
            # Copy files to a persistent location (use cache_dir if provided, else system temp)
            if cache_dir:
                persist_dir = Path(cache_dir) / "tokenizer_files" / model_name_or_path.replace("/", "--")
            else:
                persist_dir = Path(tempfile.gettempdir()) / "llm4rec_tokenizer" / model_name_or_path.replace("/", "--")
            
            persist_dir.mkdir(parents=True, exist_ok=True)
            persist_vocab = persist_dir / "vocab.json"
            persist_merges = persist_dir / "merges.txt"
            
            # Only copy if they don't exist or are different
            if not persist_vocab.exists() or not persist_merges.exists():
                shutil.copy2(vocab_file, persist_vocab)
                shutil.copy2(merges_file, persist_merges)
            
            # Use the persistent paths
            vocab_file = str(persist_vocab)
            merges_file = str(persist_merges)
        
        return cls(
            vocab_file=vocab_file,
            merges_file=merges_file,
            num_users=num_users,
            num_items=num_items,
            **kwargs,
        )


class TokenizerWithUserItemIDTokensBatch(TokenizerWithUserItemIDTokens):
    """
    Batch encoding wrapper.
    """

    def __init__(self, vocab_file: str, merges_file: str, num_users: int, num_items: int, **kwargs: Any):
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            num_users=num_users,
            num_items=num_items,
            **kwargs,
        )
        # Pad token id to 0 to match original code
        self.pad_token_id = 0

    @classmethod
    def from_pretrained_hf(
        cls,
        model_name_or_path: str,
        *,
        num_users: int,
        num_items: int,
        token: str | None = None,
        cache_dir: str | None = None,
        **kwargs: Any,
    ) -> "TokenizerWithUserItemIDTokensBatch":
        from pathlib import Path
        import os
        import tempfile
        import shutil
        
        base = GPT2Tokenizer.from_pretrained(
            model_name_or_path,
            token=token,
            cache_dir=cache_dir,
        )
        
        # Get vocab and merges file paths by saving tokenizer temporarily
        with tempfile.TemporaryDirectory() as tmpdir:
            base.save_pretrained(tmpdir)
            vocab_file = str(Path(tmpdir) / "vocab.json")
            merges_file = str(Path(tmpdir) / "merges.txt")
            
            if not os.path.exists(vocab_file) or not os.path.exists(merges_file):
                raise RuntimeError(
                    f"Could not find vocab.json and merges.txt after saving tokenizer. "
                    f"vocab.json exists: {os.path.exists(vocab_file)}, "
                    f"merges.txt exists: {os.path.exists(merges_file)}"
                )
            
            # Copy files to a persistent location
            if cache_dir:
                persist_dir = Path(cache_dir) / "tokenizer_files" / model_name_or_path.replace("/", "--")
            else:
                persist_dir = Path(tempfile.gettempdir()) / "llm4rec_tokenizer" / model_name_or_path.replace("/", "--")
            
            persist_dir.mkdir(parents=True, exist_ok=True)
            persist_vocab = persist_dir / "vocab.json"
            persist_merges = persist_dir / "merges.txt"
            
            # Only copy if they don't exist
            if not persist_vocab.exists() or not persist_merges.exists():
                shutil.copy2(vocab_file, persist_vocab)
                shutil.copy2(merges_file, persist_merges)
            
            # Use the persistent paths
            vocab_file = str(persist_vocab)
            merges_file = str(persist_merges)
        
        return cls(
            vocab_file=vocab_file,
            merges_file=merges_file,
            num_users=num_users,
            num_items=num_items,
            **kwargs,
        )

    def encode_batch(self, texts: list[str], max_length: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        encoded_inputs: list[tuple[list[int], list[int]]] = []
        max_length_batch = max(len(self._tokenize(text)) for text in texts)

        # Determine max length for padding
        if (not max_length) or max_length <= max_length_batch:
            max_length = max_length_batch

        for text in texts:
            tokens = self._tokenize(text)
            input_ids = self.convert_tokens_to_ids(tokens)
            attention_mask = [1] * len(input_ids)

            padding_length = max_length - len(input_ids)
            input_ids += [self.pad_token_id] * padding_length
            attention_mask += [0] * padding_length

            encoded_inputs.append((input_ids, attention_mask))

        input_ids_batch, attention_mask_batch = zip(*encoded_inputs)
        return np.array(input_ids_batch), np.array(attention_mask_batch)


