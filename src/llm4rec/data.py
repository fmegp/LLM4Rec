"""
Datasets / batch generators.

Adapted from `src/libs/data.py` in the original repo.
"""

from __future__ import annotations

import pickle
import random
from typing import Any

import fsspec
import torch
from torch.utils.data import Dataset

from .tokenizer import TokenizerWithUserItemIDTokensBatch


class CollaborativeGPTGeneratorBatch(Dataset):
    def __init__(self, tokenizer: TokenizerWithUserItemIDTokensBatch, train_mat: Any, max_length: int = 1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.train_mat = train_mat
        self.max_length = max_length
        self.num_users, self.num_items = train_mat.shape

    def __len__(self) -> int:
        return self.num_users

    def __getitem__(self, idx: int):
        prompt = f"user_{idx} has interacted with"
        return prompt, self.train_mat.getrow(idx).nonzero()[1]

    def collate_fn(self, batch):
        prompt_texts, item_ids = zip(*batch)
        encoded_prompt = self.tokenizer.encode_batch(list(prompt_texts))
        item_tokens = [" ".join([f"item_{item_id}" for item_id in ids]) for ids in item_ids]
        encoded_main = self.tokenizer.encode_batch(item_tokens)

        prompt_ids = torch.tensor(encoded_prompt[0])
        main_ids = torch.tensor(encoded_main[0])
        attention_mask = torch.cat((torch.tensor(encoded_prompt[1]), torch.tensor(encoded_main[1])), dim=1)

        total_length = prompt_ids.size(1) + main_ids.size(1)
        if total_length > self.max_length:
            excess_length = total_length - self.max_length
            main_ids = main_ids[:, :-excess_length]
            attention_mask = attention_mask[:, :-excess_length]

        return prompt_ids, main_ids, attention_mask


class UserItemContentGPTDatasetBatch(Dataset):
    def __init__(self, tokenizer: TokenizerWithUserItemIDTokensBatch, filepath: str, max_length: int = 1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        assert filepath.endswith(".pkl"), "we need to load from a pickle file"
        with fsspec.open(filepath, "rb") as file:
            self.data = pickle.load(file)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        prompt_text, main_text = self.data[idx][0], self.data[idx][1]
        return prompt_text, main_text

    def collate_fn(self, batch):
        prompt_texts, main_texts = zip(*batch)

        encoded_prompt = self.tokenizer.encode_batch(list(prompt_texts))
        encoded_main = self.tokenizer.encode_batch(list(main_texts))

        prompt_ids = torch.tensor(encoded_prompt[0])
        main_ids = torch.tensor(encoded_main[0])
        attention_mask = torch.cat((torch.tensor(encoded_prompt[1]), torch.tensor(encoded_main[1])), dim=1)

        total_length = prompt_ids.size(1) + main_ids.size(1)
        if total_length > self.max_length:
            excess_length = total_length - self.max_length
            main_ids = main_ids[:, :-excess_length]
            attention_mask = attention_mask[:, :-excess_length]

        return prompt_ids, main_ids, attention_mask


class RecommendationGPTTrainGeneratorBatch(Dataset):
    def __init__(
        self,
        tokenizer: TokenizerWithUserItemIDTokensBatch,
        train_mat: Any,
        max_length: int = 1024,
        predict_ratio: float = 0.2,
        shuffle: bool = True,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.train_mat = train_mat
        self.max_length = max_length
        self.num_users, self.num_items = train_mat.shape
        self.predict_ratio = predict_ratio
        self.shuffle = shuffle

    def __len__(self) -> int:
        return self.num_users

    def __getitem__(self, idx: int):
        past_interactions = self.train_mat.getrow(idx).nonzero()[1]

        num_items_to_mask = max(1, int(len(past_interactions) * self.predict_ratio))
        masked_items = random.sample(past_interactions.tolist(), num_items_to_mask)

        input_interactions = [item if item not in masked_items else None for item in past_interactions]
        if self.shuffle:
            random.shuffle(input_interactions)
        target_interactions = past_interactions

        input_prompt = (
            f"user_{idx} has interacted with "
            f"{' '.join(['item_' + str(item_id) for item_id in input_interactions if item_id is not None])}"
        )
        input_prompt += f", user_{idx} will interact with"

        target_matrix = torch.zeros(self.num_items, dtype=torch.float32)
        target_matrix[target_interactions] = 1.0
        item_ids = target_matrix.nonzero()[0]

        return input_prompt, target_matrix, item_ids

    def collate_fn(self, batch):
        prompt_texts, target_matrices, item_ids = zip(*batch)

        encoded_prompt = self.tokenizer.encode_batch(list(prompt_texts))
        target_matrices_t = torch.cat([matrix.unsqueeze(0) for matrix in target_matrices])

        item_tokens = [" ".join(["item_" + str(item_id) for item_id in ids]) for ids in item_ids]
        encoded_main = self.tokenizer.encode_batch(item_tokens)

        prompt_ids = torch.tensor(encoded_prompt[0])
        main_ids = torch.tensor(encoded_main[0])
        attention_mask = torch.tensor(encoded_prompt[1])

        total_length = prompt_ids.size(1)
        if total_length > self.max_length:
            excess_length = total_length - self.max_length
            prompt_ids = prompt_ids[:, :-excess_length]
            attention_mask = attention_mask[:, :-excess_length]

        return prompt_ids, target_matrices_t, attention_mask, main_ids


class RecommendationGPTTestGeneratorBatch(Dataset):
    def __init__(
        self,
        tokenizer: TokenizerWithUserItemIDTokensBatch,
        train_mat: Any,
        test_mat: Any,
        max_length: int = 1024,
        predict_ratio: float = 0.2,
        shuffle: bool = True,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.train_mat = train_mat
        self.test_mat = test_mat
        self.max_length = max_length
        self.num_users, self.num_items = train_mat.shape
        self.predict_ratio = predict_ratio
        self.shuffle = shuffle

    def __len__(self) -> int:
        return self.num_users

    def __getitem__(self, idx: int):
        input_interactions = self.train_mat.getrow(idx).nonzero()[1]
        if self.shuffle:
            random.shuffle(input_interactions)

        input_prompt = (
            f"user_{idx} has interacted with "
            f"{' '.join(['item_' + str(item_id) for item_id in input_interactions])}"
        )
        input_prompt += f", user_{idx} will interact with"

        train_interactions = self.train_mat.getrow(idx).nonzero()[1]
        train_matrix = torch.zeros(self.num_items, dtype=torch.float32)
        train_matrix[train_interactions] = 1.0

        target_interactions = self.test_mat.getrow(idx).nonzero()[1]
        target_matrix = torch.zeros(self.num_items, dtype=torch.float32)
        target_matrix[target_interactions] = 1.0

        return input_prompt, train_matrix, target_matrix

    def collate_fn(self, batch):
        prompt_texts, train_matrices, target_matrices = zip(*batch)

        encoded_prompt = self.tokenizer.encode_batch(list(prompt_texts))
        train_matrices_t = torch.cat([matrix.unsqueeze(0) for matrix in train_matrices])
        target_matrices_t = torch.cat([matrix.unsqueeze(0) for matrix in target_matrices])

        prompt_ids = torch.tensor(encoded_prompt[0])
        attention_mask = torch.tensor(encoded_prompt[1])

        total_length = prompt_ids.size(1)
        if total_length > self.max_length:
            excess_length = total_length - self.max_length
            prompt_ids = prompt_ids[:, :-excess_length]
            attention_mask = attention_mask[:, :-excess_length]

        return prompt_ids, train_matrices_t, target_matrices_t, attention_mask


