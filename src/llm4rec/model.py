"""
Models.

Adapted from `src/libs/model.py` in the original repo.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F


class GPT4RecommendationBaseModel(nn.Module):
    """
    GPT2 base model extended with extra user/item embeddings.
    """

    def __init__(self, config, gpt2model):
        super().__init__()
        self.num_users = int(config.num_users)
        self.num_items = int(config.num_items)
        self.vocab_size = int(config.vocab_size)
        self.config = config

        self.user_embeddings = nn.Embedding(self.num_users, config.n_embd)
        self.item_embeddings = nn.Embedding(self.num_items, config.n_embd)

        self.user_embeddings.weight.data.normal_(mean=0.0, std=config.initializer_range)
        self.item_embeddings.weight.data.normal_(mean=0.0, std=config.initializer_range)

        self.gpt2model = gpt2model

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        vocab_mask = (input_ids < self.vocab_size).long()
        user_mask = ((input_ids >= self.vocab_size) & (input_ids < self.vocab_size + self.num_users)).long()
        item_mask = (input_ids >= self.vocab_size + self.num_users).long()

        vocab_ids = (input_ids * vocab_mask).clamp_(0, self.vocab_size - 1)
        vocab_embeddings = self.gpt2model.wte(vocab_ids)
        vocab_embeddings = vocab_embeddings * vocab_mask.unsqueeze(-1)

        user_ids = ((input_ids - self.vocab_size) * user_mask).clamp_(0, self.num_users - 1)
        user_embeddings = self.user_embeddings(user_ids)
        user_embeddings = user_embeddings * user_mask.unsqueeze(-1)

        item_ids = ((input_ids - self.vocab_size - self.num_users) * item_mask).clamp_(0, self.num_items - 1)
        item_embeddings = self.item_embeddings(item_ids)
        item_embeddings = item_embeddings * item_mask.unsqueeze(-1)

        return vocab_embeddings + user_embeddings + item_embeddings

    def forward(self, input_ids: torch.Tensor | None = None, **kwargs):
        input_embeddings = self.embed(input_ids)
        return self.gpt2model(inputs_embeds=input_embeddings, **kwargs)


class CollaborativeGPTwithItemLMHeadBatch(nn.Module):
    """
    Collaborative filtering model: LM head over item space only.
    """

    def __init__(self, config, base_model: GPT4RecommendationBaseModel):
        super().__init__()
        self.num_users = int(config.num_users)
        self.num_items = int(config.num_items)
        self.vocab_size = int(config.vocab_size)
        self.base_model = base_model

        self.item_head = nn.Linear(config.n_embd, self.num_items, bias=False)
        self.item_head.weight = self.base_model.item_embeddings.weight

    def forward(
        self,
        input_ids_prompt: torch.Tensor,
        input_ids_main: torch.Tensor,
        labels_main: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        regularize: bool = False,
        lambda_V: float | None = None,
        content_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        outputs_prompt = self.base_model(input_ids=input_ids_prompt, return_dict=True, **kwargs)
        past_key_values = outputs_prompt.past_key_values

        outputs_main = self.base_model(
            input_ids=input_ids_main,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            return_dict=True,
        )

        item_logits = self.item_head(outputs_main.last_hidden_state)
        outputs = (item_logits,) + outputs_main[1:]

        if labels_main is not None:
            shift_logits = item_logits[..., :-1, :].contiguous()
            shift_labels = labels_main[..., 1:].contiguous()
            shift_labels = shift_labels - self.vocab_size - self.num_users

            loss_fct = CrossEntropyLoss()
            prompt_length = input_ids_prompt.shape[1]
            active_loss = attention_mask[:, prompt_length + 1 :].reshape(-1) == 1
            active_logits = shift_logits.view(-1, shift_logits.size(-1))[active_loss]
            active_labels = shift_labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)

            if regularize:
                assert lambda_V is not None and content_embeds is not None
                collaborative_embeds = torch.cat(
                    (self.base_model.embed(input_ids_prompt), self.base_model.embed(input_ids_main)),
                    axis=1,
                )
                regularize_loss = lambda_V * torch.mean(
                    nn.MSELoss(reduction="sum")(collaborative_embeds, content_embeds)
                )
                loss = loss + regularize_loss
                outputs = (loss, regularize_loss) + outputs
            else:
                outputs = (loss,) + outputs
        return outputs


class ContentGPTForUserItemWithLMHeadBatch(nn.Module):
    """
    Content model: LM head over vocab space.
    """

    def __init__(self, config, base_model: GPT4RecommendationBaseModel):
        super().__init__()
        self.base_model = base_model
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.base_model.gpt2model.wte.weight

    def forward(
        self,
        input_ids_prompt: torch.Tensor,
        input_ids_main: torch.Tensor,
        labels_main: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        regularize: bool = False,
        lambda_V: float | None = None,
        collaborative_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        outputs_prompt = self.base_model(input_ids=input_ids_prompt, return_dict=True, **kwargs)
        past_key_values = outputs_prompt.past_key_values

        outputs_main = self.base_model(
            input_ids=input_ids_main,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            return_dict=True,
        )

        lm_logits = self.lm_head(outputs_main.last_hidden_state)
        outputs = (lm_logits,) + outputs_main[1:]

        if labels_main is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels_main[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss()
            prompt_length = input_ids_prompt.shape[1]
            active_loss = attention_mask[:, prompt_length + 1 :].reshape(-1) == 1
            active_logits = shift_logits.view(-1, shift_logits.size(-1))[active_loss]
            active_labels = shift_labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)

            if regularize:
                assert lambda_V is not None and collaborative_embeds is not None
                content_embeds = self.base_model.embed(input_ids_prompt)
                regularize_loss = lambda_V * torch.mean(
                    nn.MSELoss(reduction="sum")(content_embeds, collaborative_embeds)
                )
                loss = loss + regularize_loss
                outputs = (loss, regularize_loss) + outputs
            else:
                outputs = (loss,) + outputs
        return outputs


class CollaborativeGPTwithItemRecommendHead(nn.Module):
    """
    Recommend items with a multinomial likelihood over item space.
    """

    def __init__(self, config, base_model: GPT4RecommendationBaseModel):
        super().__init__()
        self.num_users = int(config.num_users)
        self.num_items = int(config.num_items)
        self.base_model = base_model

        self.item_head = nn.Linear(config.n_embd, self.num_items, bias=False)
        self.item_head.weight = self.base_model.item_embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        target_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        regularize: bool = False,
        lambda_V: float | None = None,
        main_ids: torch.Tensor | None = None,
        content_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        transformer_outputs = self.base_model(input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = transformer_outputs[0]

        last_non_pad_token_indices = attention_mask.sum(dim=1) - 1
        last_token_hidden_states = torch.stack(
            [hidden_states[i, idx, :] for i, idx in enumerate(last_non_pad_token_indices)]
        )

        item_scores = self.item_head(last_token_hidden_states)
        item_log_probs = F.log_softmax(item_scores, dim=-1)

        neg_ll = -torch.mean(torch.sum(item_log_probs * target_ids, dim=-1))

        if regularize:
            assert lambda_V is not None and main_ids is not None and content_embeds is not None
            rec_embeds_prompt = self.base_model.embed(input_ids)
            rec_embeds_target = self.base_model.embed(main_ids)
            rec_embeds = torch.cat((rec_embeds_prompt, rec_embeds_target), axis=1)
            regularize_loss = lambda_V * torch.mean(nn.MSELoss(reduction="sum")(rec_embeds, content_embeds))
            neg_ll = neg_ll + regularize_loss
            return (neg_ll, regularize_loss, item_log_probs)
        return (neg_ll, item_log_probs)


