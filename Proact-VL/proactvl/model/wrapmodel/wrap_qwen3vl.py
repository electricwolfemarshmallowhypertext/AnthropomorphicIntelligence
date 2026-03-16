from transformers import Qwen3VLForConditionalGeneration
import torch.nn as nn
from transformers.activations import ACT2FN
from .base import ResponseHead
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb
import torch

from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel, create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils.generic import check_model_inputs
from transformers.utils import auto_docstring
from transformers.processing_utils import Unpack
from typing import Any, Callable, Optional, Union

@check_model_inputs
# @auto_docstring
def forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    # args for deepstack
    visual_pos_masks: Optional[torch.Tensor] = None,
    deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Union[tuple, BaseModelOutputWithPast]:
    r"""
    visual_pos_masks (`torch.Tensor` of shape `(batch_size, seqlen)`, *optional*):
        The mask of the visual positions.
    deepstack_visual_embeds (`list[torch.Tensor]`, *optional*):
        The deepstack visual embeddings. The shape is (num_layers, visual_seqlen, embed_dim).
        The feature is extracted from the different visual encoder layers, and fed to the decoder
        hidden states. It's from the paper DeepStack(https://arxiv.org/abs/2406.04334).
    """
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    # torch.jit.trace() doesn't support cache objects in the output
    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache(config=self.config)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    # the hard coded `3` is for temporal, height and width.
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    else:
        # text_position_ids = position_ids[0]
        text_position_ids = None

    attention_mask = create_causal_mask(
        config=self.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=text_position_ids,
    )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    for layer_idx, decoder_layer in enumerate(self.layers):
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = layer_outputs

        # add visual features to the hidden states of first several layers
        if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
            hidden_states = self._deepstack_process(
                hidden_states,
                visual_pos_masks,
                deepstack_visual_embeds[layer_idx],
            )

    hidden_states = self.norm(hidden_states)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )

Qwen3VLTextModel.forward = forward

class Qwen3VLMLP(nn.Module):
    def __init__(self, hidden_size, down_size, bias: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = int(hidden_size / 4)
        self.down_size =down_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.down_size, bias=bias)
        self.act_fn = ACT2FN['gelu']

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class WrapQwen3VL(Qwen3VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.state_proj = ResponseHead(self.config.text_config.hidden_size, 1)

    def get_position_ids(self, **kwargs):
        input_ids = kwargs.get('input_ids', None)
        image_grid_thw = kwargs.get('image_grid_thw', None)
        video_grid_thw = kwargs.get('video_grid_thw', None)
        attention_mask = kwargs.get('attention_mask', None)
        max_position_id = kwargs.get('max_position_id', 0)
        position_ids, mrope_position_deltas = self.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
        position_ids = position_ids.add(max_position_id+1)
        return position_ids

    def shift_position_ids(self, shift_size, key_cache, freeze_token_cnt):
        dummy_tensor =  torch.empty(1, dtype=key_cache[0].dtype, device=key_cache[0].device)
        B, H, T, D = key_cache.shape
        position_ids = torch.zeros((3, B, T-freeze_token_cnt), dtype=torch.long, device=key_cache[0].device).fill_(shift_size)
        position_embeddings = self.model.language_model.rotary_emb(dummy_tensor, position_ids)
        cos, sin = position_embeddings
        _, key_cache_shifted_not_frozen = apply_rotary_pos_emb(
            dummy_tensor, 
            key_cache[:,:,freeze_token_cnt:,:], 
            cos, sin, 
            self.model.config.text_config.rope_scaling['mrope_section']
        )
        key_cache[:,:,freeze_token_cnt:,:] = key_cache_shifted_not_frozen
        return key_cache