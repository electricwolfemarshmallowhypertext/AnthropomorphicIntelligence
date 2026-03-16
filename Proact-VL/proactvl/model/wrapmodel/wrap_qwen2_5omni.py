import torch.nn as nn
from transformers import Qwen2_5OmniForConditionalGeneration
from .base import ResponseHead
import torch
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import apply_multimodal_rotary_pos_emb


class WrapQwen2_5OmniThinker(Qwen2_5OmniForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.state_proj = ResponseHead(self.thinker.config.text_config.hidden_size, 1, 1)

    def forward(self, *args, **kwargs):
        return self.model.thinker(*args, **kwargs)

    # def get_rope_index(self, *args, **kwargs):
    #     self.llm.thinker.get_rope_index(*args, **kwargs)

    def get_position_ids(self, **kwargs):
        input_ids = kwargs.get('input_ids', None)
        image_grid_thw = kwargs.get('image_grid_thw', None)
        video_grid_thw = kwargs.get('video_grid_thw', None)
        attention_mask = kwargs.get('attention_mask', None)
        use_audio_in_video = kwargs.get('use_audio_in_video', False)
        audio_feature_lengths = kwargs.get('audio_feature_lengths', None)
        video_second_per_grid = kwargs.get('video_second_per_grid', None)
        max_position_id = kwargs.get('max_position_id', 0)
        position_ids, rope_deltas = self.thinker.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            use_audio_in_video=use_audio_in_video,
            audio_seqlens=audio_feature_lengths,
            second_per_grids=video_second_per_grid,
        )
        
        position_ids = position_ids.add(max_position_id+1)
        return position_ids

    def shift_position_ids(self, shift_size, key_cache, freeze_token_cnt):
        dummy_tensor =  torch.empty(1, dtype=key_cache[0].dtype, device=key_cache[0].device)
        B, H, T, D = key_cache.shape
        position_ids = torch.zeros((3, B, T-freeze_token_cnt), dtype=torch.long, device=key_cache[0].device).fill_(shift_size)
        position_embeddings = self.llm.thinker.model.rotary_emb(dummy_tensor, position_ids)
        cos, sin = position_embeddings
        _, key_cache_shifted_not_frozen = apply_multimodal_rotary_pos_emb(
            dummy_tensor, 
            key_cache[:,:,freeze_token_cnt:,:], 
            cos, sin, 
            self.llm.config.thinker_config.text_config.rope_scaling['mrope_section']
        )
        key_cache[:,:,freeze_token_cnt:,:] = key_cache_shifted_not_frozen
        return key_cache