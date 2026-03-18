import logging
import os
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.generation import GenerationMixin
from peft import PeftModel
from transformers import AutoProcessor


from proactvl.model.wrapmodel import WrapQwen3VL, WrapQwen2_5VL, WrapQwen2VL, WrapQwen2_5OmniThinker


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] - [%(levelname)s] - [%(name)s]: %(message)s",
)
logger = logging.getLogger(__name__)

class ProActConfig(PretrainedConfig):
    model_type = "proact_model"
    model_name_or_path: str = "Qwen/Qwen2.5-Omni-7B"
    active_layer_id: int = -2
    torch_dtype = torch.bfloat16
    attn_implementation: str = 'flash_attention_2'
    low_cpu_mem_usage: bool = True
    enable_audio_output: bool = False

    # deprecated 
    ## move to training args
    loss_active_scale: float = 1.0
    finetune_strategy: str = 'none'
    ## move to infer args
    state_threshold: float = 0.5



class ProAct_OmniModel(PreTrainedModel, GenerationMixin):
    config_class = ProActConfig
    base_model_prefix = "proact_mllm"
    supports_gradient_checkpointing = True
    # _no_split_modules = ["Qwen2_5OmniDecoderLayer", "Qwen2_5OmniVisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = True

    def __init__(self, config: ProActConfig):
        super(ProAct_OmniModel, self).__init__(config)

        self.model_name_or_path = config.model_name_or_path
        # for omni model
        self.enable_audio_output = config.enable_audio_output
        
        self.active_layer_id = config.active_layer_id if hasattr(config, 'active_layer_id') else -2
        logger.info(f'Using active_layer_id: {self.active_layer_id}')
        self.attn_implementation = config.attn_implementation

        if self.model_name_or_path in ['Qwen/Qwen3-VL-8B-Instruct', 'Qwen/Qwen3-VL-2B-Instruct']:
            self.llm = WrapQwen3VL.from_pretrained(
                self.model_name_or_path,
                torch_dtype=config.torch_dtype,
                attn_implementation=config.attn_implementation,
                low_cpu_mem_usage=config.low_cpu_mem_usage,
            )
        elif self.model_name_or_path in ['Qwen/Qwen2.5-VL-7B-Instruct', 'mit-han-lab/StreamingVLM']:
            self.llm = WrapQwen2_5VL.from_pretrained(
                self.model_name_or_path,
                torch_dtype=config.torch_dtype,
                attn_implementation=config.attn_implementation,
                low_cpu_mem_usage=config.low_cpu_mem_usage,
            )
        elif self.model_name_or_path in ['chenjoya/LiveCC-7B-Instruct', 'Qwen/Qwen2-VL-7B-Instruct', 'chenjoya/LiveCC-7B-Base']:
            self.llm = WrapQwen2VL.from_pretrained(
                self.model_name_or_path,
                torch_dtype=config.torch_dtype,
                attn_implementation=config.attn_implementation,
                low_cpu_mem_usage=config.low_cpu_mem_usage,
            )
        elif self.model_name_or_path == "Qwen/Qwen2.5-Omni-7B":
            self.llm = WrapQwen2_5OmniThinker.from_pretrained(
                self.model_name_or_path,
                torch_dtype=config.torch_dtype,
                attn_implementation=self.attn_implementation,
                low_cpu_mem_usage=config.low_cpu_mem_usage,
                enable_audio_output=self.enable_audio_output,
            )
        self.active_eos_token = ' ...'
        self.silence_eos_token = ' ...'

        self.processor = AutoProcessor.from_pretrained(self.model_name_or_path)
        self.processor.tokenizer.add_tokens(['<|elongated|>', '<|short_break|>', '<|long_break|>', '<|laugh|>'])
        self.processor.tokenizer.add_special_tokens({
            "additional_special_tokens": ['<|query_start|>', '<|query_end|>', '<|history_start|>', '<|history_end|>', '<|FLAG|>']
        })
        print(f"[OK] Added special tokens to tokenizer.")

        # response mechanism related configs
        # self.state_threshold = config.state_threshold
        self.loss_active_scale = config.loss_active_scale
        self.chunk_flag = '<|FLAG|>'
        logger.info(f'Using active eos token: {self.active_eos_token}')
        logger.info(f'Using silence oes token: {self.silence_eos_token}')
        logger.info(f'Using chunk_flag: {self.chunk_flag}')

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs):
        if hasattr(self.llm, "config"):
            self.llm.config.use_cache = False
        if hasattr(self.llm, "gradient_checkpointing_enable"):
            try:
                self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
            except TypeError:
                self.llm.gradient_checkpointing_enable()
        if hasattr(self.llm, "enable_input_require_grads"):
            self.llm.enable_input_require_grads()

    def _print_trainable_params(self):        
        trainable_params, all_params = 0, 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params += param.numel()
                print(f"thinker trainable param: {name}, param shape: {param.shape}")
            all_params += param.numel()
        
        logger.info("thinker trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
        trainable_params, all_params, 100 * trainable_params / all_params))
        print("thinker trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
        trainable_params, all_params, 100 * trainable_params / all_params))

    @staticmethod
    def _freeze_parameters(module: nn.Module):
        """
        Freeze all parameters in the given module.
        """
        for name, param in module.named_parameters():
            param.requires_grad = False

    @staticmethod
    def _unfreeze_parameters(module: nn.Module):
        """
        Unfreeze all parameters in the given module.
        """
        for name, param in module.named_parameters():
            param.requires_grad = True

    # def set_threshold(self, threshold):
    #     if hasattr(self, 'state_threhold'):
    #         self.state_threhold = threshold
    #         logger.info(f'Set state_threhold to {self.state_threhold}')
    #     else:
    #         logger.warning('No state_threhold attribute to set.')

    '''
    During training, `forward` computes main loss + active loss.
    Main loss measures text prediction accuracy, and active loss measures active-label prediction accuracy.
    There may be cases where all active labels are 0 (all silent); in that case, main loss is not computed.
    During inference,
    '''
    def forward(self, active_labels=None, output_active_logits=True, *args, **kwargs):
        if output_active_logits:
            # To output active_logits, hidden_states are needed for computation, so set this to True
            kwargs['output_hidden_states'] = True

        output = self.llm(*args, **kwargs)

        if output_active_logits and output.hidden_states is not None:
            layer_id = self.active_layer_id
            last_hidden_state = output.hidden_states[layer_id]
            active_logits = self.llm.state_proj(last_hidden_state)
            output.active_logits = active_logits
            output['active_logits'] = active_logits

        if 'output_hidden_states' in kwargs and kwargs['output_hidden_states'] is False:
            output.hidden_states = None
        # torch.cuda.empty_cache()
        
        return output

    def get_position_ids(self, **kwargs):
        return self.llm.get_position_ids(**kwargs)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        input_features=None,
        feature_attention_mask=None,
        use_audio_in_video=False,
        video_second_per_grid=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            use_audio_in_video=use_audio_in_video,
            video_second_per_grid=video_second_per_grid,
            **kwargs,
        )
        # Original Qwen 2.5 Omni sets position_ids to None; that behavior is commented out here
        # model_inputs["position_ids"] = None
        # inplace modification
        # model_kwargs["cache_position"][-1:] + num_new_tokens
        position_ids.add_(1)
        model_inputs["position_ids"] = position_ids

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs

    def shift_position_ids(self, shift_size: int, key_cache, freeze_token_cnt: int = 0):
        return self.llm.shift_position_ids(shift_size, key_cache, freeze_token_cnt)

    def save_pretrained(self, output_dir: str, **kwargs):
        """
        Simplified version:
        - Save `ProActConfig` to output_dir/config.json
        - If `self.llm` is a `PeftModel`, save only the LoRA adapter to output_dir/llm_adapter
        - Otherwise, save the full LLM to output_dir/llm
        - Always save the processor to output_dir/processor
        """
        output_dir = os.fspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # 1) Save config (outer wrapper structure)
        if hasattr(self, "config") and isinstance(self.config, PretrainedConfig):
            self.config.save_pretrained(output_dir)

        # 2) Save llm
        if isinstance(self.llm, PeftModel):
            adapter_dir = os.path.join(output_dir, "llm_adapter")
            os.makedirs(adapter_dir, exist_ok=True)
            logger.info(f"[ProAct] Saving LoRA adapter to {adapter_dir}")
            self.llm.save_pretrained(adapter_dir)  # This stores only adapter + adapter_config
        else:
            llm_dir = os.path.join(output_dir, "llm")
            os.makedirs(llm_dir, exist_ok=True)
            logger.info(f"[ProAct] Saving full llm to {llm_dir}")
            self.llm.save_pretrained(llm_dir, max_shard_size="3600MB",)

        # 3) Save processor
        proc_dir = os.path.join(output_dir, "processor")
        os.makedirs(proc_dir, exist_ok=True)
        logger.info(f"[ProAct] Saving processor to {proc_dir}")
        self.processor.save_pretrained(proc_dir)


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs):
        """
        Simplified `from_pretrained`:
        - First restore config and model_name_or_path with `ProActConfig.from_pretrained`
        - `__init__` reloads the base llm (Qwen / WrapQwen) using `config.model_name_or_path`
        - If `llm_adapter/` exists, wrap the base llm with LoRA via `PeftModel.from_pretrained`
        - If `llm/` exists, reload it directly as a full model directory
        - Restore processor from `processor/`; otherwise fall back to `model_name_or_path`
        """
        load_dir = os.fspath(pretrained_model_name_or_path)
        # 0) If this is a remote repo, download it to local cache first
        if not os.path.isdir(load_dir):
            from huggingface_hub import snapshot_download
            load_dir = snapshot_download(repo_id=str(pretrained_model_name_or_path))

        # 1) Load config
        config: ProActConfig = kwargs.pop("config", None)
        if config is None:
            config = ProActConfig.from_pretrained(load_dir)

        # 2) First construct a "shell" model (it initializes self.llm from config.model_name_or_path)
        model = cls(config, *model_args, **kwargs)

        # 3) Handle llm
        adapter_dir = os.path.join(load_dir, "llm_adapter")
        full_llm_dir = os.path.join(load_dir, "llm")

        if os.path.isdir(adapter_dir):
            # Saved content is LoRA adapter + adapter_config
            logger.info(f"[ProAct] Loading base llm from {config.model_name_or_path} and LoRA adapter from {adapter_dir}")
            base_llm = model.llm  # __init__ already loaded this from model_name_or_path
            model.llm = PeftModel.from_pretrained(base_llm, adapter_dir)
            model.llm = model.llm.merge_and_unload()  # Merge LoRA weights and release memory
        elif os.path.isdir(full_llm_dir):
            # Saved content is a full llm (e.g. when LoRA is not used)
            logger.info(f"[ProAct] Loading full llm from {full_llm_dir}")
            base_cls = type(model.llm)
            model.llm = base_cls.from_pretrained(
                full_llm_dir, 
                torch_dtype=config.torch_dtype, 
                attn_implementation=config.attn_implementation, 
                low_cpu_mem_usage=config.low_cpu_mem_usage
            )
        else:
            logger.warning(
                f"[ProAct] No llm_adapter/ or llm/ found in {load_dir}, "
                f"keep llm as freshly initialized from {config.model_name_or_path}."
            )

        # 4) Handle processor
        proc_dir = os.path.join(load_dir, "processor")
        if os.path.isdir(proc_dir):
            logger.info(f"[ProAct] Loading processor from {proc_dir}")
            model.processor = AutoProcessor.from_pretrained(proc_dir)
        else:
            logger.info(f"[ProAct] No processor/ found, loading from base {config.model_name_or_path}")
            model.processor = AutoProcessor.from_pretrained(config.model_name_or_path)

        return model