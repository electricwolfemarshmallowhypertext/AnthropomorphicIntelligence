import logging
import os
import torch

from proactvl.model.modeling_proact import ProAct_OmniModel, ProActConfig
from proactvl.train.proact_trainer import ProActTrainer
from proactvl.data.custom_commentary_dataset import CustomCommentaryDataset, DataCollatorForStream2Text
from proactvl.utils.metrics import compute_metrics, preprocess_logits_for_metrics, compute_active_metrics, preprocess_active_logits_for_metrics, compute_casual_metrics, preprocess_casual_logits_for_metrics
from transformers.trainer_utils import get_last_checkpoint

logging.basicConfig(
    format="[%(asctime)s] - [%(levelname)s] - [%(name)s]: %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def set_model(model, training_args, model_args):
    if not training_args.use_lora:
        # fully fine-tune
        model._unfreeze_parameters(model.llm)
        for name, param in model.llm.named_parameters():
            if "visual" in name:
                if training_args.freeze_visual:
                    param.requires_grad = False
            if 'audio' in name:
                if training_args.freeze_audio:
                    param.requires_grad = False
        if training_args.gradient_checkpointing and hasattr(model.llm, "enable_input_require_grads"):
            model.llm.enable_input_require_grads()
        return
        
    stage = training_args.finetune_strategy
    if stage == "strategy3":
        model._freeze_parameters(model.llm)
        # LoRA
        if training_args.lora_enable:
            from peft import LoraConfig, TaskType, get_peft_model
            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                lora_dropout=training_args.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=training_args.lora_target_modules,
                exclude_modules=r".*state_proj.*",
                modules_to_save=['state_proj'] 
            )
            # FIXME: omni model needs to adapt
            # if 'omni' in model.model_name_or_path.lower():
            #     model.llm.thinker = get_peft_model(model.llm.thinker, lora_config)
            # else:
            model.llm = get_peft_model(model.llm, lora_config)
        # if 'omni' in model.model_name_or_path.lower():
        #     for name, param in model.llm.thinker.named_parameters():
        #         if "visual" in name:
        #             if training_args.freeze_visual:
        #                 param.requires_grad = False
        #         if "audio" in name:
        #             if training_args.freeze_audio:
        #                 param.requires_grad = False
        # else:
        for name, param in model.llm.named_parameters():
            if "visual" in name:
                if training_args.freeze_visual:
                    param.requires_grad = False
            if 'audio' in name:
                if training_args.freeze_audio:
                    param.requires_grad = False


        # if 'omni' in model.model_name_or_path.lower():
        #     for name, param in model.state_proj.named_parameters():
        #         param.requires_grad = True
        # else:
        for name, param in model.llm.state_proj.named_parameters():
            param.requires_grad = True
        
        # if 'omni' in model.model_name_or_path.lower():
        #     if training_args.gradient_checkpointing and hasattr(model.llm.thinker, "enable_input_require_grads"):
        #         model.llm.thinker.enable_input_require_grads()
        # else:
        if training_args.gradient_checkpointing and hasattr(model.llm, "enable_input_require_grads"):
            model.llm.enable_input_require_grads()
    elif stage == "strategy2":
        model._freeze_parameters(model.llm)
        for name, param in model.state_proj.named_parameters():
            param.requires_grad = True
        if training_args.gradient_checkpointing and hasattr(model.llm.thinker, "enable_input_require_grads"):
            model.llm.thinker.enable_input_require_grads()
    elif stage == 'strategy1':
        model._freeze_parameters(model.llm)
        # LoRA
        if training_args.lora_enable:
                from peft import LoraConfig, TaskType, get_peft_model
                lora_config = LoraConfig(
                    r=training_args.lora_r,
                    lora_alpha=training_args.lora_alpha,
                    lora_dropout=training_args.lora_dropout,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                    target_modules=training_args.lora_target_modules,
                    
                )
                model.llm.thinker = get_peft_model(model.llm.thinker, lora_config)

        for name, param in model.llm.thinker.named_parameters():
            if "visual" in name:
                if training_args.freeze_visual:
                    param.requires_grad = False
            if "audio" in name:
                if training_args.freeze_audio:
                    param.requires_grad = False
        if training_args.gradient_checkpointing and hasattr(model.llm.thinker, "enable_input_require_grads"):
            model.llm.thinker.enable_input_require_grads()
    else:
        raise ValueError(f"Unknown finetune stage: {stage}")

def run(
    data_args,
    model_args,
    training_args
):
    logger.info("Training arguments: %s", training_args)
    logger.info("Model arguments: %s", model_args)
    logger.info("Data arguments: %s", data_args)

    config = ProActConfig(
        model_name_or_path=model_args.model_name_or_path,
        enable_audio_output=model_args.enable_audio_output,
        # state_threshold=model_args.state_threshold,
        # loss_active_scale=model_args.loss_active_scale,
        # add_special_tokens=model_args.add_special_tokens,
        active_layer_id=model_args.active_layer_id,
        # finetune_strategy=model_args.finetune_strategy,
    )
    if model_args.ckpt_path is not None:
        logger.info(f"Loading model from checkpoint: {model_args.ckpt_path}")
        model = ProAct_OmniModel.from_pretrained(config, model_args.ckpt_path)
    else:
        logger.info(f"Initializing model from scratch.")
        model = ProAct_OmniModel(config=config)

    # set model
    set_model(model, training_args, model_args)

    if torch.distributed.get_rank() == 0:
        model._print_trainable_params()

    # data
    train_dataset = CustomCommentaryDataset(
        dataset_names=data_args.train_dataset_names,
        data_dir_path=data_args.data_dir_path,
        processor=model.processor,
        use_audio_in_video=data_args.use_audio_in_video,
        is_train=True,
        active_eos_token=model.active_eos_token,
        silence_eos_token=model.silence_eos_token,
        chunk_flag=model.chunk_flag,
    )
    logger.info(f"train_dataset length: {len(train_dataset)}")
    val_dataset = CustomCommentaryDataset(
        dataset_names=data_args.val_dataset_names,
        data_dir_path=data_args.data_dir_path,
        processor=model.processor,
        use_audio_in_video=data_args.use_audio_in_video,
        is_train=False,
        active_eos_token=model.active_eos_token,
        silence_eos_token=model.silence_eos_token,
        chunk_flag=model.chunk_flag,
    )
    logger.info(f"val_dataset length: {len(val_dataset)}")

    datacollator = DataCollatorForStream2Text(tokenizer=model.processor.tokenizer)

    compute_metrics_fn = None
    preprocess_fn = None
    if training_args.finetune_strategy == "strategy1":
        compute_metrics_fn = compute_casual_metrics
        preprocess_fn = preprocess_casual_logits_for_metrics
    elif training_args.finetune_strategy == "strategy2":
        compute_metrics_fn = compute_active_metrics
        preprocess_fn = preprocess_active_logits_for_metrics
    elif training_args.finetune_strategy == "strategy3":
        compute_metrics_fn = compute_metrics
        preprocess_fn = preprocess_logits_for_metrics
    else:
        raise ValueError(f"Unknown finetune stage: {model_args.finetune_strategy}")

    # trainer
    trainer = ProActTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=datacollator,
        compute_metrics=compute_metrics_fn,
        preprocess_logits_for_metrics=preprocess_fn,
    )

    output_dir = training_args.output_dir
    last_checkpoint = None
    if os.path.exists(output_dir):
        last_checkpoint = get_last_checkpoint(output_dir)
    if last_checkpoint is not None:
        logger.info(f"Resuming training from checkpoint: {last_checkpoint}")
    else:
        logger.info("No checkpoint found, training from scratch.")
    trainer.train(resume_from_checkpoint=last_checkpoint)
    print(f'Training completed. Saving model to {os.path.join(training_args.output_dir, "final")}')
    logger.info(f'Training completed. Saving model to {os.path.join(training_args.output_dir, "final")}')
    trainer.save_model(os.path.join(training_args.output_dir, "final"))
