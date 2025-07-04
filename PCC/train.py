## Copyright (c) Microsoft Corporation.
## Licensed under the MIT license.

import logging
import os
import warnings

import numpy as np
import torch
from base_trainer import BaseTrainer, compute_metrics
from datasets import load_dataset, load_from_disk
from model.model import PCC
from transformers import HfArgumentParser
from utils.argument import DataArguments, TrainArguments
from utils.utils import DataCollator


# Configure logging and environment
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger("train.py")

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def main():
    # Parse arguments
    parser = HfArgumentParser((TrainArguments, DataArguments))
    training_args, data_args = parser.parse_args_into_dataclasses()
    
    # Setup Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )

    # Print arguments
    logger.info("-" * 100)
    logger.info("Training/evaluation parameters:")
    for arg, value in training_args.__dict__.items():
        logger.info(f"{arg}: {value}")
    
    logger.info("-" * 100)
    logger.info("Model Data parameters:")
    for arg, value in data_args.__dict__.items():
        logger.info(f"{arg}: {value}")

    # Set random seed
    set_seed(training_args.random_seed)

    # Load and prepare model
    model = PCC(training_args)
    
    # Configure data collator and trainer
    compressor_type = training_args.compressor_type
    data_collator = DataCollator(
            stage=training_args.stage,
            compress_pad_token_id=model.compressor.tokenizer.pad_token_id,
            llm_pad_token_id=model.decoder.tokenizer.pad_token_id,
            llm_eos_token_id=model.decoder.tokenizer.eos_token_id,
            compressor_type=compressor_type
    )

    # Load Datasets
    logger.info("-" * 100)
    logger.info("-" * 40 + "Dataset is loading!" + "-" * 40)

    train_dataset = None
    eval_dataset = None
    
    train_dataset = load_dataset(data_args.train_data_dir, split='train')
    
    eval_dataset = load_dataset(data_args.valid_data_dir, split='test')
        
    # Initialize trainer
    trainer = BaseTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Train or resume training
    if training_args.resume_from_checkpoint:
        logger.info("-" * 100)
        logger.info("-" * 40 + "Training is starting!" + "-" * 40)
        trainer.train(resume_from_checkpoint=training_args.last_ckpt_dir)
    else:
        logger.info("-" * 100)
        logger.info("-" * 40 + "Training is starting!" + "-" * 40)
        trainer.train(resume_from_checkpoint=False)

if __name__ == "__main__":
    main()
 