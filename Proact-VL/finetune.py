from transformers import HfArgumentParser

from proactvl.train.train import run
from proactvl.config.arguments import (CompanionDataArguments,
                                            CompanionModelArguments,
                                            CompanionTrainingArguments)
import warnings
import logging

import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"


warnings.filterwarnings(
    "ignore",
    category=SyntaxWarning,
    module=r".*processing_minicpmo.*",
)
# Logging control: suppress verbose logs during data loading
logging.getLogger("qwen_omni_utils.v2_5.vision_processor").setLevel(logging.ERROR)
logging.getLogger("root").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)


if __name__ == "__main__":
    
    parser = HfArgumentParser(
        (CompanionDataArguments, CompanionModelArguments, CompanionTrainingArguments)
    )
    data_args, model_args, training_args = parser.parse_args_into_dataclasses()
    
    
    run(
        data_args=data_args,
        model_args=model_args,
        training_args=training_args
    )
