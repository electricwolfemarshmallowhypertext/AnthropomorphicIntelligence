## Copyright (c) Microsoft Corporation.
## Licensed under the MIT license.

import os
import torch
import argparse
from model.model import PCC
from utils.argument import TrainArguments, DataArguments
from torch.cuda.amp import autocast


def infer(args: argparse.Namespace):
    model = PCC(
        args
    ).eval().to(args.device)
    tokenizer = model.compressor.tokenizer
    input_text = """In 1951, Kerner moved the team to Milwaukee, where they changed their name to the Hawks. Kerner and the team moved again in 1955 to St. Louis, where they won their only NBA Championship in 1958 and qualified to play in the NBA Finals in 1957, 1960 and 1961. The Hawks played the Boston Celtics in all four of their trips to the NBA Finals. The St. Louis Hawks moved to Atlanta in 1968, when Kerner 1958 NBA Finals The 1958 NBA World Championship Series was the championship series for the 1957–58 National Basketball Association (NBA) season, and the conclusion of the season's playoffs. It pitted the Western Division champion St. Louis Hawks against the Eastern Division champion Boston Celtics. The Hawks won the series in six games to win the club's first and so far only NBA championship title. "Hawks win series 4–2" After suffering a heartbreaking loss to the Celtics in Game 7 of the 1957 NBA Finals, St. Louis survived a sometimes difficult 1957-58 NBA season, returning to the NBA Finals to face 1971 NBA Finals The 1971 NBA World Championship Series was the championship series played at the conclusion of the National Basketball Association (NBA)'s 25th anniversary season of 1970–71. The Western Conference champion Milwaukee Bucks, who were founded just three years earlier, swept the Eastern Conference champion Baltimore Bullets in four games. Baltimore had dethroned the 1969–70 NBA champion New York Knicks. The Bucks were the first Western Conference champions to win the league's finals since the St. Louis Hawks did so in 1958. This was the first NBA Finals not played in the state of California in 10 years. It lead.Tom Heinsohn made two foul shots with 16 seconds left to cut it to 108-107. With the Boston defense converging on Pettit, Slater Martin tried a set shot that missed, but Pettit somehow fought his way through the mob of Celtics around him to tap the ball in and make a final Celtic field goal meaningless. Pettit had scored 50 points, including 18 of the Hawks' final 21 points propelling the Hawks' to the 1958 NBA Championship. The 1958 Hawks were the last team to win an NBA championship without a black player on the roster. 1958 NBA Finals The champion Celtics for more than a decade. With Bill Russell, the Celtics advanced to the 1957 NBA Finals and defeated the St. Louis Hawks in seven games, the first of a record 17 championships. Russell went on to win 11 championships, making him the most decorated player in NBA history. In 1958, the Celtics again advanced to the NBA Finals, this time losing to the Hawks in 6 games. However, with the acquisition of K.C. Jones that year, the Celtics began a dynasty that would last for more than a decade.\n
"""
    compress_ids = tokenizer(input_text, truncation=False)['input_ids']
    with torch.no_grad():
        with autocast(dtype=torch.bfloat16):
            output_text = model.generate(compress_ids, "Question: When did the hawks win the nba championship?\n\nAnswer: ", max_new_token=10)
    print(output_text)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCC Inference")
    parser.add_argument(
        "--compress_model", type=str, default="BroAlanTaps/Stage2-PCC-Lite-4x",
        help="compress model path, can be a local path or a huggingface path."
    )
    parser.add_argument(
        '--converter_model', type=str, default="BroAlanTaps/Stage2-PCC-Lite-4x",
        help="converter model path, can be a local path or a huggingface path. For common use, it should be the same as compress_model."
    )
    parser.add_argument(
        '--adapter_model', type=str, default="BroAlanTaps/Stage2-PCC-Lite-4x",
        help="converter model path, can be a local path or a huggingface path. For common use, it should be the same as compress_model."
    )
    parser.add_argument(
        '--decoder_model', type=str, default="BroAlanTaps/PCC-Decoder-Llama3-8B-Instruct",
        help="decoder model path, can be a local path or a huggingface path. For common, decoder model was freezed during training."
    )
    parser.add_argument(
        '--stage', type=int, default=2,
        help="stage 1 is pre-training, stage 2 is fine-tuning."
    )
    parser.add_argument(
        '--segment_length', type=int, default=256,
        help="per length of segment, default is 256."
    )
    parser.add_argument(
        '--ratio', type=int, default=4,
        help="ratio of compression, default is 4."
    )
    parser.add_argument(
        '--use_lora', type=bool, default=False,
        help="when using pcc-lite, set it to False. Set it to True when using pcc-large"
    )
    parser.add_argument(
        '--compressor_gradient_checkpoint', type=bool, default=False,
        help="whether to use gradient checkpointing for the compressor."
    )
    parser.add_argument(
        '--decoder_gradient_checkpoint', type=bool, default=False,
        help="whether to use gradient checkpointing for the decoder."
    )
    args = parser.parse_args()
    args.embed_len = args.segment_length // args.ratio
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    infer(args)