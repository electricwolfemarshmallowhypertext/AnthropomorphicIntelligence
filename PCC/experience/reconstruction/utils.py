## Copyright (c) Microsoft Corporation.
## Licensed under the MIT license.

from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer


class metrics:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def cal_bleu(pre:str,
                 ref:str):
        pred_text = word_tokenize(pre)
        label_text = word_tokenize(ref)
        bleu4_score = sentence_bleu([label_text], pred_text, weights=(0, 0, 0, 1))
        bleu3_score = sentence_bleu([label_text], pred_text, weights=(0, 0, 1, 0))
        bleu2_score = sentence_bleu([label_text], pred_text, weights=(0, 1, 0, 0))
        bleu1_score = sentence_bleu([label_text], pred_text, weights=(1, 0, 0, 0))
        bleu_score = sentence_bleu([label_text], pred_text,weights = (0.25,0.25,0.25,0.25))
        scorerL = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rougeL = scorerL.score(pre,ref)['rougeL'].fmeasure
        
        print(f"BLEU Score: {bleu_score}")
        print(f"BLEU-1 Score: {bleu1_score}")
        print(f"BLEU-2 Score: {bleu2_score}")
        print(f"BLEU-3 Score: {bleu3_score}")
        print(f"BLEU-4 Score: {bleu4_score}")
        print(f"Rouge-L Score: {rougeL}")
        return bleu_score,bleu1_score,bleu2_score,bleu3_score,bleu4_score,rougeL