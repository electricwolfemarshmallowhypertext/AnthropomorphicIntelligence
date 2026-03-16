import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import EvalPrediction
import torch
import logging


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] - [%(levelname)s] - [%(name)s]: %(message)s",
)
logger = logging.getLogger(__name__)

def preprocess_logits_for_metrics(logits, labels):
    """
    - main: [B, T, V] -> argmax -> [B, T]
    - active: [B, T, 1] (logits) -> sigmoid -> (>0.5) -> [B, T] (0/1)
    """
    main_logits, active_logits = logits

    main_ids = main_logits.argmax(dim=-1)  # [B, T]

    # active -> 0/1
    active_prob = torch.sigmoid(active_logits).squeeze(-1)  # [B, T]
    active_ids = (active_prob > 0.5).long()                 # [B, T]

    return (main_ids, active_ids)

def preprocess_casual_logits_for_metrics(logits, labels):
    main_ids = logits.argmax(dim=-1)  # [B, T]
    return main_ids

def preprocess_active_logits_for_metrics(logits, labels):
    # active -> 0/1
    active_prob = torch.sigmoid(logits).squeeze(-1)  # [B, T]
    active_ids = (active_prob > 0.5).long()                 # [B, T]

    # print(f'active_prob: {active_prob[torch.where(labels != -100)]}')
    # print(f'active_ids: {active_ids[torch.where(labels != -100)]}')
    # print(f'active_labels: {labels[torch.where(labels != -100)]}')
    # print(f'active prob: {active_prob[torch.where(labels == 1)]}')
    # print(f'negative prob: {active_prob[torch.where(labels == 0)]}')
    return active_ids

def compute_metrics(eval_pred: EvalPrediction):
    preds, labels = eval_pred
    main_pred_ids, active_pred_ids = preds
    main_labels,   active_labels   = labels

    # 转 numpy
    to_np = lambda x: np.asarray(x) if x is not None else None
    main_pred_ids   = to_np(main_pred_ids)
    active_pred_ids = to_np(active_pred_ids)
    main_labels     = to_np(main_labels)
    active_labels   = to_np(active_labels)

    metrics = {}

    # ---------- 主任务 ----------
    if main_pred_ids is not None and main_labels is not None:
        mask = (main_labels != -100)
        # shift logits (去掉最后一个预测) 和 labels (去掉第一个标签)
        shift_preds = main_pred_ids[:, :-1]
        shift_labels = main_labels[:, 1:]

        mask = (shift_labels != -100)
        y_true = shift_labels[mask].ravel()
        y_pred = shift_preds[mask].ravel()

        metrics["main_acc"] = accuracy_score(y_true, y_pred) if y_true.size > 0 else 0.0

    # ---------- Active 任务 ----------
    if active_pred_ids is not None and active_labels is not None:
        mask = (active_labels != -100)
        # print number of active labels and correct predictions
        # print("Number of active labels:", np.sum(active_labels == 1))
        # print("Number of correct active predictions:", np.sum((active_pred_ids == 1) & (active_labels == 1)))

        num_active = np.sum(active_labels == 1)
        num_all = np.sum(active_labels != -100)
        num_correct_pred = np.sum((active_pred_ids == 1) & (active_labels == 1))
        # active pred_ids 需要乘上 mask
        num_active_pred = np.sum(active_pred_ids[mask] == 1)
        logger.info(f'There are {num_all} labels to be predicted, among which {num_active} are active. The model predicts {num_active_pred} active labels, correctly predicts {num_correct_pred} active labels.')
        y_true = active_labels[mask].ravel()
        y_pred = active_pred_ids[mask].ravel()

        if y_true.size > 0:
            metrics["active_acc"] = accuracy_score(y_true, y_pred)
            metrics["active_f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
            
            # 精确率 / 召回率 / F1，labels=[0,1]，顺序固定
            prec, rec, f1s, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=[0, 1], zero_division=0
            )
            metrics["active_f1_response"] = float(f1s[1])
            metrics["active_prec_response"] = float(prec[1])
            metrics["active_recall_response"] = float(rec[1])
        else:
            metrics.update({
                "active_acc": 0.0,
                "active_f1_macro": 0.0,
                "active_f1_response": 0.0,
                "active_prec_response": 0.0,
                "active_recall_response": 0.0,
            })

    return metrics


def compute_casual_metrics(eval_pred: EvalPrediction):
    preds, labels = eval_pred
    main_pred_ids = preds
    main_labels   = labels

    # 转 numpy
    to_np = lambda x: np.asarray(x) if x is not None else None
    main_pred_ids   = to_np(main_pred_ids)
    main_labels     = to_np(main_labels)
    metrics = {}

    # ---------- 主任务 ----------
    if main_pred_ids is not None and main_labels is not None:
        mask = (main_labels != -100)
        # shift logits (去掉最后一个预测) 和 labels (去掉第一个标签)
        shift_preds = main_pred_ids[:, :-1]
        shift_labels = main_labels[:, 1:]

        mask = (shift_labels != -100)
        y_true = shift_labels[mask].ravel()
        y_pred = shift_preds[mask].ravel()

        metrics["main_acc"] = accuracy_score(y_true, y_pred) if y_true.size > 0 else 0.0
    return metrics

def compute_active_metrics(eval_pred: EvalPrediction):
    preds, labels = eval_pred
    active_pred_ids = preds
    active_labels   = labels
    # 转 numpy
    to_np = lambda x: np.asarray(x) if x is not None else None
    active_pred_ids = to_np(active_pred_ids)
    active_labels   = to_np(active_labels)
    metrics = {}
    # ---------- Active 任务 ----------
    if active_pred_ids is not None and active_labels is not None:
        mask = (active_labels != -100)

        num_active = np.sum(active_labels == 1)
        num_all = np.sum(active_labels != -100)
        num_correct_pred = np.sum((active_pred_ids == 1) & (active_labels == 1))
        # active pred_ids 需要乘上 mask
        num_active_pred = np.sum(active_pred_ids[mask] == 1)
        # logger.info(f'There are {num_all} labels to be predicted, among which {num_active} are active. The model predicts {num_active_pred} active labels, correctly predicts {num_correct_pred} active labels.')
        y_true = active_labels[mask].ravel()
        y_pred = active_pred_ids[mask].ravel()
        # print(f'y_tru: {y_true}')
        # print(f'y_pred: {y_pred}')
        if y_true.size > 0:
            metrics["active_acc"] = accuracy_score(y_true, y_pred)
            metrics["active_f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
            
            # 精确率 / 召回率 / F1，labels=[0,1]，顺序固定
            prec, rec, f1s, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=[0, 1], zero_division=0
            )
            metrics["active_f1_response"] = float(f1s[1])
            metrics["active_prec_response"] = float(prec[1])
            metrics["active_recall_response"] = float(rec[1])
        else:
            metrics.update({
                "active_acc": 0.0,
                "active_f1_macro": 0.0,
                "active_f1_response": 0.0,
                "active_prec_response": 0.0,
                "active_recall_response": 0.0,
            })
    return metrics