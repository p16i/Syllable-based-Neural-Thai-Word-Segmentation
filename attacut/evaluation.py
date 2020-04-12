from collections import namedtuple

import torch

import numpy as np

from nptyping import Array

EvaluationMetrics = namedtuple(
    "EvaluationMetrics",
    ["tp", "fp", "fn"]
)

def compute_metrics(
    labels: Array[torch.int],
    preds: Array[torch.int]
) -> EvaluationMetrics:

    # manually implemented due to keep no. of dependencies minimal
    tp = torch.sum(preds * labels)
    fp = torch.sum(preds * (1-labels))
    fn = torch.sum((1-preds) * labels)

    return EvaluationMetrics(
        tp=tp.cpu(),
        fp=fp.cpu(),
        fn=fn.cpu(),
    )
