import importlib
import re

import torch
import torch.nn as nn

import numpy as np

import attacut
from attacut import logger

from torchcrf import CRF

log = logger.get_logger(__name__)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


class ConvolutionBatchNorm(nn.Module):
    def __init__(self, channels, filters, kernel_size, stride=1, dilation=1):
        super(ConvolutionBatchNorm, self).__init__()

        padding = kernel_size // 2
        padding += padding * (dilation-1)

        self.conv = nn.Conv1d(
            channels,
            filters,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding
        )

        self.bn = nn.BatchNorm1d(filters)

    def forward(self, x):
        return self.bn(self.conv(x))

class ConvolutionLayer(nn.Module):
    def __init__(self, channels, filters, kernel_size, stride=1, dilation=1):
        super(ConvolutionLayer, self).__init__()

        padding = kernel_size // 2
        padding += padding * (dilation-1)

        self.conv = nn.Conv1d(
            channels,
            filters,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding
        )

    def forward(self, x):
        return self.conv(x)

class BaseModel(nn.Module):
    dataset = None
    @classmethod
    def load(cls, path, data_config, model_config, with_eval=True):
        model = cls(data_config, model_config)

        model_path = "%s/model.pth" % path
        model.load_state_dict(torch.load(model_path, map_location="cpu"))

        log.info("loaded: %s|%s (variables %d)" % (
            model_path,
            model_config,
            model.total_trainable_params()
        ))

        if with_eval:
            log.info("setting model to eval mode")
            model.eval()

        return model

    def total_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def decode_output(self, logits, y, seq):
        # logits's shape: batch ⨉ seq_length ⨉ num_tags
        # seq: seq's length
        if self.crf_model:

            yy = y.reshape(logits.shape[0], -1)

            max_len = logits.shape[1]
            mask = (torch.arange(max_len).expand(seq.shape[0], max_len) < seq.unsqueeze(1)).type(torch.uint8)

            mask = torch.t(mask)

            logits_permuted = logits.permute(1, 0, 2)

            lh = self.crf_model(
                logits_permuted, yy.permute(1, 0), mask=mask, reduction="mean"
            )

            decoded_tags = self.crf_model.decode(logits_permuted, mask=mask)

            return -lh, decoded_tags
        else:
            raise NotImplementedError("nooooo")


def get_model(model_name) -> BaseModel:
    module_path = "attacut.models.%s" % model_name
    log.info("Taking %s" % module_path)

    model_mod = importlib.import_module(module_path)
    return model_mod.Model
