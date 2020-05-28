import numpy as np
import importlib
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

import attacut
from attacut import logger, loss

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
        return F.relu(self.conv(x))

class IteratedDilatedConvolutions(nn.Module):
    # ref: https://arxiv.org/abs/1702.02098
    def __init__(self, emb_dim, filters, dropout_rate):
        super(IteratedDilatedConvolutions, self).__init__()

        self.conv1 = ConvolutionLayer(emb_dim, filters, 3, dilation=1)
        self.conv2 = ConvolutionLayer(filters, filters, 3, dilation=2)
        self.conv3 = ConvolutionLayer(filters, filters, 3, dilation=4)

        self.dropout= torch.nn.Dropout(p=dropout_rate)


    def forward(self, x):
        conv1 = self.dropout(self.conv1(x))
        conv2 = self.dropout(self.conv2(conv1))
        return self.dropout(self.conv3(conv2))

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

    def decode(self, logits, seq_lengths):
        if hasattr(self, "crf"):
            mask = loss.create_mask_with_length(seq_lengths).to(logits.device)
            return self.crf.decode(
                logits, mask=mask
            )
        else:
            _, indices = torch.max(logits, dim=2)
            return self.output_scheme.decode_condition(
                indices.cpu().detach().numpy()
            )

def get_model(model_name) -> BaseModel:
    module_path = "attacut.models.%s" % model_name
    log.info("Taking %s" % module_path)

    model_mod = importlib.import_module(module_path)
    return model_mod.Model

def prepare_embedding(data_config, model_config):
    if type(model_config["embs"]) == str:
        dict_dir = data_config["dict_dir"]

        print(f"use embedding from {dict_dir}")

        name = model_config["embs"]
        sy_embeddings = np.load(f"{dict_dir}/sy-emb-{name}.npy")

        ix = np.argwhere(np.sum(sy_embeddings, axis=1) == 0).reshape(-1)

        np.random.seed(71)
        sy_embeddings[ix, :] = np.random.normal(
            loc=0,
            scale=1,
            size=(ix.shape[0], sy_embeddings.shape[1])
        )

        assert data_config['num_tokens'] == sy_embeddings.shape[0]

        return nn.Embedding.from_pretrained(
            torch.from_numpy(sy_embeddings).float(),
            freeze=False,
            padding_idx=0
        )
    else:
        print("create new syllable embedding")
        return nn.Embedding(
            data_config['num_tokens'],
            model_config["embs"],
            padding_idx=0
        )
