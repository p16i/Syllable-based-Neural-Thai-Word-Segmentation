import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from attacut import utils, dataloaders, output_tags, char_type
from . import BaseModel, ConvolutionLayer


class Model(BaseModel):
    dataset = dataloaders.SyllableSeqDataset

    def __init__(self, data_config, model_config="embs:16|conv:48|l1:16|do:0.1|oc:BI"):
        super(Model, self).__init__()

        no_syllables = data_config["num_tokens"]

        config = utils.parse_model_params(model_config)
        conv_filters = config["conv"]
        dropout_rate = config.get("do", 0)

        self.output_scheme = output_tags.get_scheme(config["oc"])

        self.sy_embeddings = nn.Embedding(
            no_syllables,
            config["embs"],
            padding_idx=0
        )

        self.dropout= torch.nn.Dropout(p=dropout_rate)

        emb_dim = config["embs"]

        self.conv1 = ConvolutionLayer(emb_dim, conv_filters, 3)
        self.conv2 = ConvolutionLayer(emb_dim, conv_filters, 5, dilation=3)
        self.conv3 = ConvolutionLayer(emb_dim, conv_filters, 9, dilation=2)

        self.linear1 = nn.Linear(conv_filters, config['l1'])
        self.linear2 = nn.Linear(config['l1'], self.output_scheme.num_tags)

        self.model_params = model_config

    def forward(self, inputs):
        x, seq_lengths = inputs

        embedding = self.sy_embeddings(x)
        embedding = embedding.permute(0, 2, 1)

        conv1 = self.conv1(embedding).permute(0, 2, 1)
        conv2 = self.conv2(embedding).permute(0, 2, 1)
        conv3 = self.conv3(embedding).permute(0, 2, 1)

        out = torch.stack((conv1, conv2, conv3), 3)

        out, _ = torch.max(out, 3)
        out = self.dropout(out)

        out = F.relu(self.linear1(out))
        out = self.linear2(out)

        return out