import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from attacut import utils, dataloaders, output_tags, char_type
from . import BaseModel, IteratedDilatedConvolutions


class Model(BaseModel):
    dataset = dataloaders.CharacterSeqDataset

    def __init__(self, data_config, model_config="embc:16|embt:16|conv:48|l1:16|do:0.1|oc:BI"):
        super(Model, self).__init__()

        no_chars = data_config["num_tokens"]

        config = utils.parse_model_params(model_config)
        conv_filters = config["conv"]
        dropout_rate = config.get("do", 0)

        self.output_scheme = output_tags.get_scheme(config["oc"])

        self.ch_embeddings = nn.Embedding(
            no_chars,
            config["embc"],
            padding_idx=0
        )

        self.ch_type_embeddings = nn.Embedding(
            char_type.get_total_char_types(),
            config["embt"],
        )

        emb_dim = config["embc"] + config["embt"]

        self.id_conv = IteratedDilatedConvolutions(
            emb_dim, conv_filters, dropout_rate
        )

        self.linear1 = nn.Linear(conv_filters, config['l1'])
        self.linear2 = nn.Linear(config['l1'], self.output_scheme.num_tags)

        self.model_params = model_config

    def forward(self, inputs):
        x, seq_lengths = inputs

        x_char, x_type = x[:, 0, :], x[:, 1, :]

        ch_type_embedding = self.ch_type_embeddings(x_type)
        ch_embedding = self.ch_embeddings(x_char)

        embedding = torch.cat((ch_embedding, ch_type_embedding), dim=2)

        embedding = embedding.permute(0, 2, 1)

        out = self.id_conv(embedding)

        out = out.permute(0, 2, 1)

        out = F.relu(self.linear1(out))
        out = self.linear2(out)

        return out