import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from attacut import utils, dataloaders, logger, output_tags, char_type
from . import BaseModel, ConvolutionLayer

log = logger.get_logger(__name__)


class Model(BaseModel):
    dataset = dataloaders.SyllableCharacterSeqDataset

    def __init__(self, data_config, model_config="embc:16|embt:8|embs:8|cells:32|l1:16|bi:1|oc:BI|do:0.0"):
        super(Model, self).__init__()


        no_chars = data_config['num_char_tokens']
        log.info("no. characters: %d" % no_chars)

        no_syllables = data_config['num_tokens']
        log.info("no. syllables: %d" % no_syllables)

        config = utils.parse_model_params(model_config)

        self.output_scheme = output_tags.get_scheme(config["oc"])

        self.ch_type_embeddings = nn.Embedding(
            char_type.get_total_char_types(),
            config["embt"],
        )

        self.ch_embeddings = nn.Embedding(
            no_chars,
            config["embc"],
            padding_idx=0
        )

        self.sy_embeddings = nn.Embedding(
            no_syllables,
            config["embs"],
            padding_idx=0
        )

        emb_dim = config["embc"] + config["embs"] + config["embt"]

        num_cells, num_lstm_output, bi_direction = utils.compute_lstm_output_dim(
            config["cells"],
            config["bi"]
        )

        self.lstm = nn.LSTM(emb_dim, num_cells, dropout=config["do"], bidirectional=bi_direction)
        self.linear1 = nn.Linear(num_lstm_output, config["l1"])
        self.linear2 = nn.Linear(config["l1"], self.output_scheme.num_tags)

        self.model_params = model_config

    def forward(self, inputs):
        x, seq_lengths = inputs

        x_char, x_type, x_syllable = x[:, 0, :], x[:, 1, :], x[:, 2, :]

        ch_embedding = self.ch_embeddings(x_char)
        ch_type_embedding = self.ch_type_embeddings(x_type)
        sy_embedding = self.sy_embeddings(x_syllable)

        embedding = torch.cat((ch_embedding, ch_type_embedding, sy_embedding), dim=2)

        out, _ = self.lstm(embedding.permute(1, 0, 2))

        out = out.permute(1, 0, 2)

        out = self.linear1(out)
        out = self.linear2(out)

        return out