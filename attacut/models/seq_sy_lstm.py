import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from torchcrf import CRF

from attacut import utils, dataloaders, logger, output_tags, char_type, loss
from . import BaseModel, ConvolutionLayer

log = logger.get_logger(__name__)


class Model(BaseModel):
    dataset = dataloaders.SyllableSeqDataset

    def __init__(self, data_config, model_config="embs:8|cells:32|l1:16|oc:BI|crf:1"):
        super(Model, self).__init__()

        no_syllables = data_config['num_tokens']
        log.info("no. syllables: %d" % no_syllables)

        config = utils.parse_model_params(model_config)

        self.output_scheme = output_tags.get_scheme(config["oc"])

        self.sy_embeddings = nn.Embedding(
            no_syllables,
            config["embs"],
            padding_idx=0
        )

        if config["crf"]:
            assert self.output_scheme.num_tags > 2, "Using CRF doesn't work with BI tag"
            self.crf = CRF(self.output_scheme.num_tags, batch_first=True)

        emb_dim = config["embs"]

        self.lstm = nn.LSTM(emb_dim, config["cells"] // 2, bidirectional=True)
        self.linear1 = nn.Linear(config['cells'], self.output_scheme.num_tags)

        self.model_params = model_config

    def forward(self, inputs):
        x, seq_lengths = inputs

        embedding = self.sy_embeddings(x)

        out, _ = self.lstm(embedding.permute(1, 0, 2))

        out = out.permute(1, 0, 2)

        out = self.linear1(out)

        return out

    def decode(self, logits, seq_lengths):
        return self.crf.decode(
            logits, mask=loss.create_mask_with_length(seq_lengths)
        )