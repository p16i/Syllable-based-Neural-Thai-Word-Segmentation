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

    def __init__(self, data_config, model_config="embs:8|cells:32|l1:16|oc:BI|crf:1|do:0.0"):
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

        embedding = self.sy_embeddings(x)

        out, _ = self.lstm(embedding.permute(1, 0, 2))

        out = out.permute(1, 0, 2)

        out = self.linear1(out)
        out = self.linear2(out)

        return out

    def decode(self, logits, seq_lengths):
        if hasattr(self, "crf"):
            mask = loss.create_mask_with_length(seq_lengths).to(logits.device)
            return self.crf.decode(
                logits, mask=mask
            )
        else:
            return super().decode(logits, seq_lengths)