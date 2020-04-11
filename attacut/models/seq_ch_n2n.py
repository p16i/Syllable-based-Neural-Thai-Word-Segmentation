import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from attacut import utils, dataloaders, output_tags
from . import BaseModel, ConvolutionLayer, get_device

from torchcrf import CRF

class Model(BaseModel):
    dataset = dataloaders.CharacterSeqDataset

    def __init__(self, data_config, model_config="emb:32|conv:48|l1:16|do:0.1|oc:BI"):
        super(Model, self).__init__()

        no_chars = data_config['num_tokens']

        config = utils.parse_model_params(model_config)
        emb_dim = config['emb']
        conv_filters = config['conv']
        dropout_rate = config.get("do", 0)

        self.output_scheme = output_tags.get_scheme(config["oc"])

        if config["crf"]:
            self.crf_model = CRF(self.output_scheme.num_tags)

        self.embeddings = nn.Embedding(
            no_chars,
            emb_dim,
            padding_idx=0
        )

        self.dropout= torch.nn.Dropout(p=dropout_rate)

        self.conv1 = ConvolutionLayer(emb_dim, conv_filters, 3)
        self.conv2 = ConvolutionLayer(emb_dim, conv_filters, 5, dilation=3)
        self.conv3 = ConvolutionLayer(emb_dim, conv_filters, 9, dilation=2)

        self.linear1 = nn.Linear(conv_filters, config['l1'])
        self.linear_decision = nn.Linear(config["l1"], 1)

        hidden = 7
        self.lstm = nn.LSTM(config['l1'], hidden, num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden*2, self.output_scheme.num_tags)

        self.model_params = model_config

    def forward(self, inputs, threshold=0.5):
        x, seq_lengths = inputs

        embedding = self.embeddings(x).permute(0, 2, 1)

        conv1 = self.conv1(embedding).permute(0, 2, 1)
        conv2 = self.conv2(embedding).permute(0, 2, 1)
        conv3 = self.conv3(embedding).permute(0, 2, 1)

        out = torch.stack((conv1, conv2, conv3), 3)

        out, _ = torch.max(out, 3)
        out = self.dropout(out)

        out = F.relu(self.linear1(out))

        l1_out = self.linear_decision(out)

        sigmoid = torch.sigmoid(l1_out)

        max_len = torch.max(seq_lengths)
        mask = (torch.arange(max_len).expand(seq_lengths.shape[0], max_len) < seq_lengths.unsqueeze(1)).type(torch.float).to("cuda")
        # print(mask.shape)
        # print(sigmoid.shape)
        sigmoid = mask.unsqueeze(2) * sigmoid

        # print(sigmoid.shape)
        decision = (sigmoid > threshold).type(torch.uint8)
        decision[:, 0] = 1
        # print(decision.shape)
        # print(decision)


        syllable_lengths = torch.sum(decision.view(out.shape[0], out.shape[1])
        , dim=1)
            
        max_sy_lengths = torch.max(syllable_lengths)

        # print(syllable_lengths)
        # print("max length", max_sy_lengths)

        syllable_features = torch.zeros(
            sigmoid.shape[0], max_sy_lengths, out.shape[-1]
        )
        # print("sy feats", syllable_features.shape)
        
        for i in range(sigmoid.shape[0]):
            temp =  decision[i, :, 0].nonzero().view(-1)
            # if i == 0:
            #     print(temp)
            syllable_features[i, :syllable_lengths[i]] = out[i, temp, :] * sigmoid[i, temp]
        # feats = out.masked_select(decision)
        # print("feats", feats.shape)

        # print(syllable_features)

        syllable_features = syllable_features.to("cuda")


        lstm_out, _ = self.lstm(syllable_features)

        # print("lstm_out", lstm_out.shape)

        lstm_feats = self.hidden2tag(lstm_out)
        # print("lstm_feats", lstm_feats.shape)

        return lstm_feats, l1_out, decision

        # building rnn features

        # raise SystemExit("EEEE")
        # out = self.linear2(out)

        # return out