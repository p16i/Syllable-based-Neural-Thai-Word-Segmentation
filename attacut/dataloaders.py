import numpy as np
import torch
from torch.utils.data import Dataset

from attacut import logger, preprocessing, utils, char_type

log = logger.get_logger(__name__)


class SequenceDataset(Dataset):
    def __init__(self, dir: str = None, dict_dir: str = None, path: str = None, output_scheme = None):
        if path:
            self.load_preprocessed_data(path, output_scheme)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, index):
        return self.data[index]

    def load_preprocessed_data(self, path, output_scheme):
        self.data = []

        suffix = path.split("/")[-1]
        with open(path) as f, \
            utils.Timer("load-seq-data--%s" % suffix) as timer:
            for line in f:
                # we have syllables and bi tag for word boundary for each syllable here
                syllables, w_bi_labels = line.strip().split(":--:")
                w_bi_labels = np.array(list(w_bi_labels)).astype(int)
                syllables = syllables.split("~")
                self.data.append(self._process_training_line(syllables, w_bi_labels, output_scheme))

        self.total_samples = len(self.data)

    def make_feature(self, txt: str):
        raise NotImplementedError

    def setup_featurizer(self, path: str):
        raise NotImplementedError

    @staticmethod
    def prepare_model_inputs(inputs, device="cpu"):

        x, seq_lengths = inputs[0]
        x = x.to(device)

        y = inputs[1].long().to(device)
        seq_lengths = seq_lengths.to(device)

        return (x, seq_lengths), y, np.prod(y.shape)

    @staticmethod
    def _process_line(line: str, output_scheme):
        # only use when training
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch):
        # only use when training
        raise NotImplementedError

    @classmethod
    def load_preprocessed_file_with_suffix(cls, dir: str, suffix: str, output_scheme) -> "SequenceDataset":
        path = "%s/%s" % (dir, suffix)
        log.info("Loading preprocessed data from %s" % path)
        return cls(dir=dir, dict_dir=f"{dir}/dictionary", path=path, output_scheme=output_scheme)


class CharacterSeqDataset(SequenceDataset):
    def __init__(self, dir:str = None, dict_dir: str = None, path: str = None, output_scheme = None):

        self.dict = utils.load_dict(f"{dict_dir}/characters.json")
        self.ch_ix_2_ch = dict(zip(self.dict.values(), self.dict.keys()))

        super(CharacterSeqDataset, self).__init__(dir, dict_dir, path, output_scheme)

    def setup_featurizer(self):
        return dict(num_tokens=len(self.dict))


    def make_feature(self, txt):
        characters = list(txt)
        ch_ix = list(
            map(
                lambda c: preprocessing.character2ix(self.dict, c),
                characters
            )
        )

        ch_type_ix = char_type.get_char_type_ix(characters)

        features = np.stack((ch_ix, ch_type_ix), axis=0) \
            .reshape((1, 2, -1)) \
            .astype(np.int64)

        seq_lengths = np.array([features.shape[-1]], dtype=np.int64)

        return characters, (torch.from_numpy(features), torch.from_numpy(seq_lengths))

    # @staticmethod
    def _process_training_line(self, syllables, w_bi_labels, output_scheme):
        assert len(syllables) == len(w_bi_labels)

        characters, syl4chr, labels = [], [], []

        # we get syllable and its label here
        for syllable, label in zip(syllables, w_bi_labels):
            _len = len(syllable)
            if _len == 0:
                _len, _chr = 1, [""]
                _label = [label]
            else:
                _chr = list(syllable)
                _label = [label] + [0] * (_len - 1)

            assert len(_chr) == len(_label) == _len, "%d vs %d vs %d" % (len(_chr), len(_label), _len)

            characters.extend(_chr)
            labels.extend([label] + [0] * (_len - 1))
            syl4chr.extend([syllable]*_len)

        y = np.array(list(labels)).astype(int)

        ch_ix = np.array(
            list(map(lambda ch: preprocessing.character2ix(self.dict, ch), characters))
        ).astype(int)
        ct_ix = np.array(char_type.get_char_type_ix(characters)).astype(int)

        assert len(ch_ix) == len(ct_ix) == len(y)

        x = np.stack((ch_ix, ct_ix), axis=0)

        y = output_scheme.encode(y, syl4chr)

        return (x, len(y)), y

    @staticmethod
    def collate_fn(batch):
        total_samples = len(batch)

        seq_lengths = np.array(list(map(lambda x: x[0][1], batch)))
        max_length = np.max(seq_lengths)

        features = np.zeros((total_samples, 2, max_length), dtype=np.int64)
        labels = np.zeros((total_samples, max_length), dtype=np.int64)

        for i, s in enumerate(batch):
            b_feature = s[0][0]
            total_features = b_feature.shape[1]
            features[i, :, :total_features] = b_feature
            labels[i, :total_features] = s[1]

        seq_lengths = torch.from_numpy(seq_lengths)
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)

        inputs = (torch.from_numpy(features)[perm_idx], seq_lengths)

        labels = torch.from_numpy(labels)[perm_idx]

        return inputs, labels, perm_idx


class SyllableCharacterSeqDataset(SequenceDataset):
    def __init__(self, dir:str = None, dict_dir: str = None, path: str = None, output_scheme = None):

        self.ch_dict = utils.load_dict(f"{dict_dir}/characters.json")
        self.sy_dict = utils.load_dict(f"{dict_dir}/syllables.json")
        self.dict_dir = dict_dir

        print(f"we have {len(self.sy_dict)} syllables from {dict_dir}")

        self.ch_ix_2_ch = dict(zip(self.ch_dict.values(), self.ch_dict.keys()))

        super(SyllableCharacterSeqDataset, self).__init__(dir, dict_dir, path, output_scheme)

    def setup_featurizer(self):
        return dict(
            num_char_tokens=len(self.ch_dict),
            num_tokens=len(self.sy_dict),
            dict_dir=self.dict_dir
        )

    def make_feature(self, txt):
        syllables = preprocessing.syllable_tokenize(txt)

        sy2ix, ch2ix = self.sy_dict, self.ch_dict

        ch_ix, ch_type_ix, syllable_ix = [], [], []

        for syllable in syllables:
            six = preprocessing.syllable2ix(sy2ix, syllable)

            characters = list(syllable)
            chs = list(
                map(
                    lambda ch: preprocessing.character2ix(ch2ix, ch),
                    characters,
                )
            )
            ch_ix.extend(chs)
            ch_type_ix.extend(char_type.get_char_type_ix(characters))
            syllable_ix.extend([six]*len(chs))

        features = np.stack((ch_ix, ch_type_ix, syllable_ix), axis=0) \
            .reshape((1, 3, -1)) \
            .astype(np.int64)

        seq_lengths = np.array([features.shape[-1]], dtype=np.int64)

        return list(txt), (torch.from_numpy(features), torch.from_numpy(seq_lengths))

    # @staticmethod
    def _process_training_line(self, syllables, w_bi_labels, output_scheme):
        assert len(syllables) == len(w_bi_labels)

        characters, syllable_indices, labels = [], [], []

        # we get syllable and its label here
        for syllable, label in zip(syllables, w_bi_labels):
            _len = len(syllable)
            if _len == 0:
                _len, _chr = 1, [""]
                _label = [label]
            else:
                _chr = list(syllable)
                _label = [label] + [0] * (_len - 1)

            assert len(_chr) == len(_label) == _len, "%d vs %d vs %d" % (len(_chr), len(_label), _len)

            characters.extend(_chr)
            labels.extend(_label)

            sy_ix = preprocessing.syllable2ix(self.sy_dict, syllable)
            syllable_indices.extend([sy_ix]*_len)

        y = np.array(list(labels)).astype(int)

        ch_ix = np.array(
            list(map(lambda ch: preprocessing.character2ix(self.ch_dict, ch), characters))
        ).astype(int)
        ct_ix = np.array(char_type.get_char_type_ix(characters)).astype(int)

        x = np.stack((ch_ix, ct_ix, syllable_indices), axis=0)

        y = output_scheme.encode(y, syllable_indices)

        assert len(y) == len(ch_ix)
        assert len(y) == len(ct_ix)
        assert len(y) == len(syllable_indices)

        return (x, len(y)), y

    @staticmethod
    def collate_fn(batch):
        total_samples = len(batch)

        seq_lengths = np.array(list(map(lambda x: x[0][1], batch)))
        max_length = np.max(seq_lengths)

        features = np.zeros((total_samples, 3, max_length), dtype=np.int64)
        labels = np.zeros((total_samples, max_length), dtype=np.int64)

        for i, s in enumerate(batch):
            b_feature = s[0][0]

            # make sure that we have seq dimension, e.g. when length is one
            if len(b_feature.shape) == 1:
                b_feature = b_feature.unsqueeze(1)

            total_features = b_feature.shape[1]
            features[i, :, :total_features] = b_feature
            labels[i, :total_features] = s[1]

        seq_lengths = torch.from_numpy(seq_lengths)
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)

        inputs = (torch.from_numpy(features)[perm_idx], seq_lengths)

        labels = torch.from_numpy(labels)[perm_idx]

        return inputs, labels, perm_idx

class SyllableSeqDataset(SequenceDataset):
    def __init__(self, dir:str = None, dict_dir: str = None, path: str = None, output_scheme = None):

        self.sy_dict = utils.load_dict(f"{dict_dir}/syllables.json")
        print(f"we have {len(self.sy_dict)} syllables")

        super(SyllableSeqDataset, self).__init__(dir, dict_dir, path, output_scheme)

    def setup_featurizer(self):
        return dict(
            num_tokens=len(self.sy_dict)
        )

    def make_feature(self, txt):
        syllables = preprocessing.syllable_tokenize(txt)

        sy2ix = self.sy_dict

        syllable_ix = []

        for syllable in syllables:
            six = preprocessing.syllable2ix(sy2ix, syllable)
            syllable_ix.append(six)

        # dims: (len,)
        features = np.array(syllable_ix)\
            .astype(np.int64)\
            .reshape(-1)

        seq_lengths = np.array([features.shape[-1]], dtype=np.int64)
        
        features = torch.from_numpy(features)

        return syllables, (features, torch.from_numpy(seq_lengths))

    def _process_training_line(self, syllables, w_bi_labels, output_scheme):
        sy_ix = list(map(lambda s: preprocessing.syllable2ix(self.sy_dict, s), syllables))
        x = np.array(sy_ix)

        y = w_bi_labels

        assert len(sy_ix) == len(y)

        return (x, len(y)), y

    @staticmethod
    def collate_fn(batch):
        total_samples = len(batch)

        seq_lengths = np.array(list(map(lambda x: x[0][1], batch)))
        max_length = np.max(seq_lengths)

        features = np.zeros((total_samples, max_length), dtype=np.int64)
        labels = np.zeros((total_samples, max_length), dtype=np.int64)

        for i, s in enumerate(batch):
            # dims: (len, )
            b_feature = s[0][0].reshape(-1)
            total_features = b_feature.shape[0]
            features[i, :total_features] = b_feature
            labels[i, :total_features] = s[1]

        seq_lengths = torch.from_numpy(seq_lengths)
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)

        inputs = (torch.from_numpy(features)[perm_idx], seq_lengths)

        labels = torch.from_numpy(labels)[perm_idx]

        return inputs, labels, perm_idx
