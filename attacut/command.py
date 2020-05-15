import sys
import os
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


from attacut import Tokenizer, __version__, preprocessing, utils, models

# from https://github.com/pytorch/pytorch/issues/1494#issuecomment-305993854
from multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

SEP = "|"


# For models tranied with weight-decay, some weights are close to zero causing denormal
# ref: https://en.wikipedia.org/wiki/Denormal_number
# This block is aken from https://github.com/pytorch/pytorch/issues/19651#issuecomment-486170718
if not torch.set_flush_denormal(True):
    print("Unable to set flush denormal")
    print("Pytorch compiled without advanced CPU")
    print("at: https://github.com/pytorch/pytorch/blob/84b275b70f73d5fd311f62614bccc405f3d5bfa3/aten/src/ATen/cpu/FlushDenormal.cpp#L13")


def get_argument(dict, name, default):
    v = dict.get(name)
    return v if v is not None else default

class AttaCutCLIDataset(Dataset):
      def __init__(self, src, tokenizer, device):
          self.src = src
          self.total_lines = utils.wc_l(self.src)

          self.tokenizer = tokenizer

          self.device = device

          self.data = []
          with open(self.src, "r") as fin:
              for txt in fin:
                txt = preprocessing.TRAILING_SPACE_RX.sub("", txt)

                tokens, features = self.tokenizer.dataset.make_feature(txt)

                inputs = (
                    features,
                    torch.zeros(features[1])  # dummy label when won't need it here
                )

                (x, seq), labels, _= self.tokenizer.dataset.prepare_model_inputs(
                  inputs,
                )

                x = torch.squeeze(x)

                self.data.append((((x, seq), labels), tokens))

      def __len__(self):
          return self.total_lines

      def __getitem__(self, idx):
          return self.data[idx]

def collate_fn(tokenizer, batch, device):
    inputs, tokens = [], []

    for x, t in batch:
      inputs.append(x)
      tokens.append(t)

    (x, seq), labels, perm_idx = tokenizer.dataset.collate_fn(inputs)

    return ((x.to(device), seq.to(device)), labels, perm_idx), tokens

def main(src, model, num_cores=4, batch_size=32, dest=None, device="cpu"):

    assert num_cores >= 0, "Input given to <num-thread> should greather than or equal one"

    if not src:
      print(__doc__)
      sys.exit(0)

    # for a custom model, use the last dir's name.
    model_name = model.split("/")[-1]

    dest = dest if dest is not None else utils.add_suffix_to_file_path(src, f"tokenized-by-{model_name}")

    print(f"Tokenizing {src}")
    print(f"Using {src}")
    print(f"Output: {dest}")

    tokenizer = Tokenizer(model)

    total_lines = utils.wc_l(src)

    if num_cores == 0:
        print(f"Use main process processing for {total_lines} lines")
    else:
        print(f"Use {num_cores} cores for processing for {total_lines} lines")

    print(f"device={device}")

    start_time = time.time()
    ds = AttaCutCLIDataset(src, tokenizer, device)
    dataloader = DataLoader(
      ds,
      batch_size=batch_size,
      shuffle=False,
      num_workers=0, # only use main process
      collate_fn=lambda batch: collate_fn(tokenizer, batch, device)
    )

    tokenizer.model.to(device)

    tokenizer.model.eval()
    results = []

    with torch.no_grad(), \
        tqdm(total=total_lines) as tq, \
        open(dest, "w") as fout:

            for batch in dataloader:
                (x, labels, perm_idx), tokens = batch
                probs = tokenizer.model(x).cpu().detach().numpy()

                preds = tokenizer.model.output_scheme.decode_condition(
                  np.argmax(probs, axis=2)
                )

                max_seq = x[1].max().cpu().detach().numpy()

                perm_idx = perm_idx.cpu().detach()
                preds = preds.reshape((-1, max_seq))

                for ori_ix, after_sorting_ix in enumerate(np.argsort(perm_idx)):
                    pred = preds[after_sorting_ix, :]
                    token = tokens[ori_ix]

                    words = preprocessing.find_words_from_preds(token, pred)
                    fout.write("%s\n" % SEP.join(words))

                tq.update(n=preds.shape[0])

    time_took = time.time() - start_time

    return time_took