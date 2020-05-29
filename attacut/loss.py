
import torch

_cross_ent = torch.nn.CrossEntropyLoss()

def crf(model, logits, labels, seq_lengths):
    mask = create_mask_with_length(seq_lengths)

    loss = model.crf(
        logits,
        labels,
        mask=mask,
        reduction="mean"
    )

    # the module return log-likelihood
    return -loss 

def cross_ent(model, logits, labels, seq_lengths):
    # from b x seq x out -> (bxseq) x out
    logits = logits.reshape(-1, logits.shape[-1])

    # from b x seq -> (bxseq)
    labels = labels.reshape(-1)
    return _cross_ent(logits, labels)

# create mask for given sequence lenghts
def create_mask_with_length(lens):
    # ref: https://stackoverflow.com/a/53403392
    max_len = lens.max()

    mask = torch.arange(max_len)\
      .to(lens.device)\
      .expand(len(lens), max_len) < lens.unsqueeze(1)

    # there is some empty string in the dataset
    mask[:, 0] = 1

    return mask