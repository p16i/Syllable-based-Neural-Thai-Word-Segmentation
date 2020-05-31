
import torch

_cross_ent = torch.nn.CrossEntropyLoss(reduction="none")

def crf(model, logits, labels, seq_lengths):
    mask = create_mask_with_length(seq_lengths)

    loss = model.crf(
        logits,
        labels,
        mask=mask,
        reduction="mean" # mean of batch
    )

    # the module return log-likelihood
    return -loss 

def cross_ent(model, logits, labels, seq_lengths):
    # from b x seq x out -> (b x seq) x out
    logits_long = logits.reshape(-1, logits.shape[-1])

    # from b x seq -> (bxseq)
    labels_long = labels.reshape(-1)

    loss = _cross_ent(logits_long, labels_long) \
        .reshape(logits.shape[0], logits.shape[1]) # dims: b x seq

    # dims: b x seq
    mask = create_mask_with_length(seq_lengths)

    loss_masked = loss * mask

    # dims: b
    loss_per_seq = loss_masked.sum(dim=1)

    return loss_per_seq.mean()

# create mask for given sequence lenghts
def create_mask_with_length(lens):
    # ref: https://stackoverflow.com/a/53403392
    max_len = lens.max()

    mask = torch.arange(max_len) \
      .to(lens.device) \
      .expand(len(lens), max_len) < lens.unsqueeze(1)

    # there is some empty string in the dataset
    mask[:, 0] = 1

    return mask