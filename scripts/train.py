#!/usr/bin/env python

import glob
import os
import shutil
import sys
import time

sys.path.insert(0, os.getcwd())

import numpy as np

import fire
import torch
import torch.optim as optim
from torch.utils import data

from attacut import dataloaders as dl, output_tags
from attacut import evaluation, models, utils

def _create_metrics(metrics=["true_pos", "false_pos", "false_neg"]):
    return dict(zip(metrics, [0]*len(metrics)))


def accumuate_metrics(m1, m2):
    for k, v in m1.items():
        m1[k] = v + m2[k]
    return m1


def evaluate_model(preds, labels):
    metrics = evaluation.compute_metrics(labels, preds)

    return {
        "true_pos": metrics.tp,
        "false_pos": metrics.fp,
        "false_neg": metrics.fn
    }


def precision_recall(true_pos, false_pos, false_neg):
    # todo: refactor to use torch metric
    dominator = true_pos + false_pos
    precision = true_pos/dominator if dominator > 0 else 0

    dominator = true_pos + false_neg
    recall = true_pos/dominator if dominator > 0 else 0

    dominator = precision + recall 
    f1 = 2*precision*recall / dominator if dominator > 0 else 0

    return precision, recall, f1


def print_floydhub_metrics(metrics, step=0, prefix=""):
    if 'FLOYDHUB' in os.environ and os.environ['FLOYDHUB']:
        for k, v in metrics.items():
            print('{"metric": "%s:%s", "value": %f, "step": %d}' % (k, prefix, v, step))


def copy_files(path, dest):
    utils.maybe_create_dir(dest)

    for f in glob.glob(path):
        filename = f.split("/")[-1]
        shutil.copy(f, "%s/%s" % (dest, filename), follow_symlinks=True)


def do_iterate(model, generator, device,
    optimizer=None, criterion=None, prefix="", step=0):

    total_loss, total_preds = 0, 0
    metrics = _create_metrics()

    for _, batch in enumerate(generator):
        (x, seq), labels, perm_ix = batch

        xd, yd, total_batch_preds = generator.dataset.prepare_model_inputs(
            ((x, seq), labels), device
        )

        if optimizer:
            model.zero_grad()


        logits = model(xd)
        logits = logits.reshape(-1, logits.shape[-1])

        loss = criterion(
            logits,
            yd.reshape(-1)
        )

        if optimizer:
            loss.backward()
            optimizer.step()

        total_preds += total_batch_preds
        total_loss += loss.item() * total_batch_preds

        preds  = model.output_scheme.decode_condition(
            np.argmax(logits.cpu().detach().numpy(), axis=1).reshape(-1)
        )
        yd = model.output_scheme.decode_condition(yd.cpu().detach().numpy())

        accumuate_metrics(metrics, evaluate_model(preds, yd))

    avg_loss = total_loss / total_preds if total_preds > 0 else 0

    pc_values = precision_recall(**metrics)
    print("[%s] loss %f | precision %f | recall %f | f1 %f" % (
        prefix,
        avg_loss,
        *pc_values
    ))

    print_floydhub_metrics(
        dict(
            loss=avg_loss,
            precision=pc_values[0],
            recall=pc_values[1],
            f1=pc_values[2]
        ),
        step=step, prefix=prefix
    )


# taken from https://stackoverflow.com/questions/52660985/pytorch-how-to-get-learning-rate-during-training
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def main(
        model_name, data_dir, 
        epoch=10,
        lr=0.001,
        batch_size=64,
        weight_decay=0.0,
        checkpoint=5,
        model_params="",
        output_dir="",
        no_workers=4,
        lr_schedule="",
        prev_model="",
    ):

    model_cls = models.get_model(model_name)

    output_scheme = output_tags.get_scheme(
        utils.parse_model_params(model_params)["oc"]
    )

    dataset_cls = model_cls.dataset

    training_set: dl.SequenceDataset = dataset_cls.load_preprocessed_file_with_suffix(
        data_dir,
        "training.txt",
        output_scheme
    )

    validation_set: dl.SequenceDataset = dataset_cls.load_preprocessed_file_with_suffix(
        data_dir,
        "val.txt",
        output_scheme
    )

    # only required
    data_config = training_set.setup_featurizer()

    device = models.get_device()
    print("Using device: %s" % device)

    params = {}

    if model_params:
        params['model_config'] = model_params
        print(">> model configuration: %s" % model_params)

    if prev_model:
        print("Initiate model from %s" % prev_model)
        model = models.get_model(model_name).load(
            prev_model,
            data_config,
            **params
        )
    else:
        model = models.get_model(model_name)(
            data_config,
            **params
        )

    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if prev_model:
        print("Loading prev optmizer's state")
        optimizer.load_state_dict(torch.load("%s/optimizer.pth" % prev_model))
        print("Previous learning rate", get_lr(optimizer))

        # force torch to use the given lr, not previous one
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            param_group['initial_lr'] = lr

        print("Current learning rate", get_lr(optimizer))

    if lr_schedule:
        schedule_params = utils.parse_model_params(lr_schedule)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=schedule_params['step'],
            gamma=schedule_params['gamma'],
        )

    dataloader_params = dict(
        batch_size=batch_size,
        num_workers=no_workers,
        collate_fn=dataset_cls.collate_fn
    )

    print("Using dataset: %s" % type(dataset_cls).__name__)

    training_generator = data.DataLoader(
        training_set,
        shuffle=True,
        **dataloader_params
    )
    validation_generator = data.DataLoader(
        validation_set,
        shuffle=False,
        **dataloader_params
    )

    total_train_size = len(training_set) 
    total_test_size = len(validation_set)

    print("We have %d train samples and %d test samples" %
        (total_train_size, total_test_size)
    )

    # for FloydHub
    print(
        '{"metric": "%s:%s", "value": %s}' %
        ("model", model_name, model.total_trainable_params())
    )

    utils.maybe_create_dir(output_dir)

    copy_files(
        "%s/dictionary/*.json" % data_dir,
        output_dir
    )

    utils.save_training_params(
        output_dir,
        utils.ModelParams(
            name=model_name,
            params=model.model_params
        )
    )

    for e in range(1, epoch+1):
        print("===EPOCH %d ===" % (e))
        st_time = time.time()
        if lr_schedule:
            curr_lr = get_lr(optimizer)
            print_floydhub_metrics(dict(lr=curr_lr), step=e, prefix="global")
            print("lr: ", curr_lr)

        with utils.Timer("epoch-training") as timer:
            do_iterate(model, training_generator,
                prefix="training",
                step=e,
                device=device,
                optimizer=optimizer,
                criterion=criterion,
            )

        with utils.Timer("epoch-validation") as timer, \
            torch.no_grad():
            do_iterate(model, validation_generator,
                prefix="validation",
                step=e,
                device=device,
                criterion=criterion,
            )

        elapsed_time = (time.time() - st_time) / 60.
        print(f"Time took: {elapsed_time:.4f} mins")

        if lr_schedule:
            scheduler.step()

        if checkpoint and e % checkpoint == 0:
            model_path = "%s/model-e-%d.pth" % (output_dir, e)
            print("Saving model to %s" % model_path)
            torch.save(model.state_dict(), model_path)


    model_path = "%s/model.pth" % output_dir
    opt_path = "%s/optimizer.pth" % output_dir

    print("Saving model to %s" % model_path)
    torch.save(model.state_dict(), model_path)
    torch.save(optimizer.state_dict(), opt_path)

    config = utils.parse_model_params(model_params)

    if type(config["embs"]) == str:
        emb = config["embs"]
        copy_files(
            f"{data_dir}/dictionary/sy-emb-{emb}.npy",
            output_dir
        )


if __name__ == "__main__":
    fire.Fire(main)
