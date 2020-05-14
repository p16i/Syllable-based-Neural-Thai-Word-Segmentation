"""Usage: hyperopt --config=<config> --output-dir=<output> [--dry-run] --N=<N> [--max-epoch=<max-epoch>]

Options:
  -h --help     Show this screen.
  --version     Show version.
  --max-epoch=<max-epoch>   Maximum number of epoch [default: 20].
"""

from docopt import docopt

import numpy as np
import os

from sklearn.model_selection import ParameterSampler
from scipy.stats import distributions as dist

def merge_arch_params(p):
    arch_params = []

    keys = []
    for k, v in p.items():
        if "arch_" in k:
            arch_params.append("%s:%s" % (k.split("_")[1], str(v)))
            keys.append(k)

    for k in keys:
        del p[k]

    return dict(**p, arch="|".join(arch_params))

if __name__ == '__main__':
    arguments = docopt(__doc__, version='Hyperopt')
    print(arguments)
    # todo: read config

    max_epoch = int(arguments["--max-epoch"])
    # epochs

    param_grid = dict(
        model_name=["seq_sy_ch_conv_3lv"],
        batch_size=[32],
        lr=dist.loguniform(1e-6, 1e-3),
        weight_decay=dist.loguniform(1e-6, 1e-3),
        arch_oc=["BI"],
        arch_embc=[32],
        arch_embt=[32],
        arch_embs=[64],
        arch_conv=dist.randint(64, 512),
        arch_l1=dist.randint(16, 48),
        arch_do=dist.uniform(0, 0.5)
    )

    num_params = int(arguments["--N"])

    param_list = list(
        ParameterSampler(param_grid, n_iter=num_params)
    )

    os.makedirs(arguments["--output-dir"], exist_ok=True)

    cmd_template = """
sbatch jobscript.sh ./scripts/train.py --model-name {model_name} \
    --data-dir ./data/best-big \
    --epoch {max_epoch} \
    --output-dir="{output_dir}" \
    --lr {lr} \
    --batch-size={batch_size} \
    --model-params="{arch}"
    """

    for i, p in enumerate(param_list):
        output_dir = "%s/${SLURM_JOB_ID}" % arguments["--output-dir"]
        p = merge_arch_params(p)
        cmd = cmd_template.format(
            **p,
            max_epoch=max_epoch,
            output_dir=output_dir
        ).strip()

        if arguments["--dry-run"]:
            print(cmd)
        else:
            print("Executte")