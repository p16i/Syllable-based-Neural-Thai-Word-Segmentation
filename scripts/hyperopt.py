"""Usage: hyperopt --config=<config> [--dry-run] --N=<N> [--max-epoch=<max-epoch>]

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
import yaml
import time

from datetime import datetime

DATASET = "./data/best-syllable-big"


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

    config = arguments["--config"]
    param_grid = dict()

    with open(config, "r") as fh:
        for k, v in yaml.full_load(fh).items():
            if type(v) == str:
                param_grid[k] = eval("dist." + v)
            else:
                param_grid[k] = v

    _, config_name = os.path.split(config)

    dt = datetime.today().strftime("%Y-%m-%d--%H-%M")

    config_name = f"{config_name}-{dt}"
    print(config_name)

    max_epoch = int(arguments["--max-epoch"])

    n_iters = int(arguments["--N"])

    param_list = list(
        ParameterSampler(param_grid, n_iter=n_iters)
    )

    cmd_template = """
sbatch --job-name {job_name} --output "./logs/{job_name}.out" jobscript.sh ./scripts/train.py --model-name {model_name} \
    --data-dir {dataset} \
    --epoch {max_epoch} \
    --output-dir="{output_dir}" \
    --lr {lr} \
    --batch-size={batch_size} \
    --model-params="{arch}" \
    --weight-decay={weight_decay}
    """

    print("------------------------")

    for i, p in enumerate(param_list):
        job_name = f"{config_name}.{n_iters}.{i}.log"
        output_dir = f"./artifacts/{config_name}.{n_iters}/run-{i}"
        p = merge_arch_params(p)
        cmd = cmd_template.format(
            **p,
            max_epoch=max_epoch,
            output_dir=output_dir,
            job_name=job_name,
            dataset=DATASET
        ).strip()

        if arguments["--dry-run"]:
            print(cmd)
        else:
            os.system(cmd)
            if i+1 % 10 == 0:
                time.sleep(5)