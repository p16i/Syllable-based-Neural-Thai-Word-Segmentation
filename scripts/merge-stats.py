#!/usr/bin/env python

"""eval.py

Usage:
  merge-stats.py --model-group=<model-group>

"""

from docopt import docopt
import json


from glob import glob
import pandas as pd
import yaml

def load_eval_stat(model_path, file):
    with open(f"{model_path}/{file}.json") as fh:
        st = json.load(fh)

        st = dict(map(lambda x: (f"{file}:{x[0]}", x[1]), st.items()))

        return st

if __name__ == "__main__":
    arguments = docopt(__doc__)

    model_group = arguments["--model-group"]

    data = []

    for path in glob(model_group):
        # read params
        with open(f"{path}/params.yml") as fh:
            dd = yaml.full_load(fh)

        test_stat = load_eval_stat(path, "best-test")
        val_stat = load_eval_stat(path, "best-val")

        data.append(dict(
            **dd,
            **val_stat,
            **test_stat
        )

    df = pd.DataFrame(data)
    dest = "/".join(model_group.split("/")[:-1]) + "/stats.csv"
    print(f"saving file to {dest}")
    df.to_csv(dest, index=False)

