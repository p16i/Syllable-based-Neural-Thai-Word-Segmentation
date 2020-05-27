#!/usr/bin/env python

"""benchmark.py

Usage:
  benchmark.py --input=<input>  --label=<label>

Options:
  -h --help         Show this screen.
"""

import pandas as pd
from docopt import docopt
from attacut import command, __version__, benchmark
import json

columns = [
    "char_level:tp",
    "char_level:fp",
    "char_level:tn",
    "char_level:fn",
    "word_level:correctly_tokenised_words",
    "word_level:total_words_in_sample",
    "word_level:total_words_in_ref_sample",
]

def _read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = map(lambda r: r.strip(), f.readlines())
    return list(lines)

def _compute_f1(precision, recall):
    return 2*precision*recall / (precision + recall)

if __name__ == "__main__":
    arguments = docopt(__doc__, version=f"AttaCut: version {__version__}")

    actual = _read_file(arguments["--input"])
    expected = _read_file(arguments["--label"])

    df_raw = benchmark.benchmark(expected, actual)

    statistics = dict()

    for c in columns:
        statistics[c] = float(df_raw[c].sum())

    statistics["char_level:precision"] = statistics["char_level:tp"] / (
        statistics["char_level:tp"] + statistics["char_level:fp"]
    )

    statistics["char_level:recall"] = statistics["char_level:tp"] / (
        statistics["char_level:tp"] + statistics["char_level:fn"]
    )

    statistics["char_level:f1"] = _compute_f1(
        statistics["char_level:precision"],
        statistics["char_level:recall"]
    )

    statistics["word_level:precision"] = statistics["word_level:correctly_tokenised_words"] \
        / statistics["word_level:total_words_in_sample"]

    statistics["word_level:recall"] = statistics["word_level:correctly_tokenised_words"] \
        / statistics["word_level:total_words_in_ref_sample"]

    statistics["word_level:f1"] = _compute_f1(
        statistics["word_level:precision"],
        statistics["word_level:recall"]
    )

    for k in sorted(statistics.keys()):
        print(f"{k}: {statistics[k]}")