#!/usr/bin/env python

"""eval.py

Usage:
  eval.py --model=<model>  --data=<dataset> [--num-cores=<num-cores, --batch-size=<batch-size>, --gpu]

Options:
  -h --help         Show this screen.
  --num-cores=<num-cores>  Use multiple-core processing [default: 4]
  --batch-size=<batch-size>  Batch size [default: 32]
"""

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

    model_path = arguments["--model"]

    src_file = arguments["--data"] + "/input.txt"

    slug = arguments["--data"].split("/")[-1]
    dest="%s/%s.txt" % (arguments["--model"], slug)

    time_took = command.main(
        src_file,
        model_path,
        int(arguments["--num-cores"]),
        int(arguments["--batch-size"]),
        device="cuda" if arguments["--gpu"] else "cpu",
        dest=dest
    )

    # read tokenised back
    actual = _read_file(dest)
    label_file = arguments["--data"] + "/label.txt"
    expected = _read_file(label_file)

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

    statistics["time_took"] = time_took
    statistics["model_path"] = model_path

    stat_file = f"{model_path}/{slug}.json"

    with open(stat_file, "w") as fh:
        json.dump(statistics, fh)