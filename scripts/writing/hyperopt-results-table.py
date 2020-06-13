import sys

import yaml
import pandas as pd
from attacut import utils

OUTPUT = "./writing/tables/hyperopt-results-{algo}.tex"

ROW_TEMPLATE = r"""
{algo} & {seq_level} & {ch_feat} & {sy_feat} & {output_tag} & {test_score} & {avg_score} \\
"""

highlight = [
    "ID-CNN-CRF(SY)-SchemeA",
    "ID-CNN(CH+SY)-BI",
    "BiLSTM(CH+SY)-BI",
    "BiLSTM-CRF(SY)-BI",
]

results = {
    "BiLSTM" : [
        "(CH)-BI",
        "(CH+SY)-BI",
        "(SY)-BI",
        "(SY)-SchemeA",
        "(SY)-SchemeB",
    ],
    "BiLSTM-CRF": [
        "(SY)-BI",
        "(SY)-SchemeA",
        "(SY)-SchemeB",
    ],
    "ID-CNN": [
        "(CH)-BI",
        "(CH+SY)-BI",
        "(SY)-BI",
        "(SY)-SchemeA",
        "(SY)-SchemeB",
    ],
    "ID-CNN-CRF": [
        "(SY)-BI",
        "(SY)-SchemeA",
        "(SY)-SchemeB",
    ]
}

def seq_level(key):
    if "CH" in key:
        return "Character"
    elif "(SY)" in key:
        return "Syllable"
    else:
        raise ValueError(f"Can't determine sequence level for {key}")

def marker(key, cond):
    if cond(key):
        return r"\cmark"
    else:
        return r"\xmark"

if __name__ == "__main__":
    algo = sys.argv[1]
    print(f"Algorithm: {algo}")
    output = OUTPUT.format(algo=algo)

    print(f"saving table to: {output}")
    with open("./hyperopt-results.yml", "r") as fh, open(output, "w") as fw:
        data = yaml.safe_load(fh)
        data = dict(zip(map(lambda x: x["name"], data), data))

        for i, variant in enumerate(results[algo]):
            key = f"{algo}{variant}"

            print(data[key])

            df_variant = pd.read_csv(data[key]["path"])

            best_model = df_variant[
                df_variant["best-val:word_level:f1"] == df_variant["best-val:word_level:f1"].max()
            ].to_dict("row")[0]

            first_col = algo if i == 0 else ""


            score = best_model["best-val:word_level:f1"]

            print("Best Score (Test): ", best_model["best-test:word_level:f1"])
            print("Best Score (Val): ", best_model["best-val:word_level:f1"])

            score_txt = f"{score*100:.2f}\%"

            if key in highlight:
                score_txt = r"\textbf{%s}" % score_txt

            test_metric = df_variant["best-val:word_level:f1"] * 100

            row = ROW_TEMPLATE.format(
                algo=first_col,
                seq_level=seq_level(key),
                ch_feat=marker(key, lambda x: "(CH" in x),
                sy_feat=marker(key, lambda x: "SY)" in x),
                output_tag=key.split("-")[-1],
                test_score=score_txt,
                avg_score="%2.2f $\\pm$ % 2.2f" % (test_metric.mean(), test_metric.std())
            )

            fw.write(f"{row}\n")