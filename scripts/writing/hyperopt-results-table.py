import sys

import yaml
import pandas as pd
from attacut import utils

OUTPUT = "./writing/tables/hyperopt-results-{algo}.tex"
BEST_SCORE = 0.9563

ROW_TEMPLATE = r"""
{algo} & {seq_level} & {ch_feat} & {sy_feat} & {output_tag} & {crf} & {test_score} \\
"""

results = {
    "BiLSTM" : [
        # "(CH)-BI",
        "(SY)-BI",
        "(SY)-SchemeA",
        "(SY)-SchemeB",
        "-CRF(SY)-BI",
        "-CRF(SY)-SchemeA",
        "-CRF(SY)-SchemeB",
        "(CH-SY)-BI",
    ],
    "ID-CNN": [
        # "(CH)-BI",
        "(SY)-BI",
        "(SY)-SchemeA",
        "(SY)-SchemeB",
        "-CRF(SY)-BI",
        "-CRF(SY)-SchemeA",
        "-CRF(SY)-SchemeB",
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


            score = best_model["best-test:word_level:f1"]

            score_txt = f"{score*100:.2f}\%"

            if score >= BEST_SCORE:
                score_txt = r"\textbf{%s}" % score_txt

            row = ROW_TEMPLATE.format(
                algo=first_col,
                seq_level=seq_level(key),
                ch_feat=marker(key, lambda x: "(CH" in x),
                sy_feat=marker(key, lambda x: "SY)" in x),
                output_tag=key.split("-")[-1],
                crf=marker(key, lambda x: "CRF" in x),
                test_score=score_txt
            )

            fw.write(f"{row}\n")