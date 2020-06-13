import json

OUTPUT_PATH = "./writing/tables/syllable-table.tex"
TEMPLATE = "{method} & {desc}  &  {ch_f1:2.2f}\% & {wl_f1:2.2f}\% \\\\ "

data = [
    dict(
        method="CRF",
        desc="Chr (W=4), Trigram (W=4)",
        path="./writing/syllable-segmentation-eval-results/crf_trigram_w4_chr_w4.json"
    ),
    dict(
        method="CRF",
        desc="Chr (W=3), Trigram (W=3)",
        path="./writing/syllable-segmentation-eval-results/crf_trigram_w3_chr_w3.json"
    ),
    dict(
        method="CRF",
        desc="Chr (W=3), ChrSpan (W=3)",
        path="./writing/syllable-segmentation-eval-results/crf_chr_w3_span_w3.json"
    ),
    dict(
        method="MaxEnt",
        desc="Chr (W=4), Trigram (W=4)",
        path="./writing/syllable-segmentation-eval-results/maxent_trigram_w4_chr_w4.json"
    ),
    dict(
        method="MaxEnt",
        desc="Chr (W=4)",
        path="./writing/syllable-segmentation-eval-results/maxent_trigram_w4.json"
    ),
]


with open(OUTPUT_PATH, "w") as fh:

    for d in data:
        with open(d["path"], "r") as fi:
            stats = json.load(fi)

            row = TEMPLATE.format(
                method=d["method"],
                desc=d["desc"],
                ch_f1=stats["char_level:f1"]*100,
                wl_f1=stats["word_level:f1"]*100,
            )

            print(row)

            fh.write(f"{row}\n")