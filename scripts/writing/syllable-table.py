import json

OUTPUT_PATH = "./writing/tables/syllable-table.tex"
TEMPLATE = "{method} & {desc}  &  {tnc_ch_f1:2.2f}\% & {tnc_wl_f1:2.2f}\% & {best_val_ch_f1:2.2f}\% & {best_val_wl_f1:2.2f}\% \\\\ "

highlight = "CRF:Chr (W=4), Trigram (W=4)"

data = [
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
        method="CRF",
        desc="Chr (W=4), Trigram (W=4)",
        path="./writing/syllable-segmentation-eval-results/crf_trigram_w4_chr_w4.json"
    ),
    dict(
        method="MaxEnt",
        desc="Chr (W=4)",
        path="./writing/syllable-segmentation-eval-results/maxent_trigram_w4.json"
    ),
    dict(
        method="MaxEnt",
        desc="Chr (W=4), Trigram (W=4)",
        path="./writing/syllable-segmentation-eval-results/maxent_trigram_w4_chr_w4.json"
    ),
]


with open(OUTPUT_PATH, "w") as fh:

    for d in data:
        with open(d["path"], "r") as ftnc, open(d["path"].replace(".json", "_val.json"), "r") as fbv: 
            stats_tnc = json.load(ftnc)
            stats_best_val = json.load(fbv)

            method = d['method']
            if ("%s:%s" % (d['method'], d['desc'])) == highlight:
                method = f"{method}$^\star$"

            row = TEMPLATE.format(
                method=method,
                desc=d["desc"],
                tnc_ch_f1=stats_tnc["char_level:f1"]*100,
                tnc_wl_f1=stats_tnc["word_level:f1"]*100,
                best_val_ch_f1=stats_best_val["char_level:f1"]*100,
                best_val_wl_f1=stats_best_val["word_level:f1"]*100,
            )

            print(row)

            fh.write(f"{row}\n")