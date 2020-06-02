# This generates model hyperopt tables for Appendix

import yaml
import pandas as pd
from attacut import utils

OUTPUT = "./writing/tables/hyperopt-model-tables.tex"

table = r"""
\begin{table*}

\centering
\begin{tabular}{cc}
\toprule
\textbf{Best validation word-level F1}  & %(best_val_f1).4f \\
\textbf{Best model's number of trainable parameters}  & %(best_num_params)s \\
\textbf{Average training duration}  & %(avg_training).0f minutes \\
\bottomrule	
\end{tabular}

\begin{tabular}{rccc}
\toprule
\textbf{Hyperparameter} & \textbf{Search Space} & \textbf{Best Assignment} \\
learning rate &  \textit{loguniform(1e-6, 1e-3)}&  %(lr).2e \\
weight decay & \textit{loguniform(1e-6, 1e-3)} &  %(weight_decay).2e \\
%(family_params)s
\bottomrule
\end{tabular}

\caption{Best hyperparameter and search space for %(name)s.}
\label{tab:appendix-hyperopt-%(ref)s}
\end{table*}
"""

family_specific_param = {
    "ID-CNN": r"""
convolution filters & \textit{uniform-interger(64, 256)} & %(conv)d \\
linear layer &  \textit{uniform-interger(16, 48)} & %(l1)d \\
dropout & \textit{uniform(0, 0.5)} & %(do).4f \\
""",
    "ID-CNN-XL": r"""(need change)\\""",
    "BiLSTM": r"""
LSTM cells & \textit{uniform-interger(64, 256)} & %(cells)d \\
linear layer &  \textit{uniform-interger(16, 48)} & %(l1)d \\
""",
    "BiLSTM-XL": r"""(need change)\\"""
}

if __name__ == "__main__":
    with open("./hyperopt-results.yml", "r") as fh, open(OUTPUT, "w") as fw:
        data = yaml.safe_load(fh)

        for i, row in enumerate(data):
            path = row["path"]
            df = pd.read_csv(path)

            print(f"loading {path}")

            max_val_f1 = df["best-val:word_level:f1"].max()
            best_model = df[df["best-val:word_level:f1"] == max_val_f1].to_dict("row")[0]
            arch_config = utils.parse_model_params(best_model["params"])

            if "ID-CNN-XL" in row["name"]:
                fam_param_tmp = family_specific_param["ID-CNN-XL"]
            elif "ID-CNN" in row["name"]:
                fam_param_tmp = family_specific_param["ID-CNN"] 
            elif "BiLSTM-XL" in row["name"]:
                fam_param_tmp = family_specific_param["BiLSTM-XL"] 
            elif "BiLSTM" in row["name"]:
                fam_param_tmp = family_specific_param["BiLSTM"] 
            else:
                raise ValueError(row["name"], "doesn't exist!")
                
            fam_param = fam_param_tmp % arch_config

            tt = table % dict(
                best_val_f1=max_val_f1,
                best_num_params="{:,}".format(best_model["num_trainable_params"]),
                avg_training=(df["training_took"] / 60).mean(),
                name=row["name"],
                lr=best_model["lr"],
                weight_decay=best_model["weight_decay"],
                family_params=fam_param,
                ref="last" if i == len(data)-1 else i
            )

            fw.write(f"{tt} \n\n\n")