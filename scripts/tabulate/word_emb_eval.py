"""
Script to tabulate the results of the word embedding evaluation in latex
"""

from environ.constants import (MODEL, MODEL_NAMING_DICT, TABLE_PATH,
                               WORD_EMB_EVAL)
from scripts.eval.word_emb_eval import df_res

# only keep the row that in WORD_EMB_EVAL
df_res = df_res.loc[WORD_EMB_EVAL]

# tabulate the results in latex line by line
with open(f"{TABLE_PATH}/word_emb_eval.tex", "w", encoding="utf-8") as f:
    f.write(r"\begin{tabular}{ccccccccccc}" + "\n")
    f.write(r"\toprule" + "\n")
    f.write(r"Models & " + " & ".join([f"P@{i}" for i in WORD_EMB_EVAL]) + r"\\" + "\n")
    f.write(r"\midrule" + "\n")
    for model in MODEL:
        f.write(
            MODEL_NAMING_DICT[model]
            + " & "
            + " & ".join([f"{score:.2f}" for score in df_res[model]])
            + r"\\"
            + "\n"
        )
    f.write(r"\bottomrule" + "\n")
    f.write(r"\end{tabular}" + "\n")
