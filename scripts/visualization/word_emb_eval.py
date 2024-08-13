"""
Script to visualize the results of the word embedding evaluation
"""

import matplotlib.pyplot as plt
from environ.constants import MODEL, MODEL_NAMING_DICT, WORD_EMB_EVAL, FIGURE_PATH
from scripts.eval.word_emb_eval import df_res

FONT_SIZE = 12


# only keep the row that in WORD_EMB_EVAL
df_res = df_res.loc[WORD_EMB_EVAL]

# model format mapping
MODEL_FORMAT_MAPPING = {
    "cbow": {"marker": "o", "markersize": 5, "color": "blue"},
    "sg": {"marker": "*", "markersize": 8, "color": "red"},
}

# plot the results
plt.figure(figsize=(6, 4))

for model in MODEL:
    plt.plot(
        WORD_EMB_EVAL,
        df_res[model],
        label=MODEL_NAMING_DICT[model],
        marker=MODEL_FORMAT_MAPPING[model]["marker"],
        markersize=MODEL_FORMAT_MAPPING[model]["markersize"],
        color=MODEL_FORMAT_MAPPING[model]["color"],
    )

plt.xlabel("K, # most relevant words", fontsize=FONT_SIZE)
plt.ylabel("$\overline{P@K}$, average Precision@K", fontsize=FONT_SIZE)

# fix the x-axis from 0 to 100
plt.xticks(WORD_EMB_EVAL, fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.xlim(10, 100)


plt.legend(frameon=False, fontsize=FONT_SIZE)

plt.grid()

plt.tight_layout()
plt.savefig(f"{FIGURE_PATH}/word_emb_eval.pdf")
