"""
Script to find candidate keyword
"""


import re
from environ.constants import PROCESSED_DATA_PATH

with open(f"{PROCESSED_DATA_PATH}/seed/dp_scores.txt", "r", encoding="utf-8") as f:
    dp_scores = f.readlines()


# remove the \n
dp_scores = [re.sub("\n", "", dp_score) for dp_score in dp_scores]

# split the dp_scores using " "
dp_scores = [re.split(" ", dp_score) for dp_score in dp_scores]

# sort the dp_scores by the second element
dp_scores = sorted(dp_scores, key=lambda x: float(x[1]), reverse=True)