"""This file contains the configuration settings for the market environment."""

from environ.settings import PROJECT_ROOT

# Paths
DATA_PATH = PROJECT_ROOT / "data"
PROCESSED_DATA_PATH = PROJECT_ROOT / "processed_data"
FIGURE_PATH = PROJECT_ROOT / "figures"
TABLE_PATH = PROJECT_ROOT / "tables"

# Model
MODEL= ["cbow", "sg"]
MODEL_NAMING_DICT = {
    "cbow": "W2V+CBOW",
    "sg": "W2V+SG"
}

# word embedding evaluation
WORD_EMB_EVAL = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Global constants
MAXLEN = 256
BATCH_SIZE = 16
EPOCHS = 500
NUM_WORDS = 61711
MAX_FILE_NUM = 40

REP_JARGONS = [
"大饼", "太子", "姨太", "辣条", "油", 
"韭菜", "巨鲸", "瀑布", "割肉", "腰斩", 
"梭哈", "空投", "跳水", "踏空","白皮书", 
"护盘", "挖矿", "土狗","插针", "诱空"
]
